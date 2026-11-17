# %% [markdown]
# # Notebook 02 — Do time-series foundation models inherit the censoring trap?
# Kaggle: Accelerator = GPU T4 x1 (or x2), Internet ON. Expected run time ~30-60 min.
#
# Closed loop (same environment as the pilot): daily perishable newsvendor, lead time 0,
# lost sales, cu = 1, co = 1/cr - 1, critical ratios cr in {0.5, 0.8, 0.9}.
# Each day the policy orders Q_t = forecast quantile at cr; sales = min(demand, Q_t);
# the policy only ever sees its own sales (censored) unless it is the oracle.
#
# Policies
#   Empirical baselines (rolling 28 days): EMP-oracle, EMP-naive, EMP-masked, EMP-KM
#   Foundation models (zero-shot, full history as context): Chronos-2, TimesFM-2.5, TiRex
#     FM-oracle : context = true demand (model quality without censoring)
#     FM-naive  : context = own sales (the trap)
#     FM-masked : sold-out days set to NaN (native missing-value handling)
#     FM-cov    : Chronos-2 only, sold-out flag as past covariate
# Outputs saved to /kaggle/working: nb02_series_results.csv, nb02_summary.csv

# %%
# !pip -q install "chronos-forecasting==2.3.2" "timesfm[torch]==3.0.2" "tirex-ts==1.4.2" huggingface_hub pyarrow

# %%
import os, time, gc, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

CFG = dict(
    N_SERIES=1000,          # sample size from the ground-truth pool
    POOL_THR=0.20,          # max share of stockout days for ground-truth series (pilot: 0.20)
    RECON="median",         # demand on stockout days: max(sales, clean-day median) | "p80"
    START=30,               # first simulated day (days 0-29 = uncensored warm-up history)
    W_EMP=28,               # window of empirical baselines
    CRS=[0.5, 0.8, 0.9],
    MODELS=["chronos2", "timesfm25", "tirex"],
    SEED=42,
    B=2000,                 # bootstrap resamples
    MOCK=os.environ.get("NB02_MOCK", "0") == "1",   # local test without models/data
)
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
print(CFG)

# %% [markdown]
# ## 1. Data and ground truth

# %%
if not CFG["MOCK"]:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("Dingdong-Inc/FreshRetailNet-50K", "data/train.parquet", repo_type="dataset")
    df = pd.read_parquet(path, columns=["store_id", "product_id", "dt", "sale_amount", "stock_hour6_22_cnt"])
else:
    r = np.random.default_rng(1); rows = []
    for s in range(30):
        for p in range(20):
            lam = r.gamma(2, 1); pr = r.uniform(0.0, 0.3)
            for t in range(90):
                rows.append((s, p, f"d{t:03d}", r.poisson(lam) * 0.1, int(r.random() < pr) * 5))
    df = pd.DataFrame(rows, columns=["store_id", "product_id", "dt", "sale_amount", "stock_hour6_22_cnt"])

df = df.sort_values(["store_id", "product_id", "dt"])
df["oos"] = df.stock_hour6_22_cnt > 0
g = df.groupby(["store_id", "product_id"], sort=False)
share = g.oos.mean(); ndays = g.size()

print("series-days with any stockout hour: %.3f" % df.oos.mean())
print("per-series stockout share:", share.describe(percentiles=[.1, .25, .5, .75, .9]).round(3).to_dict())
print("series with 0 stockout days: %d of %d" % ((share == 0).sum(), len(share)))

L = int(ndays.max())
pool = share[(share <= CFG["POOL_THR"]) & (ndays == L)].index
rng = np.random.default_rng(CFG["SEED"])
pool = pool[rng.choice(len(pool), size=min(CFG["N_SERIES"], len(pool)), replace=False)]
pool_set = set(pool)

D, S_share = [], []
for k, grp in g:
    if k not in pool_set: continue
    x = grp.sale_amount.values.astype(float); o = grp.oos.values
    if (~o).sum() < 10 or x[~o].sum() <= 0: continue
    fill = np.median(x[~o]) if CFG["RECON"] == "median" else np.quantile(x[~o], 0.8)
    D.append(np.where(o, np.maximum(x, fill), x)); S_share.append(o.mean())
D = np.array(D); S_share = np.array(S_share); N, T = D.shape
print(f"ground truth: {N} series x {T} days, pool threshold <= {CFG['POOL_THR']:.0%}, recon = {CFG['RECON']}")
print("original stockout share in sample: mean %.3f, bins 0-5/5-10/10-20%%: %s" % (
    S_share.mean(), np.histogram(S_share, [0, .05, .10, .2001])[0].tolist()))

# %% [markdown]
# ## 2. Simulation engine (vectorised over series)

# %%
def km_quantile(x, cens, q):
    order = np.lexsort((cens, x)); x, cens = x[order], cens[order]
    n = len(x); S = 1.0; i = 0
    while i < n:
        v = x[i]; j = i
        while j < n and x[j] == v: j += 1
        ev = (~cens[i:j]).sum()
        if ev: S *= 1 - ev / (n - i)
        if 1 - S >= q: return v
        i = j
    return x.max()

def run_loop(order_fn, cr, start=CFG["START"]):
    """order_fn(t, obs, cens) -> array (N,) of order quantities for day t.
    obs: (N, t) observed history (true before START, own sales after); cens: (N, t) bool sold-out flags."""
    co = 1.0 / cr - 1.0
    obs = D[:, :start].copy(); cens = np.zeros((N, start), bool)
    under = np.zeros(N); over = np.zeros(N); served = np.zeros(N); q50err = []
    for t in range(start, T):
        Q = np.maximum(np.asarray(order_fn(t, obs, cens), float), 0.0)
        d = D[:, t]; sales = np.minimum(d, Q)
        under += d - sales; over += co * (Q - sales); served += sales
        obs = np.column_stack([obs, sales]); cens = np.column_stack([cens, d > Q])
    dem = D[:, start:].sum(1)
    return dict(cost=under + over, under=under, over=over,
                fill=served / np.maximum(dem, 1e-12), cens_rate=cens[:, start:].mean(1))

def emp_policy(mode, cr, W=CFG["W_EMP"]):
    def f(t, obs, cens):
        out = np.empty(N)
        for i in range(N):
            if mode == "oracle":
                out[i] = np.quantile(D[i, t - W:t], cr)
            else:
                h = obs[i, -W:]; m = cens[i, -W:]
                if mode == "km":
                    out[i] = km_quantile(h, m, cr); continue
                if mode == "masked" and (~m).sum() >= 7: h = h[~m]
                out[i] = np.quantile(h, cr)
        return out
    return f

# %% [markdown]
# ## 3. Foundation-model wrappers: forecast(contexts, covs=None) -> (N, 3) quantiles at 0.5/0.8/0.9

# %%
QL = [0.5, 0.8, 0.9]

def load_model(name):
    if CFG["MOCK"]:
        def f(ctx, covs=None):
            out = []
            for c in ctx:
                c = c[~np.isnan(c)]; out.append(np.quantile(c[-28:], QL))
            return np.array(out)
        return f
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "chronos2":
        from chronos import Chronos2Pipeline
        pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=dev)
        def f(ctx, covs=None):
            if covs is None:
                inp = [c.astype(np.float32) for c in ctx]
            else:
                inp = [{"target": c.astype(np.float32), "past_covariates": {"stockout": v.astype(np.float32)}}
                       for c, v in zip(ctx, covs)]
            q, _ = pipe.predict_quantiles(inp, prediction_length=1, quantile_levels=QL, batch_size=256)
            return np.stack([x.float().cpu().numpy().reshape(-1, len(QL))[0] for x in q])
        return f
    if name == "timesfm25":
        import timesfm, torch
        torch.set_float32_matmul_precision("high")
        m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
        m.compile(timesfm.ForecastConfig(max_context=128, max_horizon=16, normalize_inputs=True,
                                         use_continuous_quantile_head=True, force_flip_invariance=True,
                                         infer_is_positive=True, fix_quantile_crossing=True,
                                         per_core_batch_size=256))
        idx = [5, 8, 9]   # output layout: [mean, q0.1 ... q0.9]
        def f(ctx, covs=None):
            _, q = m.forecast(horizon=1, inputs=[c.astype(np.float32) for c in ctx])
            return np.asarray(q)[:, 0, idx]
        return f
    if name == "tirex":
        from tirex import load_model as tl
        m = tl("NX-AI/TiRex", device=dev)
        try: levels = list(m.config.quantiles)
        except Exception: levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        idx = [levels.index(q) for q in QL]
        def f(ctx, covs=None):
            q, _ = m.forecast(context=[c.astype(np.float32) for c in ctx], prediction_length=1,
                              output_type="numpy", batch_size=512)
            return np.asarray(q)[:, 0, idx]
        return f
    raise ValueError(name)

def fm_policy(fc, variant, cr):
    ci = QL.index(cr)
    def f(t, obs, cens):
        if variant == "oracle":
            ctx = [D[i, :t] for i in range(N)]; covs = None
        elif variant == "naive":
            ctx = [obs[i] for i in range(N)]; covs = None
        elif variant == "masked":
            ctx = [np.where(cens[i], np.nan, obs[i]) for i in range(N)]; covs = None
        elif variant == "cov":
            ctx = [obs[i] for i in range(N)]; covs = [cens[i].astype(float) for i in range(N)]
        return fc(ctx, covs)[:, ci]
    return f

def median_accuracy(fc):
    """Accuracy of the q0.5 forecast on true demand (no censoring), days START..T-1."""
    err = []
    for t in range(CFG["START"], T):
        q = fc([D[i, :t] for i in range(N)])[:, 0]; err.append(q - D[:, t])
    e = np.array(err)
    return np.abs(e).mean(0), e.mean(0)

# %% [markdown]
# ## 4. Run everything

# %%
res = {}   # key (policy, variant, cr) -> dict of per-series arrays
t0 = time.time()
for cr in CFG["CRS"]:
    for mode in ["oracle", "naive", "masked", "km"]:
        res[("EMP", mode, cr)] = run_loop(emp_policy(mode, cr), cr)
print("empirical baselines done in %.0fs" % (time.time() - t0))

acc = {}
for name in CFG["MODELS"]:
    t1 = time.time()
    try:
        fc = load_model(name)
    except Exception as e:
        print(f"!! could not load {name}: {e}"); continue
    acc[name] = median_accuracy(fc)
    variants = ["oracle", "naive", "masked"] + (["cov"] if name == "chronos2" else [])
    for cr in CFG["CRS"]:
        for v in variants:
            try:
                res[(name, v, cr)] = run_loop(fm_policy(fc, v, cr), cr)
            except Exception as e:
                print(f"!! {name}/{v}/cr={cr} failed: {e}")
    del fc; gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception: pass
    print(f"{name} done in {time.time()-t1:.0f}s")

# %% [markdown]
# ## 5. Results

# %%
rows = []
for (p, v, cr), r in res.items():
    for i in range(N):
        rows.append((p, v, cr, i, S_share[i], r["cost"][i], r["under"][i], r["over"][i], r["fill"][i], r["cens_rate"][i]))
ser = pd.DataFrame(rows, columns=["policy", "variant", "cr", "series", "orig_oos_share", "cost", "under", "over", "fill", "cens_rate"])
ser.to_csv(f"{OUT}/nb02_series_results.csv", index=False)

rng = np.random.default_rng(CFG["SEED"]); IDX = rng.integers(0, N, (CFG["B"], N))
def paired(a, b):
    """% excess of total cost a over b, with paired bootstrap 95% CI."""
    pt = 100 * (a.sum() / b.sum() - 1)
    bs = 100 * (a[IDX].sum(1) / b[IDX].sum(1) - 1)
    return pt, np.percentile(bs, 2.5), np.percentile(bs, 97.5)

summ = []
for (p, v, cr), r in res.items():
    ref = res[("EMP", "oracle", cr)]["cost"]
    own = res.get((p, "oracle", cr), {}).get("cost")
    e_ref = paired(r["cost"], ref)
    e_own = paired(r["cost"], own) if own is not None else (np.nan,) * 3
    summ.append(dict(policy=p, variant=v, cr=cr, cost=round(r["cost"].sum(), 1),
                     fill=round(r["fill"].mean(), 3), cens_rate=round(r["cens_rate"].mean(), 3),
                     vs_EMP_oracle=f"{e_ref[0]:+.1f} [{e_ref[1]:+.1f},{e_ref[2]:+.1f}]",
                     vs_own_oracle=f"{e_own[0]:+.1f} [{e_own[1]:+.1f},{e_own[2]:+.1f}]"))
summ = pd.DataFrame(summ).sort_values(["cr", "policy", "variant"])
summ.to_csv(f"{OUT}/nb02_summary.csv", index=False)
pd.set_option("display.width", 200)
print(summ.to_string(index=False))

# %%
print("\nMedian-forecast accuracy on true demand (no censoring), mean over series:")
e_or = res[("EMP", "oracle", 0.5)]
for name, (mae, bias) in acc.items():
    print(f"  {name:10s} MAE {mae.mean():.4f}  bias {bias.mean():+.4f}  (mean demand {D[:, CFG['START']:].mean():.4f})")

# %% [markdown]
# ## 6. Key paired contrasts (cr = 0.8 unless stated)

# %%
def show(label, a, b):
    if a in res and b in res:
        pt, lo, hi = paired(res[a]["cost"], res[b]["cost"])
        print(f"{label:55s} {pt:+6.1f}%  [{lo:+.1f}, {hi:+.1f}]")

for cr in CFG["CRS"]:
    print(f"\n--- cr = {cr}")
    show("EMP trap: naive vs oracle", ("EMP", "naive", cr), ("EMP", "oracle", cr))
    show("EMP KM vs oracle", ("EMP", "km", cr), ("EMP", "oracle", cr))
    for m in CFG["MODELS"]:
        show(f"{m} oracle vs EMP oracle (model quality)", (m, "oracle", cr), ("EMP", "oracle", cr))
        show(f"{m} trap: naive vs own oracle", (m, "naive", cr), (m, "oracle", cr))
        show(f"{m} naive vs EMP naive", (m, "naive", cr), ("EMP", "naive", cr))
        show(f"{m} masked vs own naive", (m, "masked", cr), (m, "naive", cr))
        show(f"{m} masked vs EMP KM", (m, "masked", cr), ("EMP", "km", cr))
    show("chronos2 cov vs own naive", ("chronos2", "cov", cr), ("chronos2", "naive", cr))
    show("chronos2 cov vs EMP KM", ("chronos2", "cov", cr), ("EMP", "km", cr))

# %% [markdown]
# ## 7. Sensitivity: trap size by the series' original stockout share (ground-truth quality)

# %%
bins = [(0, .05), (.05, .10), (.10, .2001)]
for cr in [0.8]:
    for p in ["EMP"] + CFG["MODELS"]:
        if (p, "naive", cr) not in res: continue
        line = []
        for lo, hi in bins:
            m = (S_share >= lo) & (S_share < hi)
            if m.sum() < 30: line.append(f"{lo:.0%}-{hi:.0%}: n<30"); continue
            a = res[(p, "naive", cr)]["cost"][m]; b = res[(p, "oracle", cr)]["cost"][m]
            line.append(f"{lo:.0%}-{hi:.0%} (n={m.sum()}): {100*(a.sum()/b.sum()-1):+.1f}%")
        print(f"{p:10s} trap by ground-truth bin | " + " | ".join(line))
print(f"\nTotal run time {time.time()-t0:.0f}s. Copy ALL printed output back to Claude; download nb02_summary.csv.")
