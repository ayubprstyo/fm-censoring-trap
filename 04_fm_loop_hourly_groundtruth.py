# %% [markdown]
# # Notebook 03 — Hourly-based ground truth, spiral trajectories, NaN diagnostics
# Kaggle: Accelerator = GPU T4, Internet ON. Expected run time ~60-90 min.
#
# Why: in Notebook 02, 993/1000 ground-truth series had 10-20% stockout days whose demand
# was filled with a constant clean-day median. Here demand on stockout days is reconstructed
# from the HOURLY data: sales in the hours the item was still in stock, scaled up by the
# series' normal intraday sales profile. Median / p80 fills are kept as sensitivity checks.
#
# Same closed loop as Notebook 02 (daily perishable newsvendor, lost sales, cu = 1, co = 1/cr - 1).
# Outputs in /kaggle/working: nb03_summary.csv, nb03_series_results.csv,
# nb03_trajectories.csv, nb03_nan_diagnostics.csv, nb03_groundtruth_check.txt

# %%
import subprocess, sys, os
if os.environ.get("NB03_MOCK", "0") != "1":
    try:
        import chronos, timesfm, tirex  # noqa
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "chronos-forecasting==2.3.2",
                               "timesfm[torch]==3.0.2", "tirex-ts==1.4.2", "huggingface_hub", "pyarrow"])
    import chronos, timesfm, tirex  # fails loudly if installation did not work
    print("model packages OK")

# %%
import os, time, gc, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

CFG = dict(
    N_SERIES=1000, POOL_THR=0.20, START=30, W_EMP=28,
    CRS=[0.5, 0.8, 0.9], MODELS=["chronos2", "timesfm25", "tirex"],
    MIN_COVER=0.20,          # min share of normal daily sales covered by in-stock hours; else fallback
    PRIOR_DAYS=10,           # shrinkage of series hourly profile towards pooled profile
    SEED=42, B=2000,
    MOCK=os.environ.get("NB03_MOCK", "0") == "1",
)
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
LOG = open(f"{OUT}/nb03_groundtruth_check.txt", "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.write(s + "\n"); LOG.flush()
print(CFG)

# %% [markdown]
# ## 1. Data, stockout encoding check

# %%
COLS = ["store_id", "product_id", "dt", "sale_amount", "stock_hour6_22_cnt", "hours_sale", "hours_stock_status"]
if not CFG["MOCK"]:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("Dingdong-Inc/FreshRetailNet-50K", "data/train.parquet", repo_type="dataset")
    df = pd.read_parquet(path, columns=COLS)
else:
    r = np.random.default_rng(1); rows = []
    prof = np.r_[np.zeros(6), np.ones(16), np.zeros(2)]; prof /= prof.sum()
    for s in range(30):
        for p in range(20):
            lam = r.gamma(2, 1) * 10; pr = r.uniform(0.0, 0.3)
            for t in range(90):
                hs = r.poisson(lam * prof * (1 + 0.3 * (t % 7 == 5))).astype(float)
                st = np.zeros(24, int)
                if r.random() < pr:
                    cut = r.integers(8, 21); st[cut:] = 1; hs[cut:] = 0
                rows.append((s, p, f"d{t:03d}", hs.sum() / 10, int(st[6:22].sum()), hs / 10, st))
    df = pd.DataFrame(rows, columns=COLS)

df = df.sort_values(["store_id", "product_id", "dt"]).reset_index(drop=True)
HS = np.stack(df.hours_sale.values).astype(float)          # (rows, 24)
ST = np.stack(df.hours_stock_status.values).astype(int)     # (rows, 24)

# Which value of hours_stock_status means "out of stock"? Pick the one consistent with stock_hour6_22_cnt.
agree = {v: np.mean((ST[:, 6:22] == v).sum(1) == df.stock_hour6_22_cnt.values) for v in [0, 1]}
OOS_VAL = max(agree, key=agree.get)
log("hours_stock_status agreement with stock_hour6_22_cnt:", {k: round(v, 4) for k, v in agree.items()}, "-> OOS value =", OOS_VAL)
ratio = HS.sum(1) / np.where(df.sale_amount.values > 0, df.sale_amount.values, np.nan)
log("sum(hours_sale)/sale_amount: median %.3f, IQR [%.3f, %.3f]" % tuple(np.nanpercentile(ratio, [50, 25, 75])))

OOSH = (ST == OOS_VAL)                         # hourly out-of-stock mask
OOSH[:, :6] = False; OOSH[:, 22:] = False      # only the 06-22 trading window counts (night flags = closed)
log("rows with OOS flag outside 06-22 (ignored): %.3f" % (ST[:, list(range(6)) + [22, 23]] == OOS_VAL).any(1).mean())
df["oos"] = OOSH.any(1)
df["oos622"] = df.stock_hour6_22_cnt > 0
g = df.groupby(["store_id", "product_id"], sort=False)
share = g.oos622.mean(); ndays = g.size()
log("series-days with any stockout hour (6-22): %.3f; with >=4 h: %.3f" % (df.oos622.mean(), (df.stock_hour6_22_cnt >= 4).mean()))
log("per-series stockout-day share:", share.describe(percentiles=[.1, .25, .5, .75, .9]).round(3).to_dict())
log("series with 0 stockout days: %d of %d" % ((share == 0).sum(), len(share)))

# %% [markdown]
# ## 2. Ground truth: hourly reconstruction (primary), median / p80 (sensitivity)

# %%
L = int(ndays.max())
pool = share[(share <= CFG["POOL_THR"]) & (ndays == L)].index
rng = np.random.default_rng(CFG["SEED"])
pool = pool[rng.choice(len(pool), size=min(CFG["N_SERIES"], len(pool)), replace=False)]
idx_of = g.indices
rows_idx = [idx_of[k] for k in pool]

# pooled intraday profile from fully in-stock days of the pool
clean_rows = np.concatenate([ri[~OOSH[ri].any(1)] for ri in rows_idx])
p_glob = HS[clean_rows].sum(0); p_glob = p_glob / p_glob.sum()

def build_D(method):
    D, S_share, n_fb, n_oos, uplift = [], [], 0, 0, []
    for ri in rows_idx:
        x = df.sale_amount.values[ri].astype(float)
        o = df.oos622.values[ri]
        clean = ~o
        if clean.sum() < 10 or x[clean].sum() <= 0: continue
        med = np.median(x[clean])
        if method == "median":
            d = np.where(o, np.maximum(x, med), x)
        elif method == "p80":
            d = np.where(o, np.maximum(x, np.quantile(x[clean], 0.8)), x)
        else:  # hourly
            hs = HS[ri]; oh = OOSH[ri]
            tot = hs[clean].sum()
            p_s = hs[clean].sum(0) / tot if tot > 0 else p_glob
            w = clean.sum() / (clean.sum() + CFG["PRIOR_DAYS"])
            p = w * p_s + (1 - w) * p_glob
            # convert hourly units to daily sale_amount units
            scale = x[clean].sum() / max(hs[clean].sum(), 1e-12)
            d = x.copy()
            for j in np.where(o)[0]:
                n_oos += 1
                inst = ~oh[j]; cover = p[inst].sum()
                if cover >= CFG["MIN_COVER"]:
                    est = hs[j, inst].sum() * scale / cover
                else:
                    est = med; n_fb += 1
                d[j] = max(x[j], est)
                if x[j] > 0: uplift.append(d[j] / x[j])
        D.append(d); S_share.append(o.mean())
    if method == "hourly":
        log(f"hourly recon: {n_oos} stockout days, fallback to median on {n_fb} ({100*n_fb/max(n_oos,1):.1f}%); "
            f"uplift demand/sales on stockout days with sales: median {np.median(uplift):.2f}, p90 {np.percentile(uplift,90):.2f}")
    return np.array(D), np.array(S_share)

GT = {m: build_D(m) for m in ["hourly", "median", "p80"]}
D, S_share = GT["hourly"]; N, T = D.shape
log(f"ground truth: {N} series x {T} days; stockout share mean {S_share.mean():.3f}; "
    f"bins 0-5/5-10/10-20%: {np.histogram(S_share, [0, .05, .10, .2001])[0].tolist()}")
for m in ["median", "p80"]:
    Dm = GT[m][0]; od = Dm != GT["hourly"][0]
    log(f"  {m} vs hourly on reconstructed days: ratio of means {Dm[od].mean() / max(D[od].mean(),1e-12):.2f}; corr {np.corrcoef(Dm[od], D[od])[0,1]:.2f}")

# %% [markdown]
# ## 3. Simulation engine (records daily trajectories)

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

def run_loop(Dx, order_fn, cr, start=CFG["START"]):
    n, T_ = Dx.shape; co = 1.0 / cr - 1.0
    obs = Dx[:, :start].copy(); cens = np.zeros((n, start), bool)
    under = np.zeros(n); over = np.zeros(n); served = np.zeros(n); traj = []
    for t in range(start, T_):
        Q = np.maximum(np.asarray(order_fn(Dx, t, obs, cens), float), 0.0)
        d = Dx[:, t]; sales = np.minimum(d, Q)
        under += d - sales; over += co * (Q - sales); served += sales
        traj.append((t, Q.mean(), sales.mean(), d.mean(), (d > Q).mean(), np.median(Q / np.maximum(d.mean(), 1e-12))))
        obs = np.column_stack([obs, sales]); cens = np.column_stack([cens, d > Q])
    return dict(cost=under + over, under=under, over=over,
                fill=served / np.maximum(Dx[:, start:].sum(1), 1e-12),
                cens_rate=cens[:, start:].mean(1), traj=traj)

def emp_policy(mode, cr, W=CFG["W_EMP"]):
    def f(Dx, t, obs, cens):
        out = np.empty(len(Dx))
        for i in range(len(Dx)):
            if mode == "oracle":
                out[i] = np.quantile(Dx[i, t - W:t], cr); continue
            h = obs[i, -W:]; m = cens[i, -W:]
            if mode == "km": out[i] = km_quantile(h, m, cr); continue
            if mode == "masked" and (~m).sum() >= 7: h = h[~m]
            out[i] = np.quantile(h, cr)
        return out
    return f

# %% [markdown]
# ## 4. Foundation-model wrappers -> (n, 3) quantiles at 0.5 / 0.8 / 0.9

# %%
QL = [0.5, 0.8, 0.9]

def load_model(name):
    if CFG["MOCK"]:
        def f(ctx, covs=None):
            return np.array([np.quantile(c[~np.isnan(c)][-28:], QL) for c in ctx])
        return f
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "chronos2":
        from chronos import Chronos2Pipeline
        pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=dev)
        def f(ctx, covs=None):
            if covs is None: inp = [c.astype(np.float32) for c in ctx]
            else: inp = [{"target": c.astype(np.float32), "past_covariates": {"stockout": v.astype(np.float32)}}
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
        idx = [5, 8, 9]
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
    def f(Dx, t, obs, cens):
        n = len(Dx)
        if variant == "oracle": ctx = [Dx[i, :t] for i in range(n)]; covs = None
        elif variant == "naive": ctx = [obs[i] for i in range(n)]; covs = None
        elif variant == "masked": ctx = [np.where(cens[i], np.nan, obs[i]) for i in range(n)]; covs = None
        elif variant == "cov": ctx = [obs[i] for i in range(n)]; covs = [cens[i].astype(float) for i in range(n)]
        return fc(ctx, covs)[:, ci]
    return f

def nan_diagnostic(name, fc, Dx, n=300, t=60, frac=0.15):
    """How does each model react to NaN gaps vs. the same context with the values present?"""
    r = np.random.default_rng(0); sub = Dx[:n, :t]
    m = r.random(sub.shape) < frac; m[:, :5] = False
    m[: n // 2, -3:] = True                          # half the series end in a gap (as after a sell-out)
    full = [s for s in sub]
    nanc = [np.where(mm, np.nan, s) for s, mm in zip(sub, m)]
    drop = [s[~mm] for s, mm in zip(sub, m)]
    interp = []
    for s, mm in zip(sub, m):
        y = s.copy(); xs = np.arange(len(y)); y[mm] = np.interp(xs[mm], xs[~mm], s[~mm]); interp.append(y)
    base = fc(full)[:, 1]
    out = {}
    for lab, ctx in [("nan", nanc), ("dropped", drop), ("interp", interp)]:
        q = fc(ctx)[:, 1]
        for grp, sel in [("ends_in_gap", slice(0, n // 2)), ("gap_inside", slice(n // 2, n))]:
            out[(name, lab, grp)] = np.median(q[sel] / np.maximum(base[sel], 1e-9))
    return out

# %% [markdown]
# ## 5. Run: primary (hourly GT, all policies), sensitivity (median / p80 GT, subset), diagnostics

# %%
res, trajs, diag = {}, [], {}
t0 = time.time()
def store(gt, p, v, cr, r):
    res[(gt, p, v, cr)] = r
    for (t, q, s, d, c, qr) in r["traj"]:
        trajs.append((gt, p, v, cr, t, q, s, d, c, qr))

for gt in ["hourly", "median", "p80"]:
    Dx = GT[gt][0]
    for cr in CFG["CRS"]:
        for mode in ["oracle", "naive", "masked", "km"]:
            store(gt, "EMP", mode, cr, run_loop(Dx, emp_policy(mode, cr), cr))
print("empirical baselines done in %.0fs" % (time.time() - t0))

for name in CFG["MODELS"]:
    t1 = time.time()
    try: fc = load_model(name)
    except Exception as e:
        print("!" * 70 + f"\n!! could not load {name}: {e!r}\n" + "!" * 70); continue
    try: diag.update(nan_diagnostic(name, fc, D))
    except Exception as e: print(f"!! NaN diagnostic {name} failed: {e}")
    for gt in ["hourly", "median", "p80"]:
        if gt != "hourly" and name != "chronos2": continue      # sensitivity runs: Chronos-2 only
        variants = ["oracle", "naive", "masked"] + (["cov"] if name == "chronos2" else [])
        if gt != "hourly": variants = ["oracle", "naive", "cov"]
        for cr in CFG["CRS"]:
            for v in variants:
                try: store(gt, name, v, cr, run_loop(GT[gt][0], fm_policy(fc, v, cr), cr))
                except Exception as e: print(f"!! {gt}/{name}/{v}/cr={cr} failed: {e}")
    del fc; gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception: pass
    print(f"{name} done in {time.time()-t1:.0f}s")

# %% [markdown]
# ## 6. Results

# %%
rows = []
for (gt, p, v, cr), r in res.items():
    Sx = GT[gt][1]
    for i in range(len(r["cost"])):
        rows.append((gt, p, v, cr, i, Sx[i], r["cost"][i], r["under"][i], r["over"][i], r["fill"][i], r["cens_rate"][i]))
pd.DataFrame(rows, columns=["gt", "policy", "variant", "cr", "series", "orig_oos_share", "cost", "under", "over",
                            "fill", "cens_rate"]).to_csv(f"{OUT}/nb03_series_results.csv", index=False)
pd.DataFrame(trajs, columns=["gt", "policy", "variant", "cr", "day", "mean_order", "mean_sales", "mean_demand",
                             "stockout_rate", "median_order_to_mean_demand"]).to_csv(f"{OUT}/nb03_trajectories.csv", index=False)
pd.DataFrame([(k[0], k[1], k[2], v) for k, v in diag.items()],
             columns=["model", "context", "group", "median_q80_ratio_vs_full"]).to_csv(f"{OUT}/nb03_nan_diagnostics.csv", index=False)

rng = np.random.default_rng(CFG["SEED"]); IDX = rng.integers(0, N, (CFG["B"], N))
def paired(a, b):
    pt = 100 * (a.sum() / b.sum() - 1); bs = 100 * (a[IDX].sum(1) / b[IDX].sum(1) - 1)
    return pt, np.percentile(bs, 2.5), np.percentile(bs, 97.5)
fmt = lambda e: f"{e[0]:+.1f} [{e[1]:+.1f},{e[2]:+.1f}]"

summ = []
for (gt, p, v, cr), r in res.items():
    ref = res[(gt, "EMP", "oracle", cr)]["cost"]; own = res.get((gt, p, "oracle", cr), {}).get("cost")
    summ.append(dict(gt=gt, policy=p, variant=v, cr=cr, cost=round(r["cost"].sum(), 1),
                     fill=round(r["fill"].mean(), 3), cens_rate=round(r["cens_rate"].mean(), 3),
                     vs_EMP_oracle=fmt(paired(r["cost"], ref)),
                     vs_own_oracle=fmt(paired(r["cost"], own)) if own is not None else ""))
summ = pd.DataFrame(summ).sort_values(["gt", "cr", "policy", "variant"])
summ.to_csv(f"{OUT}/nb03_summary.csv", index=False)
pd.set_option("display.width", 220)
print(summ[summ["gt"] == "hourly"].to_string(index=False))

# %%
def show(label, a, b):
    if a in res and b in res: print(f"{label:58s} {fmt(paired(res[a]['cost'], res[b]['cost']))}")
for gt in ["hourly", "median", "p80"]:
    for cr in CFG["CRS"]:
        print(f"\n--- GT = {gt}, cr = {cr}")
        show("EMP trap: naive vs oracle", (gt, "EMP", "naive", cr), (gt, "EMP", "oracle", cr))
        show("EMP KM vs oracle", (gt, "EMP", "km", cr), (gt, "EMP", "oracle", cr))
        for m in CFG["MODELS"]:
            show(f"{m} oracle vs EMP oracle", (gt, m, "oracle", cr), (gt, "EMP", "oracle", cr))
            show(f"{m} trap: naive vs own oracle", (gt, m, "naive", cr), (gt, m, "oracle", cr))
            show(f"{m} naive vs EMP naive", (gt, m, "naive", cr), (gt, "EMP", "naive", cr))
            show(f"{m} masked vs own naive", (gt, m, "masked", cr), (gt, m, "naive", cr))
        show("chronos2 cov vs own naive", (gt, "chronos2", "cov", cr), (gt, "chronos2", "naive", cr))
        show("chronos2 cov vs EMP KM", (gt, "chronos2", "cov", cr), (gt, "EMP", "km", cr))

# %%
print("\nSpiral check (hourly GT, cr = 0.5): mean order / mean demand in days 31-40 vs 81-90")
tr = pd.DataFrame(trajs, columns=["gt", "policy", "variant", "cr", "day", "q", "s", "d", "so", "qr"])
tr = tr[(tr["gt"] == "hourly") & (tr["cr"] == 0.5)]
for (p, v), x in tr.groupby(["policy", "variant"]):
    a = x[x.day < CFG["START"] + 10]; b = x[x.day >= T - 10]
    print(f"  {p:10s} {v:7s} early {a.q.mean()/a.d.mean():.2f}  late {b.q.mean()/b.d.mean():.2f}  "
          f"stockout rate early {a.so.mean():.2f} late {b.so.mean():.2f}")

print("\nNaN handling diagnostic (median q0.8 relative to the same context without gaps):")
for k, v in diag.items(): print(f"  {k[0]:10s} {k[1]:8s} {k[2]:12s} {v:.2f}")
print(f"\nTotal run time {time.time()-t0:.0f}s. Send ALL printed output + the nb03_*.csv / .txt files (zip) to Claude.")
LOG.close()
