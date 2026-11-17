# %% [markdown]
# # Pilot: does demand censoring cost money in FreshRetailNet-50K?
# Kaggle: Settings -> Internet ON. CPU is enough (~5-10 min).
# Question: when an ordering policy learns from its own censored sales (lost-sales,
# perishable, daily newsvendor), is the cost penalty vs. learning from true demand
# large and clear? If yes -> full study with foundation models. If no -> stop.

# %%
# !pip -q install huggingface_hub pyarrow
import numpy as np, pandas as pd
from huggingface_hub import hf_hub_download

path = hf_hub_download("Dingdong-Inc/FreshRetailNet-50K", "data/train.parquet", repo_type="dataset")
df = pd.read_parquet(path, columns=["store_id", "product_id", "dt", "sale_amount", "stock_hour6_22_cnt"])
df = df.sort_values(["store_id", "product_id", "dt"])
print(df.shape, df.dt.min(), df.dt.max())

# %% [markdown]
# ## 1. How common are stockouts?

# %%
df["oos_day"] = df.stock_hour6_22_cnt > 0
g = df.groupby(["store_id", "product_id"])
per = g.oos_day.mean()
print("share of series-days with any stockout hour: %.3f" % df.oos_day.mean())
print("share with >=4 stockout hours: %.3f" % (df.stock_hour6_22_cnt >= 4).mean())
print("per-series stockout-day share:\n", per.describe(percentiles=[.1, .25, .5, .75, .9]).round(3))
print("series with 0 stockout days:", (per == 0).sum(), "of", len(per))
print("median daily sales (normalised):", df.sale_amount.median())
# Sales on stockout days vs. clean days of the same series
ratio = (df[df.oos_day].groupby(["store_id", "product_id"]).sale_amount.mean()
         / df[~df.oos_day].groupby(["store_id", "product_id"]).sale_amount.mean()).dropna()
print("sales on stockout days / clean days, median: %.2f" % ratio.median())


# %% [markdown]
# ## 2. Closed-loop pilot
# Ground truth = series with no stockout in 90 days (sales = demand).
# Daily newsvendor, perishable (no carry-over), lead time 0.
# Order Q_t = empirical cr-quantile of last W observed days.
#   oracle : learns from true demand
#   naive  : learns from its own sales = min(demand, Q)   <- censoring trap
#   masked : drops days it knows were censored (sold out)
#   km     : Kaplan-Meier quantile, sold-out days treated as 'demand >= sales'
# Underage cost cu = 1; overage co varies -> critical ratio cr = cu/(cu+co).

# %%
def km_quantile(x, cens, q):
    """Kaplan-Meier quantile; cens=True means demand >= x (sold out)."""
    order = np.lexsort((cens, x)); x, cens = x[order], cens[order]
    n = len(x); S = 1.0; i = 0
    while i < n:
        v = x[i]; j = i
        while j < n and x[j] == v: j += 1
        events = (~cens[i:j]).sum(); at_risk = n - i
        if events: S *= 1 - events / at_risk
        if 1 - S >= q: return v
        i = j
    return x.max()

def simulate(D, mode, co, W=28, start=30):
    cr = 1.0 / (1.0 + co)
    tot = dict(under=0.0, over=0.0, demand=0.0, served=0.0)
    for d in D:
        obs = list(d[:start]); cens = [False] * start
        for t in range(start, len(d)):
            if mode == "oracle":
                h = d[t - W:t]
            else:
                h = np.array(obs[-W:]); m = np.array(cens[-W:])
                if mode == "masked" and (~m).sum() >= 7:
                    h = h[~m]
            if mode == "km":
                Q = km_quantile(np.array(obs[-W:]), np.array(cens[-W:]), cr)
            else:
                Q = np.quantile(h, cr)
            sales = min(d[t], Q)
            tot["under"] += d[t] - sales
            tot["over"] += co * (Q - sales)
            tot["demand"] += d[t]; tot["served"] += sales
            obs.append(sales); cens.append(d[t] > Q)
    cost = tot["under"] + tot["over"]
    return dict(cost=cost, fill=tot["served"] / max(tot["demand"], 1e-12))

# Ground-truth pool: fully clean series are rare, so take the cleanest series
# (smallest share of stockout days, threshold chosen to get >= 500 series) and
# reconstruct demand on their few stockout days conservatively:
# demand = max(observed sales, series median on clean days).
n_days = g.size()
for thr in [0.0, 0.02, 0.05, 0.10, 0.20, 0.30]:
    pool = per[(per <= thr) & (n_days >= 80)].index
    if len(pool) >= 500: break
print(f"ground-truth threshold: <= {thr:.0%} stockout days -> {len(pool)} series")
rng = np.random.default_rng(42)
pool = pool[rng.choice(len(pool), size=min(2000, len(pool)), replace=False)]
pool_set = set(pool)
D = []
for k, grp in g:
    if k not in pool_set: continue
    x = grp.sale_amount.values.astype(float); o = grp.oos_day.values
    if (~o).sum() >= 10 and x[~o].sum() > 0:
        x = np.where(o, np.maximum(x, np.median(x[~o])), x)
        D.append(x)
print("series in pilot:", len(D), "| mean daily demand (normalised): %.3f" % np.mean([d.mean() for d in D]))
print("series in pilot:", len(D))

rows = []
for co in [1.0, 0.25, 0.1]:          # cr = 0.50, 0.80, 0.91
    res = {m: simulate(D, m, co) for m in ["oracle", "naive", "masked", "km"]}
    for m, r in res.items():
        rows.append(dict(cr=round(1 / (1 + co), 2), policy=m, cost=round(r["cost"], 1),
                         fill=round(r["fill"], 3),
                         excess_vs_oracle_pct=round(100 * (r["cost"] / res["oracle"]["cost"] - 1), 1)))
out = pd.DataFrame(rows)
print(out.to_string(index=False))

# %% [markdown]
# ## 3. Paired bootstrap of naive - oracle (per-series cost), cr = 0.80

# %%
def per_series(mode, co=0.25):
    return np.array([simulate([d], mode, co)["cost"] for d in D])
c_or, c_nv, c_mk, c_km = per_series("oracle"), per_series("naive"), per_series("masked"), per_series("km")
B = 2000; idx = rng.integers(0, len(D), (B, len(D)))
for name, c in [("naive", c_nv), ("masked", c_mk), ("km", c_km)]:
    diff = (c[idx].sum(1) / c_or[idx].sum(1) - 1) * 100
    print(f"{name} - oracle: {100*(c.sum()/c_or.sum()-1):+.1f}%  95% CI [{np.percentile(diff,2.5):+.1f}, {np.percentile(diff,97.5):+.1f}]")

# %% [markdown]
# ## Decision rule (fixed before looking at results)
# GO if naive - oracle excess >= +5% with CI excluding 0 at cr = 0.80, AND
# fill rate drops by >= 3 points. Otherwise the censoring trap is too weak to carry the paper.
# Copy all printed output back to Claude.
