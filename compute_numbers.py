"""Paired cost contrasts with bootstrap CIs used in Tables 3, 4 and 6 of the paper.

python analysis/compute_numbers.py --results results/nb03 --out results/numbers.json
"""
import argparse, json
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results/nb03")
ap.add_argument("--out", default="results/numbers.json")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--B", type=int, default=2000)
a = ap.parse_args()

r = pd.read_csv(f"{a.results}/nb03_series_results.csv")
P = r.pivot_table(index="series", columns=["gt", "policy", "variant", "cr"], values="cost")
F = r.pivot_table(index="series", columns=["gt", "policy", "variant", "cr"], values="fill")
rng = np.random.default_rng(a.seed); N = len(P); IDX = rng.integers(0, N, (a.B, N))

def c(x, y):
    """% excess of total cost of policy x over policy y, paired bootstrap 95% CI."""
    x = P[x].values; y = P[y].values
    bs = 100 * (x[IDX].sum(1) / y[IDX].sum(1) - 1)
    return [round(float(100 * (x.sum() / y.sum() - 1)), 1), round(float(np.percentile(bs, 2.5)), 1), round(float(np.percentile(bs, 97.5)), 1)]

pols = [("EMP", "naive"), ("EMP", "masked"), ("EMP", "km"), ("chronos2", "oracle"), ("chronos2", "naive"), ("chronos2", "masked"),
        ("chronos2", "cov"), ("timesfm25", "oracle"), ("timesfm25", "naive"), ("timesfm25", "masked"),
        ("tirex", "oracle"), ("tirex", "naive"), ("tirex", "masked")]
main = {}
for cr in [0.5, 0.8, 0.9]:
    for p, v in pols + [("EMP", "oracle")]:
        k = f"{p}|{v}|{cr}"; g = "hourly"
        main[k] = dict(vsEMPor=c((g, p, v, cr), (g, "EMP", "oracle", cr)) if (p, v) != ("EMP", "oracle") else [0, 0, 0],
                       fill=round(float(F[(g, p, v, cr)].mean()), 3))
        if p != "EMP" and v != "oracle":
            main[k]["vsOwnOr"] = c((g, p, v, cr), (g, p, "oracle", cr))
            main[k]["vsEMPnaive"] = c((g, p, v, cr), (g, "EMP", "naive", cr))
            main[k]["vsKM"] = c((g, p, v, cr), (g, "EMP", "km", cr))
            main[k]["vsOwnNaive"] = c((g, p, v, cr), (g, p, "naive", cr)) if v != "naive" else None
sens = {}
for g in ["hourly", "median", "p80"]:
    for cr in [0.5, 0.8, 0.9]:
        sens[f"{g}|{cr}"] = dict(emptrap=c((g, "EMP", "naive", cr), (g, "EMP", "oracle", cr)),
                                 c2trap=c((g, "chronos2", "naive", cr), (g, "chronos2", "oracle", cr)),
                                 c2vsemp=c((g, "chronos2", "naive", cr), (g, "EMP", "naive", cr)),
                                 covkm=c((g, "chronos2", "cov", cr), (g, "EMP", "km", cr)))
json.dump(dict(main=main, sens=sens), open(a.out, "w"), indent=0)
print(f"wrote {a.out}")

# trajectory summary (Section 4.3): order/demand ratio, first vs last ten simulated days
tr = pd.read_csv(f"{a.results}/nb03_trajectories.csv"); tr = tr[tr["gt"] == "hourly"]
print("\norder/demand, days 31-40 -> 81-90 | stockout rate")
for cr in [0.5, 0.9]:
    for (p, v), x in tr[tr.cr == cr].groupby(["policy", "variant"]):
        e = x[x.day < 40]; l = x[x.day >= 80]
        print(f"  beta={cr} {p:9s} {v:6s} {e.mean_order.mean()/e.mean_demand.mean():.2f} -> {l.mean_order.mean()/l.mean_demand.mean():.2f}"
              f" | {e.stockout_rate.mean():.2f} -> {l.stockout_rate.mean():.2f}")
