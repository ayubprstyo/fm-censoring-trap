"""Robustness contrasts from Notebook 05 (equal context length, Tobit benchmark).

Reads results/nb05/nb05_series_results.csv and writes results/nb05/nb05_numbers.json.
Each entry: [percent difference in total cost, 2.5% bound, 97.5% bound]
(paired bootstrap over series, 2,000 resamples, seed 42).

Usage: python analysis/compute_nb05.py [results_dir]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "results"
s = pd.read_csv(root / "nb05" / "nb05_series_results.csv")
P = s.pivot_table(index=["cr", "series"], columns=["policy", "variant"], values="cost", aggfunc="sum")
rng = np.random.default_rng(42)
MODELS = ["chronos2", "timesfm25", "tirex"]


def contrast(cr, a, b, B=2000):
    d = P.loc[cr]
    A, Bv = d[a].to_numpy(), d[b].to_numpy()
    idx = rng.integers(0, len(A), (B, len(A)))
    boot = 100 * (A[idx].sum(1) / Bv[idx].sum(1) - 1)
    return [round(100 * (A.sum() / Bv.sum() - 1), 1), round(float(np.percentile(boot, 2.5)), 1),
            round(float(np.percentile(boot, 97.5)), 1)]


out = {}
for cr in [0.5, 0.8, 0.9]:
    r = {}
    for m in MODELS:
        r[f"{m}|naive28_vs_empnaive"] = contrast(cr, (m, "naive_28"), ("EMP", "naive"))
        r[f"{m}|naive_vs_empnaivefull"] = contrast(cr, (m, "naive"), ("EMP", "naive_full"))
        r[f"{m}|trap28"] = contrast(cr, (m, "naive_28"), (m, "oracle_28"))
        r[f"{m}|naive_vs_tobit"] = contrast(cr, (m, "naive"), ("EMP", "tobit"))
    r["chronos2|cov_vs_tobit"] = contrast(cr, ("chronos2", "cov"), ("EMP", "tobit"))
    r["chronos2|cov_vs_km"] = contrast(cr, ("chronos2", "cov"), ("EMP", "km"))
    r["tobit_vs_oracle"] = contrast(cr, ("EMP", "tobit"), ("EMP", "oracle"))
    r["tobit_vs_km"] = contrast(cr, ("EMP", "tobit"), ("EMP", "km"))
    r["tobit_vs_naive"] = contrast(cr, ("EMP", "tobit"), ("EMP", "naive"))
    r["empnaivefull_trap"] = contrast(cr, ("EMP", "naive_full"), ("EMP", "oracle_full"))
    o = P.loc[cr][("EMP", "oracle")].sum()
    r["vsEMPor"] = {f"{p}|{v}": round(100 * (P.loc[cr][(p, v)].sum() / o - 1), 1) for p, v in P.columns}
    out[str(cr)] = r

(root / "nb05" / "nb05_numbers.json").write_text(json.dumps(out, indent=1))
for cr, r in out.items():
    print(cr, {k: v for k, v in r.items() if k != "vsEMPor"})
