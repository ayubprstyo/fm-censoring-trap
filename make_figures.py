"""Figures 1-3 of the paper (300 dpi PNG, single-column width 88 mm) and an Excel workbook with the plotted data.

python analysis/make_figures.py --results results/nb03 --out figures
"""
import argparse, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results/nb03")
ap.add_argument("--out", default="figures")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "dejavuserif", "font.size": 8})

# ---- Fig. 1: closed loop
fig, ax = plt.subplots(figsize=(3.46, 4.3)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 13)
def box(x, y, w, h, t, fs=6.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", fc="white", ec="black", lw=0.9))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs)
def arr(p, q): ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, lw=0.9, color="black"))
X, W = 2.2, 7.6
box(X, 10.9, W, 1.8, "Observed history:\nsales $s_1,\\dots,s_{t-1}$\nsold-out flags $c_1,\\dots,c_{t-1}$")
box(X, 8.2, W, 1.8, "Forecaster:\nempirical quantile, Kaplan-Meier,\nor zero-shot foundation model")
box(X, 5.5, W, 1.8, "Order quantity\n$Q_t=\\hat F_t^{-1}(\\beta)$,   $\\beta=c_u/(c_u+c_o)$")
box(X, 2.8, W, 1.8, "True demand $d_t$ is realised\n(never observed by the policy)")
box(X, 0.1, W, 1.8, "Sales $s_t=\\min(d_t,Q_t)$,  flag $c_t=1[d_t>Q_t]$\ncost $c_u(d_t-s_t)+c_o(Q_t-s_t)$")
for y in [10.9, 8.2, 5.5, 2.8]: arr((X + W / 2, y - 0.08), (X + W / 2, y - 0.82))
ax.plot([X - 0.08, 0.7, 0.7], [1.0, 1.0, 11.8], color="black", lw=0.9); arr((0.7, 11.8), (X - 0.1, 11.8))
ax.text(0.35, 6.4, "next day: only sales and flags are recorded", rotation=90, ha="center", va="center", fontsize=6.4, style="italic")
fig.savefig(f"{a.out}/Fig1_closed_loop.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close()

# ---- Fig. 2: order trajectories
tr = pd.read_csv(f"{a.results}/nb03_trajectories.csv"); tr = tr[tr["gt"] == "hourly"].copy(); tr["ratio"] = tr.mean_order / tr.mean_demand
series = [("EMP", "oracle", "Empirical, oracle", "black", "-", None), ("EMP", "naive", "Empirical, naive", "black", "--", None),
          ("chronos2", "naive", "Chronos-2, naive", "0.35", "-", "o"), ("timesfm25", "naive", "TimesFM-2.5, naive", "0.35", "-", "s"),
          ("tirex", "naive", "TiRex, naive", "0.35", "-", "^"), ("chronos2", "cov", "Chronos-2, stockout covariate", "0.55", ":", "D")]
fig, axs = plt.subplots(2, 1, figsize=(3.46, 4.9), sharex=True); rows2 = []
for ax, cr, lab in zip(axs, [0.5, 0.9], ["(a)", "(b)"]):
    for p, v, lb, col, ls, mk in series:
        x = tr[(tr.policy == p) & (tr.variant == v) & (tr.cr == cr)].sort_values("day")
        ma = x.ratio.rolling(7, min_periods=1).mean()
        ax.plot(x.day + 1, ma, color=col, ls=ls, marker=mk, markevery=8, ms=3, lw=1.0, label=lb)
        rows2 += [(cr, lb, int(d), float(r_), float(m_)) for d, r_, m_ in zip(x.day + 1, x.ratio, ma)]
    ax.set_title(f"{lab} critical ratio $\\beta$ = {cr}", fontsize=8); ax.grid(alpha=0.3, lw=0.5); ax.set_ylabel("Order / true demand")
axs[1].set_xlabel("Simulation day")
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=2, fontsize=6.6, frameon=False)
fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(f"{a.out}/Fig2_order_trajectories.png", dpi=300); plt.close()
f2 = pd.DataFrame(rows2, columns=["cr", "policy", "day", "order_to_demand", "ma7"])

# ---- Fig. 3: excess cost with CIs
s = pd.read_csv(f"{a.results}/nb03_summary.csv"); s = s[s["gt"] == "hourly"]
def parse(t):
    pt, rest = t.split(" ["); lo, hi = rest.strip("]").split(","); return float(pt), float(lo), float(hi)
items = [("EMP", "naive", "Empirical, naive"), ("EMP", "km", "Empirical, Kaplan-Meier"), ("chronos2", "naive", "Chronos-2, naive"),
         ("timesfm25", "naive", "TimesFM-2.5, naive"), ("tirex", "naive", "TiRex, naive"), ("chronos2", "cov", "Chronos-2, stockout covariate")]
mk = ["o", "s", "^", "v", "D", "P"]; fills = ["black", "white", "0.5", "0.5", "0.5", "white"]
fig, ax = plt.subplots(figsize=(3.5, 3.4)); crs = [0.5, 0.8, 0.9]; rows3 = []
for k, (p, v, lab) in enumerate(items):
    pts = [parse(s[(s.policy == p) & (s.variant == v) & (s.cr == c)].vs_EMP_oracle.iloc[0]) for c in crs]
    xs = np.arange(3) + (k - 2.5) * 0.12
    ax.errorbar(xs, [q[0] for q in pts], yerr=[[q[0] - q[1] for q in pts], [q[2] - q[0] for q in pts]], fmt=mk[k],
                mfc=fills[k], mec="black", ecolor="black", ms=4, lw=0.8, capsize=1.5, label=lab)
    rows3 += [(lab, c, *q) for c, q in zip(crs, pts)]
ax.axhline(0, color="black", lw=0.6); ax.set_xticks(range(3)); ax.set_xticklabels([f"$\\beta$ = {c}" for c in crs])
ax.set_ylabel("Excess cost vs. empirical oracle (%)"); ax.grid(axis="y", alpha=0.3, lw=0.5)
fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=6.5, frameon=False)
fig.tight_layout(rect=(0, 0.17, 1, 1)); fig.savefig(f"{a.out}/Fig3_excess_cost.png", dpi=300); plt.close()
f3 = pd.DataFrame(rows3, columns=["policy", "cr", "excess_pct", "ci_low", "ci_high"])

# ---- Excel workbook with native charts (journal asks for graphs prepared in Excel)
from openpyxl import Workbook
from openpyxl.chart import LineChart, BarChart, Reference
wb = Workbook(); wb.remove(wb.active)
for cr in [0.5, 0.9]:
    p = f2[f2.cr == cr].pivot(index="day", columns="policy", values="ma7"); sh = wb.create_sheet(f"Fig2_beta_{cr}")
    sh.append(["Day"] + list(p.columns))
    for day, row in p.iterrows(): sh.append([int(day)] + [round(float(v), 4) for v in row])
    ch = LineChart(); ch.title = f"Mean order / mean true demand (7-day MA), beta = {cr}"; ch.x_axis.title = "Simulation day"; ch.y_axis.title = "Order / demand"
    ch.add_data(Reference(sh, min_col=2, max_col=1 + len(p.columns), min_row=1, max_row=1 + len(p)), titles_from_data=True)
    ch.set_categories(Reference(sh, min_col=1, min_row=2, max_row=1 + len(p))); ch.width = 20; ch.height = 10; sh.add_chart(ch, "I2")
p = f3.pivot(index="cr", columns="policy", values="excess_pct"); sh = wb.create_sheet("Fig3_data")
sh.append(["beta"] + list(p.columns))
for cr, row in p.iterrows(): sh.append([f"beta = {cr}"] + [float(v) for v in row])
ch = BarChart(); ch.title = "Excess cost vs empirical oracle (%)"; ch.y_axis.title = "%"
ch.add_data(Reference(sh, min_col=2, max_col=1 + len(p.columns), min_row=1, max_row=4), titles_from_data=True)
ch.set_categories(Reference(sh, min_col=1, min_row=2, max_row=4)); ch.width = 20; ch.height = 10; sh.add_chart(ch, "I2")
s2 = wb.create_sheet("Fig3_CI"); s2.append(list(f3.columns))
for r in f3.itertuples(index=False): s2.append(list(r))
wb.save(f"{a.out}/Figures_data_Excel.xlsx")
print(f"figures written to {a.out}/")
