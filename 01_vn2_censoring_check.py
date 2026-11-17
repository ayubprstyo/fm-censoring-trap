# %% [markdown]
# # 01 - VN2 boundary check: is the censoring trap costly in low-volume weekly retail?
# Data: SupChains/VN2 (https://github.com/SupChains/VN2), CC BY 4.0. Runs on CPU in < 1 min.
#   git clone --depth 1 https://github.com/SupChains/VN2.git
#   python notebooks/01_vn2_censoring_check.py --vn2 VN2/Data
#
# Closed loop with the official VN2 cost rules: weekly review, lead time 2 weeks,
# holding 0.2 per unit per week, lost sale 1 per unit (critical ratio 1/1.2).
# "True demand" = sales of series with <= 2% out-of-stock weeks and >= 2 years of history.
# Policies (base-stock, order-up-to = bootstrap quantile of 3-week demand from the last 52 weeks):
#   oracle : learns from true demand
#   naive  : learns from its own (censored) sales
#   masked : drops weeks it knows were censored

# %%
import argparse, numpy as np, pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("--vn2", default="VN2/Data")
args, _ = ap.parse_known_args()
s = pd.read_csv(f"{args.vn2}/Week 8 - 2024-06-03 - Sales.csv").set_index(["Store", "Product"])
f = pd.read_csv(f"{args.vn2}/Week 0 - In Stock.csv").set_index(["Store", "Product"])[s.columns].astype(str).eq("True")

act = (s > 0).cumsum(axis=1) > 0          # active from first positive sale
oos = (~f) & act
print(f"series: {len(s)}, weeks: {s.shape[1]}")
print(f"share of active weeks out of stock: {oos.values.sum() / act.values.sum():.3f}")
for y in sorted(set(pd.to_datetime(s.columns).year)):
    c = s.columns[pd.to_datetime(s.columns).year == y]
    print(f"  {y}: {oos[c].values.sum() / act[c].values.sum():.3f}")
print(f"OOS weeks with zero sales: {((s == 0) & oos).values.sum()} of {oos.values.sum()}")
o = oos.values; a = act.values
print("P(OOS t+1 | OOS t) = %.2f, P(OOS t+1 | in stock t) = %.2f" % (
    (o[:, 1:] & o[:, :-1]).sum() / o[:, :-1].sum(), (o[:, 1:] & ~o[:, :-1] & a[:, :-1]).sum() / (~o[:, :-1] & a[:, :-1]).sum()))
print(f"share of active weeks with zero sales: {((s == 0) & act).values.sum() / act.values.sum():.3f}")
per = oos.sum(axis=1) / act.sum(axis=1)
clean = (per <= 0.02) & (act.sum(axis=1) >= 104)
print(f"clean series (<=2% OOS, >=2y): {clean.sum()}; median of mean weekly sales: {s[clean].mean(axis=1).median():.2f}")

# %%
D = s[clean].values.astype(float)
start = list(s.columns).index("2023-04-03"); T = 52; W = 52; q = 1 / 1.2

def run(mode, quant=q):
    hold = short = served = dem = 0.0
    for d in D:
        obs = list(d[:start]); flag = [False] * start
        def S():
            h = np.array(obs[-W:]); m = np.array(flag[-W:])
            if mode == "oracle": h = d[len(obs) - W:len(obs)]
            elif mode == "masked": h = h[~m] if (~m).sum() >= 8 else h
            sums = np.random.default_rng(0).choice(h, (2000, 3)).sum(1)   # 3-week protection period
            return np.ceil(np.quantile(sums, quant))
        onhand = S(); pipe = [0, 0]
        for t in range(start, start + T):
            onhand += pipe.pop(0)
            pipe.append(max(0, S() - onhand - sum(pipe)))   # arrives in 2 weeks
            sales = min(d[t], onhand); lost = d[t] - sales; onhand -= sales
            hold += 0.2 * onhand; short += lost; served += sales; dem += d[t]
            obs.append(sales); flag.append(onhand == 0 and lost > 0)
    return dict(cost=round(hold + short, 1), holding=round(hold, 1), shortage=round(short, 1), fill=round(served / dem, 3))

res = {m: run(m) for m in ["oracle", "naive", "masked"]}
for m, r in res.items(): print(m, r)
o_, n_ = res["oracle"], res["naive"]
print(f"naive vs oracle: shortage {100*(n_['shortage']/o_['shortage']-1):+.1f}%, fill {100*(n_['fill']-o_['fill']):+.1f} pts, "
      f"total cost {100*(n_['cost']/o_['cost']-1):+.1f}%")
