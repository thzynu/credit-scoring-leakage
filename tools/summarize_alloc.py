"""Allocation stability of the same-fold decomposition (C1.2). Reads results/alloc_*.csv.
Allocation 0 (the allocation of exp_joint.py) is a reproducibility check and is excluded from
the statistics. Per (dataset, model): mean of each effect over split seeds, allocation SD (mean of the
within-split SDs), split SD (SD of the per-split means), the mean 95% allocation interval, the share
of (split, allocation) runs whose effect has the sign of the mean, and the share of runs in which
the interaction is negative. Also: how stable is the ordering of models by leakage effect.
SE across split seeds uses the split means only (allocation draws within one split are not
independent samples of the data). Outputs: results/tables/new_alloc_stability.csv, new_alloc_order.csv."""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent.parent / "results"
T = R / "tables"
T.mkdir(exist_ok=True)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
fs = sorted(R.glob("alloc_*.csv"))
if not fs:
    raise SystemExit("no results/alloc_*.csv")
A = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
A0 = A[A["alloc_seed"] == 0]
A = A[A["alloc_seed"] > 0]
EFF = ["leak_eff", "score_eff", "joint_eff", "interaction"]
rows = []
for (d, m), g in A.groupby(["dataset", "model"]):
    r = dict(dataset=d, model=m, n_split=g["split_seed"].nunique(), n_alloc=g["alloc_seed"].nunique())
    for e in EFF:
        per = g.groupby("split_seed")[e]
        mean = per.mean().mean()
        r[e] = mean
        r[e + "_alloc_sd"] = per.std(ddof=1).mean()
        r[e + "_split_sd"] = per.mean().std(ddof=1) if g["split_seed"].nunique() > 1 else np.nan
        r[e + "_lo"] = per.quantile(0.025).mean()
        r[e + "_hi"] = per.quantile(0.975).mean()
        r[e + "_same_sign"] = float((np.sign(g[e]) == np.sign(mean)).mean())
    r["inter_negative_share"] = float((g["interaction"] < 0).mean())
    r["leak_share"] = r["leak_eff"] / r["joint_eff"] if abs(r["joint_eff"]) >= 0.01 else np.nan
    r["inter_flag"] = bool(r["interaction_alloc_sd"] > abs(r["interaction"]) or r["inter_negative_share"] < 0.95)
    rows.append(r)
S = pd.DataFrame(rows)
S.round(4).to_csv(T / "new_alloc_stability.csv", index=False)
print("\n===== allocation stability (alloc 1..N)")
print(S[["dataset", "model", "n_split", "n_alloc", "leak_eff", "leak_eff_alloc_sd", "leak_eff_split_sd",
         "leak_eff_lo", "leak_eff_hi", "leak_eff_same_sign", "score_eff", "score_eff_alloc_sd",
         "interaction", "interaction_alloc_sd", "inter_negative_share", "joint_eff_alloc_sd",
         "leak_share", "inter_flag"]].round(4).to_string(index=False))

rows = []
for d, g in A.groupby("dataset"):
    models = sorted(g["model"].unique())
    if len(models) < 2:
        continue
    mean_rank = g.groupby("model")["leak_eff"].mean().rank(ascending=False)
    top_ref = mean_rank.idxmin()
    same_order, same_top, taus = [], [], []
    for (s, a), h in g.groupby(["split_seed", "alloc_seed"]):
        if h["model"].nunique() < len(models):
            continue
        rk = h.set_index("model")["leak_eff"].rank(ascending=False)
        same_order.append(bool((rk.loc[models] == mean_rank.loc[models]).all()))
        same_top.append(rk.idxmin() == top_ref)
        c = d_ = 0
        for x, y in combinations(models, 2):
            c += (rk[x] - rk[y]) * (mean_rank[x] - mean_rank[y]) > 0
            d_ += (rk[x] - rk[y]) * (mean_rank[x] - mean_rank[y]) < 0
        taus.append((c - d_) / (c + d_) if c + d_ else np.nan)
    rows.append(dict(dataset=d, n_models=len(models), n_runs=len(same_order),
                     ordering_by_mean=" > ".join(mean_rank.sort_values().index),
                     share_identical_order=np.mean(same_order), share_same_top=np.mean(same_top),
                     mean_tau_vs_mean_order=np.nanmean(taus)))
O = pd.DataFrame(rows)
O.round(4).to_csv(T / "new_alloc_order.csv", index=False)
print("\n===== stability of the ordering of models by leakage effect")
print(O.round(3).to_string(index=False))

if len(A0):
    print("\n(allocation 0 is stored; compare with results/joint_*_smote.csv with tools/check_alloc0.py)")