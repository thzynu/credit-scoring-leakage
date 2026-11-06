"""Model-selection regret and the split of the optimism on Lending Club (C3.2, C3.3).

For every seed the 7 main models are compared. Three selections are defined:
  leaky choice   : model with the highest leaky CV AUC (SMOTE before the folds)  [leak_both]
  clean choice   : model with the highest CLEAN VALIDATION AUC available at selection time, never
                   the test set: the correct-pipeline cross-validated AUC [cv_dung] for German,
                   Australian and Taiwan; the random hold-out drawn from the 2013-2015 training years
                   [holdout_rand] for Lending Club
  oracle choice  : model with the best TEST AUC (hold-out; out-of-time 2016 for Lending Club). It uses test
                   results, so it is a retrospective oracle comparison, not a selection procedure.
Test AUC = hold-out AUC (out-of-time AUC for Lending Club). Reported per dataset (mean over seeds, SE):
  excess_regret = test[clean choice] - test[leaky choice]   (AUC lost by selecting with leaky CV)
  oracle_regret_leaky = test[oracle] - test[leaky choice];  oracle_regret_clean = test[oracle] - test[clean choice]
  share_same = share of seeds in which both procedures pick the same model.
Lending Club optimism split (per model, mean over seeds):
  optimism_oot = leaky CV - out-of-time AUC = (leaky CV - random hold-out) + (random hold-out - out-of-time)
  the first term has no temporal shift, the second is the temporal shift.
Outputs: results/tables/new_selection_regret.csv, new_selection_regret_seeds.csv, new_optimism_split.csv."""
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent.parent / "results"
T = R / "tables"
T.mkdir(exist_ok=True)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
MAIN = ["logreg", "knn5", "rf_depth5", "rf_full", "hgb_depth2", "hgb_depth4", "hgb_full"]


def load():
    out = {}
    for d in ["australian", "german", "taiwan"]:
        v = pd.read_csv(R / f"{d}_v2.csv")
        v = v[(v["frac"] == 1.0) & v["model"].isin(MAIN)]
        out[d] = v.rename(columns={"cv_dung": "clean_val", "holdout": "test"})[["model", "seed", "clean_val", "leak_both", "test"]]
    t = pd.read_csv(R / "lendingclub_time.csv")
    t = t[(t["frac"] == 1.0) & (t["split"] == "train2013-2014-2015_test2016") & t["model"].isin(MAIN)]
    out["lendingclub"] = t.rename(columns={"holdout_rand": "clean_val", "holdout_oot": "test"})[
        ["model", "seed", "clean_val", "leak_both", "test"]]
    return out, t


data, tlc = load()
seed_rows, rows = [], []
for d, df in data.items():
    for s, g in df.groupby("seed"):
        if g["model"].nunique() < len(MAIN):
            continue
        g = g.set_index("model")
        lk, cl, orc = g["leak_both"].idxmax(), g["clean_val"].idxmax(), g["test"].idxmax()
        seed_rows.append(dict(dataset=d, seed=s, leaky_choice=lk, clean_choice=cl, oracle_choice=orc,
                              test_leaky=g.loc[lk, "test"], test_clean=g.loc[cl, "test"], test_oracle=g.loc[orc, "test"],
                              excess_regret=g.loc[cl, "test"] - g.loc[lk, "test"],
                              oracle_regret_leaky=g.loc[orc, "test"] - g.loc[lk, "test"],
                              oracle_regret_clean=g.loc[orc, "test"] - g.loc[cl, "test"],
                              same=lk == cl))
S = pd.DataFrame(seed_rows)
S.round(5).to_csv(T / "new_selection_regret_seeds.csv", index=False)
for d, g in S.groupby("dataset"):
    n = len(g)
    se = lambda c: g[c].std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    rows.append(dict(dataset=d, n_seed=n, excess_regret=g["excess_regret"].mean(), excess_se=se("excess_regret"),
                     share_seeds_excess_gt0=(g["excess_regret"] > 0).mean(), share_same_choice=g["same"].mean(),
                     oracle_regret_leaky=g["oracle_regret_leaky"].mean(), oracle_regret_clean=g["oracle_regret_clean"].mean(),
                     leaky_choice_modal=g["leaky_choice"].mode().iloc[0], clean_choice_modal=g["clean_choice"].mode().iloc[0],
                     oracle_choice_modal=g["oracle_choice"].mode().iloc[0]))
Rg = pd.DataFrame(rows)
Rg.round(4).to_csv(T / "new_selection_regret.csv", index=False)
print("===== model-selection regret (per seed, then mean)")
print(Rg.round(4).to_string(index=False))

L = tlc.copy()
L["inflation_vs_random_holdout"] = L["leak_both"] - L["holdout_rand"]
L["temporal_shift"] = L["holdout_rand"] - L["holdout_oot"]
L["optimism_oot"] = L["leak_both"] - L["holdout_oot"]
O = L.groupby("model")[["leak_both", "holdout_rand", "holdout_oot", "inflation_vs_random_holdout", "temporal_shift",
                        "optimism_oot"]].mean().reindex(MAIN).reset_index()
O["n_seed"] = L.groupby("model")["seed"].nunique().reindex(MAIN).values
O["share_temporal"] = O["temporal_shift"] / O["optimism_oot"]
O.round(4).to_csv(T / "new_optimism_split.csv", index=False)
print("\n===== Lending Club: optimism against the out-of-time test = protocol part + temporal shift")
print(O.round(4).to_string(index=False))