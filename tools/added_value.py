"""What does the decomposition add beyond the joint inflation? (C1.3)
Reads results/tables/new_decomp_closed.csv (made by summarize_new.py). Within each dataset, finds pairs
of model settings whose JOINT inflation is nearly the same (difference <= --tol) but whose leakage effect
differs by at least --min-leak-diff. For such pairs the conventional difference between leaky and
leakage-free cross-validation cannot tell the models apart, while the decomposition shows that different
mechanisms are at work. Writes results/tables/new_added_value_pairs.csv."""
import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--tol", type=float, default=0.005)
ap.add_argument("--min-leak-diff", type=float, default=0.03)
ap.add_argument("--min-joint", type=float, default=0.03)
args = ap.parse_args()
R = Path(__file__).resolve().parent.parent / "results"
D = pd.read_csv(R / "tables" / "new_decomp_closed.csv")
D = D[~D["model"].str.endswith("_noes")]
rows = []
for d, g in D.groupby("dataset"):
    for (_, a), (_, b) in combinations(g.iterrows(), 2):
        if min(a["joint"], b["joint"]) < args.min_joint or abs(a["joint"] - b["joint"]) > args.tol:
            continue
        if abs(a["leak"] - b["leak"]) < args.min_leak_diff:
            continue
        hi, lo = (a, b) if a["leak"] > b["leak"] else (b, a)
        rows.append(dict(dataset=d, model_leak_dominated=hi["model"], joint_1=hi["joint"], leak_1=hi["leak"],
                         score_1=hi["score"], inter_1=hi["interaction"], model_scoring_dominated=lo["model"],
                         joint_2=lo["joint"], leak_2=lo["leak"], score_2=lo["score"], inter_2=lo["interaction"],
                         joint_diff=abs(a["joint"] - b["joint"]), leak_diff=hi["leak"] - lo["leak"]))
P = pd.DataFrame(rows)
if len(P):
    P = P.sort_values(["dataset", "joint_diff"])
P.round(4).to_csv(R / "tables" / "new_added_value_pairs.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print(P.round(4).to_string(index=False))
print(f"\n{len(P)} pairs; datasets: {sorted(P['dataset'].unique()) if len(P) else []}")