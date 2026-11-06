"""Allocation 0 of exp_alloc.py must reproduce results/joint_<data>_smote.csv exactly."""
from pathlib import Path
import pandas as pd

R = Path(__file__).resolve().parent.parent / "results"
for f in sorted(R.glob("alloc_*.csv")):
    d = f.stem.replace("alloc_", "")
    j = R / f"joint_{d}_smote.csv"
    if not j.exists():
        continue
    a = pd.read_csv(f)
    a = a[a["alloc_seed"] == 0]
    m = a.merge(pd.read_csv(j), left_on=["model", "split_seed"], right_on=["model", "seed"], suffixes=("_a", "_j"))
    print(d, "rows", len(m), {c: float((m[c + "_a"] - m[c + "_j"]).abs().max()) for c in
                              ["cv_in", "leak_eff", "score_eff", "joint_eff", "interaction"]})