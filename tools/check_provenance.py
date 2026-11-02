"""Check that exp_provenance.py reproduces the stored same-fold results exactly.

Compares, on the (model, seed) pairs present in both files:
  cv_in         vs cv_in_same   of results/samefold_<data>_smote.csv
  leak_all      vs leak_same    of results/samefold_<data>_smote.csv
  leak_matched  vs leak_same    of results/samematch_<data>_smote.csv
Expected: max difference 0 for German and Australian; about 1e-5 or less for Taiwan and
Lending Club (the parallel-jobs effect described in the README).

Usage (from the project root):  python tools/check_provenance.py
"""
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parent.parent / "results"
for f in sorted(R.glob("provenance_*_smote.csv")):
    data = f.stem.split("_")[1]
    p = pd.read_csv(f)
    for tag, ref_file, cols in [("samefold", f"samefold_{data}_smote.csv",
                                 [("cv_in", "cv_in_same"), ("leak_all", "leak_same")]),
                                ("samematch", f"samematch_{data}_smote.csv",
                                 [("leak_matched", "leak_same")])]:
        ref = R / ref_file
        if not ref.exists():
            print(f"{data}: {ref_file} missing, skipped")
            continue
        m = p.merge(pd.read_csv(ref), on=["model", "seed"], suffixes=("", "_ref"))
        if m.empty:
            print(f"{data} vs {tag}: no common (model, seed)")
            continue
        for a, b in cols:
            b = b if b != a else b + "_ref"
            print(f"{data} vs {tag}: {a:13s} rows={len(m):3d} max|diff|={(m[a] - m[b]).abs().max():.2e}")
