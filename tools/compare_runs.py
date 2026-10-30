import sys, glob, os
import pandas as pd
a, b = sys.argv[1], sys.argv[2]
KEYS = ["dataset", "balancer", "model", "seed", "frac", "kfrac", "variant", "split"]
for fa in sorted(glob.glob(os.path.join(a, "*.csv"))):
    name = os.path.basename(fa)
    fb = os.path.join(b, name)
    if not os.path.exists(fb):
        print(f"{name}: MISSING in repro")
        continue
    A, B = pd.read_csv(fa), pd.read_csv(fb)
    keys = [k for k in KEYS if k in A.columns and k in B.columns]
    num = [c for c in A.columns if c in B.columns and c not in keys and c != "giay"
           and pd.api.types.is_numeric_dtype(A[c])]
    m = A.merge(B, on=keys, suffixes=("_a", "_b"), how="outer", indicator=True)
    both = m[m["_merge"] == "both"]
    diffs = [(both[c + "_a"] - both[c + "_b"]).abs() for c in num]
    diff = pd.concat(diffs, axis=1).max().max() if diffs and len(both) else float("nan")
    print(f"{name}: both={len(both)} only_orig={(m['_merge']=='left_only').sum()} "
          f"only_repro={(m['_merge']=='right_only').sum()} max_abs_diff={diff:.1e}")
