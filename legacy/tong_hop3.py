import sys, glob
import pandas as pd
frac = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
d = pd.concat([pd.read_csv(f) for f in glob.glob("results/*_v2.csv")])
d = d[abs(d["frac"] - frac) < 1e-9]
for name, g in d.groupby("dataset"):
    t = g.groupby("model")[["holdout", "leak_impute", "leak_both", "phong"]].mean().round(4)
    t["phong_sd"] = g.groupby("model")["phong"].std().round(4)
    t["n_seed"] = g.groupby("model")["seed"].nunique()
    t["hang_that"] = t["holdout"].rank(ascending=False).astype(int)
    t["hang_ro_ri"] = t["leak_both"].rank(ascending=False).astype(int)
    print("=====", name, "frac", frac, "n_rows", int(g["n_rows"].iloc[0]))
    print(t.sort_values("hang_that").to_string())
