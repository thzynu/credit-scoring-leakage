import glob
import pandas as pd
files = glob.glob("results/*_v2.csv")
d = pd.concat([pd.read_csv(f) for f in files])
d = d[d["frac"] == 1.0]
for name, g in d.groupby("dataset"):
    t = g.groupby("model")[["holdout", "cv_dung", "leak_impute", "leak_prep", "leak_both", "phong"]].mean().round(4)
    t["phong_sd"] = g.groupby("model")["phong"].std().round(4)
    t["n_seed"] = g.groupby("model")["seed"].nunique()
    t["hang_that"] = t["holdout"].rank(ascending=False).astype(int)
    t["hang_ro_ri"] = t["leak_both"].rank(ascending=False).astype(int)
    print("=====", name)
    print(t.sort_values("hang_that").to_string())
