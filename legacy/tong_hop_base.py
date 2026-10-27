import glob
import pandas as pd

for f in sorted(glob.glob("results/*_nosmote.csv")):
    b = pd.read_csv(f)
    ds = b["dataset"].iloc[0]
    v = pd.read_csv(f"results/{ds}_v2.csv")
    v = v[v["frac"] == 1.0]
    b = b[b["frac"] == 1.0]
    m = b.merge(v[["model", "seed", "holdout", "leak_both"]], on=["model", "seed"])
    g = m.groupby("model")[["holdout_nosmote", "holdout", "leak_both"]].mean()
    g["delta_smote"] = g["holdout"] - g["holdout_nosmote"]
    g["n_seed"] = m.groupby("model")["seed"].nunique()
    print("=====", ds)
    print(g.round(4).sort_values("holdout_nosmote", ascending=False).to_string())
