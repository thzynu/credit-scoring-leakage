import glob
import pandas as pd
rows = []
for f in sorted(glob.glob("results/*_nosmote.csv")):
    b = pd.read_csv(f)
    ds = b["dataset"].iloc[0]
    v = pd.read_csv(f"results/{ds}_v2.csv")
    m = b[b["frac"] == 1.0].merge(
        v[v["frac"] == 1.0][["model", "seed", "holdout"]], on=["model", "seed"])
    m["d"] = m["holdout"] - m["holdout_nosmote"]
    for name, g in m.groupby("model"):
        rows.append(dict(dataset=ds, model=name, n=len(g), mean=g["d"].mean(),
                         se=g["d"].std() / len(g) ** 0.5, n_neg=int((g["d"] < 0).sum())))
t = pd.DataFrame(rows)
t["z"] = t["mean"] / t["se"]
print(t.round(4).to_string(index=False))
