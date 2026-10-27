import pandas as pd
d = pd.read_csv("results/lendingclub_time.csv")
d = d[d["frac"] == 1.0]
t = d.groupby("model")[["holdout_rand", "holdout_oot", "leak_both"]].mean().round(4)
t["phong_oot"] = (t["leak_both"] - t["holdout_oot"]).round(4)
t["rand_hon_oot"] = (t["holdout_rand"] - t["holdout_oot"]).round(4)
for c in ["holdout_rand", "holdout_oot", "leak_both"]:
    t["hang_" + c] = t[c].rank(ascending=False).astype(int)
t["n_seed"] = d.groupby("model")["seed"].nunique()
print(t.sort_values("hang_holdout_oot").to_string())
