import glob
import pandas as pd


def se(s):
    return s.std() / len(s) ** 0.5


v = pd.concat([pd.read_csv(f) for f in glob.glob("results/*_v2.csv")])
v = v[v["frac"] == 1.0].copy()
v["prep_tang"] = v["leak_prep"] - v["cv_dung"]
v["smote_tang"] = v["leak_both"] - v["leak_prep"]
v["phong"] = v["leak_both"] - v["holdout"]
a = v.groupby(["dataset", "model"]).agg(
    n_seed=("seed", "nunique"), holdout_smote=("holdout", "mean"),
    cv_dung=("cv_dung", "mean"), leak_both=("leak_both", "mean"),
    prep_tang=("prep_tang", "mean"), smote_tang=("smote_tang", "mean"),
    smote_tang_se=("smote_tang", se), phong=("phong", "mean"))

b = pd.concat([pd.read_csv(f) for f in glob.glob("results/*_nosmote.csv")])
b = b[b["frac"] == 1.0]
m = b.merge(v[["dataset", "model", "seed", "holdout"]], on=["dataset", "model", "seed"])
m["delta_smote"] = m["holdout"] - m["holdout_nosmote"]
d = m.groupby(["dataset", "model"]).agg(
    holdout_nosmote=("holdout_nosmote", "mean"),
    delta_smote=("delta_smote", "mean"), delta_smote_se=("delta_smote", se))

s = pd.concat([pd.read_csv(f) for f in glob.glob("results/*_sel.csv")])
s = s[abs(s["kfrac"] - 0.25) < 1e-9]
sg = s.groupby(["dataset", "model"]).agg(
    sel_tang=("sel_tang", "mean"), sel_tang_se=("sel_tang", se))

out = a.join(d).join(sg).round(4)
out.to_csv("results/tong_hop_chinh.csv")
print(out.to_string())
