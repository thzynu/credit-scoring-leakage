import glob
import pandas as pd
d = pd.concat([pd.read_csv(f) for f in glob.glob("results/*_sel.csv")])
g = d.groupby(["dataset", "model", "kfrac"]).agg(
    n_feat=("n_feat", "first"), k=("k", "first"), n_seed=("seed", "nunique"),
    cv_dung=("cv_dung", "mean"), leak_sel=("leak_sel", "mean"),
    holdout=("holdout", "mean"), sel_tang=("sel_tang", "mean"), sel_sd=("sel_tang", "std"))
g["sel_z"] = g["sel_tang"] / (g["sel_sd"] / g["n_seed"] ** 0.5)
print(g.round(4).to_string())
