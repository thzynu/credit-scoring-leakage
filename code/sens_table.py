import glob
import pandas as pd
d = pd.concat([pd.read_csv(f) for f in glob.glob("results/samematch_*.csv")])
d = d[d["model"].isin(["knn5", "knn25", "rf_depth5", "rf_leaf5", "rf_full"])]
g = d.groupby(["dataset", "model"])["real_same"].agg(["mean", "sem", "count"]).round(4)
g.to_csv("results/tables/bang_nhay_sieu_tham_so.csv")
print(g.to_string())
