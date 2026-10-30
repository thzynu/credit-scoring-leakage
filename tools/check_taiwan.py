import sys, pandas as pd
sys.path.insert(0, r"leakage_project\code")
import exp as E
cfg = E.DATASETS["taiwan"]
df = pd.read_csv(r"leakage_project\data\taiwan.csv")
y = (df[cfg["target"]] == cfg["positive"]).astype(int)
X = df.drop(columns=[cfg["target"]] + cfg["drop"], errors="ignore")
cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
num = [c for c in X.columns if c not in cat]
orig = pd.read_csv(r"leakage_project\results\taiwan_v2.csv")
rep = pd.read_csv(r"repro_full\results\taiwan_v2.csv")
def pick(d, name, seed):
    return d[(d.model == name) & (d.seed == seed) & (d.frac == 1.0)].iloc[0]
for name in ["logreg", "knn5"]:
    seed = 4
    o, r = pick(orig, name, seed), pick(rep, name, seed)
    print(name, seed, "ORIG ", round(o.cv_dung, 6), round(o.leak_prep, 6))
    print(name, seed, "REPRO", round(r.cv_dung, 6), round(r.leak_prep, 6))
    for j in (1, 2, 4):
        v = E.run(E.MODELS[name](), X, y, num, cat, cfg["min_freq"], seed, j)
        print(name, seed, "now jobs=%d" % j, round(v["cv_dung"], 6), round(v["leak_prep"], 6))
