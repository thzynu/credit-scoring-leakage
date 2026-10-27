from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "gmsc.csv")
y = df["SeriousDlqin2yrs"]
X = df.drop(columns="SeriousDlqin2yrs")

def lr(cw=None):
    return LogisticRegression(max_iter=1000, class_weight=cw)

def steps(quantile):
    sc = QuantileTransformer(output_distribution="normal", random_state=0) if quantile else StandardScaler()
    return [("imp", SimpleImputer(strategy="median")), ("sc", sc)]

rows = []
for seed in range(1, 4):
    Xd, Xt, yd, yt = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    cfgs = {
        "plain": Pipeline(steps(False) + [("clf", lr())]),
        "balanced": Pipeline(steps(False) + [("clf", lr("balanced"))]),
        "smote": ImbPipeline(steps(False) + [("smote", SMOTE(random_state=seed)), ("clf", lr())]),
        "quantile_plain": Pipeline(steps(True) + [("clf", lr())]),
        "quantile_smote": ImbPipeline(steps(True) + [("smote", SMOTE(random_state=seed)), ("clf", lr())]),
    }
    for name, p in cfgs.items():
        p.fit(Xd, yd)
        rows.append(dict(method=name, seed=seed, auc=roc_auc_score(yt, p.predict_proba(Xt)[:, 1])))
    print("seed", seed, "done", flush=True)
print(pd.DataFrame(rows).groupby("method")["auc"].mean().round(4).to_string())
