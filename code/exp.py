import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent

# positive = value of the label column treated as the positive class (default = 1)
DATASETS = {
    "taiwan": dict(path="taiwan.csv", target="target", positive=1, drop=[], min_freq=None),
    "german": dict(path="german.csv", target="target", positive=2, drop=[], min_freq=None),
    "australian": dict(path="australian.csv", target="target", positive=1, drop=[], min_freq=None),
    "lendingclub": dict(path="lendingclub_100k.csv", target="Default", positive=1,
                        drop=["id", "issue_d", "year", "zip_code"], min_freq=0.01),
}

MODELS = {
    "logreg": lambda: LogisticRegression(max_iter=1000),
    "hgb_depth2": lambda: HistGradientBoostingClassifier(max_depth=2, max_iter=100, random_state=42),
    "hgb_depth4": lambda: HistGradientBoostingClassifier(max_depth=4, max_iter=100, random_state=42),
    "hgb_full": lambda: HistGradientBoostingClassifier(random_state=42),
    "knn5": lambda: KNeighborsClassifier(n_neighbors=5),
    "knn25": lambda: KNeighborsClassifier(n_neighbors=25),
    "rf_depth5": lambda: RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    "rf_full": lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    "rf_leaf5": lambda: RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=42),
}


def make_prep(num_cols, cat_cols, min_freq):
    parts = []
    if num_cols:
        num = SkPipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())])
        parts.append(("num", num, num_cols))
    if cat_cols:
        cat = SkPipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                               min_frequency=min_freq,
                                               sparse_output=False))])
        parts.append(("cat", cat, cat_cols))
    return ColumnTransformer(parts, sparse_threshold=0)


def impute_all(Xdev, num_cols, cat_cols):
    # impute using statistics from the WHOLE dev set (deliberately leaky)
    Xi = Xdev.copy()
    if num_cols:
        Xi[num_cols] = Xi[num_cols].fillna(Xi[num_cols].median())
    for c in cat_cols:
        if Xi[c].isna().any():
            Xi[c] = Xi[c].fillna(Xi[c].mode().iloc[0])
    return Xi


def run(model, X, y, num_cols, cat_cols, min_freq, seed, jobs):
    Xdev, Xte, ydev, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)

    def sm():
        return SMOTE(random_state=seed)

    def prep():
        return make_prep(num_cols, cat_cols, min_freq)

    def score(est, A, b):
        return cross_val_score(est, A, b, cv=cv, scoring="roc_auc", n_jobs=jobs).mean()

    # CORRECT: preprocessing + SMOTE inside each fold
    pipe = Pipeline([("prep", prep()), ("smote", sm()), ("clf", clone(model))])
    cv_dung = score(pipe, Xdev, ydev)

    # LEAK: imputation (only if the data has NaN)
    leak_impute = np.nan
    if Xdev.isna().any().any():
        leak_impute = score(pipe, impute_all(Xdev, num_cols, cat_cols), ydev)

    # LEAK: preprocessing: fit on the whole dev set, SMOTE still inside the fold
    Xp = prep().fit_transform(Xdev)
    p1 = Pipeline([("smote", sm()), ("clf", clone(model))])
    leak_prep = score(p1, Xp, ydev)

    # LEAK: preprocessing + SMOTE on the whole dev set before splitting folds
    Xr, yr = sm().fit_resample(Xp, ydev)
    leak_both = score(clone(model), Xr, yr)

    # TRUTH: train correctly on dev, score on the hold-out
    pipe.fit(Xdev, ydev)
    holdout = roc_auc_score(yte, pipe.predict_proba(Xte)[:, 1])

    return dict(cv_dung=cv_dung, leak_impute=leak_impute, leak_prep=leak_prep,
                leak_both=leak_both, holdout=holdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, choices=list(DATASETS))
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--nseeds", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--frac", type=float, default=1.0, help="fraction of the data, used for trial runs")
    args = ap.parse_args()

    cfg = DATASETS[args.data]
    df = pd.read_csv(ROOT / "data" / cfg["path"])
    if args.frac < 1.0:
        df = df.sample(frac=args.frac, random_state=0)
    y = (df[cfg["target"]] == cfg["positive"]).astype(int)
    X = df.drop(columns=[cfg["target"]] + cfg["drop"], errors="ignore")
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num_cols = [c for c in X.columns if c not in cat_cols]
    print(f"{args.data}: {X.shape}, default rate={y.mean():.3f}, "
          f"numeric={len(num_cols)}, categorical={len(cat_cols)}, "
          f"max missing rate={X.isna().mean().max():.3f}")

    out = ROOT / "results" / f"{args.data}_v2.csv"
    out.parent.mkdir(exist_ok=True)
    done = set()
    if out.exists():
        old = pd.read_csv(out)
        done = set(zip(old["model"], old["seed"], old["frac"]))

    for name in args.models:
        for seed in range(1, args.nseeds + 1):
            if (name, seed, args.frac) in done:
                print("skipping (already done):", name, seed)
                continue
            t = time.time()
            r = run(MODELS[name](), X, y, num_cols, cat_cols, cfg["min_freq"], seed, args.jobs)
            r.update(dataset=args.data, model=name, seed=seed, frac=args.frac,
                     n_rows=len(df), phong=r["leak_both"] - r["holdout"],
                     giay=round(time.time() - t, 1))
            pd.DataFrame([r]).to_csv(out, mode="a", header=not out.exists(), index=False)
            print(f"{name} seed {seed}: holdout={r['holdout']:.4f} "
                  f"leak_both={r['leak_both']:.4f} inflation={r['phong']:.4f} {r['giay']}s")


if __name__ == "__main__":
    main()
