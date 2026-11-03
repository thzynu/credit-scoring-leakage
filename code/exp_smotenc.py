"""Categorical-aware oversampling: SMOTE on one-hot columns (as in the paper) versus SMOTENC.
SMOTENC works on ordinal-coded categoricals (synthetic categories are the majority category of the
nearest neighbours, so no fractional dummies); one-hot encoding is applied AFTER oversampling.
Both variants use the same seeds, folds and models. tang = cv_before - cv_in; phong = cv_before - holdout.
Output: results/smotenc_<data>.csv (column balancer = smote | smotenc; resumable)."""
import argparse
import time
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.pipeline import Pipeline

import exp as E

ROOT = Path(__file__).resolve().parent.parent


def prep_ord(num_cols, cat_cols):
    parts = []
    if num_cols:
        parts.append(("num", SkPipeline([("imp", SimpleImputer(strategy="median")),
                                         ("sc", StandardScaler())]), num_cols))
    parts.append(("cat", SkPipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                     ("ord", OrdinalEncoder(handle_unknown="use_encoded_value",
                                                            unknown_value=-1))]), cat_cols))
    return ColumnTransformer(parts, sparse_threshold=0)


def onehot_after(idx, min_freq):
    return ColumnTransformer([("oh", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                                   min_frequency=min_freq, sparse_output=False), idx)],
                             remainder="passthrough", sparse_threshold=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="german", choices=["german", "lendingclub"])
    ap.add_argument("--models", nargs="+", default=["logreg", "knn5", "rf_depth5", "hgb_depth2"])
    ap.add_argument("--nseeds", type=int, default=30)
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()

    cfg = E.DATASETS[args.data]
    df = pd.read_csv(ROOT / "data" / cfg["path"])
    y = (df[cfg["target"]] == cfg["positive"]).astype(int)
    X = df.drop(columns=[cfg["target"]] + cfg["drop"], errors="ignore")
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num_cols = [c for c in X.columns if c not in cat_cols]
    idx = list(range(len(num_cols), len(num_cols) + len(cat_cols)))
    mf = cfg["min_freq"]
    out = ROOT / "results" / f"smotenc_{args.data}.csv"
    out.parent.mkdir(exist_ok=True)
    done = set()
    if out.exists():
        o = pd.read_csv(out)
        done = set(zip(o["model"], o["seed"], o["balancer"]))

    for name in args.models:
        for seed in range(1, args.nseeds + 1):
            Xdev, Xte, ydev, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
            cv = StratifiedKFold(5, shuffle=True, random_state=seed)
            for bal in ("smote", "smotenc"):
                if (name, seed, bal) in done:
                    continue
                t = time.time()
                model = E.MODELS[name]()
                if bal == "smote":
                    prep = lambda: E.make_prep(num_cols, cat_cols, mf)
                    Xp = prep().fit_transform(Xdev)
                    inside = Pipeline([("bal", SMOTE(random_state=seed)), ("clf", clone(model))])
                    Xr, yr = SMOTE(random_state=seed).fit_resample(Xp, ydev)
                    full = Pipeline([("prep", prep()), ("bal", SMOTE(random_state=seed)),
                                     ("clf", clone(model))])
                else:
                    prep = lambda: prep_ord(num_cols, cat_cols)
                    mk = lambda: SMOTENC(categorical_features=idx, random_state=seed)
                    Xp = prep().fit_transform(Xdev)
                    inside = Pipeline([("bal", mk()), ("oh", onehot_after(idx, mf)), ("clf", clone(model))])
                    Xr, yr = mk().fit_resample(Xp, ydev)
                    Xr = onehot_after(idx, mf).fit_transform(Xr)
                    full = Pipeline([("prep", prep()), ("bal", mk()), ("oh", onehot_after(idx, mf)),
                                     ("clf", clone(model))])
                cv_in = cross_val_score(inside, Xp, ydev, cv=cv, scoring="roc_auc", n_jobs=args.jobs).mean()
                cv_before = cross_val_score(clone(model), Xr, yr, cv=cv, scoring="roc_auc",
                                            n_jobs=args.jobs).mean()
                full.fit(Xdev, ydev)
                holdout = roc_auc_score(yte, full.predict_proba(Xte)[:, 1])
                r = dict(dataset=args.data, balancer=bal, model=name, seed=seed, cv_in=cv_in,
                         cv_before=cv_before, holdout=holdout, tang=cv_before - cv_in,
                         phong=cv_before - holdout, giay=round(time.time() - t, 1))
                pd.DataFrame([r]).to_csv(out, mode="a", header=not out.exists(), index=False)
                print(f"{bal} {name} seed {seed}: cv_in={cv_in:.4f} cv_before={cv_before:.4f} "
                      f"holdout={holdout:.4f} tang={r['tang']:+.4f} {r['giay']}s", flush=True)


if __name__ == "__main__":
    main()