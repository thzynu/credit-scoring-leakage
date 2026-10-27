import time
import argparse
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "taiwan.csv"
OUT = ROOT / "results" / "taiwan_results.csv"

MODELS = {
    "logreg": lambda: LogisticRegression(max_iter=1000),
    "hgb_depth2": lambda: HistGradientBoostingClassifier(max_depth=2, max_iter=100, random_state=42),
    "hgb_depth4": lambda: HistGradientBoostingClassifier(max_depth=4, max_iter=100, random_state=42),
    "hgb_full": lambda: HistGradientBoostingClassifier(random_state=42),
    "knn5": lambda: KNeighborsClassifier(n_neighbors=5),
    "rf_depth5": lambda: RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    "rf_full": lambda: RandomForestClassifier(n_estimators=100, random_state=42),
}


def run(model, X, y, seed, jobs):
    Xdev, Xte, ydev, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)

    def sm():
        return SMOTE(random_state=seed)

    # CORRECT: scaling and SMOTE inside each fold
    pipe = Pipeline([("sc", StandardScaler()), ("smote", sm()), ("clf", clone(model))])
    cv_dung = cross_val_score(pipe, Xdev, ydev, cv=cv, scoring="roc_auc", n_jobs=jobs).mean()

    # LEAK: scaling
    Xs = StandardScaler().fit_transform(Xdev)
    p1 = Pipeline([("smote", sm()), ("clf", clone(model))])
    leak_scale = cross_val_score(p1, Xs, ydev, cv=cv, scoring="roc_auc", n_jobs=jobs).mean()

    # LEAK: scaling + SMOTE on the whole dev set before splitting folds
    Xr, yr = sm().fit_resample(Xs, ydev)
    leak_both = cross_val_score(clone(model), Xr, yr, cv=cv, scoring="roc_auc", n_jobs=jobs).mean()

    # TRUTH: train correctly on dev, score on the hold-out
    pipe.fit(Xdev, ydev)
    holdout = roc_auc_score(yte, pipe.predict_proba(Xte)[:, 1])

    return dict(cv_dung=cv_dung, leak_scale=leak_scale,
                leak_both=leak_both, holdout=holdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--frac", type=float, default=1.0, help="fraction of the data, used for trial runs")
    args = ap.parse_args()

    df = pd.read_csv(DATA)
    if args.frac < 1.0:
        df = df.sample(frac=args.frac, random_state=0)
    X, y = df.drop(columns="target"), df["target"]

    OUT.parent.mkdir(exist_ok=True)
    done = set()
    if OUT.exists():
        old = pd.read_csv(OUT)
        done = set(zip(old["model"], old["seed"], old["frac"]))

    for name in args.models:
        for seed in args.seeds:
            if (name, seed, args.frac) in done:
                print("skipping (already done):", name, seed)
                continue
            t = time.time()
            r = run(MODELS[name](), X, y, seed, args.jobs)
            r.update(model=name, seed=seed, frac=args.frac,
                     phong=r["leak_both"] - r["holdout"],
                     giay=round(time.time() - t, 1))
            pd.DataFrame([r]).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
            print(name, seed, {k: round(v, 4) for k, v in r.items()
                                if k in ("holdout", "leak_both", "phong")}, r["giay"], "sec")


if __name__ == "__main__":
    main()
