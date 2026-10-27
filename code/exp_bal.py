import argparse
import time
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline

import exp as E

ROOT = Path(__file__).resolve().parent.parent
BAL = {
    "smote": lambda s: SMOTE(random_state=s),
    "ros": lambda s: RandomOverSampler(random_state=s),
    "borderline": lambda s: BorderlineSMOTE(random_state=s),
    "adasyn": lambda s: ADASYN(random_state=s),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, choices=list(E.DATASETS))
    ap.add_argument("--bal", required=True, choices=list(BAL))
    ap.add_argument("--models", nargs="+", default=["logreg", "hgb_depth2", "rf_depth5", "knn5"])
    ap.add_argument("--nseeds", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()

    cfg = E.DATASETS[args.data]
    df = pd.read_csv(ROOT / "data" / cfg["path"])
    y = (df[cfg["target"]] == cfg["positive"]).astype(int)
    X = df.drop(columns=[cfg["target"]] + cfg["drop"], errors="ignore")
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num_cols = [c for c in X.columns if c not in cat_cols]

    out = ROOT / "results" / f"{args.data}_bal_{args.bal}.csv"
    out.parent.mkdir(exist_ok=True)
    done = set()
    if out.exists():
        old = pd.read_csv(out)
        done = set(zip(old["model"], old["seed"]))

    for name in args.models:
        for seed in range(1, args.nseeds + 1):
            if (name, seed) in done:
                continue
            t = time.time()
            Xdev, Xte, ydev, yte = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=seed)
            cv = StratifiedKFold(5, shuffle=True, random_state=seed)
            model = E.MODELS[name]()

            def prep():
                return E.make_prep(num_cols, cat_cols, cfg["min_freq"])

            def bal():
                return BAL[args.bal](seed)

            Xp = prep().fit_transform(Xdev)
            inside = Pipeline([("bal", bal()), ("clf", clone(model))])
            cv_in = cross_val_score(inside, Xp, ydev, cv=cv, scoring="roc_auc",
                                    n_jobs=args.jobs).mean()
            Xr, yr = bal().fit_resample(Xp, ydev)
            cv_before = cross_val_score(clone(model), Xr, yr, cv=cv, scoring="roc_auc",
                                        n_jobs=args.jobs).mean()
            full = Pipeline([("prep", prep()), ("bal", bal()), ("clf", clone(model))])
            full.fit(Xdev, ydev)
            holdout = roc_auc_score(yte, full.predict_proba(Xte)[:, 1])

            r = dict(dataset=args.data, balancer=args.bal, model=name, seed=seed,
                     cv_in=cv_in, cv_before=cv_before, holdout=holdout,
                     tang=cv_before - cv_in, giay=round(time.time() - t, 1))
            pd.DataFrame([r]).to_csv(out, mode="a", header=not out.exists(), index=False)
            print(f"{args.bal} {name} seed {seed}: cv_in={cv_in:.4f} cv_before={cv_before:.4f} "
                  f"holdout={holdout:.4f} {r['giay']}s", flush=True)


if __name__ == "__main__":
    main()
