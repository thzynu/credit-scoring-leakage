"""Effect of the oversampling ratio on the pre-split inflation (SMOTE).
sampling_strategy = desired minority/majority ratio after oversampling (1.0 = fully balanced, the
default used elsewhere). Ratios not above the current ratio of the development set are skipped.
Same procedure as exp_bal.py. tang = cv_before - cv_in; phong = cv_before - holdout.
Output: results/ratio_<data>.csv (resumable per model, seed, ratio)."""
import argparse
import time
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

import exp as E

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, choices=list(E.DATASETS))
    ap.add_argument("--models", nargs="+", default=["logreg", "knn5", "rf_depth5", "hgb_depth2"])
    ap.add_argument("--ratios", nargs="+", type=float, default=[0.4, 0.6, 0.8, 1.0])
    ap.add_argument("--nseeds", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()

    cfg = E.DATASETS[args.data]
    df = pd.read_csv(ROOT / "data" / cfg["path"])
    y = (df[cfg["target"]] == cfg["positive"]).astype(int)
    X = df.drop(columns=[cfg["target"]] + cfg["drop"], errors="ignore")
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num_cols = [c for c in X.columns if c not in cat_cols]
    cur = y.mean() / (1 - y.mean())
    ratios = [r for r in args.ratios if r > cur + 0.01]
    print(f"{args.data}: current minority/majority ratio {cur:.3f}; ratios used: {ratios}")

    out = ROOT / "results" / f"ratio_{args.data}.csv"
    out.parent.mkdir(exist_ok=True)
    done = set()
    if out.exists():
        o = pd.read_csv(out)
        done = set(zip(o["model"], o["seed"], o["ratio"].round(3)))
    for name in args.models:
        for seed in range(1, args.nseeds + 1):
            Xdev, Xte, ydev, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
            cv = StratifiedKFold(5, shuffle=True, random_state=seed)
            for r in ratios:
                if (name, seed, round(r, 3)) in done:
                    continue
                t = time.time()

                def prep():
                    return E.make_prep(num_cols, cat_cols, cfg["min_freq"])

                def bal():
                    return SMOTE(sampling_strategy=r, random_state=seed)

                model = E.MODELS[name]()
                Xp = prep().fit_transform(Xdev)
                cv_in = cross_val_score(Pipeline([("bal", bal()), ("clf", clone(model))]), Xp, ydev,
                                        cv=cv, scoring="roc_auc", n_jobs=args.jobs).mean()
                Xr, yr = bal().fit_resample(Xp, ydev)
                cv_before = cross_val_score(clone(model), Xr, yr, cv=cv, scoring="roc_auc",
                                            n_jobs=args.jobs).mean()
                full = Pipeline([("prep", prep()), ("bal", bal()), ("clf", clone(model))]).fit(Xdev, ydev)
                holdout = roc_auc_score(yte, full.predict_proba(Xte)[:, 1])
                row = dict(dataset=args.data, balancer="smote", model=name, seed=seed, ratio=r,
                           cv_in=cv_in, cv_before=cv_before, holdout=holdout,
                           tang=cv_before - cv_in, phong=cv_before - holdout,
                           giay=round(time.time() - t, 1))
                pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
                print(f"{name} seed {seed} ratio {r}: cv_in={cv_in:.4f} cv_before={cv_before:.4f} "
                      f"holdout={holdout:.4f} tang={row['tang']:+.4f}", flush=True)


if __name__ == "__main__":
    main()