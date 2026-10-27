import argparse
import time
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

import exp as E

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, choices=list(E.DATASETS))
    ap.add_argument("--models", nargs="+", default=["logreg", "hgb_depth2", "rf_depth5", "knn5"])
    ap.add_argument("--nseeds", type=int, default=10)
    ap.add_argument("--kfrac", type=float, default=0.25)
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()

    cfg = E.DATASETS[args.data]
    df = pd.read_csv(ROOT / "data" / cfg["path"])
    y = (df[cfg["target"]] == cfg["positive"]).astype(int)
    X = df.drop(columns=[cfg["target"]] + cfg["drop"], errors="ignore")
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num_cols = [c for c in X.columns if c not in cat_cols]

    out = ROOT / "results" / f"{args.data}_sel.csv"
    out.parent.mkdir(exist_ok=True)
    done = set()
    if out.exists():
        old = pd.read_csv(out)
        done = set(zip(old["model"], old["seed"], old["kfrac"]))

    for name in args.models:
        for seed in range(1, args.nseeds + 1):
            if (name, seed, args.kfrac) in done:
                continue
            t = time.time()
            Xdev, Xte, ydev, yte = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=seed)
            cv = StratifiedKFold(5, shuffle=True, random_state=seed)

            def prep():
                return E.make_prep(num_cols, cat_cols, cfg["min_freq"])

            Xp = prep().fit_transform(Xdev)
            k = max(3, int(Xp.shape[1] * args.kfrac))
            model = E.MODELS[name]()

            def sel():
                return SelectKBest(f_classif, k=k)

            # CORRECT: feature selection inside each fold
            good = Pipeline([("prep", prep()), ("sel", sel()), ("clf", clone(model))])
            cv_dung = cross_val_score(good, Xdev, ydev, cv=cv, scoring="roc_auc",
                                      n_jobs=args.jobs).mean()
            # LEAK: select features on the whole dev set, then split folds
            Xs = sel().fit_transform(Xp, ydev)
            leak_sel = cross_val_score(clone(model), Xs, ydev, cv=cv, scoring="roc_auc",
                                       n_jobs=args.jobs).mean()
            # TRUTH: train correctly on dev, score on the hold-out
            good.fit(Xdev, ydev)
            holdout = roc_auc_score(yte, good.predict_proba(Xte)[:, 1])

            r = dict(dataset=args.data, model=name, seed=seed, kfrac=args.kfrac, k=k,
                     n_feat=Xp.shape[1], cv_dung=cv_dung, leak_sel=leak_sel,
                     holdout=holdout, sel_tang=leak_sel - cv_dung,
                     giay=round(time.time() - t, 1))
            pd.DataFrame([r]).to_csv(out, mode="a", header=not out.exists(), index=False)
            print(f"{name} seed {seed}: k={k}/{Xp.shape[1]} cv_dung={cv_dung:.4f} "
                  f"leak_sel={leak_sel:.4f} holdout={holdout:.4f} {r['giay']}s", flush=True)


if __name__ == "__main__":
    main()
