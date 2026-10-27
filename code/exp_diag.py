import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.pipeline import Pipeline

import exp as E

ROOT = Path(__file__).resolve().parent.parent
BAL = {"smote": SMOTE, "borderline": BorderlineSMOTE}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, choices=list(E.DATASETS))
    ap.add_argument("--bal", default="smote", choices=list(BAL))
    ap.add_argument("--models", nargs="+", default=["logreg", "hgb_depth2", "rf_depth5", "knn5"])
    ap.add_argument("--nseeds", type=int, default=10)
    args = ap.parse_args()

    cfg = E.DATASETS[args.data]
    df = pd.read_csv(ROOT / "data" / cfg["path"])
    y = (df[cfg["target"]] == cfg["positive"]).astype(int)
    X = df.drop(columns=[cfg["target"]] + cfg["drop"], errors="ignore")
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num_cols = [c for c in X.columns if c not in cat_cols]

    out = ROOT / "results" / f"diag_{args.data}_{args.bal}.csv"
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
            Xdev, _, ydev, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
            cv = StratifiedKFold(5, shuffle=True, random_state=seed)
            model = E.MODELS[name]()
            Xp = E.make_prep(num_cols, cat_cols, cfg["min_freq"]).fit_transform(Xdev)
            yd = ydev.to_numpy()
            n = len(yd)

            inside = Pipeline([("bal", BAL[args.bal](random_state=seed)), ("clf", clone(model))])
            cv_in = cross_val_score(inside, Xp, yd, cv=cv, scoring="roc_auc").mean()

            Xr, yr = BAL[args.bal](random_state=seed).fit_resample(Xp, yd)
            assert np.allclose(Xr[:n], Xp) and (yr[:n] == yd).all(), "sample order is not as expected"
            # synthetic points must belong to the positive (minority) class; otherwise the auc_syn split is wrong
            assert (yr[n:] == 1).all(), "synthetic points are not the positive class: check positive / minority class"

            a_all, a_orig, a_syn = [], [], []
            for tr, te in cv.split(Xr, yr):
                p = clone(model).fit(Xr[tr], yr[tr]).predict_proba(Xr[te])[:, 1]
                yt = yr[te]
                orig = te < n
                a_all.append(roc_auc_score(yt, p))
                a_orig.append(roc_auc_score(yt[orig], p[orig]))
                sel = (~orig) | (yt == 0)
                a_syn.append(roc_auc_score(yt[sel], p[sel]))

            r = dict(dataset=args.data, balancer=args.bal, model=name, seed=seed,
                     cv_in=cv_in, auc_all=np.mean(a_all), auc_orig=np.mean(a_orig),
                     auc_syn=np.mean(a_syn), giay=round(time.time() - t, 1))
            pd.DataFrame([r]).to_csv(out, mode="a", header=not out.exists(), index=False)
            print(f"{name} seed {seed}: cv_in={cv_in:.4f} all={r['auc_all']:.4f} "
                  f"orig={r['auc_orig']:.4f} syn={r['auc_syn']:.4f} {r['giay']}s", flush=True)


if __name__ == "__main__":
    main()
