import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

import exp as E

ROOT = Path(__file__).resolve().parent.parent
PAIRS = [("logreg", "hgb_full"), ("logreg", "rf_depth5"), ("hgb_full", "rf_depth5"),
         ("rf_depth5", "rf_full"), ("logreg", "knn5"), ("hgb_full", "hgb_depth2")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(E.MODELS))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--nboot", type=int, default=1000)
    args = ap.parse_args()

    df = pd.read_csv(ROOT / "data" / "lendingclub_100k.csv")
    tr = df[df["year"].isin([2013, 2014, 2015])]
    te = df[df["year"] == 2016]
    drop = ["Default", "id", "issue_d", "zip_code", "year"]
    Xtr, ytr = tr.drop(columns=drop), tr["Default"]
    Xte, yte = te.drop(columns=drop), te["Default"].to_numpy()
    cat_cols = [c for c in Xtr.columns if not pd.api.types.is_numeric_dtype(Xtr[c])]
    num_cols = [c for c in Xtr.columns if c not in cat_cols]
    Xdev, _, ydev, _ = train_test_split(
        Xtr, ytr, test_size=0.2, stratify=ytr, random_state=args.seed)

    preds = {}
    for name in args.models:
        pipe = Pipeline([("prep", E.make_prep(num_cols, cat_cols, 0.01)),
                         ("smote", SMOTE(random_state=args.seed)),
                         ("clf", E.MODELS[name]())])
        pipe.fit(Xdev, ydev)
        preds[name] = pipe.predict_proba(Xte)[:, 1]
        print(name, "AUC 2016 =", round(roc_auc_score(yte, preds[name]), 4), flush=True)

    rng = np.random.default_rng(0)
    n = len(yte)
    aucs = {m: [] for m in preds}
    for _ in range(args.nboot):
        i = rng.integers(0, n, n)
        for m, p in preds.items():
            aucs[m].append(roc_auc_score(yte[i], p[i]))
    aucs = {m: np.array(v) for m, v in aucs.items()}

    print("\n95% confidence interval of each model's AUC:")
    for m, v in aucs.items():
        print(f"  {m:11s} {v.mean():.4f}  [{np.percentile(v, 2.5):.4f}, {np.percentile(v, 97.5):.4f}]")

    print("\nAUC difference (first model minus second), paired on the same bootstrap samples:")
    for a, b in PAIRS:
        if a in aucs and b in aucs:
            d = aucs[a] - aucs[b]
            print(f"  {a} - {b}: {d.mean():+.4f}  [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]"
                  f"  P(first>second)={np.mean(d > 0):.3f}")


if __name__ == "__main__":
    main()
