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
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                    help="training splits / SMOTE seeds; each is a separate bootstrap run")
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
    rows = []
    for seed in args.seeds:
        Xdev, _, ydev, _ = train_test_split(
            Xtr, ytr, test_size=0.2, stratify=ytr, random_state=seed)
        preds = {}
        for name in args.models:
            pipe = Pipeline([("prep", E.make_prep(num_cols, cat_cols, 0.01)),
                             ("smote", SMOTE(random_state=seed)),
                             ("clf", E.MODELS[name]())])
            pipe.fit(Xdev, ydev)
            preds[name] = pipe.predict_proba(Xte)[:, 1]
            print(f"seed {seed} {name}: AUC 2016 = {roc_auc_score(yte, preds[name]):.4f}", flush=True)

        # bootstrap of the 2016 test set, paired across models, one run per training seed
        rng = np.random.default_rng(seed)
        n = len(yte)
        aucs = {m: [] for m in preds}
        for _ in range(args.nboot):
            i = rng.integers(0, n, n)
            for m, p in preds.items():
                aucs[m].append(roc_auc_score(yte[i], p[i]))
        aucs = {m: np.array(v) for m, v in aucs.items()}
        for m, v in aucs.items():
            rows.append(dict(kind="auc", seed=seed, a=m, b="", mean=v.mean(),
                             lo=np.percentile(v, 2.5), hi=np.percentile(v, 97.5), p_first_gt=np.nan))
        for a, b in PAIRS:
            if a in aucs and b in aucs:
                d = aucs[a] - aucs[b]
                rows.append(dict(kind="diff", seed=seed, a=a, b=b, mean=d.mean(),
                                 lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5),
                                 p_first_gt=np.mean(d > 0)))

    out = pd.DataFrame(rows)
    path = ROOT / "results" / "lendingclub_boot.csv"
    out.to_csv(path, index=False)
    print("\nsaved", path)

    # Two sources of uncertainty are shown separately: test-set resampling (CI within a seed)
    # and training randomness (spread of the point estimate across seeds).
    print("\nAUC 2016: mean over seeds, sd across seeds, mean width of the within-seed 95% CI")
    for m, g in out[out["kind"] == "auc"].groupby("a"):
        print(f"  {m:11s} {g['mean'].mean():.4f}  sd_seeds={g['mean'].std():.4f}  ci_width={(g['hi'] - g['lo']).mean():.4f}")
    print("\nAUC difference (first minus second): mean over seeds, and in how many seeds the 95% CI excludes 0")
    for (a, b), g in out[out["kind"] == "diff"].groupby(["a", "b"]):
        excl = int(((g["lo"] > 0) | (g["hi"] < 0)).sum())
        print(f"  {a} - {b}: {g['mean'].mean():+.4f}  CI excludes 0 in {excl}/{len(g)} seeds")


if __name__ == "__main__":
    main()
