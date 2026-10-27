import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

import exp as E

ROOT = Path(__file__).resolve().parent.parent
RATIOS = [3, 5, 10]


def cost(y, pred, c_fn):
    fn = np.sum((y == 1) & (pred == 0))
    fp = np.sum((y == 0) & (pred == 1))
    return (fn * c_fn + fp) / len(y)


def best_threshold(y, p, c_fn):
    grid = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 197)))
    return min(grid, key=lambda t: cost(y, (p >= t).astype(int), c_fn))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["logreg", "hgb_full", "rf_depth5"])
    ap.add_argument("--nseeds", type=int, default=3)
    args = ap.parse_args()

    df = pd.read_csv(ROOT / "data" / "lendingclub_100k.csv")
    tr = df[df["year"].isin([2013, 2014, 2015])]
    te = df[df["year"] == 2016]
    drop = ["Default", "id", "issue_d", "zip_code", "year"]
    Xtr, ytr = tr.drop(columns=drop), tr["Default"]
    Xte, yte = te.drop(columns=drop), te["Default"].to_numpy()
    cat_cols = [c for c in Xtr.columns if not pd.api.types.is_numeric_dtype(Xtr[c])]
    num_cols = [c for c in Xtr.columns if c not in cat_cols]
    base = yte.mean()
    print(f"2016 default rate = {base:.3f}; train 2013-2015 = {ytr.mean():.3f}")
    triv = {r: min(base * r, 1 - base) for r in RATIOS}
    print("trivial cost (approve all or reject all):", {r: round(v, 4) for r, v in triv.items()})

    rows = []
    for name in args.models:
        for variant in ["smote", "khong_smote"]:
            for seed in range(1, args.nseeds + 1):
                Xdev, Xval, ydev, yval = train_test_split(
                    Xtr, ytr, test_size=0.2, stratify=ytr, random_state=seed)
                steps = [("prep", E.make_prep(num_cols, cat_cols, 0.01))]
                if variant == "smote":
                    steps.append(("smote", SMOTE(random_state=seed)))
                steps.append(("clf", E.MODELS[name]()))
                pipe = Pipeline(steps).fit(Xdev, ydev)
                pv = pipe.predict_proba(Xval)[:, 1]
                pt = pipe.predict_proba(Xte)[:, 1]
                r = dict(model=name, variant=variant, seed=seed,
                         auc=roc_auc_score(yte, pt), brier=brier_score_loss(yte, pt),
                         mean_pred=pt.mean())
                for c in RATIOS:
                    t = best_threshold(yval.to_numpy(), pv, c)
                    r[f"cost{c}"] = cost(yte, (pt >= t).astype(int), c)
                rows.append(r)
                print(name, variant, seed, round(r["auc"], 4), flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "lendingclub_cost.csv", index=False)
    t = out.groupby(["model", "variant"]).mean(numeric_only=True).drop(columns="seed").round(4)
    print(t.to_string())


if __name__ == "__main__":
    main()
