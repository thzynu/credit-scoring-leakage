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


def run(model, Xtr, ytr, Xoot, yoot, num_cols, cat_cols, seed, jobs):
    Xdev, Xrnd, ydev, yrnd = train_test_split(
        Xtr, ytr, test_size=0.2, stratify=ytr, random_state=seed)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)

    def prep():
        return E.make_prep(num_cols, cat_cols, 0.01)

    def sm():
        return SMOTE(random_state=seed)

    # LEAK: preprocessing + SMOTE on the whole dev set before splitting folds
    Xp = prep().fit_transform(Xdev)
    Xr, yr = sm().fit_resample(Xp, ydev)
    leak_both = cross_val_score(clone(model), Xr, yr, cv=cv,
                                scoring="roc_auc", n_jobs=jobs).mean()

    # CORRECT: train correctly on dev, score on the random hold-out and the out-of-time set
    pipe = Pipeline([("prep", prep()), ("smote", sm()), ("clf", clone(model))])
    pipe.fit(Xdev, ydev)
    holdout_rand = roc_auc_score(yrnd, pipe.predict_proba(Xrnd)[:, 1])
    holdout_oot = roc_auc_score(yoot, pipe.predict_proba(Xoot)[:, 1])
    return dict(leak_both=leak_both, holdout_rand=holdout_rand, holdout_oot=holdout_oot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(E.MODELS))
    ap.add_argument("--nseeds", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--train-years", nargs="+", type=int, default=[2013, 2014, 2015])
    ap.add_argument("--test-years", nargs="+", type=int, default=[2016])
    ap.add_argument("--frac", type=float, default=1.0)
    args = ap.parse_args()

    df = pd.read_csv(ROOT / "data" / "lendingclub_100k.csv")
    print(df.groupby("year")["Default"].agg(["size", "mean"]).round(3).T.to_string())
    tr = df[df["year"].isin(args.train_years)]
    te = df[df["year"].isin(args.test_years)]
    if args.frac < 1.0:
        tr = tr.sample(frac=args.frac, random_state=0)
    drop = ["Default", "id", "issue_d", "zip_code", "year"]
    Xtr, ytr = tr.drop(columns=drop), tr["Default"]
    Xoot, yoot = te.drop(columns=drop), te["Default"]
    cat_cols = [c for c in Xtr.columns if not pd.api.types.is_numeric_dtype(Xtr[c])]
    num_cols = [c for c in Xtr.columns if c not in cat_cols]
    split = "train" + "-".join(map(str, args.train_years)) + "_test" + "-".join(map(str, args.test_years))
    print(split, "n_train", len(tr), "n_test", len(te), "cat", cat_cols)

    out = ROOT / "results" / "lendingclub_time.csv"
    out.parent.mkdir(exist_ok=True)
    done = set()
    if out.exists():
        old = pd.read_csv(out)
        done = set(zip(old["model"], old["seed"], old["frac"], old["split"]))

    for name in args.models:
        for seed in range(1, args.nseeds + 1):
            if (name, seed, args.frac, split) in done:
                print("skipping (already done):", name, seed)
                continue
            t = time.time()
            r = run(E.MODELS[name](), Xtr, ytr, Xoot, yoot, num_cols, cat_cols, seed, args.jobs)
            r.update(model=name, seed=seed, frac=args.frac, split=split,
                     n_train=len(tr), n_test=len(te), giay=round(time.time() - t, 1))
            pd.DataFrame([r]).to_csv(out, mode="a", header=not out.exists(), index=False)
            print(f"{name} seed {seed}: rand={r['holdout_rand']:.4f} oot={r['holdout_oot']:.4f} "
                  f"leak_both={r['leak_both']:.4f} {r['giay']}s")


if __name__ == "__main__":
    main()
