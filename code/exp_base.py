import argparse
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import exp as E

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, choices=list(E.DATASETS))
    ap.add_argument("--models", nargs="+", default=list(E.MODELS))
    ap.add_argument("--nseeds", type=int, default=10)
    ap.add_argument("--frac", type=float, default=1.0)
    args = ap.parse_args()

    cfg = E.DATASETS[args.data]
    df = pd.read_csv(ROOT / "data" / cfg["path"])
    if args.frac < 1.0:
        df = df.sample(frac=args.frac, random_state=0)
    y = (df[cfg["target"]] == cfg["positive"]).astype(int)
    X = df.drop(columns=[cfg["target"]] + cfg["drop"], errors="ignore")
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num_cols = [c for c in X.columns if c not in cat_cols]

    out = ROOT / "results" / f"{args.data}_nosmote.csv"
    out.parent.mkdir(exist_ok=True)
    done = set()
    if out.exists():
        old = pd.read_csv(out)
        done = set(zip(old["model"], old["seed"], old["frac"]))

    for name in args.models:
        for seed in range(1, args.nseeds + 1):
            if (name, seed, args.frac) in done:
                continue
            t = time.time()
            Xdev, Xte, ydev, yte = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=seed)
            pipe = Pipeline([("prep", E.make_prep(num_cols, cat_cols, cfg["min_freq"])),
                             ("clf", E.MODELS[name]())])
            pipe.fit(Xdev, ydev)
            auc = roc_auc_score(yte, pipe.predict_proba(Xte)[:, 1])
            r = dict(dataset=args.data, model=name, seed=seed, frac=args.frac,
                     holdout_nosmote=auc, giay=round(time.time() - t, 1))
            pd.DataFrame([r]).to_csv(out, mode="a", header=not out.exists(), index=False)
            print(f"{name} seed {seed}: holdout_nosmote={auc:.4f} {r['giay']}s", flush=True)


if __name__ == "__main__":
    main()
