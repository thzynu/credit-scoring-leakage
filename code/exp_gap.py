"""Train-validation generalisation gap of the correct pipeline (C2.1).

For every fold of the folds used by exp_joint.py (same split, same SMOTE points), the correct
model (original training part + SMOTE fitted on the training part only) is fitted and scored on
  train_orig : the ORIGINAL training observations      -> train_auc
  train_res  : the oversampled training set it was fitted on -> train_auc_res
  validation : the ORIGINAL validation observations    -> val_auc (= cv_in of exp_joint.py)
gap = train_auc - val_auc. The gap is NOT a measure of memorisation; it is a supporting
indicator of how closely a model reproduces its own training data. For k-nearest neighbours a
training point is its own nearest neighbour, which raises train_auc (k = 1 gives 1.0).
Output: results/gap_<data>.csv (resumable). Joined with results/joint_<data>_smote.csv by
tools/summarize_gap.py.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

import exp as E
import exp_joint as J
import exp_provenance as P

ROOT = Path(__file__).resolve().parent.parent


def run(model, Xp, yd, folds):
    acc = {}
    for F in folds:
        tr, te = F["tr"], F["te"]
        m = clone(model).fit(F["Xb"], F["yb"])
        r = dict(train_auc=roc_auc_score(yd[tr], m.predict_proba(Xp[tr])[:, 1]),
                 train_auc_res=roc_auc_score(F["yb"], m.predict_proba(F["Xb"])[:, 1]),
                 val_auc=roc_auc_score(yd[te], m.predict_proba(Xp[te])[:, 1]))
        for k, v in r.items():
            acc.setdefault(k, []).append(v)
    out = {k: float(np.mean(v)) for k, v in acc.items()}
    out["gap"] = out["train_auc"] - out["val_auc"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, choices=list(E.DATASETS))
    ap.add_argument("--models", nargs="+", default=list(J.MODELS), choices=list(J.MODELS))
    ap.add_argument("--nseeds", type=int, default=5)
    args = ap.parse_args()

    cfg = E.DATASETS[args.data]
    df = pd.read_csv(ROOT / "data" / cfg["path"])
    y = (df[cfg["target"]] == cfg["positive"]).astype(int)
    X = df.drop(columns=[cfg["target"]] + cfg["drop"], errors="ignore")
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num_cols = [c for c in X.columns if c not in cat_cols]
    out = ROOT / "results" / f"gap_{args.data}.csv"
    out.parent.mkdir(exist_ok=True)
    done = set()
    if out.exists():
        o = pd.read_csv(out)
        done = set(zip(o["model"], o["seed"]))
    for seed in range(1, args.nseeds + 1):
        todo = [m for m in args.models if (m, seed) not in done]
        if not todo:
            continue
        Xp, yd, Xs, folds = P.prepare(X, y, cfg, num_cols, cat_cols, seed)
        print(f"seed {seed}", flush=True)
        for name in todo:
            t = time.time()
            r = run(J.MODELS[name](), Xp, yd, folds)
            r.update(dataset=args.data, model=name, seed=seed, giay=round(time.time() - t, 1))
            pd.DataFrame([r]).to_csv(out, mode="a", header=not out.exists(), index=False)
            print(f"  {name}: train {r['train_auc']:.4f} val {r['val_auc']:.4f} gap {r['gap']:.4f} "
                  f"{r['giay']}s", flush=True)


if __name__ == "__main__":
    main()