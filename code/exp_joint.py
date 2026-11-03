"""Closed decomposition of the leaky-CV inflation: every component is measured directly.

Same folds and synthetic points as exp_provenance.py. In each fold the synthetic points are split
into A (what the leaky model trains on: random subset as large as the correct pipeline's own
synthetic set, = leak_matched of exp_provenance.py) and B (the rest, about T/5 points, which play
the role of the synthetic points that sit in a validation fold of the leaky cross-validation).
Two models per fold:
  correct model  : original train part + SMOTE fitted on the train part only
  leaky model    : original train part + A (A contains points built from validation points)
Two scoring sets: originals of the validation part (orig) and orig + B (orig+B, B labelled 1).
  cv_in  = correct model on orig          leak  = leaky model on orig
  score  = correct model on orig+B        joint = leaky model on orig+B  (what leaky CV reports)
Effects (all differences against cv_in, same folds):
  leak_eff = leak - cv_in      score_eff = score - cv_in      joint_eff = joint - cv_in
  interaction = joint - leak - score + cv_in
so joint_eff = leak_eff + score_eff + interaction holds exactly, with nothing obtained by subtraction
across different designs. Extra capacity settings are defined here (exp.py is untouched).
Output: results/joint_<data>_smote.csv (resumable).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier as HGB, RandomForestClassifier as RF
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier as KNN

import exp as E
import exp_provenance as P

ROOT = Path(__file__).resolve().parent.parent
MODELS = dict(P.MODELS)
for k in (1, 3, 15, 50, 100):
    MODELS[f"knn{k}"] = lambda k=k: KNN(n_neighbors=k)
for d in (3, 10):
    MODELS[f"rf_depth{d}"] = lambda d=d: RF(n_estimators=100, max_depth=d, random_state=42)
for m in (2, 10, 20):
    MODELS[f"rf_leaf{m}"] = lambda m=m: RF(n_estimators=100, min_samples_leaf=m, random_state=42)
for d in (1, 3, 6):
    MODELS[f"hgb_depth{d}"] = lambda d=d: HGB(max_depth=d, max_iter=100, random_state=42)
MODELS["hgb_depth4_iter300"] = lambda: HGB(max_depth=4, max_iter=300, random_state=42)


def run_model(model, Xp, yd, Xs, folds):
    T = len(Xs)
    acc = {}
    for F in folds:
        tr, te = F["tr"], F["te"]
        A = F["matched"]
        B = np.setdiff1d(np.arange(T), A)
        yte = yd[te]
        Xsc = np.vstack([Xp[te], Xs[B]])
        ysc = np.concatenate([yte, np.ones(len(B), dtype=yte.dtype)])
        k = len(te)
        pc = clone(model).fit(F["Xb"], F["yb"]).predict_proba(Xsc)[:, 1]
        XL = np.vstack([Xp[tr], Xs[A]])
        yL = np.concatenate([yd[tr], np.ones(len(A), dtype=yd.dtype)])
        pl = clone(model).fit(XL, yL).predict_proba(Xsc)[:, 1]
        r = dict(cv_in=roc_auc_score(yte, pc[:k]), leak=roc_auc_score(yte, pl[:k]),
                 score=roc_auc_score(ysc, pc), joint=roc_auc_score(ysc, pl),
                 n_A=len(A), n_B=len(B), contam_share_A=F["contam"][A].mean(),
                 contam_share_B=F["contam"][B].mean() if len(B) else np.nan)
        for kk, v in r.items():
            acc.setdefault(kk, []).append(v)
    out = {kk: float(np.mean(v)) for kk, v in acc.items()}
    out["leak_eff"] = out["leak"] - out["cv_in"]
    out["score_eff"] = out["score"] - out["cv_in"]
    out["joint_eff"] = out["joint"] - out["cv_in"]
    out["interaction"] = out["joint"] - out["leak"] - out["score"] + out["cv_in"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, choices=list(E.DATASETS))
    ap.add_argument("--models", nargs="+", default=["logreg", "knn5", "rf_full", "hgb_depth2", "hgb_full"],
                    choices=list(MODELS))
    ap.add_argument("--nseeds", type=int, default=5)
    args = ap.parse_args()

    cfg = E.DATASETS[args.data]
    df = pd.read_csv(ROOT / "data" / cfg["path"])
    y = (df[cfg["target"]] == cfg["positive"]).astype(int)
    X = df.drop(columns=[cfg["target"]] + cfg["drop"], errors="ignore")
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num_cols = [c for c in X.columns if c not in cat_cols]

    out = ROOT / "results" / f"joint_{args.data}_smote.csv"
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
        print(f"seed {seed}: {len(Xs)} synthetic points", flush=True)
        for name in todo:
            t = time.time()
            r = run_model(MODELS[name](), Xp, yd, Xs, folds)
            r.update(dataset=args.data, balancer="smote", model=name, seed=seed,
                     giay=round(time.time() - t, 1))
            pd.DataFrame([r]).to_csv(out, mode="a", header=not out.exists(), index=False)
            print(f"  {name}: cv_in={r['cv_in']:.4f} leak={r['leak_eff']:+.4f} score={r['score_eff']:+.4f} "
                  f"inter={r['interaction']:+.4f} joint={r['joint_eff']:+.4f} {r['giay']}s", flush=True)


if __name__ == "__main__":
    main()