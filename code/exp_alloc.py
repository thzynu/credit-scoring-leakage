"""Allocation stability of the same-fold decomposition (C1.2).

exp_joint.py splits the T synthetic points of a development set into A (what the leaky model
trains on, as many as the correct pipeline creates) and B (the points added to the validation
fold). That split is random. Here the split, the folds and the synthetic points are kept fixed
(one 'split seed') and ONLY the random allocation of the synthetic points to A and B is redrawn
(allocation seeds 1..N; allocation 0 is the allocation of exp_joint.py and must reproduce
results/joint_<data>_smote.csv).

Per (model, split seed, allocation seed): cv_in, leakage, scoring, joint effect and interaction,
averaged over the 5 folds, plus the share of contaminated points in A and in B.
The correct model does not depend on the allocation, so it is fitted once per fold and its
predictions on the original validation points and on all synthetic points are cached.
Output: results/alloc_<data>.csv (resumable).
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


def run(model, Xp, yd, Xs, folds, alloc_seeds, split_seed):
    T = len(Xs)
    cache = []
    for F in folds:
        te = F["te"]
        pc = clone(model).fit(F["Xb"], F["yb"]).predict_proba(np.vstack([Xp[te], Xs]))[:, 1]
        cache.append(pc)
    out = {}
    for a in alloc_seeds:
        acc = {}
        for fi, (F, pc) in enumerate(zip(folds, cache)):
            tr, te = F["tr"], F["te"]
            k = len(te)
            if a == 0:
                A = F["matched"]
            else:
                A = np.random.RandomState([a, split_seed, fi]).choice(T, size=F["n_in"], replace=False)
            B = np.setdiff1d(np.arange(T), A)
            yte = yd[te]
            ysc = np.concatenate([yte, np.ones(len(B), dtype=yte.dtype)])
            pcs = np.concatenate([pc[:k], pc[k + B]])
            XL = np.vstack([Xp[tr], Xs[A]])
            yL = np.concatenate([yd[tr], np.ones(len(A), dtype=yd.dtype)])
            pl = clone(model).fit(XL, yL).predict_proba(np.vstack([Xp[te], Xs[B]]))[:, 1]
            r = dict(cv_in=roc_auc_score(yte, pc[:k]), leak=roc_auc_score(yte, pl[:k]),
                     score=roc_auc_score(ysc, pcs), joint=roc_auc_score(ysc, pl),
                     contam_A=F["contam"][A].mean(), contam_B=F["contam"][B].mean())
            for kk, v in r.items():
                acc.setdefault(kk, []).append(v)
        m = {kk: float(np.mean(v)) for kk, v in acc.items()}
        m["leak_eff"] = m["leak"] - m["cv_in"]
        m["score_eff"] = m["score"] - m["cv_in"]
        m["joint_eff"] = m["joint"] - m["cv_in"]
        m["interaction"] = m["joint"] - m["leak"] - m["score"] + m["cv_in"]
        out[a] = m
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, choices=list(E.DATASETS))
    ap.add_argument("--models", nargs="+", default=["logreg", "knn5", "rf_depth5", "hgb_depth2", "hgb_full"],
                    choices=list(J.MODELS))
    ap.add_argument("--split-seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--nalloc", type=int, default=10)
    args = ap.parse_args()

    cfg = E.DATASETS[args.data]
    df = pd.read_csv(ROOT / "data" / cfg["path"])
    y = (df[cfg["target"]] == cfg["positive"]).astype(int)
    X = df.drop(columns=[cfg["target"]] + cfg["drop"], errors="ignore")
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num_cols = [c for c in X.columns if c not in cat_cols]
    out = ROOT / "results" / f"alloc_{args.data}.csv"
    out.parent.mkdir(exist_ok=True)
    done = set()
    if out.exists():
        o = pd.read_csv(out)
        done = set(zip(o["model"], o["split_seed"], o["alloc_seed"]))
    allocs = list(range(0, args.nalloc + 1))
    for s in args.split_seeds:
        todo = [m for m in args.models if any((m, s, a) not in done for a in allocs)]
        if not todo:
            continue
        Xp, yd, Xs, folds = P.prepare(X, y, cfg, num_cols, cat_cols, s)
        print(f"split seed {s}: {len(Xs)} synthetic points", flush=True)
        for name in todo:
            t = time.time()
            need = [a for a in allocs if (name, s, a) not in done]
            res = run(J.MODELS[name](), Xp, yd, Xs, folds, need, s)
            rows = []
            for a, m in res.items():
                m.update(dataset=args.data, model=name, split_seed=s, alloc_seed=a)
                rows.append(m)
            pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
            v = np.array([m["leak_eff"] for a, m in res.items() if a > 0])
            print(f"  {name}: leak {v.mean():+.4f} (alloc SD {v.std(ddof=1):.4f}) "
                  f"inter {np.mean([m['interaction'] for a, m in res.items() if a > 0]):+.4f} "
                  f"{time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()