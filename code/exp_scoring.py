"""Scoring-set effect of synthetic points, measured on a CORRECTLY trained model.

Question: in the leaky cross-validation, part of the inflation could come from the validation
fold containing synthetic points (they are interpolations of observations, so they are easy to
classify). exp_samefold.py / exp_provenance.py score only original observations, so they never
measure this part; it was previously only the "remainder" obtained by subtraction.

Here the model is trained exactly as in the correct pipeline (original training part + SMOTE
fitted on that part only, i.e. the training set behind cv_in). Nothing about the validation
observations reaches training. It is then scored on
  orig          the original validation observations only (= cv_in, asserted against provenance)
  + all         the originals + s synthetic points drawn at random from ALL globally generated ones
                (this mimics the test fold of the leaky cross-validation, where s = T / 5)
  + clean       the originals + s synthetic points whose two parents are both in the training part
  + contam      the originals + s synthetic points with at least one parent in the validation part
Any change of AUC against `orig` is a pure scoring effect: the evaluation set changed, the model
did not. (AUC is computed per fold on originals + the s added points, then averaged over folds.)
Also stored: the mean predicted probability of the added points and of the validation positives.

Output: results/scoring_<data>_smote.csv (resumable per model-seed).
Usage:  python code/exp_scoring.py --data german --nseeds 30
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

import exp as E
import exp_provenance as P

ROOT = Path(__file__).resolve().parent.parent


def run_model(model, Xp, yd, Xs, folds, seed):
    T = len(Xs)
    s = int(round(T / 5))
    rng = np.random.RandomState(200000 + seed)
    acc = {}
    for F in folds:
        te = F["te"]
        m = clone(model).fit(F["Xb"], F["yb"])
        p_val = m.predict_proba(Xp[te])[:, 1]
        y_val = yd[te]

        def plus(pool):
            idx = rng.choice(pool, size=min(s, len(pool)), replace=False)
            p_syn = m.predict_proba(Xs[idx])[:, 1]
            auc = roc_auc_score(np.concatenate([y_val, np.ones(len(idx), dtype=y_val.dtype)]),
                                np.concatenate([p_val, p_syn]))
            return auc, p_syn.mean()

        r = dict(cv_in=roc_auc_score(y_val, p_val), p_valpos=p_val[y_val == 1].mean())
        r["plus_all"], r["p_all"] = plus(np.arange(T))
        r["plus_clean"], r["p_clean"] = plus(F["clean_idx"])
        r["plus_contam"], r["p_contam"] = plus(F["contam_idx"])
        for k, v in r.items():
            acc.setdefault(k, []).append(v)
    out = {k: float(np.mean(v)) for k, v in acc.items()}
    out["n_added"] = s
    for k in ["all", "clean", "contam"]:
        out["score_" + k] = out["plus_" + k] - out["cv_in"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, choices=list(E.DATASETS))
    ap.add_argument("--models", nargs="+",
                    default=["logreg", "knn5", "rf_full", "hgb_depth2", "hgb_full"])
    ap.add_argument("--nseeds", type=int, default=10)
    args = ap.parse_args()

    cfg = E.DATASETS[args.data]
    df = pd.read_csv(ROOT / "data" / cfg["path"])
    y = (df[cfg["target"]] == cfg["positive"]).astype(int)
    X = df.drop(columns=[cfg["target"]] + cfg["drop"], errors="ignore")
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num_cols = [c for c in X.columns if c not in cat_cols]

    out = ROOT / "results" / f"scoring_{args.data}_smote.csv"
    ref = ROOT / "results" / f"provenance_{args.data}_smote.csv"
    ref = pd.read_csv(ref) if ref.exists() else None
    done = set()
    if out.exists():
        old = pd.read_csv(out)
        done = set(zip(old["model"], old["seed"]))

    for seed in range(1, args.nseeds + 1):
        todo = [m for m in args.models if (m, seed) not in done]
        if not todo:
            continue
        t0 = time.time()
        Xp, yd, Xs, folds = P.prepare(X, y, cfg, num_cols, cat_cols, seed)
        print(f"seed {seed}: {len(Xs)} synthetic points, {int(round(len(Xs) / 5))} added per fold "
              f"({time.time() - t0:.1f}s)", flush=True)
        for name in todo:
            t = time.time()
            r = run_model(P.MODELS[name](), Xp, yd, Xs, folds, seed)
            r.update(dataset=args.data, balancer="smote", model=name, seed=seed,
                     giay=round(time.time() - t, 1))
            msg = ""
            if ref is not None:
                q = ref[(ref["model"] == name) & (ref["seed"] == seed)]
                if len(q):
                    msg = f" [cv_in vs provenance: {abs(r['cv_in'] - q['cv_in'].iloc[0]):.1e}]"
            pd.DataFrame([r]).to_csv(out, mode="a", header=not out.exists(), index=False)
            print(f"  {name}: orig={r['cv_in']:.4f} +all={r['score_all']:+.4f} "
                  f"+clean={r['score_clean']:+.4f} +contam={r['score_contam']:+.4f} "
                  f"{r['giay']}s{msg}", flush=True)


if __name__ == "__main__":
    main()
