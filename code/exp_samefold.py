"""Same-scoring-set decomposition of the inflation caused by balancing before CV.

Old design (exp_diag.py): cv_in and auc_orig used DIFFERENT fold splits, because the
folds of the oversampled array are not the folds of the original array.

This design fixes the folds once, on the ORIGINAL dev data. For every fold (tr, te):
  correct : oversample Xp[tr] only, train, score on Xp[te]          -> cv_in
  leaky   : oversample ALL of Xp first, train on every synthetic point plus the
            original points of tr (the original points of te are dropped), score on
            the SAME Xp[te]                                          -> leak_same
Both are scored on exactly the same original points, so
  real = leak_same - cv_in
is the effect of leaking through the training set only (synthetic points built from
test points), with no change of scoring set and no change of fold composition.

Note: the leaky variant sees all synthetic points, so it also has more synthetic points
than the correct variant (global oversampling target). We record n_syn_leak / n_syn_in.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE, RandomOverSampler

import exp as E

ROOT = Path(__file__).resolve().parent.parent
BAL = {"smote": SMOTE, "borderline": BorderlineSMOTE, "adasyn": ADASYN,
       "ros": RandomOverSampler}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, choices=list(E.DATASETS))
    ap.add_argument("--bal", default="smote", choices=list(BAL))
    ap.add_argument("--models", nargs="+",
                    default=["logreg", "hgb_depth2", "rf_depth5", "knn5"])
    ap.add_argument("--nseeds", type=int, default=10)
    ap.add_argument("--match-syn", action="store_true",
                    help="leaky pipeline uses a random subset of the synthetic points, "
                         "as many as the correct pipeline creates in that fold; "
                         "writes samematch_*.csv instead of samefold_*.csv")
    args = ap.parse_args()

    cfg = E.DATASETS[args.data]
    df = pd.read_csv(ROOT / "data" / cfg["path"])
    y = (df[cfg["target"]] == cfg["positive"]).astype(int)
    X = df.drop(columns=[cfg["target"]] + cfg["drop"], errors="ignore")
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num_cols = [c for c in X.columns if c not in cat_cols]

    prefix = "samematch" if args.match_syn else "samefold"
    out = ROOT / "results" / f"{prefix}_{args.data}_{args.bal}.csv"
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
            Xdev, _, ydev, _ = train_test_split(X, y, test_size=0.2, stratify=y,
                                                random_state=seed)
            cv = StratifiedKFold(5, shuffle=True, random_state=seed)
            model = E.MODELS[name]()
            # Same global preprocessing as exp_diag.py (prep_tang ~ 0, see tong_hop_chinh)
            Xp = E.make_prep(num_cols, cat_cols, cfg["min_freq"]).fit_transform(Xdev)
            yd = ydev.to_numpy()
            n = len(yd)

            Xr, yr = BAL[args.bal](random_state=seed).fit_resample(Xp, yd)
            assert np.allclose(Xr[:n], Xp) and (yr[:n] == yd).all(), \
                "sample order is not as expected"
            assert (yr[n:] == 1).all(), "synthetic points are not the positive class"
            syn_idx = np.arange(n, len(yr))
            rng = np.random.RandomState(seed)

            a_in, a_leak, ns_in, ns_leak = [], [], [], []
            for tr, te in cv.split(Xp, yd):
                # correct: balance the training fold only
                Xb, yb = BAL[args.bal](random_state=seed).fit_resample(Xp[tr], yd[tr])
                p = clone(model).fit(Xb, yb).predict_proba(Xp[te])[:, 1]
                a_in.append(roc_auc_score(yd[te], p))
                ns_in.append(len(yb) - len(tr))
                # leaky: globally balanced data, original test points removed
                syn_use = syn_idx
                if args.match_syn:
                    syn_use = rng.choice(syn_idx, size=len(yb) - len(tr), replace=False)
                keep = np.concatenate([tr, syn_use])
                p = clone(model).fit(Xr[keep], yr[keep]).predict_proba(Xp[te])[:, 1]
                a_leak.append(roc_auc_score(yd[te], p))
                ns_leak.append(len(syn_use))

            r = dict(dataset=args.data, balancer=args.bal, model=name, seed=seed,
                     cv_in_same=np.mean(a_in), leak_same=np.mean(a_leak),
                     real_same=np.mean(a_leak) - np.mean(a_in),
                     n_syn_in=np.mean(ns_in), n_syn_leak=np.mean(ns_leak),
                     giay=round(time.time() - t, 1))
            pd.DataFrame([r]).to_csv(out, mode="a", header=not out.exists(), index=False)
            print(f"{name} seed {seed}: cv_in_same={r['cv_in_same']:.4f} "
                  f"leak_same={r['leak_same']:.4f} real_same={r['real_same']:+.4f} "
                  f"{r['giay']}s", flush=True)


if __name__ == "__main__":
    main()
