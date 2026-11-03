"""Provenance-tracked SMOTE: which synthetic points are built from validation observations?

exp_samefold.py estimates the leakage through the training set as a DIFFERENCE between two
pipelines. This script measures it more directly. It re-implements imblearn's binary SMOTE
(same neighbours, same random numbers, so the synthetic points are identical; this is
asserted for every seed) and records for each synthetic point its two parents: the base
observation and the neighbour it was interpolated towards.

For every fold (tr, te) of the folds used in exp_samefold.py, a synthetic point is
  contaminated  if its base OR its neighbour belongs to the validation part te,
  clean         if both parents belong to tr.
Only contaminated points carry information about validation observations.

Training sets compared on the SAME validation points (all scored by AUC on Xp[te]):
  cv_in         original tr + SMOTE fitted on tr only (the correct pipeline)
  nosyn         original tr only (no oversampling)
  leak_all      original tr + ALL globally generated synthetic points  (= exp_samefold.py leak_same)
  leak_matched  original tr + random subset of the synthetic points, as many as the correct
                pipeline creates (= exp_samefold.py --match-syn leak_same; same random numbers)
  leak_m05, leak_m2   matched subset with 0.5x and 2x that size (sensitivity of the matching)
  clean_all     original tr + all clean synthetic points
  contam_eq, clean_eq   original tr + m contaminated points, or + m clean points, same m.
                direct_eq = contam_eq - clean_eq is the effect of having a validation parent,
                with the number of synthetic points held fixed.

Also writes (once per seed, model independent) the geometry between validation points and the
training set: distances to the nearest original / clean / contaminated point, and the share of
contaminated synthetic points among the 5 nearest training neighbours of validation points
(the neighbours a k-nearest-neighbour model would use).

Distances are Euclidean in the preprocessed (standardised, one-hot) feature space.
Output: results/provenance_<data>_smote.csv and results/provgeom_<data>_smote.csv (resumable).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state
from imblearn.over_sampling import SMOTE

import exp as E

ROOT = Path(__file__).resolve().parent.parent

# Extra models defined here so that exp.py (and every stored result) stays untouched.
# "_noes" = same boosting model with early stopping switched off (scikit-learn's default
# "auto" switches it on when the training set has more than 10,000 rows).
MODELS = dict(E.MODELS)
MODELS["hgb_depth2_noes"] = lambda: HistGradientBoostingClassifier(
    max_depth=2, max_iter=100, early_stopping=False, random_state=42)
MODELS["hgb_depth4_noes"] = lambda: HistGradientBoostingClassifier(
    max_depth=4, max_iter=100, early_stopping=False, random_state=42)
MODELS["hgb_full_noes"] = lambda: HistGradientBoostingClassifier(
    early_stopping=False, random_state=42)


def smote_provenance(X, y, seed, k=5):
    """Binary SMOTE identical to imblearn.SMOTE(random_state=seed), plus the parents."""
    counts = np.bincount(y)
    minority = int(np.argmin(counts))
    n_new = int(counts.max() - counts.min())
    idx = np.flatnonzero(y == minority)
    Xc = X[idx]
    nns = NearestNeighbors(n_neighbors=k + 1).fit(Xc).kneighbors(Xc, return_distance=False)[:, 1:]
    rs = check_random_state(seed)
    pick = rs.randint(low=0, high=nns.size, size=n_new)
    steps = rs.uniform(size=n_new)[:, np.newaxis]
    rows = np.floor_divide(pick, nns.shape[1])
    cols = np.mod(pick, nns.shape[1])
    nbr = nns[rows, cols]
    Xs = (Xc[rows] + steps * (Xc[nbr] - Xc[rows])).astype(X.dtype)
    return Xs, idx[rows], idx[nbr], minority


def prepare(X, y, cfg, num_cols, cat_cols, seed):
    """Everything that does not depend on the classifier (computed once per seed)."""
    Xdev, _, ydev, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    Xp = E.make_prep(num_cols, cat_cols, cfg["min_freq"]).fit_transform(Xdev)
    yd = ydev.to_numpy()
    n = len(yd)

    Xs, base, nbr, minority = smote_provenance(Xp, yd, seed)
    assert minority == 1, "the minority class is not the positive class"
    ref, yref = SMOTE(random_state=seed).fit_resample(Xp, yd)
    assert np.allclose(ref[:n], Xp) and (yref[n:] == 1).all(), "unexpected sample order"
    assert np.allclose(ref[n:], Xs), "re-implemented SMOTE differs from imblearn"
    T = len(Xs)

    rng_m = np.random.RandomState(seed)             # reproduces exp_samefold.py --match-syn
    rng_x = np.random.RandomState(100000 + seed)    # extra draws, separate stream
    folds = []
    for tr, te in cv.split(Xp, yd):
        Xb, yb = SMOTE(random_state=seed).fit_resample(Xp[tr], yd[tr])
        n_in = len(yb) - len(tr)
        in_te = np.zeros(n, dtype=bool)
        in_te[te] = True
        contam = in_te[base] | in_te[nbr]
        clean_idx = np.flatnonzero(~contam)
        contam_idx = np.flatnonzero(contam)
        m_eq = min(len(clean_idx), len(contam_idx))
        folds.append(dict(
            tr=tr, te=te, Xb=Xb, yb=yb, n_in=n_in, contam=contam,
            clean_idx=clean_idx, contam_idx=contam_idx,
            matched=rng_m.choice(np.arange(T), size=n_in, replace=False),
            m05=rng_x.choice(T, size=min(T, int(round(0.5 * n_in))), replace=False),
            m2=rng_x.choice(T, size=min(T, 2 * n_in), replace=False),
            clean_sub=rng_x.choice(clean_idx, size=m_eq, replace=False),
            contam_sub=rng_x.choice(contam_idx, size=m_eq, replace=False),
            m_eq=m_eq, n_base=int(in_te[base].sum()),
            n_nbr_only=int((~in_te[base] & in_te[nbr]).sum())))
    return Xp, yd, Xs, folds


def run_model(model, Xp, yd, Xs, folds):
    T = len(Xs)
    acc = {}
    for F in folds:
        tr, te = F["tr"], F["te"]

        def auc(Xt, yt):
            p = clone(model).fit(Xt, yt).predict_proba(Xp[te])[:, 1]
            return roc_auc_score(yd[te], p)

        def plus(idx):
            return (np.vstack([Xp[tr], Xs[idx]]),
                    np.concatenate([yd[tr], np.ones(len(idx), dtype=yd.dtype)]))

        r = dict(cv_in=auc(F["Xb"], F["yb"]), nosyn=auc(Xp[tr], yd[tr]),
                 leak_all=auc(*plus(np.arange(T))), leak_matched=auc(*plus(F["matched"])),
                 leak_m05=auc(*plus(F["m05"])), leak_m2=auc(*plus(F["m2"])),
                 clean_all=auc(*plus(F["clean_idx"])))
        if F["m_eq"] > 0:
            r["contam_eq"] = auc(*plus(F["contam_sub"]))
            r["clean_eq"] = auc(*plus(F["clean_sub"]))
        else:
            r["contam_eq"] = r["clean_eq"] = np.nan
        r.update(n_syn_in=F["n_in"], n_syn_total=T, n_clean=len(F["clean_idx"]),
                 n_contam=len(F["contam_idx"]), n_contam_base=F["n_base"],
                 n_contam_nbr_only=F["n_nbr_only"], m_eq=F["m_eq"])
        for k, v in r.items():
            acc.setdefault(k, []).append(v)
    out = {k: float(np.mean(v)) for k, v in acc.items()}
    out["real_all"] = out["leak_all"] - out["cv_in"]
    out["real_matched"] = out["leak_matched"] - out["cv_in"]
    out["direct_eq"] = out["contam_eq"] - out["clean_eq"]
    return out


def geometry(Xp, yd, Xs, folds):
    acc = {}
    for F in folds:
        tr, te, contam = F["tr"], F["te"], F["contam"]
        Xtr, ytr = Xp[tr], yd[tr]
        Xall = np.vstack([Xtr, Xs])
        typ = np.concatenate([ytr.astype(int), np.where(contam, 3, 2)])  # 0 neg, 1 pos, 2 clean, 3 contam
        nn5 = NearestNeighbors(n_neighbors=5).fit(Xall)
        r = {}
        for label, name in [(1, "pos"), (0, "neg")]:
            V = Xp[te][yd[te] == label]
            t = typ[nn5.kneighbors(V, return_distance=False)]
            r[f"nn5_contam_{name}"] = (t == 3).mean()
            r[f"nn5_clean_{name}"] = (t == 2).mean()
            r[f"nn5_origpos_{name}"] = (t == 1).mean()
            r[f"nn5_origneg_{name}"] = (t == 0).mean()
        V = Xp[te][yd[te] == 1]

        def dist(A):
            if len(A) == 0:
                return np.full(len(V), np.nan)
            return NearestNeighbors(n_neighbors=1).fit(A).kneighbors(V)[0][:, 0]

        d_o, d_c, d_k = dist(Xtr[ytr == 1]), dist(Xs[~contam]), dist(Xs[contam])
        r.update(d_origpos=np.median(d_o), d_clean=np.median(d_c), d_contam=np.median(d_k),
                 share_nearest_syn_contam=float(np.mean(d_k < d_c)), n_val_pos=len(V))
        for k, v in r.items():
            acc.setdefault(k, []).append(v)
    return {k: float(np.nanmean(v)) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, choices=list(E.DATASETS))
    ap.add_argument("--models", nargs="+",
                    default=["logreg", "knn5", "rf_full", "hgb_depth2", "hgb_full"],
                    choices=list(MODELS))
    ap.add_argument("--nseeds", type=int, default=5)
    ap.add_argument("--geometry", choices=["auto", "on", "off"], default="auto",
                    help="auto = only when the development set has at most 30,000 rows")
    args = ap.parse_args()

    cfg = E.DATASETS[args.data]
    df = pd.read_csv(ROOT / "data" / cfg["path"])
    y = (df[cfg["target"]] == cfg["positive"]).astype(int)
    X = df.drop(columns=[cfg["target"]] + cfg["drop"], errors="ignore")
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num_cols = [c for c in X.columns if c not in cat_cols]

    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    f_mod = res / f"provenance_{args.data}_smote.csv"
    f_geo = res / f"provgeom_{args.data}_smote.csv"
    done = set()
    if f_mod.exists():
        o = pd.read_csv(f_mod)
        done = set(zip(o["model"], o["seed"]))
    done_geo = set(pd.read_csv(f_geo)["seed"]) if f_geo.exists() else set()
    do_geo = args.geometry == "on" or (args.geometry == "auto" and 0.8 * len(df) <= 30000)

    for seed in range(1, args.nseeds + 1):
        todo = [m for m in args.models if (m, seed) not in done]
        need_geo = do_geo and seed not in done_geo
        if not todo and not need_geo:
            continue
        t0 = time.time()
        Xp, yd, Xs, folds = prepare(X, y, cfg, num_cols, cat_cols, seed)
        print(f"seed {seed}: {len(Xs)} synthetic points; contaminated per fold "
              f"{np.mean([len(F['contam_idx']) for F in folds]):.0f}, clean "
              f"{np.mean([len(F['clean_idx']) for F in folds]):.0f} "
              f"({time.time() - t0:.1f}s)", flush=True)
        if need_geo:
            g = geometry(Xp, yd, Xs, folds)
            g.update(dataset=args.data, seed=seed)
            pd.DataFrame([g]).to_csv(f_geo, mode="a", header=not f_geo.exists(), index=False)
            print(f"  geometry: median distance of validation positives to nearest original "
                  f"{g['d_origpos']:.3f}, clean synthetic {g['d_clean']:.3f}, contaminated "
                  f"{g['d_contam']:.3f}; contaminated share of 5-NN {g['nn5_contam_pos']:.3f}",
                  flush=True)
        for name in todo:
            t = time.time()
            r = run_model(MODELS[name](), Xp, yd, Xs, folds)
            r.update(dataset=args.data, balancer="smote", model=name, seed=seed,
                     giay=round(time.time() - t, 1))
            pd.DataFrame([r]).to_csv(f_mod, mode="a", header=not f_mod.exists(), index=False)
            print(f"  {name}: cv_in={r['cv_in']:.4f} leak_all={r['leak_all']:.4f} "
                  f"matched={r['leak_matched']:.4f} clean_all={r['clean_all']:.4f} "
                  f"direct_eq={r['direct_eq']:+.4f} {r['giay']}s", flush=True)


if __name__ == "__main__":
    main()
