"""Lending Club (train 2013-2015, test 2016): discrimination, calibration and decision cost of
no resampling, SMOTE and class weighting, with uncertainty.
Per model, variant, seed: AUC, Brier, mean prediction, calibration intercept and slope (logistic
regression of the outcome on the logit of the prediction; ideal 0 and 1), ECE (10 equal-count
bins), the same after Platt recalibration fitted on the VALIDATION split only, and decision cost
at cost ratios 3, 5, 10 with the threshold chosen on the validation split only (the 2016 data never
influence a threshold or a recalibration). Percentile bootstrap (resampling the 2016 loans) gives a
95% interval for each metric and for its difference from the no-resampling variant on the same seed.
Output: results/calib_lendingclub.csv and results/calibcurve_lendingclub.csv (resumable per model-seed)."""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

import exp as E
import exp_cost as C

ROOT = Path(__file__).resolve().parent.parent
VARIANTS = ["nosmote", "smote", "classweight"]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def cal_fit(p, y):
    lr = LogisticRegression(C=np.inf, max_iter=1000).fit(logit(p).reshape(-1, 1), y)
    return float(lr.intercept_[0]), float(lr.coef_[0, 0]), lr


def ece(p, y, nb=10):
    b = np.digitize(p, np.quantile(p, np.linspace(0, 1, nb + 1))[1:-1])
    return float(sum((b == i).mean() * abs(y[b == i].mean() - p[b == i].mean())
                     for i in range(nb) if (b == i).any()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["logreg", "rf_depth5", "hgb_full"])
    ap.add_argument("--nseeds", type=int, default=10)
    ap.add_argument("--nboot", type=int, default=300)
    ap.add_argument("--frac", type=float, default=1.0, help="fraction of training rows (trial runs)")
    args = ap.parse_args()

    df = pd.read_csv(ROOT / "data" / "lendingclub_100k.csv")
    tr = df[df["year"].isin([2013, 2014, 2015])]
    te = df[df["year"] == 2016]
    if args.frac < 1.0:
        tr = tr.sample(frac=args.frac, random_state=0)
    drop = ["Default", "id", "issue_d", "zip_code", "year"]
    Xtr, ytr = tr.drop(columns=drop), tr["Default"]
    Xte, yte = te.drop(columns=drop), te["Default"].to_numpy()
    cat_cols = [c for c in Xtr.columns if not pd.api.types.is_numeric_dtype(Xtr[c])]
    num_cols = [c for c in Xtr.columns if c not in cat_cols]
    N = len(yte)
    out = ROOT / "results" / "calib_lendingclub.csv"
    outc = ROOT / "results" / "calibcurve_lendingclub.csv"
    out.parent.mkdir(exist_ok=True)
    done = set()
    if out.exists():
        o = pd.read_csv(out)
        done = set(zip(o["model"], o["seed"]))

    for name in args.models:
        for seed in range(1, args.nseeds + 1):
            if (name, seed) in done:
                continue
            t0 = time.time()
            Xdev, Xval, ydev, yval = train_test_split(Xtr, ytr, test_size=0.2, stratify=ytr,
                                                      random_state=seed)
            yv = yval.to_numpy()
            pt, pv = {}, {}
            for v in VARIANTS:
                clf = E.MODELS[name]()
                if v == "classweight":
                    clf.set_params(class_weight="balanced")
                steps = [("prep", E.make_prep(num_cols, cat_cols, 0.01))]
                if v == "smote":
                    steps.append(("smote", SMOTE(random_state=seed)))
                pipe = Pipeline(steps + [("clf", clf)]).fit(Xdev, ydev)
                pv[v], pt[v] = pipe.predict_proba(Xval)[:, 1], pipe.predict_proba(Xte)[:, 1]
            idx = np.random.RandomState(1000 + seed).randint(0, N, size=(args.nboot, N))
            loss, rows = {}, {}
            for v in VARIANTS:
                p = pt[v]
                a, b, _ = cal_fit(p, yte)
                _, _, plat = cal_fit(pv[v], yv)
                pr = plat.predict_proba(logit(p).reshape(-1, 1))[:, 1]
                a2, b2, _ = cal_fit(pr, yte)
                r = dict(model=name, variant=v, seed=seed, auc=roc_auc_score(yte, p),
                         brier=brier_score_loss(yte, p), mean_pred=p.mean(), cal_int=a, cal_slope=b,
                         ece=ece(p, yte), brier_recal=brier_score_loss(yte, pr),
                         mean_pred_recal=pr.mean(), cal_int_recal=a2, cal_slope_recal=b2,
                         ece_recal=ece(pr, yte), base_rate=yte.mean())
                loss[v] = {"brier": (p - yte) ** 2}
                for c in C.RATIOS:
                    th = C.best_threshold(yv, pv[v], c)
                    d = (p >= th).astype(int)
                    loss[v][f"cost{c}"] = c * ((yte == 1) & (d == 0)) + ((yte == 0) & (d == 1))
                    r[f"cost{c}"] = loss[v][f"cost{c}"].mean()
                rows[v] = r
            for m in loss["nosmote"]:
                bs = {v: loss[v][m][idx].mean(axis=1) for v in VARIANTS}
                for v in VARIANTS:
                    rows[v][m + "_lo"], rows[v][m + "_hi"] = np.percentile(bs[v], [2.5, 97.5])
                    if v != "nosmote":
                        dd = bs[v] - bs["nosmote"]
                        rows[v]["d_" + m] = rows[v][m] - rows["nosmote"][m]
                        rows[v]["d_" + m + "_lo"], rows[v]["d_" + m + "_hi"] = np.percentile(dd, [2.5, 97.5])
            pd.DataFrame([rows[v] for v in VARIANTS]).assign(giay=round(time.time() - t0, 1)).to_csv(
                out, mode="a", header=not out.exists(), index=False)
            cur = []
            for v in VARIANTS:
                b = np.digitize(pt[v], np.quantile(pt[v], np.linspace(0, 1, 11))[1:-1])
                for i in range(10):
                    k = b == i
                    cur.append(dict(model=name, variant=v, seed=seed, bin=i, n=int(k.sum()),
                                    mean_pred=pt[v][k].mean(), frac_pos=yte[k].mean()))
            pd.DataFrame(cur).to_csv(outc, mode="a", header=not outc.exists(), index=False)
            print(name, seed, {v: (round(rows[v]["brier"], 4), round(rows[v]["mean_pred"], 3),
                                   round(rows[v]["cal_slope"], 2)) for v in VARIANTS},
                  f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()