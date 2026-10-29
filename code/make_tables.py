"""Rebuild ALL summary tables from the raw result files in results/.

Run:     python code/make_tables.py
Output:  results/tables/*.csv  (also printed to screen)

Convention: se = standard deviation / sqrt(number of seeds). SE only reflects the variation
between data splits on the SAME dataset; it is not a general measure of statistical confidence.
"""
import glob
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "tables"
OUT.mkdir(exist_ok=True)


def se(s):
    return s.std() / len(s) ** 0.5


def read_all(pattern):
    files = sorted(glob.glob(str(RES / pattern)))
    if not files:
        raise SystemExit(f"no file matches results/{pattern}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def save(df, name):
    df.to_csv(OUT / name)
    print(f"\n===== {name}")
    print(df.to_string())


# ---------------------------------------------------------------- Table 1
# Leakage of each step (preprocessing, SMOTE before splitting folds, feature selection)
# and the correct hold-out result. Main columns: smote_tang, prep_tang, sel_tang, phong.
v = read_all("*_v2.csv")
v = v[v["frac"] == 1.0].copy()
v["prep_tang"] = v["leak_prep"] - v["cv_dung"]
v["smote_tang"] = v["leak_both"] - v["leak_prep"]
v["phong"] = v["leak_both"] - v["holdout"]
a = v.groupby(["dataset", "model"]).agg(
    n_seed=("seed", "nunique"), holdout_smote=("holdout", "mean"),
    cv_dung=("cv_dung", "mean"), leak_both=("leak_both", "mean"),
    prep_tang=("prep_tang", "mean"), smote_tang=("smote_tang", "mean"),
    smote_tang_se=("smote_tang", se), phong=("phong", "mean"))

b = read_all("*_nosmote.csv")
b = b[b["frac"] == 1.0]
m = b.merge(v[["dataset", "model", "seed", "holdout"]], on=["dataset", "model", "seed"])
m["delta_smote"] = m["holdout"] - m["holdout_nosmote"]
d = m.groupby(["dataset", "model"]).agg(
    holdout_nosmote=("holdout_nosmote", "mean"),
    delta_smote=("delta_smote", "mean"), delta_smote_se=("delta_smote", se))

s = read_all("*_sel.csv")
s = s[abs(s["kfrac"] - 0.25) < 1e-9]
sg = s.groupby(["dataset", "model"]).agg(
    sel_tang=("sel_tang", "mean"), sel_tang_se=("sel_tang", se))

save(a.join(d).join(sg).round(4), "tong_hop_chinh.csv")

# ---------------------------------------------------------------- Table 2
# Decomposition of the inflation when classes are balanced BEFORE splitting folds:
#   real  = auc_orig - cv_in   : true leakage (scored on original points)
#   synth = auc_all - auc_orig : due to synthetic points landing in the scoring set
#   total = auc_all - cv_in    : the inflation a reader of the leaky paper sees
dg = read_all("diag_*.csv")
dg["real"] = dg["auc_orig"] - dg["cv_in"]
dg["synth"] = dg["auc_all"] - dg["auc_orig"]
dg["total"] = dg["auc_all"] - dg["cv_in"]
g = dg.groupby(["dataset", "balancer", "model"]).agg(
    n=("seed", "nunique"), total=("total", "mean"), real=("real", "mean"),
    real_se=("real", se), synth=("synth", "mean"), auc_syn=("auc_syn", "mean"))
g["share_synth"] = g["synth"] / g["total"]   # only meaningful when |total| > ~0.01
bad = (g["total"] < 0.01) | (g["real"] < 0) | (g["synth"] < 0)
g.loc[bad, "share_synth"] = float("nan")   # leave blank when the split is meaningless
save(g.round(4), "bang_phan_ra.csv")

# ---------------------------------------------------------------- Table 2b
# Same-scoring-set decomposition (exp_samefold.py). Folds are fixed on the original data, so
# cv_in_same == cv_in exactly and the scoring set is identical for both pipelines:
#   real_same = leak_same - cv_in_same : leakage through the TRAINING set only
#   rest      = total - real_same      : inflation NOT explained by leakage into training
#                                        (synthetic points in the scoring set; also any
#                                        fold-composition effect of the old design)
# total and real_old come from the old design (Table 2) and are shown for comparison.
sf = read_all("samefold_*.csv")
t2b = sf.groupby(["dataset", "balancer", "model"]).agg(
    n=("seed", "nunique"), real_same=("real_same", "mean"), real_same_se=("real_same", se),
    n_syn_in=("n_syn_in", "mean"), n_syn_leak=("n_syn_leak", "mean"))
old = dg.groupby(["dataset", "balancer", "model"]).agg(
    n_old=("seed", "nunique"), total=("total", "mean"), real_old=("real", "mean"))
t2b = t2b.join(old)
t2b["rest"] = t2b["total"] - t2b["real_same"]
t2b["share_real_same"] = t2b["real_same"] / t2b["total"]
t2b.loc[t2b["total"] < 0.01, "share_real_same"] = float("nan")   # meaningless when total is tiny
save(t2b.round(4), "bang_phan_ra_samefold.csv")

# ---------------------------------------------------------------- Table 3
# Leakage when balancing classes before splitting folds, by balancer (SMOTE/ROS/Borderline)
bl = read_all("*_bal_*.csv")
t3 = bl.groupby(["dataset", "balancer", "model"]).agg(
    n=("seed", "nunique"), tang=("tang", "mean"), tang_se=("tang", se),
    cv_in=("cv_in", "mean"), cv_before=("cv_before", "mean"), holdout=("holdout", "mean"))
save(t3.round(4), "bang_can_bang.csv")

# ---------------------------------------------------------------- Table 4
# Lending Club: random split vs out-of-time split
tm = pd.read_csv(RES / "lendingclub_time.csv")
tm = tm[tm["frac"] == 1.0]
t4 = tm.groupby(["split", "model"])[["holdout_rand", "holdout_oot", "leak_both"]].mean()
t4["phong_oot"] = t4["leak_both"] - t4["holdout_oot"]
t4["rand_hon_oot"] = t4["holdout_rand"] - t4["holdout_oot"]
t4["n_seed"] = tm.groupby(["split", "model"])["seed"].nunique()
save(t4.round(4), "bang_thoi_gian.csv")

# ---------------------------------------------------------------- Table 5
# Lending Club: decision cost (threshold chosen on the validation set)
cs = pd.read_csv(RES / "lendingclub_cost.csv")
t5 = cs.groupby(["model", "variant"]).mean(numeric_only=True).drop(columns="seed")
t5["n_seed"] = cs.groupby(["model", "variant"])["seed"].nunique()
save(t5.round(4), "bang_chi_phi.csv")

# ---------------------------------------------------------------- Table 6
# Coverage: how many seeds each result file has for each model.
# This lets the reader see at once where the seed count is low (figures there are less reliable).
rows = []
for f in sorted(glob.glob(str(RES / "*.csv"))):
    name = Path(f).name
    if name in ("tong_hop_chinh.csv", "bang_phan_ra.csv"):
        continue
    dd = pd.read_csv(f)
    keys = [c for c in ("model", "variant", "frac", "kfrac") if c in dd.columns]
    for k, gg in dd.groupby(keys):
        k = k if isinstance(k, tuple) else (k,)
        rows.append(dict(file=name, **dict(zip(keys, k)), n_seed=gg["seed"].nunique()))
cov = pd.DataFrame(rows).set_index(["file", "model"]).sort_index()
save(cov, "bao_phu_seed.csv")
