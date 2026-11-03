"""Consistency audit of the raw results and of the headline numbers quoted in the manuscript.
Run `python code/make_tables.py` first. Prints (1) differences between Table 5 and Table 6 totals and
which term explains them, (2) rounding mismatches in the decomposition, (3) provenance versus
samematch agreement, (4) duplicate or missing result rows, (5) seed coverage below the planned
number, (6) manuscript claims versus numbers recomputed from the raw files. Nothing is modified."""
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent.parent / "results"
T = R / "tables"
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 500)
EXPECT = {"german": 30, "australian": 30, "taiwan": 10, "lendingclub": 5}

print("\n(1) Table 5 smote increase (exp.py: leak_both - leak_prep) vs Table 6 total (exp_diag.py: auc_all - cv_in)")
v = pd.concat([pd.read_csv(f) for f in sorted(R.glob("*_v2.csv")) if "_q_" not in f.name])
v = v[v["frac"] == 1.0]
dg = pd.concat([pd.read_csv(f) for f in sorted(R.glob("diag_*_smote.csv"))])
m = v.merge(dg, on=["dataset", "model", "seed"], suffixes=("_v2", "_diag"))
m["t5"] = m["leak_both"] - m["leak_prep"]
m["t6"] = m["auc_all"] - m["cv_in"]
m["d_balance_cv"] = m["leak_both"] - m["auc_all"]
m["d_inside_cv"] = m["leak_prep"] - m["cv_in"]
a = m.groupby(["dataset", "model"]).agg(n_common=("seed", "nunique"), t5=("t5", "mean"), t6=("t6", "mean"),
                                        leak_both_minus_auc_all=("d_balance_cv", "mean"),
                                        leak_prep_minus_cv_in=("d_inside_cv", "mean")).reset_index()
a["diff"] = a["t5"] - a["t6"]
n5 = v.groupby(["dataset", "model"])["seed"].nunique().rename("n_seed_v2").reset_index()
n6 = dg.groupby(["dataset", "model"])["seed"].nunique().rename("n_seed_diag").reset_index()
a = a.merge(n5).merge(n6)
print(a[a["diff"].abs() > 5e-5].round(5).to_string(index=False) or "none")
print("-> if leak_both_minus_auc_all is 0 and leak_prep_minus_cv_in is not, the gap comes from the two\n"
      "   'inside the fold' runs (different n_jobs / different seeds), not from the leaky estimate.")

print("\n(2) rounding: round(real_match,4)+round(rest,4) vs round(total,4)")
s = pd.read_csv(T / "bang_phan_ra_samefold.csv")
s = s[s["balancer"] == "smote"]
bad = s.dropna(subset=["total", "real_match"])
bad = bad[(bad["real_match"].round(4) + bad["rest"].round(4) - bad["total"].round(4)).abs() > 1e-9]
print(bad[["dataset", "model", "total", "real_match", "rest"]].to_string(index=False) if len(bad) else "none")
print("-> in the manuscript tables print rest = round(total,4) - round(real_match,4) so the rounded columns add up.")

print("\n(3) provenance real_matched vs samematch real_same (same seeds)")
for d in EXPECT:
    fp, fs = R / f"provenance_{d}_smote.csv", R / f"samematch_{d}_smote.csv"
    if fp.exists() and fs.exists():
        x = pd.read_csv(fp).merge(pd.read_csv(fs), on=["model", "seed"])
        print(f"{d}: rows={len(x)}  max|diff|={(x['real_matched'] - x['real_same']).abs().max():.2e}")

print("\n(4) duplicate keys / missing values in result files")
KEYS = ["dataset", "balancer", "model", "variant", "seed", "frac", "kfrac", "split", "ratio", "bin"]
for f in sorted(R.glob("*.csv")):
    x = pd.read_csv(f)
    if "model" not in x or "seed" not in x:
        continue
    k = [c for c in KEYS if c in x.columns]
    dup = int(x.duplicated(k).sum())
    skip = [c for c in x.columns if c == "leak_impute" or c.startswith("d_")]
    nan = {c: int(n) for c, n in x.drop(columns=skip).isna().sum().items() if n}
    if dup or nan:
        print(f"{f.name}: duplicates={dup} NaN={nan}")

print("\n(5) cells with fewer seeds than planned (german 30, australian 30, taiwan 10, lendingclub 5)")
rows = []
for f in sorted(R.glob("*.csv")):
    d = next((k for k in EXPECT if k in f.name), None)
    x = pd.read_csv(f)
    if d is None or "model" not in x or "seed" not in x:
        continue
    for mname, g in x.groupby("model"):
        n = g["seed"].nunique()
        if n < EXPECT[d]:
            rows.append(dict(file=f.name, model=mname, n_seed=n, planned=EXPECT[d]))
print(pd.DataFrame(rows).to_string(index=False) if rows else "none")

print("\n(6) manuscript claims vs recomputed values")
tc = pd.read_csv(T / "tong_hop_chinh.csv")
tc_q = tc[tc["dataset"].str.endswith("_q")]
tc = tc[tc["dataset"].isin(["german", "australian", "taiwan"])]
cl = []


def add(name, claimed, got, tol=0.0006):
    cl.append(dict(claim=name, manuscript=claimed, recomputed=round(float(got), 4),
                   status="ok" if abs(claimed - got) <= tol else "CHECK"))


add("German max SMOTE increase", 0.146, tc[tc.dataset == "german"]["smote_tang"].max())
add("Taiwan max SMOTE increase", 0.176, tc[tc.dataset == "taiwan"]["smote_tang"].max())
add("Australian min SMOTE increase", 0.005, tc[tc.dataset == "australian"]["smote_tang"].min())
add("Australian max SMOTE increase", 0.014, tc[tc.dataset == "australian"]["smote_tang"].max())
add("max |preprocessing change|, standard scaling (4.2.1: 0.0010)", 0.0010, tc["prep_tang"].abs().max(), 0.00006)
add("max |preprocessing change|, quantile scaling (4.2.1: below 0.002)", 0.002, tc_q["prep_tang"].abs().max(), 0.002)
tt = pd.read_csv(T / "bang_thoi_gian.csv")
add("max inflation vs out-of-time (abstract 0.32)", 0.32, tt["phong_oot"].max(), 0.005)
add("min inflation vs out-of-time (0.006)", 0.006, tt["phong_oot"].min(), 0.0006)
sf = s.copy()
kn = sf[sf["model"] == "knn5"]["share_real"].dropna()
add("knn5 share min (50%)", 0.50, kn.min(), 0.006)
add("knn5 share max (79%)", 0.79, kn.max(), 0.006)
rf = sf[(sf["model"] == "rf_full")]["share_real"].dropna()
add("rf_full share min (34%)", 0.34, rf.min(), 0.006)
add("rf_full share max (59%)", 0.59, rf.max(), 0.006)
hb = sf[sf["model"].str.startswith("hgb") & sf["dataset"].isin(["german", "taiwan", "lendingclub"])]["share_real"].dropna()
add("boosting share max on German/Taiwan/LC (0.20)", 0.20, hb.max(), 0.006)
ck = pd.read_csv(T / "bang_chi_phi.csv")
for mname, var, claimed in [("logreg", "smote", 0.471), ("logreg", "khong_smote", 0.192), ("hgb_full", "smote", 0.265),
                            ("rf_depth5", "smote", 0.473)]:
    add(f"LC mean prediction {mname} {var}", claimed,
        ck[(ck["model"] == mname) & (ck["variant"] == var)]["mean_pred"].iloc[0], 0.0006)
for d, claimed in [("australian", 0.64), ("german", 0.65), ("taiwan", 0.64), ("lendingclub", 0.60)]:
    x = s[s["dataset"] == d].set_index("model")["real_match"]
    add(f"k 5 -> 25 leakage reduction {d}", claimed, 1 - x["knn25"] / x["knn5"], 0.006)
print(pd.DataFrame(cl).to_string(index=False))