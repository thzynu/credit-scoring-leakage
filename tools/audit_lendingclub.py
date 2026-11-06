"""Lending Club data audit (C3.1). Prints and writes
  results/tables/new_lc_variable_audit.csv   one row per column: role in the models, type, missing, range, flags
  results/tables/new_lc_year_audit.csv       loans and default rate per issue year, and a maturity flag
  results/tables/new_lc_shift_audit.csv      covariate drift 2013-2015 vs 2016 and adversarial-validation AUC
The columns 'available at application time' and 'default definition' CANNOT be decided from the data:
they must be checked against the data dictionary of the Zenodo dataset and of the Lending Club file
(DOI 10.5281/zenodo.11295916) and written in by hand."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
T = ROOT / "results" / "tables"
T.mkdir(exist_ok=True)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
df = pd.read_csv(ROOT / "data" / "lendingclub_100k.csv")
NOT_USED = {"Default": "target", "id": "excluded (identifier; also a proxy for time)", "issue_d": "excluded (issue date)",
            "zip_code": "excluded (3-digit zip, high cardinality)", "year": "excluded (derived from issue_d; used only to split)"}
rows = []
for c in df.columns:
    s = df[c]
    r = dict(variable=c, role=NOT_USED.get(c, "predictor"), dtype=str(s.dtype), n_unique=int(s.nunique()),
             missing=int(s.isna().sum()))
    if pd.api.types.is_numeric_dtype(s):
        r.update(min=s.min(), p01=s.quantile(.01), median=s.median(), p99=s.quantile(.99), max=s.max())
        flags = []
        if c == "dti_n" and (s >= 999).any():
            flags.append(f"{int((s >= 999).sum())} rows with dti >= 999 (sentinel?)")
        if c == "revenue" and s.max() > 20 * s.quantile(.99):
            flags.append(f"max {s.max():.0f} is {s.max() / s.quantile(.99):.0f}x the 99th percentile")
        if s.nunique() <= 2 and c not in ("Default",):
            flags.append(f"almost constant ({s.value_counts(normalize=True).iloc[0]:.4f} share of the commonest value)")
        r["flags"] = "; ".join(flags)
    else:
        top = s.value_counts(normalize=True)
        r.update(flags=f"commonest value '{top.index[0]}' {top.iloc[0]:.3f}; rare (<1%) categories {int((top < 0.01).sum())}")
    r["available_at_application_TO_VERIFY"] = ""
    rows.append(r)
V = pd.DataFrame(rows)
V.round(3).to_csv(T / "new_lc_variable_audit.csv", index=False)
print("===== variables\n", V.drop(columns=["available_at_application_TO_VERIFY"]).round(2).to_string(index=False))
print("\nduplicated ids:", int(df["id"].duplicated().sum()), "| duplicated rows without id:", int(df.drop(columns="id").duplicated().sum()))

Y = df.groupby("year")["Default"].agg(n_loans="size", default_rate="mean").reset_index()
Y["share_of_sample"] = Y["n_loans"] / len(df)
Y["role"] = np.where(Y["year"].isin([2013, 2014, 2015]), "training years",
                     np.where(Y["year"] == 2016, "out-of-time test", "not used"))
Y["maturity_flag"] = np.where(Y["year"] >= 2016, "CHECK: loans issued late may be missing if still running when the file was built", "")
Y.round(4).to_csv(T / "new_lc_year_audit.csv", index=False)
print("\n===== loans and default rate by issue year\n", Y.round(4).to_string(index=False))

tr, te = df[df["year"].isin([2013, 2014, 2015])], df[df["year"] == 2016]
pred = [c for c in df.columns if c not in NOT_USED]
num = [c for c in pred if pd.api.types.is_numeric_dtype(df[c])]
drift = []
for c in num:
    sd = tr[c].std()
    drift.append(dict(variable=c, mean_2013_15=tr[c].mean(), mean_2016=te[c].mean(),
                      std_diff=(te[c].mean() - tr[c].mean()) / sd if sd > 0 else np.nan))
for c in [c for c in pred if c not in num]:
    a, b = tr[c].value_counts(normalize=True), te[c].value_counts(normalize=True)
    drift.append(dict(variable=c, mean_2013_15=np.nan, mean_2016=np.nan,
                      std_diff=float(0.5 * (a.reindex(a.index.union(b.index), fill_value=0)
                                           - b.reindex(a.index.union(b.index), fill_value=0)).abs().sum())))
D = pd.DataFrame(drift)
D["note"] = np.where(D["mean_2013_15"].isna(), "categorical: total-variation distance of the category shares", "standardised mean difference")
X = pd.concat([tr, te])
Xm = X[pred].copy()
for c in Xm.columns:
    if not pd.api.types.is_numeric_dtype(Xm[c]):
        Xm[c] = Xm[c].astype("category").cat.codes
yadv = np.r_[np.zeros(len(tr)), np.ones(len(te))]
p = cross_val_predict(HistGradientBoostingClassifier(random_state=0), Xm, yadv, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                      method="predict_proba")[:, 1]
auc_adv = roc_auc_score(yadv, p)
D.loc[len(D)] = dict(variable="ADVERSARIAL AUC (train years vs 2016, predictors only)", std_diff=auc_adv,
                     note="0.5 = no covariate shift; values well above 0.5 = the test year is distinguishable")
D.round(4).to_csv(T / "new_lc_shift_audit.csv", index=False)
print("\n===== drift 2013-2015 vs 2016\n", D.round(4).to_string(index=False))