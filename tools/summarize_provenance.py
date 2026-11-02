"""Summarise the provenance-tracked decomposition (exp_provenance.py).

Reads results/provenance_<data>_smote.csv and results/provgeom_<data>_smote.csv and writes
  results/tables/bang_provenance.csv       one row per dataset x model
  results/tables/bang_provenance_geom.csv  one row per dataset (distances, neighbour shares)

Columns of bang_provenance.csv (all AUC differences are on the SAME validation points):
  n_seed
  cv_in            AUC of the correct pipeline (SMOTE fitted inside the training part)
  syn_effect_all   leak_all  - cv_in   original training part + ALL globally generated synthetic points
  syn_effect_m05   leak_m05  - cv_in   matched count x 0.5
  syn_effect_m1    leak_matched - cv_in matched count x 1 (as many synthetic points as the correct pipeline)
  syn_effect_m2    leak_m2   - cv_in   matched count x 2
  clean_effect     clean_all - cv_in   only synthetic points whose two parents are both in the training part
  direct_eq        contam_eq - clean_eq  same number of synthetic points, with vs without a validation parent
  direct_eq_se     standard error over seeds (seeds are repeated splits of one dataset, so they are
                   NOT independent samples: use as a measure of split-to-split variation only)
  direct_pos_share share of seeds with direct_eq > 0
  contam_share     contaminated synthetic points / all synthetic points (mean over folds and seeds)
Directly measured: direct_eq, clean_effect, syn_effect_*. Not measured: why a model profits from a
contaminated point (memorisation is a hypothesis).
"""
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parent.parent / "results"
T = R / "tables"
T.mkdir(exist_ok=True)


def se(s):
    return s.std() / len(s) ** 0.5 if len(s) > 1 else float("nan")


rows, geo = [], []
for f in sorted(R.glob("provenance_*_smote.csv")):
    d = pd.read_csv(f)
    d["contam_share"] = d["n_contam"] / d["n_syn_total"]
    for name, g in d.groupby("model"):
        rows.append(dict(
            dataset=g["dataset"].iloc[0], model=name, n_seed=g["seed"].nunique(),
            cv_in=g["cv_in"].mean(),
            syn_effect_all=(g["leak_all"] - g["cv_in"]).mean(),
            syn_effect_m05=(g["leak_m05"] - g["cv_in"]).mean(),
            syn_effect_m1=(g["leak_matched"] - g["cv_in"]).mean(),
            syn_effect_m2=(g["leak_m2"] - g["cv_in"]).mean(),
            clean_effect=(g["clean_all"] - g["cv_in"]).mean(),
            clean_effect_se=se(g["clean_all"] - g["cv_in"]),
            direct_eq=g["direct_eq"].mean(), direct_eq_se=se(g["direct_eq"]),
            direct_pos_share=(g["direct_eq"] > 0).mean(),
            contam_share=g["contam_share"].mean()))
for f in sorted(R.glob("provgeom_*_smote.csv")):
    d = pd.read_csv(f)
    geo.append(dict(dataset=d["dataset"].iloc[0], n_seed=d["seed"].nunique(),
                    d_nearest_original_pos=d["d_origpos"].mean(),
                    d_nearest_clean_syn=d["d_clean"].mean(),
                    d_nearest_contaminated_syn=d["d_contam"].mean(),
                    nn5_share_contaminated=d["nn5_contam_pos"].mean(),
                    nearest_syn_is_contaminated=d["share_nearest_syn_contam"].mean()))

out = pd.DataFrame(rows).sort_values(["dataset", "direct_eq"], ascending=[True, False])
out.round(4).to_csv(T / "bang_provenance.csv", index=False)
g = pd.DataFrame(geo)
g.round(3).to_csv(T / "bang_provenance_geom.csv", index=False)
pd.set_option("display.width", 220)
print(out.round(4).to_string(index=False))
print()
print(g.round(3).to_string(index=False))
