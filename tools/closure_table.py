"""Closure table: does measuring the two channels reproduce the total inflation?

For every dataset x model:
  total             CV-to-CV inflation of the leaky CV (SMOTE before the fold split minus the correct CV):
                    smote_tang from tong_hop_chinh.csv (German, Australian, Taiwan) or total from
                    bang_phan_ra.csv (Lending Club, SMOTE rows)
  training_channel  syn_effect_all of bang_provenance.csv: the model trained with all globally generated
                    synthetic points, scored on original validation points only (measured)
  scoring_channel   score_all of results/scoring_*_smote.csv: the correctly trained model scored on the
                    validation points plus synthetic points (measured)
  interaction       total - training_channel - scoring_channel (what the two measured channels do not
                    explain; negative = the channels overlap, positive = something else adds)
Shares are left empty when |total| < 0.01 (too small to divide by).
Output: results/tables/bang_khep_kin.csv
"""
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parent.parent / "results"
T = R / "tables"

sc = pd.concat([pd.read_csv(f) for f in sorted(R.glob("scoring_*_smote.csv"))])
sc = sc.groupby(["dataset", "model"]).agg(
    n_seed_scoring=("seed", "nunique"), scoring_channel=("score_all", "mean")).reset_index()

pr = pd.read_csv(T / "bang_provenance.csv")[["dataset", "model", "n_seed", "syn_effect_all"]]
pr = pr.rename(columns={"n_seed": "n_seed_training", "syn_effect_all": "training_channel"})

main = pd.read_csv(T / "tong_hop_chinh.csv")[["dataset", "model", "n_seed", "smote_tang"]]
main = main.rename(columns={"n_seed": "n_seed_total", "smote_tang": "total"})
lc = pd.read_csv(T / "bang_phan_ra.csv")
lc = lc[(lc["dataset"] == "lendingclub") & (lc["balancer"] == "smote")][["dataset", "model", "n", "total"]]
lc = lc.rename(columns={"n": "n_seed_total"})
tot = pd.concat([main, lc], ignore_index=True)

d = tot.merge(pr, on=["dataset", "model"]).merge(sc, on=["dataset", "model"])
d["sum_channels"] = d["training_channel"] + d["scoring_channel"]
d["interaction"] = d["total"] - d["sum_channels"]
big = d["total"].abs() >= 0.01
d["share_training"] = (d["training_channel"] / d["total"]).where(big)
d["share_scoring"] = (d["scoring_channel"] / d["total"]).where(big)
d = d.sort_values(["dataset", "total"], ascending=[True, False])
d.round(4).to_csv(T / "bang_khep_kin.csv", index=False)
pd.set_option("display.width", 220)
print(d.round(4).to_string(index=False))
