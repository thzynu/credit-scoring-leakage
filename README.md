# Preprocessing leakage and class balancing in credit scoring: experiment code

This package contains the code and raw results of experiments that measure how much AUC is
"inflated" when preprocessing steps (scaling, imputation, feature selection, class balancing) are
performed before splitting folds instead of inside each fold. It accompanies the manuscript
"Where Does the Inflation Come From? Decomposing Cross-Validation Leakage from Pre-Split Oversampling
in Credit Scoring" (in preparation; see Citation below).

## Status

| Item | Status |
|---|---|
| Experiment scripts | Logic kept exactly as it was run. Only changed: root path, removed BOM/CRLF, added 1 safety assert in `exp_diag.py` |
| Re-run check | German, seed 1: `exp.py`, `exp_base.py`, `exp_bal.py`, `exp_diag.py`, `exp_sel.py` match `results/` digit for digit (difference 0). `exp_time.py`, `exp_cost.py`, `exp_boot.py`, `exp_q.py` were only smoke-tested (they run), numbers not compared |
| `make_tables.py` | Rebuilds every summary table from the raw files in `results/` |
| Full run from scratch | Partly done (`repro_full`, 2026-10-29). Every row present in both runs agrees exactly (max difference 0) for the `_nosmote`, `_sel`, `_q_*`, `bal_*`, `lendingclub_time`, `lendingclub_cost`, `diag_german_*` and `diag_taiwan_*` files and for `_v2` of German and Australian. Exception: `cv_dung` and `leak_prep` in `taiwan_v2.csv` differ by up to 1.4e-3 (limitation 10). Second check (`repro_check`, 2026-10-30, `tools/compare_runs.py`): every row present in both runs agrees exactly (max difference 0, no extra rows) for `diag_australian_{smote,borderline,adasyn}`, `diag_german_{borderline,adasyn}`, `german_bal_smote`, `samefold_{german,australian}_smote` and `samematch_{german,australian}_smote` (all models, seeds 1 to 3), and for `diag_taiwan_adasyn`, `diag_lendingclub_smote`, `samematch_taiwan_smote`, `samematch_lendingclub_smote` (`logreg` and `hgb_depth2` only, seed 1). Not re-run: `samefold_taiwan`, `samefold_lendingclub`, the other models on Taiwan and Lending Club, and seeds above 3 for German and Australian |
| Data sources and licences | Documented in `DATA.md`. GMSC was excluded (Kaggle licence: Competition Use only) |

Environment (see `requirements-lock.txt`, Python 3.13.12): scikit-learn 1.8.0, imbalanced-learn
0.14.1, pandas 3.0.2, numpy 2.4.3. German seed 1 numbers were also reproduced on a slightly
different environment (imbalanced-learn 0.14.2, numpy 2.4.4, Python 3.12.3), but that is not
evidence for every cell.

## Citation

If you use this code or the results, please cite the manuscript (details to be added on publication) and the
archived release of this repository: https://doi.org/10.5281/zenodo.23797523 (all versions: https://doi.org/10.5281/zenodo.23797522).

## License

The code in `code/` and `tools/` is released under the MIT License (see `LICENSE`). The data files in `data/`
keep the licences of their original sources, listed in `DATA.md` (all CC BY 4.0).

## Structure

```
code/             experiment scripts + make_tables.py
data/             CSV files (see DATA.md)
results/          raw results of every run (each row = one model x one seed)
results/tables/   summary tables produced by make_tables.py
legacy/           old scripts and results that have been superseded, kept for comparison
tools/            helper scripts used to check reproducibility (`check_taiwan.py`, `compare_runs.py`)
DATA.md           source, licence and citation of every dataset
```

## Install and run

```
pip install -r requirements.txt
python code/exp.py --data german --models logreg --nseeds 1
python code/make_tables.py
```

All scripts are resumable: if a (model, seed) pair is already in the result file, it is skipped.
So if you re-run inside a `results/` folder that already has data, the script will not recompute
anything. To check reproducibility, copy the folder elsewhere and delete `results/`.
`run_all.ps1` (Windows) and `run_all.sh` list the commands equivalent to the saved results.

## What each script measures

| Script | Measures | Output |
|---|---|---|
| `exp.py` | The correct pipeline versus the leaky variants: imputation, whole-data preprocessing, SMOTE before splitting folds, and hold-out. `--data` = taiwan, german, australian, lendingclub | `results/<data>_v2.csv` |
| `exp_base.py` | Hold-out without SMOTE (baseline) | `<data>_nosmote.csv` |
| `exp_sel.py` | Leakage from feature selection (`SelectKBest`) before splitting folds, at ratio `--kfrac` | `<data>_sel.csv` |
| `exp_bal.py` | Comparison of class balancers: SMOTE, RandomOverSampler, BorderlineSMOTE, ADASYN | `<data>_bal_<balancer>.csv` |
| `exp_diag.py` | Splits the inflation into "true leakage" and "synthetic points ending up in the scoring set" | `diag_<data>_<balancer>.csv` |
| `exp_samefold.py` | Repeats the decomposition with the correct and the leaky model scored on exactly the same folds and original points; `--match-syn` also matches the number of synthetic points | `samefold_*.csv`, `samematch_*.csv` |
| `exp_time.py` | Lending Club: random split vs out-of-time split (train 2013 to 2015, test 2016) | `lendingclub_time.csv` |
| `exp_cost.py` | Lending Club: decision cost with cost ratios 3, 5, 10; threshold chosen on the validation set | `lendingclub_cost.csv` |
| `exp_boot.py` | Bootstrap confidence intervals of the 2016 AUC and of the paired AUC differences between models | printed to screen and saved to `results/lendingclub_boot.csv` |
| `exp_q.py`, `exp_base_q.py` | Variants using `QuantileTransformer` instead of `StandardScaler` (dataset names get the suffix `_q`) | `<data>_q_*.csv` |
| `make_tables.py` | Builds every summary table from the raw files | `results/tables/*.csv` |

## How to read the tables

- `tong_hop_chinh.csv`: the CV score increase caused by each type of leakage (`prep_tang`,
  `smote_tang`, `sel_tang`), the correct hold-out, and the effect of SMOTE (`delta_smote`).
- `bang_phan_ra.csv`: `real = auc_orig - cv_in` is true leakage; `synth = auc_all - auc_orig` is due
  to synthetic points being in the scoring set; `total = auc_all - cv_in` is the inflation a reader
  sees. `share_synth` is only meaningful when |total| is larger than about 0.01 (it is left empty
  otherwise).
- `bang_phan_ra_samefold.csv`: same-folds decomposition. `real_match` (matched synthetic count) is the main estimate of true leakage; `rest = total - real_old` mixes two designs, so call it "the remainder", not a measured component.
- `bang_can_bang.csv`: `tang = cv_before - cv_in`, by balancer.
- `bang_thoi_gian.csv`, `bang_chi_phi.csv`: Lending Club.
- `bao_phu_seed.csv`: number of seeds in each cell. **Look at this table before quoting any number.**

## Coverage (see `bao_phu_seed.csv` for every cell)

| Dataset | Seeds | What is covered |
|---|---|---|
| German, Australian | 30 | all 7 models in `_v2`, `_nosmote`, and `diag_*_smote`; 4 models for ROS, Borderline, ADASYN and feature selection |
| Taiwan | 10 | all 7 models in `_v2` and `_nosmote`; 4 models for ROS, Borderline, feature selection; `diag_*_smote`: all 7 models with 10 seeds; `diag_*_borderline` and `diag_*_adasyn`: 4 models (`logreg`, `hgb_depth2`, `rf_depth5`, `knn5`) with 10 seeds |
| Lending Club | 3 to 5 | `diag_*_smote`: `logreg`, `hgb_depth2`, `hgb_depth4`, `hgb_full`, `rf_depth5`, `knn5` (5 seeds); `samematch_*`: `logreg`, `hgb_depth2`, `hgb_depth4`, `hgb_full`, `rf_depth5`, `rf_leaf5`, `knn5`, `knn25` (5 seeds); out-of-time: 7 models (5 seeds); cost: 3 models (3 seeds). No `rf_full` in the decomposition and no `_v2` hold-out run |

## Known limitations (must be stated in the paper)

1. **Uneven number of seeds** (table above). Quote the seed count next to every number.
2. **The decomposition is incomplete.** ADASYN and Borderline were run for German, Australian and Taiwan
   (4 models each: `logreg`, `hgb_depth2`, `rf_depth5`, `knn5`); Lending Club has SMOTE only, without `rf_full`.
3. `german_bal_smote.csv` now has 30 seeds for 4 models. For all-model SMOTE results use `diag_*`
   or the `smote_tang` column in `tong_hop_chinh.csv`.
4. Hyperparameters are left at their defaults, not tuned.
5. `cv_in` and `auc_orig` use two different fold splits (arrays of different sizes), so part of the
   difference may come from fold composition. Averaging over many seeds reduces this but does not
   eliminate it. `exp_samefold.py` (with `--match-syn`) repeats the decomposition on identical folds and a matched
   number of synthetic points; it agrees with the original decomposition to within about 0.002
   (`bang_phan_ra_samefold.csv`), so this is a robustness check rather than a flaw. It has only
   5 seeds for Lending Club (3 without `--match-syn`).
6. The SEs in the tables only reflect the variation between splits on the same dataset. Do not use the
   very large z values as confidence in the conclusions.
7. `exp_boot.py` runs one bootstrap per training seed (default seeds 1 to 5) and resamples only the 2016
   test set. It reports test-set uncertainty and the spread across training seeds separately; it gives no
   confidence interval for the leakage inflation itself.
8. The mechanism explanations (the model "memorizing" the original points; Borderline flipping sign
   because synthetic points lie close to the decision boundary) are hypotheses, not directly tested.
9. Data provenance: see `DATA.md`. The Lending Club sample is verified row by row against the Zenodo
   file (by `id`), but the sampling script and seed were not recorded. GMSC was
   excluded (Kaggle licence: Competition Use only).
10. **Parallelism changes a few numbers.** `cv_dung` and `leak_prep` of `exp.py` (cross-validation with SMOTE
   inside the fold) depend on `--jobs`: for Taiwan, `--jobs 4` (original run) and `--jobs 2` (re-run) differ by up
   to 1.4e-3, and `--jobs 1` gives a third value. `leak_both` and `holdout` do not change. The likely cause (not
   tested) is that the number of BLAS threads changes with `--jobs`, which changes floating-point sums and the
   nearest-neighbour choice in SMOTE when distances are almost tied. The effect is small against the inflation
   (0.07 to 0.17) but similar to the standard error of the means (about 0.001), so read `prep_tang` as 0 within
   about +-0.002. To re-run Taiwan use `--jobs 4` as in the original; other machines may still differ by about 1e-3.

## Glossary of Vietnamese names kept in code and result files

Column, file, and label names were left unchanged so the code still matches the saved `results/`.

| Name | Meaning |
|---|---|
| `cv_dung` | correct CV AUC (preprocessing/SMOTE inside each fold) |
| `phong` | inflation = `leak_both - holdout` |
| `tang` | increase (`*_tang`: increase caused by a given leakage) |
| `giay` | seconds (run time) |
| `khong_smote` | "no SMOTE" variant label in `lendingclub_cost.csv` |
| `tong_hop_chinh` | main summary |
| `bang_phan_ra` | decomposition table |
| `bang_can_bang` | balancing table |
| `bang_thoi_gian` | time (out-of-time) table |
| `bang_chi_phi` | cost table |
| `bao_phu_seed` | seed coverage |
| `legacy/tong_hop_*`, `xem_du_lieu` | old summary scripts, "view data" |
| `hang_that`, `hang_ro_ri` | true rank, leaky rank |
