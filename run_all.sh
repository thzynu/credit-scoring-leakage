#!/usr/bin/env bash
# Commands equivalent to the results saved in results/ (seed counts taken from the result files).
# NOT run end to end; takes many hours. The scripts are resumable (they skip model-seed pairs already present).
# Run from the project root. To check reproducibility: copy the project elsewhere and delete results/.
set -e
cd "$(dirname "$0")"
P="python code"
J=2

# --- Correct pipeline vs leaks (exp.py) and the no-SMOTE baseline (exp_base.py)
for d in german australian; do $P/exp.py --data $d --nseeds 30 --jobs $J; $P/exp_base.py --data $d --nseeds 30; done
$P/exp.py --data taiwan --nseeds 10 --jobs $J;  $P/exp_base.py --data taiwan --nseeds 10

# --- Leakage from feature selection
for d in german australian; do $P/exp_sel.py --data $d --nseeds 30 --jobs $J; done
$P/exp_sel.py --data taiwan --nseeds 10 --jobs $J

# --- Class balancers
for b in ros borderline; do
  $P/exp_bal.py --data german --bal $b --nseeds 30 --jobs $J
  $P/exp_bal.py --data taiwan --bal $b --nseeds 10 --jobs $J
done

# --- Decomposition of the inflation (SMOTE). Seeds: german/australian 30, taiwan 10, lendingclub 5.
# Old design (different folds for cv_in and the scoring set): exp_diag.py
for d in german australian; do $P/exp_diag.py --data $d --bal smote --models logreg hgb_depth2 hgb_depth4 hgb_full rf_depth5 rf_full knn5 --nseeds 30; done
for b in borderline adasyn; do for d in german australian; do $P/exp_diag.py --data $d --bal $b --models logreg hgb_depth2 rf_depth5 knn5 --nseeds 30; done; done
$P/exp_diag.py --data taiwan --bal smote --models logreg hgb_depth2 hgb_depth4 hgb_full rf_depth5 rf_full knn5 --nseeds 10
for b in borderline adasyn; do $P/exp_diag.py --data taiwan --bal $b --models logreg hgb_depth2 rf_depth5 knn5 --nseeds 10; done
$P/exp_diag.py --data lendingclub --bal smote --models logreg hgb_depth2 hgb_depth4 hgb_full rf_depth5 knn5 --nseeds 5
# New design (same folds and same scoring set): exp_samefold.py. --match-syn gives the leaky pipeline the same
# number of synthetic points as the correct one; samematch_* is the main result, samefold_* a sensitivity check.
for d in german australian; do
  $P/exp_samefold.py --data $d --bal smote --models logreg hgb_depth2 hgb_depth4 hgb_full rf_depth5 rf_full knn5 knn25 rf_leaf5 --nseeds 30 --match-syn
  $P/exp_samefold.py --data $d --bal smote --models logreg hgb_depth2 hgb_depth4 hgb_full rf_depth5 rf_full knn5 --nseeds 30
done
$P/exp_samefold.py --data taiwan --bal smote --models logreg hgb_depth2 hgb_depth4 hgb_full rf_depth5 rf_full knn5 knn25 rf_leaf5 --nseeds 10 --match-syn
$P/exp_samefold.py --data taiwan --bal smote --models logreg hgb_depth2 hgb_depth4 hgb_full rf_depth5 rf_full knn5 --nseeds 10
$P/exp_samefold.py --data lendingclub --bal smote --models logreg hgb_depth2 hgb_depth4 hgb_full rf_depth5 rf_leaf5 knn5 knn25 --nseeds 5 --match-syn
# Optional and slow (rf_full is NOT in the saved results for Lending Club):
# $P/exp_samefold.py --data lendingclub --bal smote --models rf_full --nseeds 3 --match-syn

# --- QuantileTransformer variant
for d in german_q australian_q; do $P/exp_q.py --data $d --models logreg knn5 --nseeds 30 --jobs $J; $P/exp_base_q.py --data $d --models logreg knn5 --nseeds 30; done
$P/exp_q.py --data taiwan_q --models logreg knn5 --nseeds 10 --jobs $J; $P/exp_base_q.py --data taiwan_q --models logreg knn5 --nseeds 10

# --- Lending Club: out-of-time split, decision cost, bootstrap
$P/exp_time.py --nseeds 5 --jobs $J
$P/exp_cost.py --nseeds 3
$P/exp_boot.py --seeds 1 2 3 4 5

# --- Summary tables
$P/make_tables.py
$P/sens_table.py
$P/make_extra.py

# --- Added for the revision (closed decomposition, ratio, SMOTENC, calibration, scoring)
python code/exp_joint.py --data german --models logreg knn5 rf_depth5 rf_full hgb_depth2 hgb_depth4 hgb_full knn1 knn3 knn15 knn25 knn50 knn100 rf_depth3 rf_depth10 rf_leaf2 rf_leaf5 rf_leaf10 rf_leaf20 hgb_depth1 hgb_depth3 hgb_depth6 hgb_depth4_iter300 hgb_depth2_noes hgb_full_noes --nseeds 30
python code/exp_joint.py --data australian --models logreg knn5 rf_depth5 rf_full hgb_depth2 hgb_depth4 hgb_full knn1 knn3 knn15 knn25 knn50 knn100 rf_depth3 rf_depth10 rf_leaf2 rf_leaf5 rf_leaf10 rf_leaf20 hgb_depth1 hgb_depth3 hgb_depth6 hgb_depth4_iter300 hgb_depth2_noes hgb_full_noes --nseeds 30
python code/exp_joint.py --data taiwan --models logreg knn5 rf_depth5 rf_full hgb_depth2 hgb_depth4 hgb_full knn1 knn3 knn15 knn25 knn50 knn100 rf_depth3 rf_depth10 rf_leaf2 rf_leaf5 rf_leaf10 rf_leaf20 hgb_depth1 hgb_depth3 hgb_depth6 hgb_depth4_iter300 hgb_depth2_noes hgb_full_noes --nseeds 10
python code/exp_joint.py --data lendingclub --models logreg knn5 knn25 rf_depth5 hgb_depth2 hgb_depth4 hgb_full hgb_depth2_noes hgb_full_noes --nseeds 5
python code/exp_provenance.py --data german --models rf_depth5 hgb_depth4 knn25 rf_leaf5 --nseeds 30
python code/exp_provenance.py --data australian --models rf_depth5 hgb_depth4 knn25 rf_leaf5 --nseeds 30
python code/exp_provenance.py --data taiwan --models rf_depth5 rf_full rf_leaf5 --nseeds 10
python code/exp_provenance.py --data lendingclub --models logreg knn5 knn25 hgb_depth2 hgb_depth2_noes hgb_full hgb_full_noes hgb_depth4 --nseeds 5
python code/exp_scoring.py --data german --models rf_depth5 hgb_depth4 knn25 rf_leaf5 --nseeds 30
python code/exp_scoring.py --data australian --models rf_depth5 hgb_depth4 knn25 rf_leaf5 --nseeds 30
python code/exp_scoring.py --data lendingclub --models rf_depth5 hgb_depth4 --nseeds 5
python code/exp_ratio.py --data german --nseeds 30
python code/exp_ratio.py --data australian --ratios 0.85 0.9 1.0 --nseeds 30
python code/exp_ratio.py --data taiwan --models logreg knn5 hgb_depth2 --nseeds 10
python code/exp_smotenc.py --data german --nseeds 30
python code/exp_bal.py --data australian --bal ros --models logreg hgb_depth2 rf_depth5 knn5 --nseeds 30
python code/exp_calib.py --nseeds 10
python tools/summarize_provenance.py
python tools/summarize_new.py
python tools/rank_stats.py
python tools/check_consistency.py
