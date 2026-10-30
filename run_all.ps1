# Commands equivalent to the results saved in results\ (seed counts taken from the result files).
# NOT run end to end (takes many hours) and NOT tested on PowerShell. The scripts are resumable:
# a (model, seed) pair already present in the result file is skipped.
# To check reproducibility: copy the folder elsewhere and delete results\.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$J = 2
function Run { python @args; if ($LASTEXITCODE -ne 0) { throw "error while running: python $args" } }

# --- Correct pipeline vs leaks (exp.py) and the no-SMOTE baseline (exp_base.py)
foreach ($d in "german", "australian") {
  Run code\exp.py --data $d --nseeds 30 --jobs $J
  Run code\exp_base.py --data $d --nseeds 30
}
Run code\exp.py --data taiwan --nseeds 10 --jobs $J
Run code\exp_base.py --data taiwan --nseeds 10

# --- Leakage from feature selection
foreach ($d in "german", "australian") { Run code\exp_sel.py --data $d --nseeds 30 --jobs $J }
Run code\exp_sel.py --data taiwan --nseeds 10 --jobs $J

# --- Class balancers
foreach ($b in "ros", "borderline") {
  Run code\exp_bal.py --data german --bal $b --nseeds 30 --jobs $J
  Run code\exp_bal.py --data taiwan --bal $b --nseeds 10 --jobs $J
}

# --- Decomposition of the inflation (SMOTE). Seeds: german/australian 30, taiwan 10, lendingclub 5.
# Old design (different folds for cv_in and the scoring set): exp_diag.py
foreach ($d in "german", "australian") { Run code\exp_diag.py --data $d --bal smote --models logreg hgb_depth2 hgb_depth4 hgb_full rf_depth5 rf_full knn5 --nseeds 30 }
foreach ($b in "borderline", "adasyn") { foreach ($d in "german", "australian") { Run code\exp_diag.py --data $d --bal $b --models logreg hgb_depth2 rf_depth5 knn5 --nseeds 30 } }
Run code\exp_diag.py --data taiwan --bal smote --models logreg hgb_depth2 hgb_depth4 hgb_full rf_depth5 rf_full knn5 --nseeds 10
foreach ($b in "borderline", "adasyn") { Run code\exp_diag.py --data taiwan --bal $b --models logreg hgb_depth2 rf_depth5 knn5 --nseeds 10 }
Run code\exp_diag.py --data lendingclub --bal smote --models logreg hgb_depth2 hgb_depth4 hgb_full rf_depth5 knn5 --nseeds 5
# New design (same folds and same scoring set): exp_samefold.py. --match-syn gives the leaky pipeline the same
# number of synthetic points as the correct one; samematch_* is the main result, samefold_* a sensitivity check.
foreach ($d in "german", "australian") {
  Run code\exp_samefold.py --data $d --bal smote --models logreg hgb_depth2 hgb_depth4 hgb_full rf_depth5 rf_full knn5 knn25 rf_leaf5 --nseeds 30 --match-syn
  Run code\exp_samefold.py --data $d --bal smote --models logreg hgb_depth2 hgb_depth4 hgb_full rf_depth5 rf_full knn5 --nseeds 30
}
Run code\exp_samefold.py --data taiwan --bal smote --models logreg hgb_depth2 hgb_depth4 hgb_full rf_depth5 rf_full knn5 knn25 rf_leaf5 --nseeds 10 --match-syn
Run code\exp_samefold.py --data taiwan --bal smote --models logreg hgb_depth2 hgb_depth4 hgb_full rf_depth5 rf_full knn5 --nseeds 10
Run code\exp_samefold.py --data lendingclub --bal smote --models logreg hgb_depth2 rf_depth5 rf_leaf5 knn5 knn25 --nseeds 5 --match-syn
# Optional and slow (not in the saved results unless you ran them): remaining Lending Club models
# Run code\exp_samefold.py --data lendingclub --bal smote --models hgb_depth4 hgb_full --nseeds 5 --match-syn
# Run code\exp_samefold.py --data lendingclub --bal smote --models rf_full --nseeds 3 --match-syn

# --- QuantileTransformer variant
foreach ($d in "german_q", "australian_q") {
  Run code\exp_q.py --data $d --models logreg knn5 --nseeds 30 --jobs $J
  Run code\exp_base_q.py --data $d --models logreg knn5 --nseeds 30
}
Run code\exp_q.py --data taiwan_q --models logreg knn5 --nseeds 10 --jobs $J
Run code\exp_base_q.py --data taiwan_q --models logreg knn5 --nseeds 10

# --- Lending Club: out-of-time split, decision cost, bootstrap
Run code\exp_time.py --nseeds 5 --jobs $J
Run code\exp_cost.py --nseeds 3
Run code\exp_boot.py --seeds 1 2 3 4 5

# --- Summary tables
Run code\make_tables.py
Run code\sens_table.py
Run code\make_extra.py
