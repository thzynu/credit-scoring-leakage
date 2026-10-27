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
# GMSC: knn5 and rf_full are very slow; the saved version has only 3 seeds for these two models
Run code\exp.py --data gmsc --models logreg hgb_depth2 hgb_depth4 hgb_full rf_depth5 --nseeds 5 --jobs $J
Run code\exp_base.py --data gmsc --models logreg hgb_depth2 hgb_depth4 hgb_full rf_depth5 --nseeds 5

# --- Leakage from feature selection
foreach ($d in "german", "australian") { Run code\exp_sel.py --data $d --nseeds 30 --jobs $J }
Run code\exp_sel.py --data taiwan --nseeds 10 --jobs $J
Run code\exp_sel.py --data gmsc --models logreg hgb_depth2 rf_depth5 --nseeds 5 --jobs $J

# --- Class balancers
foreach ($b in "ros", "borderline") {
  Run code\exp_bal.py --data german --bal $b --nseeds 30 --jobs $J
  Run code\exp_bal.py --data taiwan --bal $b --nseeds 10 --jobs $J
}

# --- Decomposition: "true leakage" vs "due to synthetic points"
Run code\exp_diag.py --data german --bal smote --models logreg hgb_depth2 hgb_depth4 hgb_full rf_depth5 rf_full knn5 --nseeds 30
Run code\exp_diag.py --data german --bal borderline --nseeds 30
Run code\exp_diag.py --data taiwan --bal smote --models logreg hgb_depth2 rf_depth5 knn5 --nseeds 5
Run code\exp_diag.py --data taiwan --bal smote --models hgb_full rf_full --nseeds 3
Run code\exp_diag.py --data taiwan --bal borderline --models logreg hgb_depth2 --nseeds 5

# --- QuantileTransformer variant
foreach ($d in "german_q", "australian_q") {
  Run code\exp_q.py --data $d --models logreg knn5 --nseeds 30 --jobs $J
  Run code\exp_base_q.py --data $d --models logreg knn5 --nseeds 30
}
Run code\exp_q.py --data taiwan_q --models logreg knn5 --nseeds 10 --jobs $J
Run code\exp_base_q.py --data taiwan_q --models logreg knn5 --nseeds 10
Run code\exp_q.py --data gmsc_q --models logreg --nseeds 5 --jobs $J
Run code\exp_base_q.py --data gmsc_q --models logreg --nseeds 5

# --- Lending Club: out-of-time split, decision cost, bootstrap
Run code\exp_time.py --nseeds 5 --jobs $J
Run code\exp_cost.py --nseeds 3
Run code\exp_boot.py

# --- Summary tables
Run code\make_tables.py
