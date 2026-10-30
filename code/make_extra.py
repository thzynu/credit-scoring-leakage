"""Extra outputs for the paper, built only from files already in results/.

Run (after make_tables.py):  python code/make_extra.py
Writes:
  results/tables/bang_thu_hang_mo_hinh.csv   rank of each model by correct hold-out vs by leaky CV
  results/tables/bang_thu_hang_tom_tat.csv   per dataset: Kendall tau between the two rankings, top model
  results/tables_en/*.csv                    every table in results/tables with English column names
  results/figures/fig1_decomposition.(png|pdf)  true leakage vs remainder, by model
  results/figures/fig2_memorisation.(png|pdf)   true leakage vs model memorisation (kNN k, RF depth)
  results/figures/fig3_leaky_vs_true.(png|pdf)  leaky CV AUC vs correct hold-out AUC
"""
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
TAB = RES / "tables"
TAB_EN = RES / "tables_en"
FIG = RES / "figures"
for d in (TAB, TAB_EN, FIG):
    d.mkdir(exist_ok=True)

MODEL_ORDER = ["logreg", "hgb_depth2", "hgb_depth4", "hgb_full",
               "rf_depth5", "rf_leaf5", "rf_full", "knn25", "knn5"]
DATASETS = ["german", "australian", "taiwan", "lendingclub"]


def read_all(pattern):
    files = sorted(glob.glob(str(RES / pattern)))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def save_fig(fig, name, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=300)
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    print("figure:", name)


# ------------------------------------------------------------------ rank tables
# Correct hold-out AUC vs leaky CV AUC (SMOTE and preprocessing before the fold split).
v = read_all("*_v2.csv")
v = v[(v["frac"] == 1.0) & (~v["dataset"].str.endswith("_q"))]
a = v.groupby(["dataset", "model"]).agg(n_seed=("seed", "nunique"), holdout=("holdout", "mean"),
                                        leaky_cv=("leak_both", "mean")).reset_index()
t = read_all("lendingclub_time.csv")                      # Lending Club: out-of-time test year
if len(t):
    t = t[t["frac"] == 1.0]
    b = t.groupby("model").agg(n_seed=("seed", "nunique"), holdout=("holdout_oot", "mean"),
                               leaky_cv=("leak_both", "mean")).reset_index()
    b.insert(0, "dataset", "lendingclub")
    a = pd.concat([a, b], ignore_index=True)
a["rank_true"] = a.groupby("dataset")["holdout"].rank(ascending=False, method="min").astype(int)
a["rank_leaky"] = a.groupby("dataset")["leaky_cv"].rank(ascending=False, method="min").astype(int)
a["rank_change"] = a["rank_true"] - a["rank_leaky"]       # >0: the leaky CV makes the model look better
a["inflation"] = a["leaky_cv"] - a["holdout"]
a = a.sort_values(["dataset", "rank_true"])
a.round(4).to_csv(TAB / "bang_thu_hang_mo_hinh.csv", index=False)

rows = []
for ds, g in a.groupby("dataset"):
    tau, p = kendalltau(g["holdout"], g["leaky_cv"])
    rows.append(dict(dataset=ds, n_models=len(g), kendall_tau=tau, p_value=p,
                     top_true=g.loc[g["rank_true"].idxmin(), "model"],
                     top_leaky=g.loc[g["rank_leaky"].idxmin(), "model"],
                     min_seeds=g["n_seed"].min()))
s = pd.DataFrame(rows)
s.round(4).to_csv(TAB / "bang_thu_hang_tom_tat.csv", index=False)
print("\n== ranking summary\n", s.round(4).to_string(index=False))

# ------------------------------------------------------------------ English tables
RENAME = {
    "cv_dung": "cv_correct", "leak_impute": "cv_leak_impute", "leak_prep": "cv_leak_prep",
    "leak_both": "cv_leak_balance", "holdout": "holdout_auc", "phong": "inflation",
    "prep_tang": "prep_increase", "smote_tang": "balance_increase", "sel_tang": "selection_increase",
    "smote_tang_se": "balance_increase_se", "sel_tang_se": "selection_increase_se",
    "delta_smote": "smote_effect", "delta_smote_se": "smote_effect_se",
    "holdout_smote": "holdout_with_smote", "holdout_nosmote": "holdout_no_smote",
    "tang": "increase", "tang_se": "increase_se", "n_seed": "n_seeds",
    "hang_that": "rank_true", "hang_ro_ri": "rank_leaky", "giay": "seconds",
    "phong_oot": "inflation_oot", "rand_hon_oot": "random_minus_oot",
    "holdout_rand": "holdout_random", "holdout_oot": "holdout_out_of_time",
    "real": "true_leakage", "real_se": "true_leakage_se", "synth": "synthetic_in_scoring",
    "share_synth": "share_synthetic", "auc_syn": "auc_synthetic_only",
    "real_same": "true_leakage_same_folds", "real_same_se": "true_leakage_same_folds_se",
    "real_match": "true_leakage_matched", "real_match_se": "true_leakage_matched_se",
    "real_old": "true_leakage_old_design", "rest": "remainder", "share_real": "share_true_leakage",
}
VALUES = {"khong_smote": "no_smote"}
for f in sorted(TAB.glob("*.csv")):
    d = pd.read_csv(f)
    d = d.rename(columns=RENAME)
    for c in d.columns:
        if d[c].dtype == object:
            d[c] = d[c].replace(VALUES)
    d.to_csv(TAB_EN / f.name, index=False)
print("English tables:", len(list(TAB_EN.glob('*.csv'))))

# ------------------------------------------------------------------ Figure 1: decomposition
dec = pd.read_csv(TAB / "bang_phan_ra_samefold.csv")
dec = dec[dec["balancer"] == "smote"].copy()
col = "real_match" if "real_match" in dec.columns else "real_same"
dec = dec.dropna(subset=["total", col])
fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=False)
for ax, ds in zip(axes, DATASETS):
    g = dec[dec["dataset"] == ds]
    g = g.set_index("model").reindex([m for m in MODEL_ORDER if m in set(g["model"])])
    x = np.arange(len(g))
    ax.bar(x - 0.2, g[col], 0.4, label="true leakage (training set)")
    ax.bar(x + 0.2, g["total"] - g[col], 0.4, label="remainder (total minus true leakage)")
    ax.set_xticks(x)
    ax.set_xticklabels(g.index, rotation=60, ha="right", fontsize=8)
    ax.set_title(ds)
    ax.axhline(0, color="k", lw=0.5)
axes[0].set_ylabel("AUC inflation")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, fontsize=8, frameon=False)
fig.subplots_adjust(bottom=0.32)
save_fig(fig, "fig1_decomposition", tight=False)

# ------------------------------------------------------------------ Figure 2: memorisation
sm = read_all("samematch_*.csv")
sm = sm[sm["balancer"] == "smote"]
g = sm.groupby(["dataset", "model"])["real_same"].agg(["mean", "sem"]).reset_index()
groups = [("kNN: larger k memorises less", ["knn5", "knn25"]),
          ("Random forest: deeper trees memorise more", ["rf_depth5", "rf_leaf5", "rf_full"])]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for ax, (title, models) in zip(axes, groups):
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        h = g[g["model"] == m].set_index("dataset").reindex(DATASETS)
        ax.bar(np.arange(len(DATASETS)) + i * w - 0.4 + w / 2, h["mean"], w,
               yerr=h["sem"], label=m, capsize=2)
    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels(DATASETS)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
axes[0].set_ylabel("true leakage (AUC)")
save_fig(fig, "fig2_memorisation")

# ------------------------------------------------------------------ Figure 3: leaky vs true
fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
for ax, ds in zip(axes, DATASETS):
    h = a[a["dataset"] == ds]
    ax.scatter(h["holdout"], h["leaky_cv"])
    for _, r in h.iterrows():
        ax.annotate(r["model"], (r["holdout"], r["leaky_cv"]), fontsize=6,
                    xytext=(2, 2), textcoords="offset points")
    lo = min(h["holdout"].min(), h["leaky_cv"].min()) - 0.01
    hi = max(h["holdout"].max(), h["leaky_cv"].max()) + 0.01
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.6)
    ax.set_title(ds)
    ax.set_xlabel("correct hold-out AUC")
axes[0].set_ylabel("leaky CV AUC")
save_fig(fig, "fig3_leaky_vs_true")
