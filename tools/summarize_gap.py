"""Generalisation gap versus leakage effect (C2.1, C2.2). Joins results/gap_<data>.csv with
results/joint_<data>_smote.csv on (model, seed), averages over the common seeds, and reports
Spearman correlations between the gap and the leakage effect (all models, and inside each family).
The gap is a supporting indicator, not a measure of memorisation. Writes
results/tables/new_gap_leak.csv, new_gap_corr.csv, new_gap_exceptions.csv and the figures
results/figures/new_fig_gap_leak.png (gap vs leakage) and new_fig_capacity_gap.png (capacity)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

R = Path(__file__).resolve().parent.parent / "results"
T, FG = R / "tables", R / "figures"
T.mkdir(exist_ok=True)
FG.mkdir(exist_ok=True)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
FAM = {"knn": "k-NN", "rf": "Random forest", "hgb": "Boosting", "logreg": "Logistic regression"}
COL = {"k-NN": "#c0392b", "Random forest": "#27ae60", "Boosting": "#2980b9", "Logistic regression": "#7f8c8d"}
ORDER = {"knn": ["knn1", "knn3", "knn5", "knn15", "knn25", "knn50", "knn100"],
         "rf": ["rf_depth3", "rf_depth5", "rf_depth10", "rf_leaf20", "rf_leaf10", "rf_leaf5", "rf_leaf2", "rf_full"],
         "hgb": ["hgb_depth1", "hgb_depth2", "hgb_depth3", "hgb_depth4", "hgb_depth6", "hgb_full"]}


def family(m):
    for k, v in FAM.items():
        if m.startswith(k):
            return v
    return "other"


rows = []
for f in sorted(R.glob("gap_*.csv")):
    d = f.stem.replace("gap_", "")
    j = R / f"joint_{d}_smote.csv"
    if not j.exists():
        continue
    m = pd.read_csv(f).merge(pd.read_csv(j), on=["model", "seed"], suffixes=("", "_j"))
    m = m[~m["model"].str.endswith("_noes") & (m["model"] != "hgb_depth4_iter300")]
    g = m.groupby("model").agg(n_seed=("seed", "nunique"), train_auc=("train_auc", "mean"), val_auc=("val_auc", "mean"),
                               gap=("gap", "mean"), leak=("leak_eff", "mean"), score=("score_eff", "mean"),
                               joint=("joint_eff", "mean")).reset_index()
    g.insert(0, "dataset", d)
    g["family"] = g["model"].map(family)
    rows.append(g)
if not rows:
    raise SystemExit("no matching gap_*.csv and joint_*_smote.csv")
G = pd.concat(rows, ignore_index=True)
G.round(4).to_csv(T / "new_gap_leak.csv", index=False)
print(G.round(4).to_string(index=False))

cr = []
for d, g in G.groupby("dataset"):
    for name, h in [("all models", g)] + [(fam, g[g["family"] == fam]) for fam in FAM.values()]:
        if len(h) >= 4:
            rho, p = stats.spearmanr(h["gap"], h["leak"])
            rho2, p2 = stats.spearmanr(h["gap"], h["joint"])
            cr.append(dict(dataset=d, group=name, n_models=len(h), rho_gap_leak=rho, p_leak=p, rho_gap_joint=rho2, p_joint=p2))
C = pd.DataFrame(cr)
C.round(3).to_csv(T / "new_gap_corr.csv", index=False)
print("\nSpearman correlation of the generalisation gap with the leakage / joint effect (model-level means)")
print(C.round(3).to_string(index=False))

ex = []
for d, g in G.groupby("dataset"):
    g = g.copy()
    g["r_gap"] = g["gap"].rank()
    g["r_leak"] = g["leak"].rank()
    g["rank_diff"] = g["r_leak"] - g["r_gap"]
    ex.append(g.reindex(g["rank_diff"].abs().sort_values(ascending=False).index).head(5)[
        ["dataset", "model", "gap", "leak", "r_gap", "r_leak", "rank_diff"]])
E_ = pd.concat(ex)
E_.round(4).to_csv(T / "new_gap_exceptions.csv", index=False)
print("\nLargest rank disagreements between gap and leakage effect (exceptions)")
print(E_.round(3).to_string(index=False))

ds = list(G["dataset"].unique())
fig, axs = plt.subplots(1, len(ds), figsize=(4.3 * len(ds), 3.9), squeeze=False)
for ax, d in zip(axs[0], ds):
    g = G[G["dataset"] == d]
    for fam, h in g.groupby("family"):
        ax.scatter(h["gap"], h["leak"], s=28, color=COL.get(fam, "k"), label=fam)
    ax.set_title(d)
    ax.set_xlabel("train - validation AUC (correct pipeline)")
axs[0][0].set_ylabel("leakage effect (AUC)")
axs[0][0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(FG / "new_fig_gap_leak.png", dpi=200)
plt.close(fig)

fig, axs = plt.subplots(len(ds), 3, figsize=(11, 3.1 * len(ds)), squeeze=False)
for i, d in enumerate(ds):
    g = G[G["dataset"] == d].set_index("model")
    for j, (fk, ms) in enumerate(ORDER.items()):
        ms = [m for m in ms if m in g.index]
        ax = axs[i][j]
        if not ms:
            ax.axis("off")
            continue
        x = np.arange(len(ms))
        ax.plot(x, g.loc[ms, "gap"], "o-", color="#e67e22", label="gap")
        ax.plot(x, g.loc[ms, "leak"], "s-", color="#c0392b", label="leakage effect")
        ax.set_xticks(x)
        ax.set_xticklabels(ms, rotation=60, ha="right", fontsize=7)
        ax.set_title(f"{d}: {FAM[fk] if fk in FAM else fk}", fontsize=9)
axs[0][0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(FG / "new_fig_capacity_gap.png", dpi=200)