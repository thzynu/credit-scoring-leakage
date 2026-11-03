"""Summarise the new experiments (joint, scoring, provenance counts, ratio, SMOTENC, calibration).
Writes results/tables/new_*.csv and results/figures/new_*.png. Missing inputs are skipped.
SE = standard deviation / sqrt(number of seeds): split-to-split variation only (seeds are
repeated splits of one dataset, not independent samples)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent.parent / "results"
T, FG = R / "tables", R / "figures"
T.mkdir(exist_ok=True)
FG.mkdir(exist_ok=True)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)


def se(s):
    return s.std() / len(s) ** 0.5 if len(s) > 1 else np.nan


def rd(pattern):
    fs = sorted(R.glob(pattern))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else None


def save(df, name):
    df.round(4).to_csv(T / name, index=False)
    print(f"\n===== {name}\n{df.round(4).to_string(index=False)}")


# 1. synthetic points per fold (model independent, so one row per dataset-seed)
p = rd("provenance_*_smote.csv")
if p is not None:
    q = p.drop_duplicates(["dataset", "seed"])
    c = q.groupby("dataset").agg(
        n_seed=("seed", "nunique"), synthetic_total=("n_syn_total", "mean"),
        created_by_correct_pipeline=("n_syn_in", "mean"), contaminated=("n_contam", "mean"),
        clean=("n_clean", "mean"), contaminated_via_base=("n_contam_base", "mean"),
        contaminated_via_neighbour_only=("n_contam_nbr_only", "mean")).reset_index()
    c["not_used_by_matched_leaky_model"] = c["synthetic_total"] - c["created_by_correct_pipeline"]
    c["contaminated_share"] = c["contaminated"] / c["synthetic_total"]
    save(c, "new_synthetic_counts.csv")

# 2. closed decomposition + comparison with the first design (exp_diag.py)
j = rd("joint_*_smote.csv")
if j is not None:
    dg = rd("diag_*_smote.csv")
    if dg is not None:
        dg["diag_total"] = dg["auc_all"] - dg["cv_in"]
        j = j.merge(dg[["dataset", "model", "seed", "diag_total"]], on=["dataset", "model", "seed"], how="left")
    else:
        j["diag_total"] = np.nan
    g = j.groupby(["dataset", "model"])
    t = g.agg(n_seed=("seed", "nunique"), cv_in=("cv_in", "mean"),
              leak=("leak_eff", "mean"), leak_se=("leak_eff", se),
              score=("score_eff", "mean"), score_se=("score_eff", se),
              interaction=("interaction", "mean"), interaction_se=("interaction", se),
              joint=("joint_eff", "mean"), joint_se=("joint_eff", se),
              diag_total=("diag_total", "mean")).reset_index()
    t["joint_minus_diag_total"] = t["joint"] - t["diag_total"]
    big = t["joint"].abs() >= 0.01
    for k in ("leak", "score", "interaction"):
        t[k + "_share"] = np.where(big, t[k] / t["joint"], np.nan)
    save(t, "new_decomp_closed.csv")

    rows = []
    for (d, m), gg in j.groupby(["dataset", "model"]):
        gg = gg.sort_values("seed")
        for met in ("leak_eff", "score_eff", "joint_eff"):
            v = gg[met].to_numpy()
            th = [x.mean() for x in np.array_split(v, 3)] if len(v) >= 3 else [np.nan] * 3
            rows.append(dict(dataset=d, model=m, metric=met, n_seed=len(v), mean=v.mean(),
                             sd=v.std(ddof=1) if len(v) > 1 else np.nan, min=v.min(), max=v.max(),
                             share_positive=(v > 0).mean(), third1=th[0], third2=th[1], third3=th[2]))
    save(pd.DataFrame(rows), "new_seed_stability.csv")

    order = {"knn": ["knn1", "knn3", "knn5", "knn15", "knn25", "knn50", "knn100"],
             "rf": ["rf_depth3", "rf_depth5", "rf_depth10", "rf_leaf20", "rf_leaf10", "rf_leaf5",
                    "rf_leaf2", "rf_full"],
             "hgb": ["hgb_depth1", "hgb_depth2", "hgb_depth3", "hgb_depth4", "hgb_depth6", "hgb_full",
                     "hgb_depth4_iter300", "hgb_depth2_noes", "hgb_depth4_noes", "hgb_full_noes"]}
    cap = []
    for fam, ms in order.items():
        for i, m in enumerate(ms):
            s = t[t["model"] == m]
            for _, r in s.iterrows():
                cap.append(dict(family=fam, order=i, setting=m, dataset=r["dataset"], n_seed=r["n_seed"],
                                leak=r["leak"], score=r["score"], interaction=r["interaction"], joint=r["joint"]))
    if cap:
        save(pd.DataFrame(cap).sort_values(["dataset", "family", "order"]).drop(columns="order"),
             "new_capacity.csv")

    # figure: closed decomposition
    sets = [d for d in ["german", "australian", "taiwan", "lendingclub"] if d in t["dataset"].unique()]
    main = ["logreg", "knn5", "rf_full", "hgb_depth2", "hgb_full"]
    fig, axs = plt.subplots(1, len(sets), figsize=(4.2 * len(sets), 3.6), squeeze=False)
    for ax, d in zip(axs[0], sets):
        s = t[(t["dataset"] == d) & t["model"].isin(main)].set_index("model").reindex(
            [m for m in main if m in set(t[t["dataset"] == d]["model"])])
        x = np.arange(len(s))
        pos = np.zeros(len(s))
        neg = np.zeros(len(s))
        for k, col in (("leak", "#c0392b"), ("score", "#2980b9"), ("interaction", "#7f8c8d")):
            v = s[k].to_numpy()
            b = np.where(v >= 0, pos, neg)
            ax.bar(x, v, bottom=b, color=col, label=k)
            pos += np.where(v >= 0, v, 0)
            neg += np.where(v < 0, v, 0)
        ax.plot(x, s["joint"], "k_", ms=18, label="joint (total)")
        ax.set_xticks(x)
        ax.set_xticklabels(s.index, rotation=45, ha="right")
        ax.set_title(d)
        ax.axhline(0, color="k", lw=.5)
    axs[0][0].set_ylabel("AUC inflation")
    axs[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FG / "new_fig_closed_decomposition.png", dpi=200)

# 3. scoring channel measured on a correctly trained model
s = rd("scoring_*_smote.csv")
if s is not None:
    t = s.groupby(["dataset", "model"]).agg(
        n_seed=("seed", "nunique"), score_all=("score_all", "mean"), score_all_se=("score_all", se),
        score_clean=("score_clean", "mean"), score_contam=("score_contam", "mean"),
        p_added=("p_all", "mean"), p_val_positive=("p_valpos", "mean")).reset_index()
    save(t, "new_scoring.csv")

# 4. geometry figure
gm = rd("provgeom_*_smote.csv")
if gm is not None:
    a = gm.groupby("dataset")[["nn5_contam_pos", "nn5_clean_pos", "nn5_origpos_pos", "nn5_origneg_pos"]].mean()
    a.columns = ["contaminated synthetic", "clean synthetic", "original positive", "original negative"]
    ax = a.plot(kind="bar", stacked=True, figsize=(5.5, 3.6), color=["#c0392b", "#e59866", "#2980b9", "#aab7b8"])
    ax.set_ylabel("share of the 5 nearest training points")
    ax.set_title("Neighbours of validation positives")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FG / "new_fig_geometry.png", dpi=200)
    plt.close()

# 5. oversampling ratio
r = rd("ratio_*.csv")
if r is not None:
    save(r.groupby(["dataset", "model", "ratio"]).agg(
        n_seed=("seed", "nunique"), tang=("tang", "mean"), tang_se=("tang", se), phong=("phong", "mean"),
        cv_in=("cv_in", "mean"), cv_before=("cv_before", "mean"), holdout=("holdout", "mean")).reset_index(),
        "new_ratio.csv")

# 6. SMOTE versus SMOTENC
n = rd("smotenc_*.csv")
if n is not None:
    save(n.groupby(["dataset", "model", "balancer"]).agg(
        n_seed=("seed", "nunique"), tang=("tang", "mean"), tang_se=("tang", se), phong=("phong", "mean"),
        holdout=("holdout", "mean")).reset_index(), "new_smotenc.csv")
    w = n.pivot_table(index=["dataset", "model", "seed"], columns="balancer", values=["tang", "holdout"]).dropna()
    w.columns = [a + "_" + b for a, b in w.columns]
    w["d_tang"] = w["tang_smotenc"] - w["tang_smote"]
    w["d_holdout"] = w["holdout_smotenc"] - w["holdout_smote"]
    save(w.groupby(["dataset", "model"]).agg(
        n_seed=("d_tang", "size"), d_tang=("d_tang", "mean"), d_tang_se=("d_tang", se),
        d_holdout=("d_holdout", "mean"), d_holdout_se=("d_holdout", se)).reset_index(), "new_smotenc_paired.csv")

# 7. calibration, class weighting, decision cost
cb = rd("calib_lendingclub.csv")
if cb is not None:
    cols = ["auc", "brier", "mean_pred", "cal_int", "cal_slope", "ece", "brier_recal", "cal_slope_recal",
            "cost3", "cost5", "cost10", "d_brier", "d_brier_lo", "d_brier_hi", "d_cost3", "d_cost3_lo",
            "d_cost3_hi", "d_cost5", "d_cost5_lo", "d_cost5_hi", "d_cost10", "d_cost10_lo", "d_cost10_hi"]
    cols = [c for c in cols if c in cb.columns]
    a = cb.groupby(["model", "variant"])[cols].mean()
    a["n_seed"] = cb.groupby(["model", "variant"])["seed"].nunique()
    a["brier_se"] = cb.groupby(["model", "variant"])["brier"].apply(se)
    a["base_rate"] = cb["base_rate"].mean()
    save(a.reset_index(), "new_calibration.csv")
    cc = pd.read_csv(R / "calibcurve_lendingclub.csv")
    ms = list(cc["model"].unique())
    fig, axs = plt.subplots(1, len(ms), figsize=(4 * len(ms), 3.8), squeeze=False)
    for ax, m in zip(axs[0], ms):
        for v, col in (("nosmote", "#2980b9"), ("smote", "#c0392b"), ("classweight", "#27ae60")):
            z = cc[(cc["model"] == m) & (cc["variant"] == v)].groupby("bin")[["mean_pred", "frac_pos"]].mean()
            ax.plot(z["mean_pred"], z["frac_pos"], "o-", ms=3, color=col, label=v)
        ax.plot([0, 1], [0, 1], "k--", lw=.7)
        ax.set_title(m)
        ax.set_xlabel("mean predicted probability")
    axs[0][0].set_ylabel("observed default rate (2016)")
    axs[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FG / "new_fig_calibration_lc.png", dpi=200)