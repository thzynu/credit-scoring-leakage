"""Ranking stability (clean vs leaky model ranking, seed by seed) and paired statistics.

Rankings: for every seed the 7 main models are ranked by clean AUC (hold-out; out-of-time AUC for
Lending Club) and by leaky cross-validated AUC (SMOTE before the fold split). Kendall tau is
computed per seed; the interval is a percentile bootstrap over seeds (seeds are repeated splits of
one dataset, so it describes split-to-split variation only). Discordant pairs whose clean AUCs differ
by less than 0.005 are counted separately: those reversals carry little practical meaning.
Paired statistics: per seed differences (SMOTE minus no SMOTE on the hold-out; leaky CV minus
hold-out). Two intervals are given: the naive t interval, and the Nadeau-Bengio corrected one for
repeated random splits (variance factor 1/J + n_test/n_train, here 0.2/0.8 = 0.25), which is wider
because the splits overlap. Outputs: results/tables/new_rank_stability.csv, new_paired_ci.csv."""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

R = Path(__file__).resolve().parent.parent / "results"
T = R / "tables"
T.mkdir(exist_ok=True)
pd.set_option("display.width", 250)
MAIN = ["logreg", "knn5", "rf_depth5", "rf_full", "hgb_depth2", "hgb_depth4", "hgb_full"]
SETS = ["australian", "german", "taiwan"]
rng = np.random.RandomState(0)

data, paired = {}, []
for d in SETS:
    v = pd.read_csv(R / f"{d}_v2.csv")
    v = v[v["frac"] == 1.0]
    data[d] = v.rename(columns={"holdout": "clean"})[["model", "seed", "clean", "leak_both"]]
    b = pd.read_csv(R / f"{d}_nosmote.csv")
    b = b[b["frac"] == 1.0]
    m = v.merge(b[["model", "seed", "holdout_nosmote"]], on=["model", "seed"])
    for name, g in m.groupby("model"):
        paired.append((d, name, "delta_holdout_smote_minus_nosmote", (g["holdout"] - g["holdout_nosmote"]).to_numpy()))
        paired.append((d, name, "inflation_leaky_cv_minus_holdout", (g["leak_both"] - g["holdout"]).to_numpy()))
t = pd.read_csv(R / "lendingclub_time.csv")
t = t[(t["frac"] == 1.0) & (t["split"] == "train2013-2014-2015_test2016")]
data["lendingclub"] = t.rename(columns={"holdout_oot": "clean"})[["model", "seed", "clean", "leak_both"]]
for name, g in t.groupby("model"):
    paired.append(("lendingclub", name, "inflation_leaky_cv_minus_out_of_time", (g["leak_both"] - g["holdout_oot"]).to_numpy()))

rows = []
for d, df in data.items():
    df = df[df["model"].isin(MAIN)]
    seeds = df.groupby("seed")["model"].nunique()
    seeds = seeds[seeds == len(MAIN)].index
    res = []
    for s in seeds:
        g = df[df["seed"] == s].set_index("model").loc[MAIN]
        bc, bl = g["clean"].idxmax(), g["leak_both"].idxmax()
        disc = near = 0
        for i, j in combinations(range(len(MAIN)), 2):
            a = g["clean"].iloc[i] - g["clean"].iloc[j]
            b = g["leak_both"].iloc[i] - g["leak_both"].iloc[j]
            if a * b < 0:
                disc += 1
                near += abs(a) < 0.005
        res.append(dict(tau=stats.kendalltau(g["clean"], g["leak_both"]).statistic, top1=bc == bl,
                        leaky_top_rf_full=bl == "rf_full",
                        rank_of_clean_best=int(g["leak_both"].rank(ascending=False)[bc]),
                        disc=disc, near=near))
    r = pd.DataFrame(res)
    mm = df[df["seed"].isin(seeds)].groupby("model")[["clean", "leak_both"]].mean().loc[MAIN]
    tm = stats.kendalltau(mm["clean"], mm["leak_both"])
    bs = [r["tau"].to_numpy()[rng.randint(0, len(r), len(r))].mean() for _ in range(2000)]
    rows.append(dict(dataset=d, n_seed=len(r), tau_mean=r["tau"].mean(), tau_sd=r["tau"].std(),
                     tau_lo=np.percentile(bs, 2.5), tau_hi=np.percentile(bs, 97.5),
                     share_tau_positive=(r["tau"] > 0).mean(), top1_agree_share=r["top1"].mean(),
                     leaky_top_is_rf_full_share=r["leaky_top_rf_full"].mean(),
                     mean_rank_of_clean_best_under_leaky=r["rank_of_clean_best"].mean(),
                     discordant_pairs=r["disc"].mean(), discordant_pairs_clean_gap_lt_0005=r["near"].mean(),
                     tau_of_seed_means=tm.statistic, p_of_seed_means=tm.pvalue))
rk = pd.DataFrame(rows)
rk.round(4).to_csv(T / "new_rank_stability.csv", index=False)
print(rk.round(3).to_string(index=False))

rows = []
for d, m, q, x in paired:
    J = len(x)
    if J < 3:
        continue
    mean, sd = x.mean(), x.std(ddof=1)
    tq = stats.t.ppf(0.975, J - 1)
    se_n, se_nb = sd / np.sqrt(J), sd * np.sqrt(1 / J + 0.25)
    rows.append(dict(dataset=d, model=m, quantity=q, n_seed=J, mean=mean, se_naive=se_n,
                     ci_naive_lo=mean - tq * se_n, ci_naive_hi=mean + tq * se_n, se_nb=se_nb,
                     ci_nb_lo=mean - tq * se_nb, ci_nb_hi=mean + tq * se_nb,
                     nb_excludes_zero=bool((mean - tq * se_nb > 0) or (mean + tq * se_nb < 0))))
pc = pd.DataFrame(rows)
pc.round(4).to_csv(T / "new_paired_ci.csv", index=False)
print()
print(pc[pc["quantity"].str.startswith("delta")].round(4).to_string(index=False))