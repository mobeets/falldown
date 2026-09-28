# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#     jupytext_version: 1.19.4
#   kernelspec:
#     display_name: analysis
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Cross-dataset figures (cloud study x EMU neural)
#
# Reads the CSVs from `cross_dataset_planning.py` and renders one figure per
# finding. Output: analysis/cross_dataset_outputs/figures/fig_xd_*.png
#
#   .venv-analysis\Scripts\python.exe analysis\cross_dataset_figures.py

# %%
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parent.parent
OUT = _REPO / "analysis" / "cross_dataset_outputs"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

EMU_RUNS = ["yfz_1", "yfz_2", "yga_1", "yga_2"]
EMU_COLORS = {"yfz_1": "#1f6fb4", "yfz_2": "#5aa9e6",
              "yga_1": "#e07b39", "yga_2": "#f0b27a"}
CLOUD_COLOR = "#7f8c8d"


def _load(name):
    p = OUT / name
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def _save(fig, name):
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {name}")


def _tsuf(trials):
    """Filename suffix for a trial set."""
    return "conflict" if trials == "conflict" else "alltrials"


def _tslabel(trials):
    return ("conflict trials only" if trials == "conflict"
            else "all trials (conflict + agreement)")


# %% [markdown]
# ## Part 1

# %%
def fig_prevalence(summ):
    cloud = summ[summ["source"] == "cloud"]
    emu = summ[summ["source"] == "emu"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(cloud))
    ax.scatter(x, cloud["p_plan"], color=CLOUD_COLOR, s=45, edgecolor="k",
               zorder=3, label="cloud (all datasets)")
    ax.axhline(cloud["p_plan"].mean(), color=CLOUD_COLOR, ls="--", lw=1,
               label=f"cloud mean {cloud['p_plan'].mean():.2f}")
    for j, r in enumerate(emu.itertuples()):
        ax.scatter(len(cloud) + j, r.p_plan, color=EMU_COLORS[r.dataset], s=90,
                   marker="D", edgecolor="k", zorder=4, label=f"EMU {r.dataset}")
    ax.axhline(0.5, color="k", ls=":", lw=1)
    ax.set_xticks(list(x) + [len(cloud) + j for j in range(len(emu))])
    ax.set_xticklabels([str(p)[:6] for p in cloud["participant"]] +
                       [r.dataset for r in emu.itertuples()],
                       rotation=90, fontsize=7)
    ax.set_ylabel("P(plan | conflict)")
    ax.set_title("Planning prevalence: EMU runs sit inside the cloud range")
    ax.legend(fontsize=8, loc="upper left")
    _save(fig, "fig_xd_prevalence.png")


def fig_rt_tradeoff(summ):
    d = summ.copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, cols, title in [
            (axes[0], ("rt_dec_plan", "rt_dec_greedy"), "Decision RT (entry\u2192choice)"),
            (axes[1], ("rt_exec_plan", "rt_exec_greedy"), "Execution RT (choice\u2192goal)")]:
        width = 0.35
        for i, (src, color) in enumerate([("cloud", CLOUD_COLOR), ("emu", "#1f6fb4")]):
            s = d[d["source"] == src]
            off = (i - 0.5) * width
            ax.scatter(np.full(len(s), 0 + off), s[cols[0]], color=color, s=25,
                       alpha=0.6, zorder=3)
            ax.scatter(np.full(len(s), 1 + off), s[cols[1]], color=color, s=25,
                       alpha=0.6, zorder=3)
            ax.plot([0 + off, 1 + off],
                    [s[cols[0]].mean(), s[cols[1]].mean()], "-o", color=color,
                    lw=2.5, ms=9, zorder=4, label=src)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["planning", "greedy"])
        ax.set_ylabel("RT (ms)")
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.suptitle("The planning/greedy RT trade-off is present in EMU as in cloud "
                 "(planning slower to commit, faster to execute)", y=1.03)
    _save(fig, "fig_xd_rt_tradeoff.png")


def fig_mixture(summ, mix):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, col, title in [(axes[0], "p_lapse", "Lapse rate (agreement trials)"),
                           (axes[1], "w1", "Mixture threat coefficient w1")]:
        if col == "p_lapse":
            d = summ
        else:
            d = mix
        cloud = d[d["source"] == "cloud"]; emu = d[d["source"] == "emu"]
        ax.scatter(np.zeros(len(cloud)) + np.random.default_rng(0).uniform(-0.1, 0.1, len(cloud)),
                   cloud[col], color=CLOUD_COLOR, s=35, edgecolor="k", label="cloud")
        for j, r in enumerate(emu.itertuples()):
            ax.scatter([1 + (j - 1.5) * 0.08], [getattr(r, col)],
                       color=EMU_COLORS[r.dataset], s=80, marker="D",
                       edgecolor="k", zorder=4)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["cloud", "EMU"])
        ax.set_title(title)
    axes[1].text(0.02, 0.02, "w1 hits the \u00b18 bound for several participants\n"
                             "\u2192 treat as unstable", transform=axes[1].transAxes,
                 fontsize=8, bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    fig.suptitle("Model-based summaries: lapse rate comparable; mixture w1 unstable",
                 y=1.03)
    _save(fig, "fig_xd_mixture.png")


# %% [markdown]
# ## Part 2

# %%
def fig_neural_choice(link):
    fig, ax = plt.subplots(figsize=(7, 4.4))
    x = np.arange(len(link)); w = 0.35
    ax.bar(x - w / 2, link["acc_geometry"], w, label="geometry only",
           color="0.7", edgecolor="k")
    ax.bar(x + w / 2, link["acc_geometry_neural"], w, label="+ neural score",
           color="#1f6fb4", edgecolor="k")
    ax.axhline(0.5, color="k", ls="--", lw=1)
    for i, r in enumerate(link.itertuples()):
        ax.text(i, max(r.acc_geometry, r.acc_geometry_neural) + 0.01,
                f"{r.acc_geometry:.2f}\u2192{r.acc_geometry_neural:.2f}",
                ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(link["run_id"])
    ax.set_ylabel("choice accuracy (CV)")
    ax.set_ylim(0.5, 1.0)
    ax.legend(fontsize=8)
    ax.set_title("Neural planning score adds choice information beyond geometry "
                 "(post-choice)")
    _save(fig, "fig_xd_neural_choice.png")


def fig_neural_rt(link, trials):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axes[0]
    ax.bar(np.arange(len(link)), link["corr_conflict_rt_resid"],
           color=[EMU_COLORS[r] for r in link["run_id"]], edgecolor="k")
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_xticks(range(len(link))); ax.set_xticklabels(link["run_id"])
    ax.set_ylabel("corr(neural conflict_mag, RT residual)")
    ax.set_title("Neural conflict_mag vs kinematic-adjusted RT")
    ax = axes[1]
    t = trials[trials["run_id"] == "yfz_1"].dropna(
        subset=["neural_conflict_pred", "chosen_1step_dist", "rt_decision"])
    if len(t):
        x = t["chosen_1step_dist"].to_numpy(float)
        y = t["rt_decision"].to_numpy(float)
        b = np.polyfit(x, y, 1)
        res = y - (b[0] * x + b[1])
        ax.scatter(t["neural_conflict_pred"], res, s=8, alpha=0.3, color="#1f6fb4")
        ax.axhline(0, color="k", ls="--", lw=1)
        ax.set_xlabel("out-of-fold neural conflict_mag prediction")
        ax.set_ylabel("RT residual (ms)")
        ax.set_title("yfz_1 (trial level)")
    fig.suptitle("The conflict representation does not track decision time", y=1.03)
    _save(fig, "fig_xd_neural_rt.png")


def fig_predecision(link):
    fig, ax = plt.subplots(figsize=(7, 4.4))
    x = np.arange(len(link)); w = 0.35
    ax.bar(x - w / 2, link["acc_pre_geometry"], w, label="geometry only",
           color="0.7", edgecolor="k")
    ax.bar(x + w / 2, link["acc_pre_neural"], w, label="+ pre-decision neural",
           color="#c0392b", edgecolor="k")
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(link["run_id"])
    ax.set_ylabel("choice accuracy (CV)")
    ax.set_ylim(0.5, 1.0)
    ax.legend(fontsize=8)
    ax.set_title("Pre-decision neural activity adds no choice information "
                 "(entry-anchored)")
    _save(fig, "fig_xd_predecision.png")


# %% [markdown]
# ## Part 3

# %%
def fig_transfer(tr):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    x = np.arange(len(tr))
    axes[0].bar(x, tr["acc"], color=["#1f6fb4", "#e07b39"], edgecolor="k")
    axes[0].axhline(0.5, color="k", ls="--", lw=1)
    axes[0].set_ylabel("accuracy predicting EMU choices")
    axes[0].set_title("Choice prediction")
    axes[1].bar(x, tr["loglik"], color=["#1f6fb4", "#e07b39"], edgecolor="k")
    axes[1].set_ylabel("mean log-likelihood")
    axes[1].set_title("Held-out log-likelihood")
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(tr["model"], fontsize=8)
    fig.suptitle("A model fit on the cloud all-drift cohort predicts EMU choices "
                 "as well as an EMU-fitted model", y=1.03)
    _save(fig, "fig_xd_transfer.png")


def fig_model_neural(mn):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(np.arange(len(mn)), mn["corr_neural_cm_model_conflict"],
           color=[EMU_COLORS[r] for r in mn["run_id"]], edgecolor="k")
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_xticks(range(len(mn))); ax.set_xticklabels(mn["run_id"])
    ax.set_ylabel("corr(neural conflict_mag, analytic conflict_mag)")
    ax.set_title("The neural conflict representation matches the model quantity\n"
                 "(strong in YFZ, weak in YGA)")
    _save(fig, "fig_xd_model_neural.png")


# %% [markdown]
# ## Part 4

# %%
def fig_threat(threat, trials):
    suf = _tsuf(trials)
    rate_lbl = ("P(plan | conflict)" if trials == "conflict"
                else "P(choose optimal)")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axes[0]
    cloud = threat[threat["source"] == "cloud"].dropna(subset=["p_plan_near"])
    if len(cloud):
        for r in cloud.itertuples():
            ax.plot([0, 1], [r.p_plan_safe, r.p_plan_near], "-", color="0.7", alpha=0.6)
        ax.plot([0, 1], [cloud["p_plan_safe"].mean(), cloud["p_plan_near"].mean()],
                "-o", color=CLOUD_COLOR, lw=3, label="cloud mean")
    emu = threat[threat["source"] == "emu"]
    for r in emu.itertuples():
        ax.plot([0, 1], [r.p_plan_safe, r.p_plan_near], "-o",
                color=EMU_COLORS[r.dataset], lw=2, ms=8, label=r.dataset)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["safe\n(high ball_y)", "near death\n(low ball_y)"])
    ax.set_ylabel(rate_lbl)
    ax.set_title("Behavioral threat effect")
    ax.legend(fontsize=8)
    ax = axes[1]
    for src, color in [("cloud", CLOUD_COLOR), ("emu", "#1f6fb4")]:
        s = threat[threat["source"] == src]["slope_ball_y"].dropna()
        ax.scatter(np.full(len(s), 0 if src == "cloud" else 1) +
                   np.random.default_rng(0).uniform(-0.1, 0.1, len(s)),
                   s, color=color, s=40, edgecolor="k", alpha=0.7, label=src)
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["cloud", "EMU"])
    ax.set_ylabel("slope of planning on ball_y (per px)" if trials == "conflict"
                  else "slope of choice on ball_y (per px)")
    ax.set_title("Threat sensitivity (slope)")
    ax.legend(fontsize=8)
    fig.suptitle("Threat shifts planning in EMU and cloud\n"
                 f"(cloud: drift-exposed participants, drift blocks, {_tslabel(trials)})",
                 y=1.03)
    _save(fig, f"fig_xd_threat_{suf}.png")


def fig_threat_neural(tn):
    if tn.empty or "acc_near_death" not in tn.columns:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(tn)); w = 0.35
    ax.bar(x - w / 2, tn["acc_near_death"], w, label="near death",
           color="#c0392b", edgecolor="k")
    ax.bar(x + w / 2, tn["acc_far"], w, label="far from death", color="0.7",
           edgecolor="k")
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(tn["run_id"])
    ax.set_ylabel("planning_vs_greedy balanced accuracy")
    ax.legend(fontsize=8)
    ax.set_title("planning_vs_greedy decodability near death vs far\n"
                 "(rel_y: small = near death)")
    _save(fig, "fig_xd_threat_neural.png")


# %% [markdown]
# ## Part 5

# %%
def fig_environment(env):
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for src, color, label in [("cloud", CLOUD_COLOR, "cloud"),
                              ("emu", "#1f6fb4", "EMU")]:
        s = env[env["source"] == src]["slope"].dropna()
        x = 0 if src == "cloud" else 1
        ax.scatter(np.full(len(s), x) + np.random.default_rng(0).uniform(-0.12, 0.12, len(s)),
                   s, color=color, s=45, edgecolor="k", alpha=0.7, label=label)
        ax.errorbar([x], [s.mean()], yerr=[s.std(ddof=1) / np.sqrt(len(s))],
                    fmt="D", color="k", ms=10, capsize=6, zorder=4)
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["cloud", "EMU"])
    ax.set_ylabel("slope of planning on block conflict rate")
    ax.set_title("Within-subject sensitivity to the block's conflict rate\n"
                 "(cloud positive; EMU near zero)")
    ax.legend(fontsize=8)
    _save(fig, "fig_xd_environment.png")


def fig_block_neural(bn):
    if bn.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(np.arange(len(bn)), bn["r_score_conflict"],
           color=[EMU_COLORS[r] for r in bn["run_id"]], edgecolor="k")
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_xticks(range(len(bn))); ax.set_xticklabels(bn["run_id"])
    ax.set_ylabel("corr(per-block neural score, block conflict rate)")
    ax.set_title("Per-block neural planning score vs block conflict rate\n"
                 "(weakly positive, n.s.)")
    _save(fig, "fig_xd_block_neural.png")


# %% [markdown]
# ## Part 6

# %%
def fig_individual(summ, threat, ind, trials):
    fig, ax = plt.subplots(figsize=(7, 5))
    slopes = threat.drop_duplicates("participant").set_index("participant")["slope_ball_y"]
    s = summ.copy(); s["slope_ball_y"] = s["participant"].map(slopes)
    cloud = s[s["source"] == "cloud"].dropna(subset=["p_plan", "slope_ball_y"])
    emu = s[s["source"] == "emu"].dropna(subset=["p_plan", "slope_ball_y"])
    ax.scatter(cloud["p_plan"], cloud["slope_ball_y"], color=CLOUD_COLOR, s=50,
               edgecolor="k", label="cloud", zorder=3)
    for r in emu.itertuples():
        ax.scatter([r.p_plan], [r.slope_ball_y], color=EMU_COLORS[r.dataset],
                   s=140, marker="D", edgecolor="k", zorder=4, label=r.dataset)
        ax.annotate(r.dataset, (r.p_plan, r.slope_ball_y), fontsize=7,
                    xytext=(5, 4), textcoords="offset points")
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_xlabel("planning propensity P(plan | conflict)")
    ax.set_ylabel("threat sensitivity (slope of planning on ball_y)")
    ax.set_title("EMU participants in the cloud trait space (n=2, case study)\n"
                 f"({_tslabel(trials)})")
    ax.legend(fontsize=8)
    _save(fig, f"fig_xd_individual_{_tsuf(trials)}.png")


# %% [markdown]
# ## Model-based planning weight (coefficient space)

# %%
def fig_prevalence_weight(summ):
    cloud = summ[(summ["source"] == "cloud")].dropna(subset=["planning_weight"])
    emu = summ[(summ["source"] == "emu")].dropna(subset=["planning_weight"])
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(cloud))
    ax.scatter(x, cloud["planning_weight"], color=CLOUD_COLOR, s=45, edgecolor="k",
               zorder=3, label="cloud (all datasets)")
    ax.axhline(cloud["planning_weight"].mean(), color=CLOUD_COLOR, ls="--", lw=1,
               label=f"cloud mean {cloud['planning_weight'].mean():.2f}")
    for j, r in enumerate(emu.itertuples()):
        ax.scatter(len(cloud) + j, r.planning_weight, color=EMU_COLORS[r.dataset],
                   s=90, marker="D", edgecolor="k", zorder=4, label=f"EMU {r.dataset}")
    ax.set_xticks(list(x) + [len(cloud) + j for j in range(len(emu))])
    ax.set_xticklabels([str(p)[:6] for p in cloud["participant"]] +
                       [r.dataset for r in emu.itertuples()], rotation=90, fontsize=7)
    ax.set_ylabel("planning weight  |b_plan| / (|b_plan| + |b_1step|)")
    ax.set_title("Model-based planning weight: EMU runs sit inside the cloud range")
    ax.legend(fontsize=8, loc="upper left")
    _save(fig, "fig_xd_prevalence_weight.png")


def fig_threat_weight(tw, trials):
    if tw.empty:
        return
    suf = _tsuf(trials)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axes[0]
    for src, color, label in [("cloud", CLOUD_COLOR, "cloud"), ("emu", "#1f6fb4", "EMU")]:
        s = tw[tw["source"] == src]
        for r in s.itertuples():
            ax.plot([0, 1], [r.pw_low, r.pw_high], "-", color=color, alpha=0.45)
        ax.plot([0, 1], [s["pw_low"].mean(), s["pw_high"].mean()], "-o",
                color=color, lw=3, ms=9, label=label)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["near death\n(ball high)", "safe\n(ball low)"])
    ax.set_ylabel("planning weight")
    ax.set_title("Normalized planning weight")
    ax.legend(fontsize=8)
    ax = axes[1]
    for src, color, label in [("cloud", CLOUD_COLOR, "cloud"), ("emu", "#1f6fb4", "EMU")]:
        s = tw[tw["source"] == src]
        for r in s.itertuples():
            ax.plot([0, 1], [r.beta_plan_low, r.beta_plan_high], "-", color=color, alpha=0.45)
        ax.plot([0, 1], [s["beta_plan_low"].mean(), s["beta_plan_high"].mean()], "-o",
                color=color, lw=3, ms=9, label=label)
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["near death\n(ball high)", "safe\n(ball low)"])
    ax.set_ylabel("beta_plan  (more negative = more planning)")
    ax.set_title("Raw planning coefficient")
    ax.legend(fontsize=8)
    fig.suptitle("Coefficient-space threat effect by ball height\n"
                 f"(drift blocks, {_tslabel(trials)})", y=1.03)
    _save(fig, f"fig_xd_threat_weight_{suf}.png")


def fig_environment_weight(ew):
    if ew.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for src, color, label in [("cloud", CLOUD_COLOR, "cloud"), ("emu", "#1f6fb4", "EMU")]:
        s = ew[ew["source"] == src]["inter_mod_x_plan"].dropna()
        x = 0 if src == "cloud" else 1
        ax.scatter(np.full(len(s), x) + np.random.default_rng(0).uniform(-0.12, 0.12, len(s)),
                   s, color=color, s=45, edgecolor="k", alpha=0.7, label=label)
        ax.errorbar([x], [s.mean()], yerr=[s.std(ddof=1) / np.sqrt(len(s))],
                    fmt="D", color="k", ms=10, capsize=6, zorder=4)
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["cloud", "EMU"])
    ax.set_ylabel("block-conflict-rate x planning coefficient")
    ax.set_title("Coefficient-space environment sensitivity\n(planning weight vs block conflict rate)")
    ax.legend(fontsize=8)
    _save(fig, "fig_xd_environment_weight.png")


def fig_individual_weight(summ, threat, trials):
    fig, ax = plt.subplots(figsize=(7, 5))
    slopes = threat.drop_duplicates("participant").set_index("participant")["slope_ball_y"]
    s = summ.copy(); s["slope_ball_y"] = s["participant"].map(slopes)
    cloud = s[(s["source"] == "cloud")].dropna(subset=["planning_weight", "slope_ball_y"])
    emu = s[(s["source"] == "emu")].dropna(subset=["planning_weight", "slope_ball_y"])
    ax.scatter(cloud["planning_weight"], cloud["slope_ball_y"], color=CLOUD_COLOR, s=50,
               edgecolor="k", label="cloud", zorder=3)
    for r in emu.itertuples():
        ax.scatter([r.planning_weight], [r.slope_ball_y], color=EMU_COLORS[r.dataset],
                   s=140, marker="D", edgecolor="k", zorder=4, label=r.dataset)
        ax.annotate(r.dataset, (r.planning_weight, r.slope_ball_y), fontsize=7,
                    xytext=(5, 4), textcoords="offset points")
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_xlabel("planning weight  |b_plan| / (|b_plan| + |b_1step|)")
    ax.set_ylabel("threat sensitivity (slope of planning on ball_y)")
    ax.set_title("EMU participants in the cloud trait space (model-based, n=2)\n"
                 f"({_tslabel(trials)})")
    ax.legend(fontsize=8)
    _save(fig, f"fig_xd_individual_weight_{_tsuf(trials)}.png")


# %% [markdown]
# ## Part 4c - Coefficient profiles across ball-y positions

# %%
COEF_TERMS = [("greedy", "greedy (1-step)"),
              ("planning", "planning (2-step)"),
              ("incoming", "incoming direction")]


def _robust_ylim(ax, vals):
    vals = np.asarray([v for v in vals if np.isfinite(v)], float)
    if len(vals) == 0:
        return
    lo, hi = np.percentile(vals, [2, 98])
    lo = min(lo, 0.0); hi = max(hi, 0.0)
    pad = 0.15 * (hi - lo) if hi > lo else 1.0
    ax.set_ylim(lo - pad, hi + pad)


def _plot_profile(ax, d, term, value, nbin=5):
    sub = d[d["term"] == term]
    cloud = sub[sub["source"] == "cloud"]
    emu = sub[sub["source"] == "emu"]
    for _, g in cloud.groupby("participant"):
        g = g.sort_values("bin")
        ax.plot(g["bin"], g[value], "-", color="0.82", lw=0.8, zorder=1)
    lims = []
    if len(cloud):
        agg = cloud.groupby("bin")[value].agg(["mean", "sem"]).reset_index()
        ax.errorbar(agg["bin"], agg["mean"], yerr=agg["sem"], fmt="-o",
                    color=CLOUD_COLOR, lw=3, ms=8, capsize=4, zorder=4,
                    label="cloud")
        lims += list(agg["mean"])
    for run, g in emu.groupby("dataset"):
        g = g.sort_values("bin")
        ax.plot(g["bin"], g[value], "-o", color=EMU_COLORS.get(run, "k"),
                lw=2, ms=7, zorder=3, label=run)
        lims += list(g[value])
    ax.axhline(0, color="k", ls="--", lw=1)
    _robust_ylim(ax, lims)
    ax.set_xticks(range(nbin))
    ax.set_xticklabels(["near\ndeath" if i == 0 else
                        ("safe" if i == nbin - 1 else str(i))
                        for i in range(nbin)], fontsize=8)
    ax.set_xlabel("ball-y bin")


def fig_coef_profile(d, trials):
    if d.empty:
        return
    suf = _tsuf(trials)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for j, (term, title) in enumerate(COEF_TERMS):
        ax = axes[0, j]
        _plot_profile(ax, d, term, "beta")
        ax.set_title(title)
        if j == 0:
            ax.set_ylabel("coefficient (log-odds)")
            ax.legend(fontsize=7)
        ax = axes[1, j]
        if term in ("greedy", "planning"):
            _plot_profile(ax, d, term, "share")
            if j == 0:
                ax.set_ylabel("normalized share")
        else:
            ax.text(0.5, 0.5, "incoming direction has no\nnormalized share",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="0.45")
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Coefficient profile across ball-y positions "
                 f"(drift blocks, {_tslabel(trials)})", y=1.0)
    fig.tight_layout()
    _save(fig, f"fig_xd_coef_profile_{suf}.png")


def _plot_drift_pair(ax, d, term, value):
    sub = d[d["term"] == term]
    for cond, color in [("drift", "#c0392b"), ("follow", "#1f6fb4")]:
        s = sub[sub["condition"] == cond][value].dropna()
        if len(s):
            ax.errorbar([0 if cond == "drift" else 1], [s.mean()],
                        yerr=[s.std(ddof=1) / np.sqrt(len(s))], fmt="-o",
                        color=color, lw=3, ms=9, capsize=5, zorder=4,
                        label=cond)
    piv = sub.pivot_table(index="participant", columns="condition",
                          values=value)
    if {"drift", "follow"}.issubset(piv.columns):
        piv = piv.dropna()
        for _, row in piv.iterrows():
            ax.plot([0, 1], [row["drift"], row["follow"]], "-", color="0.8",
                    lw=1, marker="o", ms=3, zorder=1)
        if len(piv) > 1:
            _, p = stats.ttest_rel(piv["drift"], piv["follow"])
            ax.text(0.5, 0.97, f"paired p={p:.3f}", ha="center", va="top",
                    transform=ax.transAxes, fontsize=8, color="0.3")
        lims = list(piv["drift"]) + list(piv["follow"])
        _robust_ylim(ax, lims)
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["drift", "follow"])


def fig_coef_profile_drift(d, trials):
    if d.empty:
        return
    suf = _tsuf(trials)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for j, (term, title) in enumerate(COEF_TERMS):
        ax = axes[0, j]
        _plot_drift_pair(ax, d, term, "beta")
        ax.set_title(title)
        if j == 0:
            ax.set_ylabel("coefficient (log-odds)")
            ax.legend(fontsize=7)
        ax = axes[1, j]
        if term in ("greedy", "planning"):
            _plot_drift_pair(ax, d, term, "share")
            if j == 0:
                ax.set_ylabel("normalized share")
        else:
            ax.text(0.5, 0.5, "incoming direction has no\nnormalized share",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="0.45")
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Drift vs follow coefficients, within subject "
                 f"(exp3_altdrift only, {_tslabel(trials)})", y=1.0)
    fig.tight_layout()
    _save(fig, f"fig_xd_coef_profile_drift_{suf}.png")


# %% [markdown]
# ## Part 2b - Neural planning profiles across ball height

# %%
def _run_lines(ax, df, xcol, ycol, chance=None, nbin=5):
    for run, g in df.groupby("run_id"):
        g = g.sort_values(xcol)
        ax.plot(g[xcol], g[ycol], "-o", color=EMU_COLORS.get(run, "k"),
                lw=1.5, ms=5, label=run)
    if len(df):
        mean = df.groupby(xcol)[ycol].mean().sort_index()
        ax.plot(mean.index, mean.values, "-", color="k", lw=3, label="mean")
    if chance is not None:
        ax.axhline(chance, color="k", ls="--", lw=1)
    ax.set_xticks(range(nbin))
    ax.set_xticklabels(["near\ndeath" if i == 0 else
                        ("far" if i == nbin - 1 else str(i))
                        for i in range(nbin)], fontsize=8)
    ax.set_xlabel("ball-y bin")


def fig_neural_planning_bally(plan):
    if plan.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, window, title in [(axes[0], "post", "choice-anchored post"),
                              (axes[1], "entry", "entry-anchored pre")]:
        d = plan[(plan["window"] == window) & (plan["rep"] == "rate")]
        _run_lines(ax, d, "bin", "acc_mean", chance=0.5)
        ax.set_title(f"planning_vs_greedy ({title})")
        if ax is axes[0]:
            ax.set_ylabel("balanced accuracy")
            ax.legend(fontsize=7)
    fig.suptitle("Neural planning decodability across ball-y positions "
                 "(EMU all-drift)", y=1.03)
    _save(fig, "fig_xd_neural_planning_bally.png")


def fig_neural_choice_bally(cb):
    if cb.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, window, title in [(axes[0], "post", "choice-anchored post"),
                              (axes[1], "entry", "entry-anchored pre")]:
        d = cb[(cb["window"] == window) &
               (cb["term"].isin(["neural", "neural_x_ball_y"]))]
        piv = d.pivot_table(index="run_id", columns="term", values="coef")
        x = np.arange(len(piv)); w = 0.38
        ax.bar(x - w / 2, piv.get("neural", np.nan),
               w, label="neural", color="#1f6fb4", edgecolor="k")
        ax.bar(x + w / 2, piv.get("neural_x_ball_y", np.nan),
               w, label="neural x ball_y", color="#c0392b", edgecolor="k")
        ax.axhline(0, color="k", ls="--", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(piv.index, fontsize=8)
        ax.set_title(title)
        if ax is axes[0]:
            ax.set_ylabel("logistic coefficient")
            ax.legend(fontsize=8)
    fig.suptitle("Neural planning score and its ball-height interaction on "
                 "choice", y=1.03)
    _save(fig, "fig_xd_neural_choice_bally.png")


def fig_neural_value_bally(val):
    if val.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), sharey=True)
    for ax, target in zip(axes, ["greedy_gap", "planning_gap", "conflict_mag"]):
        d = val[(val["window"] == "post") & (val["control"] == "both") &
                (val["target"] == target)]
        _run_lines(ax, d, "bin", "corr_mean", chance=0.0)
        ax.set_title(target)
        if ax is axes[0]:
            ax.set_ylabel("decoding correlation (r)")
            ax.legend(fontsize=7)
    fig.suptitle("Model-value decoding across ball-y positions "
                 "(post-choice, nuisance-controlled)", y=1.03)
    _save(fig, "fig_xd_neural_value_bally.png")


def fig_neural_region_bally(reg):
    if reg.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for region, g in reg.groupby("region"):
        mean = g.groupby("bin")["acc_mean"].mean().sort_index()
        ax.plot(mean.index, mean.values, "-o", lw=2, ms=6, label=region)
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["near\ndeath", "1", "2", "3", "far"], fontsize=8)
    ax.set_xlabel("ball-y bin")
    ax.set_ylabel("planning_vs_greedy balanced accuracy")
    ax.legend(fontsize=8)
    ax.set_title("Per-region planning decodability across ball-y positions")
    _save(fig, "fig_xd_neural_region_bally.png")


def fig_neural_threat_bally(tb):
    if tb.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    _run_lines(ax, tb, "bin", "acc_mean", chance=0.5)
    ax.set_ylabel("planning_vs_greedy balanced accuracy")
    ax.set_title("Neural planning decodability across proximity to death\n"
                 "(bin 0 = smallest ball_y - camera_y = nearest death)")
    ax.legend(fontsize=8)
    _save(fig, "fig_xd_neural_threat_bally.png")


def fig_neural_controlled(ctrl):
    if ctrl.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    controls = list(dict.fromkeys(ctrl["control"]))
    for ax, val, title in [(axes[0], "near_acc", "near death"),
                           (axes[1], "far_acc", "far from death")]:
        for run, g in ctrl.groupby("run_id"):
            g = g.set_index("control").reindex(controls)
            ax.plot(range(len(controls)), g[val], "-o",
                    color=EMU_COLORS.get(run, "k"), label=run)
        ax.set_xticks(range(len(controls)))
        ax.set_xticklabels(controls, rotation=20, fontsize=7)
        ax.axhline(0.5, color="k", ls="--", lw=1)
        ax.set_title(f"planning decode - {title}")
        if ax is axes[0]:
            ax.set_ylabel("balanced accuracy")
            ax.legend(fontsize=7)
    fig.suptitle("Ball-height planning decode under confound controls", y=1.03)
    _save(fig, "fig_xd_neural_controlled.png")


def fig_neural_cross(cross):
    if cross.empty:
        return
    d = cross[cross["direction"].isin(["near_to_far", "far_to_near"])
              & (cross["rep"] == "rate")]
    windows = list(dict.fromkeys(d["window"]))
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for run, g in d.groupby("run_id"):
        for dirn, mk, color in [("near_to_far", "o", "#c0392b"),
                                ("far_to_near", "s", "#1f6fb4")]:
            gg = g[g["direction"] == dirn].set_index("window") \
                .reindex(windows)
            ax.plot(range(len(windows)), gg["acc"], marker=mk, ls="-",
                    color=color, alpha=0.4, ms=4)
    for dirn, mk, color, label in [("near_to_far", "o", "#c0392b",
                                    "near -> far"),
                                   ("far_to_near", "s", "#1f6fb4",
                                    "far -> near")]:
        m = [d[(d["window"] == wn) & (d["direction"] == dirn)]["acc"].mean()
             for wn in windows]
        ax.plot(range(len(windows)), m, marker=mk, ls="-", color=color, lw=3,
                ms=9, label=label)
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.set_xticks(range(len(windows))); ax.set_xticklabels(windows)
    ax.set_ylabel("cross-decoded balanced accuracy")
    ax.legend(fontsize=8)
    ax.set_title("Planning cross-decoding across ball height")
    _save(fig, "fig_xd_neural_cross_bally.png")


def fig_neural_unit_threat(ut):
    if ut.empty:
        return
    ut = ut.copy()
    ut["region"] = ut["region"].fillna("unassigned").astype(str)
    regions = sorted(ut["region"].unique())
    data = [ut[ut["region"] == r]["interaction"].values for r in regions]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.boxplot(data, showfliers=False)
    ax.set_xticks(range(1, len(regions) + 1))
    ax.set_xticklabels(regions, rotation=15, fontsize=8)
    sig = ut[ut["significant"].astype(str).str.lower() == "true"]
    if len(sig):
        for i, r in enumerate(regions):
            s = sig[sig["region"] == r]["interaction"]
            ax.scatter(np.full(len(s), i + 1), s, color="#c0392b", s=40,
                       zorder=4, edgecolor="k")
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_ylabel("planning modulation  (near - far)")
    ax.set_title("Single-unit threat x planning interaction\n"
                 "(red = significant after BH-FDR)")
    _save(fig, "fig_xd_neural_unit_threat.png")


def fig_neural_rsa(rsa):
    if rsa.empty:
        return
    piv = rsa.pivot_table(index="subset", columns="model", values="spearman_r")
    order = [s for s in ["all", "near", "far"] + [f"bin{i}" for i in range(5)]
             if s in piv.index]
    piv = piv.reindex(order)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-0.1, vmax=0.1,
                   aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=30, fontsize=8)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index, fontsize=8)
    fig.colorbar(im, ax=ax, label="Spearman r")
    ax.set_title("Neural-model RSA by ball-y subset")
    _save(fig, "fig_xd_neural_rsa.png")


def fig_neural_dpca_threat(dp):
    if dp.empty:
        return
    var = dp[dp["analysis"] == "variance"]
    sim = dp[dp["analysis"] == "subspace_similarity"]
    piv = var.groupby(["term", "condition"])["explained_variance"].sum() \
        .unstack()
    terms = [t for t in ["t", "c", "s", "cs", "ct", "st", "cst"]
             if t in piv.index]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    x = np.arange(len(terms)); w = 0.38
    if "near" in piv.columns:
        axes[0].bar(x - w / 2, piv.loc[terms, "near"], w, label="near death",
                    color="#c0392b", edgecolor="k")
    if "far" in piv.columns:
        axes[0].bar(x + w / 2, piv.loc[terms, "far"], w, label="far",
                    color="0.7", edgecolor="k")
    axes[0].set_xticks(x); axes[0].set_xticklabels(terms)
    axes[0].legend(fontsize=8)
    axes[0].set_ylabel("sum explained variance")
    axes[0].set_title("dPCA variance near vs far")
    if len(sim):
        axes[1].bar(sim["term"], sim["explained_variance"], color="#1f6fb4",
                    edgecolor="k")
        axes[1].set_ylim(0, 1)
        axes[1].set_ylabel("mean subspace cosine (near vs far)")
        axes[1].set_title("Subspace similarity")
    fig.suptitle("Threat-conditioned dPCA", y=1.03)
    _save(fig, "fig_xd_neural_dpca_threat.png")


# %% [markdown]
# ## Summary

# %%
def fig_summary(summ, link, tr, threat, env, mn):
    emu = summ[summ["source"] == "emu"]
    cloud = summ[summ["source"] == "cloud"]
    inc = float((link["acc_geometry_neural"] - link["acc_geometry"]).mean())
    pre = float((link["acc_pre_neural"] - link["acc_pre_geometry"]).mean())
    transfer_acc = float(tr[tr["model"] == "cloud->EMU"]["acc"].iloc[0])
    mn_corr = float(mn["corr_neural_cm_model_conflict"].mean())
    cloud_env = float(env[env["source"] == "cloud"]["slope"].mean())
    emu_env = float(env[env["source"] == "emu"]["slope"].mean())
    rows = [
        ("prevalence inside cloud range", "pass",
         f"EMU {emu['p_plan'].mean():.2f} vs cloud "
         f"{cloud['p_plan'].min():.2f}-{cloud['p_plan'].max():.2f}"),
        ("model planning weight in range", "pass",
         f"EMU {emu['planning_weight'].mean():.2f} vs cloud "
         f"{cloud['planning_weight'].min():.2f}-{cloud['planning_weight'].max():.2f}"),
        ("planning/greedy RT trade-off", "pass",
         "planning slower to commit, faster to execute"),
        ("neural score adds choice info", "pass", f"+{inc:.3f} accuracy (4/4)"),
        ("pre-decision adds nothing", "pass", f"{pre:+.3f} accuracy (4/4)"),
        ("cloud model transfers to EMU", "pass", f"accuracy {transfer_acc:.2f}"),
        ("neural matches model conflict", "partial",
         f"r = {mn_corr:.2f} (strong in YFZ, weak in YGA)"),
        ("behavioral threat effect", "pass",
         "cloud drift<follow; EMU near-death<safe (3/4)"),
        ("environment sensitivity", "fail in EMU",
         f"cloud slope {cloud_env:.3f} vs EMU {emu_env:.3f}"),
    ]
    colors = {"pass": "#27ae60", "partial": "#e08a39", "fail in EMU": "#c0392b"}
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, (label, status, detail) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.scatter(0, y, s=220, color=colors[status], edgecolor="k", zorder=3)
        ax.text(0.06, y, label, va="center", fontsize=9, fontweight="bold")
        ax.text(0.55, y, detail, va="center", fontsize=8.5, color="0.25")
    ax.set_xlim(-0.05, 1.3); ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.axis("off")
    ax.set_title("Cross-dataset scorecard: what replicates in EMU and what does not",
                 fontsize=12)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=s,
                          markeredgecolor="k", markersize=10)
               for s, c in colors.items()]
    ax.legend(handles=handles, loc="lower right", fontsize=8, ncol=3,
              frameon=False, bbox_to_anchor=(1.0, -0.05))
    _save(fig, "fig_xd_summary.png")


# %%
def main():
    print("Making cross-dataset figures ...")
    summ = _load("xd_behavioral_summary.csv")
    mix = _load("xd_mixture.csv")
    link = _load("xd_neural_linkage.csv")
    trials = _load("xd_neural_trials.csv")
    tr = _load("xd_transfer.csv")
    mn = _load("xd_model_neural.csv")
    threat = _load("xd_threat_behavior.csv")
    tn = _load("xd_threat_neural.csv")
    env = _load("xd_environment.csv")
    bn = _load("xd_block_neural.csv")
    ind = _load("xd_individual.csv")
    tw = _load("xd_threat_weight.csv")
    ew = _load("xd_environment_weight.csv")
    cp = _load("xd_coef_profile.csv")
    cpd = _load("xd_coef_profile_drift.csv")
    nplan = _load("xd_neural_planning_bally.csv")
    nchoice = _load("xd_neural_choice_bally.csv")
    nvalue = _load("xd_neural_value_bally.csv")
    nregion = _load("xd_neural_region_bally.csv")
    nthreat = _load("xd_neural_threat_bally.csv")
    nctrl = _load("xd_neural_planning_controlled.csv")
    ncross = _load("xd_neural_cross_bally.csv")
    nunit = _load("xd_neural_unit_threat.csv")
    nrsa = _load("xd_neural_rsa.csv")
    ndpca = _load("xd_neural_dpca_threat.csv")

    fig_prevalence(summ)
    fig_rt_tradeoff(summ)
    fig_mixture(summ, mix)
    fig_neural_choice(link)
    fig_neural_rt(link, trials)
    fig_predecision(link)
    fig_transfer(tr)
    fig_model_neural(mn)
    fig_threat_neural(tn)
    fig_environment(env)
    fig_block_neural(bn)
    fig_prevalence_weight(summ)
    fig_environment_weight(ew)
    # ball-height figures are emitted once per trial set (conflict vs all)
    for trials in ("conflict", "all"):
        t = threat[threat["trials"] == trials]
        w = tw[tw["trials"] == trials]
        fig_threat(t, trials)
        fig_threat_weight(w, trials)
        fig_individual(summ, t, ind, trials)
        fig_individual_weight(summ, t, trials)
        fig_coef_profile(cp[cp["trials"] == trials], trials)
        fig_coef_profile_drift(cpd[cpd["trials"] == trials], trials)
    fig_neural_planning_bally(nplan)
    fig_neural_choice_bally(nchoice)
    fig_neural_value_bally(nvalue)
    fig_neural_region_bally(nregion)
    fig_neural_threat_bally(nthreat)
    fig_neural_controlled(nctrl)
    fig_neural_cross(ncross)
    fig_neural_unit_threat(nunit)
    fig_neural_rsa(nrsa)
    fig_neural_dpca_threat(ndpca)
    fig_summary(summ, link, tr, threat, env, mn)
    print(f"Done. Figures in {FIG}")


if __name__ == "__main__":
    main()
