# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: analysis
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Planning experiments: emphasis figures
#
# Reads the outputs of `planning_experiments.py` and produces clear,
# annotated figures that foreground the key findings:
#
#   Exp 1  people plan ~30% of conflict trials; no excess sequential
#          persistence beyond a memoryless null.
#   Exp 2  planning rate rises with the block's conflict rate / expected
#          planning advantage (sensitivity, not insensitivity); the designed
#          environment shift was never reached by the only participant who ran it.
#   Exp 3  threat (drift / proximity to death) is associated with LESS
#          planning; planning slows decisions but speeds execution.
#
# Run after planning_experiments.py:
#   .venv-analysis/Scripts/python.exe analysis/planning_figures.py

# %%
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy import stats
import seaborn as sns

import planning_experiments as pe

FIG_DIR = pe.OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

C = {
    "planning": "#1f6fb4",
    "greedy": "#e07b39",
    "drift": "#c0392b",
    "follow": "#27ae60",
    "neutral": "#555555",
    "agree": "#7f8c8d",
    "lapse": "#bdc3c7",
}

# one colour per online dataset (used for the all-participant figures)
DS_COLORS = {
    "exp1_nodrift": "#27ae60",
    "exp3_altdrift": "#e07b39",
    "exp3_alldrift": "#c0392b",
}
DS_LABELS = {
    "exp1_nodrift": "nodrift",
    "exp3_altdrift": "alt-drift",
    "exp3_alldrift": "all-drift",
}


def _ds_label(ds):
    return DS_LABELS.get(ds, ds)


def _load(name):
    path = pe.OUT_DIR / name
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return pd.DataFrame()


def _save(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path.name}")


def _sig(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


# %% [markdown]
# ## Exp 1 — prevalence and temporal structure

# %%
def fig_planning_prevalence(prev, tbl):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4),
                             gridspec_kw={"width_ratios": [1.15, 1]})

    # ---- left: per-participant P(plan | conflict), all datasets ----
    ax = axes[0]
    p = prev.sort_values("p_plan").reset_index(drop=True)
    y = np.arange(len(p))
    for ds, g in p.groupby("dataset"):
        ax.scatter(g["p_plan"], y[g.index], s=55, color=DS_COLORS[ds],
                   edgecolor="k", zorder=3,
                   label=f"{_ds_label(ds)} (mean {g['p_plan'].mean():.2f})")
    ax.axvline(0.5, color="k", ls="--", lw=1.2)
    t, pv = stats.ttest_1samp(p["p_plan"], 0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(p["participant"].str[:8], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("P(plan | conflict)")
    ax.set_title(f"Planning prevalence per participant (N={len(p)})")
    ax.legend(loc="lower right", fontsize=8)
    ax.text(0.02, 0.97, f"vs 0.5: t={t:.2f}, p={pv:.1e} {_sig(pv)}",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.7"))

    # ---- right: per-participant proportions grouped by agreement ----
    ax = axes[1]
    specs = [
        (0.0, "optimal", "agree", prev["p_optimal"], C["agree"]),
        (1.0, "lapse", "agree", prev["p_lapse"], C["lapse"]),
        (2.8, "planning", "conflict", prev["p_plan"], C["planning"]),
        (3.8, "greedy", "conflict", prev["p_greedy"], C["greedy"]),
    ]
    rng = np.random.default_rng(0)
    for x, name, group, vals, color in specs:
        vals = vals.dropna().values
        m = vals.mean()
        se = vals.std(ddof=1) / np.sqrt(len(vals))
        ax.bar(x, m, width=0.8, color=color, edgecolor="k", alpha=0.9)
        ax.errorbar([x], [m], yerr=[se], fmt="none", ecolor="k",
                    capsize=5, lw=1.5, zorder=4)
        ax.scatter(x + rng.uniform(-0.18, 0.18, len(vals)), vals, s=18,
                   color="k", zorder=5, alpha=0.7)
        ax.text(x, m + se + 0.03, f"{m:.2f}", ha="center", fontsize=9)
    ax.axvline(1.9, color="0.5", ls="--", lw=1)
    ax.text(0.5, 1.005, "greedy = planning (agree)", ha="center", va="bottom",
            fontsize=9, transform=ax.get_xaxis_transform())
    ax.text(3.3, 1.005, "greedy \u2260 planning (conflict)", ha="center",
            va="bottom", fontsize=9, transform=ax.get_xaxis_transform())
    ax.set_xticks([0, 1, 2.8, 3.8])
    ax.set_xticklabels(["optimal", "lapse", "planning", "greedy"], fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("proportion of trials (per-participant mean \u00b1 SEM)")
    ax.set_title("Choice proportions by agreement", pad=24)

    fig.suptitle("Planning prevalence across all online participants (N=18); "
                 "choices split by whether greedy and planning agree",
                 fontsize=13, y=1.02)
    _save(fig, "fig_planning_prevalence.png")


def fig_planning_temporal(trans):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    lo = min(trans["perm_lo"].min(), trans["stay_prob"].min()) - 0.03
    hi = max(trans["perm_hi"].max(), trans["stay_prob"].max()) + 0.03

    ax = axes[0]
    for ds, g in trans.groupby("dataset"):
        ax.errorbar(g["perm_mean"], g["stay_prob"],
                    xerr=[g["perm_mean"] - g["perm_lo"],
                          g["perm_hi"] - g["perm_mean"]],
                    fmt="o", color=DS_COLORS[ds], ms=8, capsize=3, zorder=3,
                    label=_ds_label(ds))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("memoryless null stay prob (permutation 95% CI)")
    ax.set_ylabel("observed stay prob")
    ax.set_title("Observed persistence vs independence")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    t = trans.copy()
    t["excess"] = t["stay_prob"] - t["perm_mean"]
    t["lo"] = t["perm_lo"] - t["perm_mean"]
    t["hi"] = t["perm_hi"] - t["perm_mean"]
    t = t.sort_values("excess").reset_index(drop=True)
    y = np.arange(len(t))
    ax.errorbar(t["excess"], y,
                xerr=[t["excess"] - t["lo"], t["hi"] - t["excess"]],
                fmt="o", color="k", ms=4, capsize=3, zorder=4)
    for ds, g in t.groupby("dataset"):
        ax.barh(y[g.index], g["excess"], color=DS_COLORS[ds], alpha=0.6,
                height=0.5)
    ax.axvline(0, color="k", lw=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(t["participant"].str[:8], fontsize=7)
    ax.set_xlabel("observed - memoryless stay probability (95% perm CI)")
    n_sig = int((t["perm_p"] < 0.05).sum())
    ax.set_title(f"Excess persistence ({n_sig}/{len(t)} at perm p<0.05)")
    ax.text(0.02, 0.98, f"{n_sig} of {len(t)} participants\n"
                        f"survive permutation p<0.05",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.7"))

    fig.suptitle("Strategy choices look memoryless once base rate is controlled "
                 "(all online participants, N=18)", fontsize=13, y=1.02)
    _save(fig, "fig_planning_temporal.png")


# %% [markdown]
# ## Exp 2 — environmental statistics

# %%
def fig_exp2_within(block_env, within):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    specs = [
        (axes[0], "block_conflict_rate", "block conflict rate", "slope"),
        (axes[1], "block_plan_advantage", "block expected planning advantage",
         "slope_adv"),
    ]
    for ax, xcol, label, key in specs:
        d = block_env.dropna(subset=["planning_rate", xcol]).copy()
        d["x_c"] = d[xcol] - d.groupby("participant")[xcol].transform("mean")
        d["y_c"] = (d["planning_rate"]
                    - d.groupby("participant")["planning_rate"].transform("mean"))
        for pid, g in d.groupby("participant"):
            if len(g) >= 3:
                ax.plot(g["x_c"], g["y_c"], "-", color="0.8", lw=0.7, zorder=1)
        ax.scatter(d["x_c"], d["y_c"], s=16, color=C["planning"], alpha=0.55,
                   zorder=2)
        b, a = np.polyfit(d["x_c"], d["y_c"], 1)
        xs = np.linspace(d["x_c"].min(), d["x_c"].max(), 50)
        ax.plot(xs, a + b * xs, color=C["drift"], lw=2.6, zorder=3)
        ax.axhline(0, color="0.6", lw=0.8, ls=":")
        ax.axvline(0, color="0.6", lw=0.8, ls=":")
        mean_slope = within[key].mean()
        t, p = stats.ttest_1samp(within[key].dropna(), 0)
        ax.set_xlabel(f"{label}  (participant-centred)")
        ax.set_ylabel("P(plan | conflict)  (centred)")
        ax.set_title(f"vs {label.replace('block ', '')}")
        ax.text(0.03, 0.96,
                f"mean per-participant slope = {mean_slope:.2f}\n"
                f"t={t:.2f}, p={p:.3f} {_sig(p)}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec="0.7"))

    ax = axes[2]
    rng = np.random.default_rng(0)
    specs2 = [("slope", "conflict rate", C["drift"]),
              ("slope_adv", "planning advantage", C["planning"])]
    for i, (key, label, color) in enumerate(specs2):
        v = within[key].dropna().values
        x = i + rng.uniform(-0.12, 0.12, len(v))
        ax.scatter(x, v, color=color, s=40, alpha=0.8, edgecolor="k", zorder=3)
        m, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
        ax.errorbar([i], [m], yerr=[se], fmt="D", color="k", ms=10, capsize=6,
                    zorder=4)
        t, p = stats.ttest_1samp(v, 0)
        ax.text(i, max(v.max(), m + se) + 0.03,
                f"mean={m:.2f}\np={p:.3f} {_sig(p)}", ha="center", fontsize=9)
    ax.axhline(0, color="k", ls="--", lw=1.2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["vs conflict rate", "vs planning advantage"])
    ax.set_ylabel("per-participant slope")
    ax.set_title("all slopes shift positive")

    fig.suptitle("Exp 2: planning tracks the environment "
                 "(within-participant; sensitivity, not insensitivity)",
                 fontsize=13, y=1.03)
    _save(fig, "fig_exp2_within.png")


def fig_exp2_between(between):
    if between.empty:
        return
    order = [e for e in ["balanced", "high_agree", "shift"]
             if e in set(between["environment"])]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    rng = np.random.default_rng(0)
    colors = {"balanced": C["follow"], "high_agree": C["greedy"],
              "shift": C["planning"]}
    for i, env in enumerate(order):
        g = between[between["environment"] == env]
        x = i + rng.uniform(-0.12, 0.12, len(g))
        ax.scatter(x, g["p_plan"], color=colors[env], s=42, alpha=0.8,
                   edgecolor="k", zorder=3)
        m, se = g["p_plan"].mean(), g["p_plan"].std(ddof=1) / np.sqrt(len(g))
        ax.errorbar([i], [m], yerr=[se], fmt="D", color="k", ms=10,
                    capsize=6, zorder=4)
        ax.text(i, m + se + 0.03, f"n={len(g)}", ha="center", fontsize=9)
    groups = [between[between["environment"] == e]["p_plan"].values for e in order]
    if len(groups) >= 2:
        t, p = stats.ttest_ind(groups[0], groups[1], equal_var=False)
        ax.text(0.98, 0.02,
                f"balanced vs high-agreement:\nt={t:.2f}, p={p:.3f} {_sig(p)}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    ax.set_xticks(range(len(order)))
    labels = {"balanced": "balanced\n(~0.53 agree)",
              "high_agree": "high-agreement\n(~0.79 agree)",
              "shift": "shift pilot\n(YFX)"}
    ax.set_xticklabels([labels[e] for e in order])
    ax.set_ylabel("P(plan | conflict)")
    ax.set_title("Exp 2: high-agreement environment does NOT reduce planning\n"
                 "(between-cohort, confounded with drift)")
    _save(fig, "fig_exp2_between.png")


def config_agreement_curve(config_rel):
    path = pe._REPO_ROOT / config_rel
    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    rates = []
    for b in cfg:
        lv = b.get("levels", [])
        if len(lv) < 9:
            continue
        a = t = 0
        for i in range(len(lv) // 3):
            entry = lv[3 * i][0]; ch = lv[3 * i + 1]; goal = lv[3 * i + 2][0]
            if len(ch) != 2:
                continue
            g = {h: abs(entry - h) for h in ch}
            p = {h: abs(entry - h) + abs(h - goal) for h in ch}
            a += (min(g, key=g.get) == min(p, key=p.get)); t += 1
        if t:
            rates.append(a / t)
    return np.array(rates)


def fig_exp2_shift(shift):
    if shift is None or shift.empty:
        return
    cfg = config_agreement_curve("app/configs/default_experiment.json")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(np.arange(1, len(cfg) + 1), cfg, "-", color="0.6", lw=1.8,
            label="default_experiment conflict rate (designed)")
    ax.scatter(shift["block_number"] + 1, shift["conflict_rate"],
               color=C["drift"], s=45, zorder=3, label="YFX observed blocks")
    ax.axvspan(46, len(cfg), color=C["drift"], alpha=0.06)
    ax.annotate("designed low-agreement /\nzero-agreement regime\n(never reached)",
                xy=(60, 0.15), xytext=(30, 0.55),
                arrowprops=dict(arrowstyle="->", color=C["drift"]),
                color=C["drift"], fontsize=9)
    ax.set_xlabel("experimental block")
    ax.set_ylabel("block conflict rate (1 - agreement)")
    ax.set_ylim(-0.02, 1.0)
    ax.set_title("Exp 2: the only participant who ran the shifted environment\n"
                 "stopped inside the high-agreement phase")
    ax.legend(fontsize=9, loc="upper right")
    _save(fig, "fig_exp2_shift.png")


# %% [markdown]
# ## Exp 3 — threat, planning, and execution benefit

# %%
def fig_exp3_threat(drift_per, threat_bins):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    ax = axes[0]
    if not drift_per.empty:
        piv = drift_per.pivot_table(index="participant", columns="block_drift",
                                    values="p_plan").dropna()
        for pid, row in piv.iterrows():
            ax.plot([0, 1], [row[0], row[1]], "-o", color="0.6", alpha=0.7, ms=5)
        ax.plot([0, 1], [piv[0].mean(), piv[1].mean()], "-o", color=C["drift"],
                lw=3, ms=10, zorder=5, label="group mean")
        t, p = stats.ttest_rel(piv[1], piv[0])
        ax.text(0.5, 0.03, f"drift - follow = {piv[1].mean()-piv[0].mean():+.3f}\n"
                           f"t={t:.2f}, p={p:.3f} {_sig(p)}",
                transform=ax.transAxes, ha="center", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    ax.set_xticks([0, 1]); ax.set_xticklabels(["follow\n(no threat)",
                                               "drift\n(threat)"])
    ax.set_ylabel("P(plan | conflict)")
    ax.set_title("Block-level threat")
    ax.legend(fontsize=9)

    ax = axes[1]
    if not threat_bins.empty:
        # x-axis ordered safe -> near death so that, like the left panel,
        # moving right = more threat and "less planning under threat" is a
        # negative slope.
        for _, row in threat_bins.iterrows():
            ax.plot([0, 1], [row["p_plan_safe"], row["p_plan_near_death"]],
                    "-o", color="0.6", alpha=0.7, ms=5)
        ax.plot([0, 1], [threat_bins["p_plan_safe"].mean(),
                         threat_bins["p_plan_near_death"].mean()],
                "-o", color=C["planning"], lw=3, ms=10, zorder=5,
                label="group mean")
        t, p = stats.ttest_rel(threat_bins["p_plan_near_death"],
                               threat_bins["p_plan_safe"])
        ax.text(0.5, 0.03, f"near - safe = "
                           f"{threat_bins['p_plan_near_death'].mean()-threat_bins['p_plan_safe'].mean():+.3f}\n"
                           f"t={t:.2f}, p={p:.3f} {_sig(p)}",
                transform=ax.transAxes, ha="center", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["safe\n(high ball_y)", "near death\n(low ball_y)"])
    ax.set_ylabel("P(plan | conflict)")
    ax.set_title("Continuous proximity to death")
    ax.legend(fontsize=9)

    fig.suptitle("Exp 3: threat is associated with LESS planning (both measures)",
                 fontsize=13, y=1.03)
    _save(fig, "fig_exp3_threat.png")


_PRETTY = {
    "const": "intercept",
    "diff_planning": "planning diff",
    "z_ball_y": "ball_y (threat)",
    "block_drift": "drift block",
    "ballXdrift": "ball_y \u00d7 drift",
    "incoming_direction": "incoming dir",
    "chosen_1step_dist": "1-step dist",
    "chosen_2step_dist": "2-step path",
    "chose_planning": "chose planning",
}


def _forest(ax, res, title, color):
    if res is None:
        ax.axis("off"); return
    ci = res.conf_int()
    ci.columns = ["lo", "hi"]
    names = list(res.params.index)
    y = np.arange(len(names))
    ax.errorbar(res.params.values, y,
                xerr=[res.params.values - ci["lo"].values,
                      ci["hi"].values - res.params.values],
                fmt="o", color=color, ms=7, capsize=4, lw=1.5)
    ax.axvline(0, color="k", lw=1, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{_PRETTY.get(n, n)}  {_sig(p)}"
                        for n, p in zip(names, res.pvalues.values)], fontsize=9)
    ax.set_title(title)
    ax.invert_yaxis()


def fig_exp3_forest(tbl):
    gee_res, _ = pe.exp3_gee(tbl)
    exec_res, dec_res, total_res, _ = pe.exp3_execution_benefit(tbl)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.subplots_adjust(wspace=0.55)
    _forest(axes[0], gee_res, "P(chose planning)", C["planning"])
    _forest(axes[1], dec_res, "log decision RT (1-step controlled)", C["greedy"])
    _forest(axes[2], exec_res, "log execution RT (2-step controlled)", C["follow"])
    for ax in axes:
        ax.set_xlabel("coefficient (log-odds or log-RT)")
    fig.suptitle("Exp 3: threat reduces planning; planning's RT effects shrink "
                 "to a small residual once kinematics are controlled",
                 fontsize=13, y=1.05)
    _save(fig, "fig_exp3_forest.png")


def fig_exp3_exec(tbl):
    """Distance-adjusted RT only.

    Conflict trials force the planning hole to be farther from the entry and
    closer to the goal, so raw RT differences are kinematic. We therefore
    residualize decision RT on chosen_1step_dist, execution RT on
    chosen_2step_dist, and total RT on chosen_2step_dist, then compare choices.
    """
    _, _, _, d = pe.exp3_execution_benefit(tbl)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    fig.subplots_adjust(wspace=0.35, bottom=0.22)
    panels = [
        (axes[0], "rt_decision_adj", "Decision RT\n(adjusted for 1-step dist)"),
        (axes[1], "rt_exec_adj", "Execution RT\n(adjusted for path geometry)"),
        (axes[2], "rt_total_adj", "Total RT\n(adjusted for path geometry)"),
    ]
    for k, (ax, col, title) in enumerate(panels):
        means, sems = [], []
        for plan in (0, 1):
            v = d[d["chose_planning"] == plan][col].values
            means.append(np.mean(v))
            sems.append(np.std(v, ddof=1) / np.sqrt(len(v)))
        ax.bar([0, 1], means, yerr=sems, capsize=6,
               color=[C["greedy"], C["planning"]], edgecolor="k", alpha=0.9)
        for i, (m, s) in enumerate(zip(means, sems)):
            ax.text(i, m + (s if m >= 0 else -s), f"{m:+.0f}",
                    ha="center", va="bottom" if m >= 0 else "top", fontsize=9)
        delta = means[1] - means[0]
        ax.axhline(0, color="k", lw=0.8)
        ax.text(0.5, 0.96, f"planning - greedy = {delta:+.0f} ms",
                transform=ax.transAxes, ha="center", va="top", fontsize=9,
                color=C["drift"] if delta > 0 else C["follow"],
                bbox=dict(boxstyle="round", fc="white", ec="0.7"))
        ax.set_xticks([0, 1]); ax.set_xticklabels(["greedy", "planning"])
        ax.set_title(title)
        if k == 0:
            ax.set_ylabel("residual RT (ms)")
    fig.suptitle("Exp 3: after removing the kinematic confound, planning has only "
                 "a small RT residual", fontsize=13, y=1.0)
    fig.text(0.5, 0.02,
             "On conflict trials planning is always the farther 1-step hole and the "
             "shorter 2-step path, so the raw RT gaps are geometric.\n"
             "After residualizing on those distances the gaps collapse to +33 / "
             "\u22124 / +4 ms; a kinematic-controlled log-RT GEE gives decision +0.04 "
             "(n.s.), execution \u22120.28 (p<.001), total +0.01 (n.s.).",
             ha="center", fontsize=8.5, color="0.35")
    _save(fig, "fig_exp3_exec.png")


# %% [markdown]
# ## HMM state-count selection and state coefficients

# %%
def fig_hmm_selection(sel):
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
    fig.subplots_adjust(wspace=0.42)
    for ax, exp, color in [(axes[0], "exp1", C["planning"]),
                           (axes[1], "exp3", C["drift"])]:
        d = sel[sel["experiment"] == exp]
        for _, g in d.groupby("participant"):
            g = g.sort_values("K")
            ax.plot(g["K"], g["bic"], "-o", color="0.75", lw=1, ms=4)
        s = d.groupby("K")["bic"].sum()
        ax.plot(s.index, s.values, "-o", color=color, lw=3, ms=9,
                label="group sum BIC")
        Kstar = int(s.idxmin())
        ax.axvline(Kstar, color=color, ls="--", alpha=0.5)
        ax.set_xticks(sorted(d["K"].unique()))
        ax.set_xlabel("number of states K")
        ax.set_ylabel("BIC")
        ax.set_title(f"{exp}: BIC vs K  (K* = {Kstar})")
        ax.legend(fontsize=8)

    # group held-out log-likelihood (higher = better)
    ax = axes[2]
    for exp, color in [("exp1", C["planning"]), ("exp3", C["drift"])]:
        d = sel[sel["experiment"] == exp]
        s = d.groupby("K")["ll_test"].sum()
        ax.plot(s.index, s.values, "-o", color=color, lw=2.5, ms=8,
                label=f"{exp} (K*={int(s.idxmax())})")
    ax.set_xticks(sorted(sel["K"].unique()))
    ax.set_xlabel("number of states K")
    ax.set_ylabel("sum held-out log-likelihood")
    ax.set_title("Held-out LL vs K (higher = better)")
    ax.legend(fontsize=8)

    # per-participant optimal K under each criterion (pooled experiments)
    ax = axes[3]
    bic_opt = sel.loc[sel.groupby(["experiment", "participant"])["bic"].idxmin()]
    ll_opt = sel.loc[sel.groupby(["experiment", "participant"])["ll_test"].idxmax()]
    bc = bic_opt["K"].value_counts().sort_index()
    lc = ll_opt["K"].value_counts().sort_index()
    Ks = sorted(set(bc.index) | set(lc.index))
    x = np.arange(len(Ks))
    ax.bar(x - 0.2, [bc.get(k, 0) for k in Ks], width=0.4, color=C["neutral"],
           edgecolor="k", label="BIC")
    ax.bar(x + 0.2, [lc.get(k, 0) for k in Ks], width=0.4, color=C["planning"],
           edgecolor="k", hatch="//", label="held-out LL")
    ax.set_xticks(x); ax.set_xticklabels(Ks)
    ax.set_xlabel("per-participant optimal K")
    ax.set_ylabel("participants (both experiments)")
    ax.set_title("Criterion changes individual K")
    ax.legend(fontsize=8)

    fig.suptitle("HMM model selection: BIC favors K*=1; held-out LL is flat/noisy "
                 "and train LL always picks the max K", fontsize=13, y=1.03)
    _save(fig, "fig_hmm_selection.png")


def fig_hmm_weights(weights_all):
    d = weights_all[weights_all["experiment"] == "exp1"]
    feats = pe.FEATURE_NAMES
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    ax = axes[0]
    d1 = d[d["num_states"] == 1]
    for i, f in enumerate(feats):
        v = d1[d1["feature"] == f]["weight"].values
        ax.bar(i, v.mean(), color=C["planning"], alpha=0.85, edgecolor="k")
        ax.scatter([i] * len(v), v, color="k", s=20, zorder=3)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels(feats, rotation=30, ha="right")
    ax.set_ylabel("emission coefficient")
    ax.set_title(f"Justified model: single state (K*=1, n={d1['participant'].nunique()})")

    ax = axes[1]
    d2 = d[d["num_states"] == 2]
    width = 0.38
    for s in sorted(d2["state"].unique()):
        ds = d2[d2["state"] == s]
        for i, f in enumerate(feats):
            v = ds[ds["feature"] == f]["weight"].values
            ax.bar(i + (s - 0.5) * width, v.mean(), width=width,
                   color=[C["greedy"], C["planning"]][s], alpha=0.9,
                   edgecolor="k", label=f"state {s}" if i == 0 else None)
            ax.scatter([i + (s - 0.5) * width] * len(v), v, color="k", s=12,
                       zorder=3)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels(feats, rotation=30, ha="right")
    ax.set_title("Forced K=2 (not BIC-supported)")
    ax.legend(fontsize=8)

    ax = axes[2]
    piv = d2.pivot_table(index="participant", columns=["state", "feature"],
                         values="weight")
    cols = [(s, f) for s in sorted(d2["state"].unique()) for f in feats]
    piv = piv.reindex(columns=pd.MultiIndex.from_tuples(cols))
    sns.heatmap(piv, cmap="coolwarm", center=0, ax=ax,
                cbar_kws={"label": "coefficient"},
                yticklabels=[str(p)[:8] for p in piv.index])
    ax.set_title("Individual coefficients (forced K=2)")
    ax.set_xlabel("(state, feature)")

    fig.suptitle("HMM emission coefficients: one justified state; the forced "
                 "two-state solution separates only weakly", fontsize=13, y=1.04)
    _save(fig, "fig_hmm_weights.png")


def fig_hmm_k2_states(glmhmm_k2, posteriors):
    if glmhmm_k2.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    tm = np.mean([np.array(json.loads(s)) for s in glmhmm_k2["transition_matrix"]],
                 axis=0)
    ax = axes[0]
    ax.imshow(tm, cmap="Blues", vmin=0, vmax=1)
    for i in range(tm.shape[0]):
        for j in range(tm.shape[1]):
            ax.text(j, i, f"{tm[i, j]:.2f}", ha="center", va="center",
                    color="white" if tm[i, j] > 0.5 else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xlabel("to state"); ax.set_ylabel("from state")
    ax.set_title("Mean transition matrix (forced K=2)")

    ax = axes[1]
    occ = np.array([json.loads(s) for s in glmhmm_k2["occupancy"]])
    bottom = np.zeros(len(occ))
    for k in range(occ.shape[1]):
        ax.bar(range(len(occ)), occ[:, k], bottom=bottom,
               color=[C["greedy"], C["planning"]][k], edgecolor="k",
               label=f"state {k}")
        bottom += occ[:, k]
    ax.set_xticks(range(len(occ)))
    ax.set_xticklabels([str(p)[:6] for p in glmhmm_k2["participant"]],
                       rotation=45, ha="right")
    ax.set_ylabel("occupancy"); ax.set_ylim(0, 1)
    ax.set_title("State occupancy per participant")
    ax.legend(fontsize=8)

    ax = axes[2]
    for pid, g in posteriors.groupby("participant"):
        g = g[g["state"] == 1].sort_values("trial")
        ax.plot(g["trial"], g["posterior"], lw=1, alpha=0.8,
                label=str(pid)[:6])
    ax.set_xlabel("choice trial"); ax.set_ylabel("P(state 2)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("State-2 posterior over time")
    ax.legend(fontsize=7, ncol=2)

    fig.suptitle("Forced K=2 solution (for interpretation only \u2014 BIC "
                 "rejects it): states are only weakly separated",
                 fontsize=13, y=1.04)
    _save(fig, "fig_hmm_k2_states.png")


# %% [markdown]
# ## Main

# %%
def main():
    print("Loading outputs ...")
    tbl = pe.add_features(_load("sequence_table.csv"))
    prev = _load("exp1_prevalence.csv")
    trans = _load("exp1_transitions.csv")
    hmm_sel = _load("exp1_hmm_model_selection.csv")
    hmm_weights = _load("exp1_hmm_state_weights.csv")
    hmm_post = _load("exp1_hmm_state_posteriors.csv")
    hmm_k2 = _load("exp1_glmhmm_k2.csv")
    block_env = _load("exp2_block_environment.csv")
    within = _load("exp2_within_subject.csv")
    between = _load("exp2_between_cohort.csv")
    shift = _load("exp2_shift_blocks.csv")
    drift_per = _load("exp3_drift_contrast.csv")
    threat_bins = _load("exp3_threat_bins.csv")

    print("Making figures ...")
    fig_planning_prevalence(prev, tbl)
    # NOTE: fig_planning_temporal is SUPERSEDED. Its one-sided "excess stay"
    # test missed a negative within-block lag-1 dependence (alternation); see
    # analysis/planning_reanalysis_sequential.py and
    # analysis/superseded_figures/SUPERSEDED.md. Do not regenerate it here.
    fig_hmm_selection(hmm_sel)
    fig_hmm_weights(hmm_weights)
    fig_hmm_k2_states(hmm_k2, hmm_post)
    fig_exp2_within(block_env, within)
    fig_exp2_between(between)
    fig_exp2_shift(shift)
    fig_exp3_threat(drift_per, threat_bins)
    fig_exp3_forest(tbl)
    fig_exp3_exec(tbl)
    print(f"Done. Figures in {FIG_DIR}")
    fig_exp3_threat(drift_per, threat_bins)
    fig_exp3_forest(tbl)
    fig_exp3_exec(tbl)
    print(f"Done. Figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
