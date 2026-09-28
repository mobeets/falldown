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
# # Full results summary: cloud study x EMU
#
# One reproducible place that reads every headline result CSV (behavioural,
# cross-dataset, neural) and emits a single tidy results table plus a master
# figure. This backs `docs/full_results_synthesis.md`; it does not re-fit any
# model, it only aggregates existing outputs.
#
# Run with:
#   .venv-analysis\Scripts\python.exe analysis\full_results_summary.py

# %%
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")

_REPO = Path(__file__).resolve().parent.parent
PO = _REPO / "analysis" / "planning_outputs"
XD = _REPO / "analysis" / "cross_dataset_outputs"
AGG = _REPO / "analysis" / "neural_outputs" / "aggregate"

C_CLOUD = "0.45"
C_EMU = "#d1495b"
C_PLAN = "#d1495b"
C_GREED = "#00798c"


def _load(path):
    p = Path(path)
    return pd.read_csv(p) if p.exists() else None


def build_master():
    """Tidy long-format ledger of headline results."""
    rows = []

    def add(domain, source, cohort, metric, value, n=None, note=""):
        rows.append({"domain": domain, "source_file": source, "cohort": cohort,
                     "metric": metric, "value": value, "n": n, "note": note})

    # ---- behaviour: planning propensity -------------------------------------
    prev = _load(PO / "exp1_prevalence.csv")
    if prev is not None:
        add("behaviour", "exp1_prevalence.csv", "cloud",
            "P(plan|conflict) mean", prev["p_plan"].mean(), len(prev),
            "18 online participants")
        add("behaviour", "exp1_prevalence.csv", "cloud",
            "P(plan|conflict) min", prev["p_plan"].min(), len(prev))
        add("behaviour", "exp1_prevalence.csv", "cloud",
            "P(plan|conflict) max", prev["p_plan"].max(), len(prev))
        add("behaviour", "exp1_prevalence.csv", "cloud",
            "P(optimal|agreement) mean", prev["p_optimal"].mean(), len(prev))
        add("behaviour", "exp1_prevalence.csv", "cloud",
            "lapse rate mean", prev["p_lapse"].mean(), len(prev))

    xd = _load(XD / "xd_behavioral_summary.csv")
    if xd is not None:
        emu = xd[xd["source"] == "emu"]
        add("behaviour", "xd_behavioral_summary.csv", "EMU",
            "P(plan|conflict) mean", emu["p_plan"].mean(), len(emu))
        add("behaviour", "xd_behavioral_summary.csv", "EMU",
            "planning_weight mean", emu["planning_weight"].mean(), len(emu))
        cloud = xd[xd["source"] == "cloud"]
        add("behaviour", "xd_behavioral_summary.csv", "cloud",
            "planning_weight mean", cloud["planning_weight"].mean(), len(cloud))

    # ---- behaviour: latent states -------------------------------------------
    hmm = _load(PO / "exp1_hmm_optimal_k_per_participant.csv")
    if hmm is not None:
        add("behaviour", "exp1_hmm_optimal_k_per_participant.csv", "cloud",
            "participants with K*=1 (BIC)", int((hmm["optimal_K"] == 1).sum()),
            len(hmm))
    hmm_e = _load(XD / "xd_hmm_optimal_k.csv")
    if hmm_e is not None:
        add("behaviour", "xd_hmm_optimal_k.csv", "EMU",
            "runs with K*=1 (BIC)", int((hmm_e["optimal_K"] == 1).sum()),
            len(hmm_e))

    # ---- behaviour: environment sensitivity ---------------------------------
    env = _load(XD / "xd_environment.csv")
    if env is not None:
        for cohort, sub in env.groupby(env["source"]):
            add("behaviour", "xd_environment.csv",
                "cloud" if cohort == "cloud" else "EMU",
                "within-session planning~conflict slope mean", sub["slope"].mean(),
                len(sub))

    # ---- behaviour: threat --------------------------------------------------
    drift = _load(PO / "exp3_drift_contrast.csv")
    if drift is not None:
        piv = drift.pivot_table(index="participant", columns="block_drift",
                                values="p_plan")
        if {0, 1}.issubset(piv.columns):
            d = (piv[1] - piv[0]).dropna()
            t, p = stats.ttest_rel(piv.loc[d.index, 1], piv.loc[d.index, 0])
            add("behaviour", "exp3_drift_contrast.csv", "cloud",
                "P(plan) drift-follow difference", d.mean(), len(d),
                f"paired t={t:.2f}, p={p:.3f}")

    # ---- threat: 5-bin ball-height profile ---------------------------------
    tb = _load(PO / "threat_bins5_tests.csv")
    if tb is not None:
        for _, r in tb.iterrows():
            coh = "cloud" if str(r["test"]).startswith("cloud") else "EMU"
            val = r.get("coef_quadratic")
            if (not isinstance(val, (int, float))) or (
                    isinstance(val, float) and not np.isfinite(val)):
                val = r.get("coef_linear")
            add("behaviour", "threat_bins5_tests.csv", coh,
                f"ball-y 5-bin: {r['test']}", val,
                int(r["n"]) if pd.notna(r.get("n")) else None,
                f"quad p={r['p_quadratic']:.3f}" if pd.notna(r.get("p_quadratic"))
                else "")

    # ---- timing -------------------------------------------------------------
    tsum = _load(PO / "timing_phase_summary.csv")
    if tsum is not None:
        for _, r in tsum.iterrows():
            tag = f"{r['cohort']} {r['interval']} {r['choice']}"
            add("timing", "timing_phase_summary.csv", r["cohort"],
                f"{r['interval']}_roll_ms [{r['choice']}]", r["roll_ms"],
                int(r["n"]))
            add("timing", "timing_phase_summary.csv", r["cohort"],
                f"{r['interval']}_fall_ms [{r['choice']}]", r["fall_ms"],
                int(r["n"]))
    dwell = _load(PO / "timing_dwell_tests.csv")
    if dwell is not None:
        for _, r in dwell.iterrows():
            add("timing", "timing_dwell_tests.csv", r["cohort"],
                f"dwell planning-greedy {r['interval']}", r["diff_ms"],
                int(r["n_participants"]), f"p={r['p']:.3f}")
    carry = _load(PO / "timing_carryover_models.csv")
    if carry is not None:
        for _, r in carry.iterrows():
            add("timing", "timing_carryover_models.csv", r["cohort"],
                f"{r['interval']} planning coef [{r['spec']}]",
                r["coef_planning"], int(r["n"]), f"p={r['p_planning']:.3f}")

    # ---- neural -------------------------------------------------------------
    ss = _load(AGG / "aggregate_sign_consistency.csv")
    if ss is not None:
        for _, r in ss.iterrows():
            add("neural", "aggregate_sign_consistency.csv", "EMU",
                f"{r['metric']} {r['window']} {r['control']}", r["mean"], 4,
                f"same sign {int(r['n_same_sign'])}/4")
    link = _load(XD / "xd_neural_linkage.csv")
    if link is not None:
        add("neural_behaviour", "xd_neural_linkage.csv", "EMU",
            "choice acc geometry-only (mean)", link["acc_geometry"].mean(),
            len(link))
        add("neural_behaviour", "xd_neural_linkage.csv", "EMU",
            "choice acc +neural (mean)", link["acc_geometry_neural"].mean(),
            len(link))
        add("neural_behaviour", "xd_neural_linkage.csv", "EMU",
            "choice acc +pre-decision (mean)", link["acc_pre_neural"].mean(),
            len(link))
    transfer = _load(XD / "xd_transfer.csv")
    if transfer is not None:
        for _, r in transfer.iterrows():
            add("neural_behaviour", "xd_transfer.csv", "EMU",
                f"choice acc {r['model']}", r["acc"], int(r["n_test"]))

    return pd.DataFrame(rows)


# %% [markdown]
# ## Master figure

# %%
def make_master_figure():
    fig, axes = plt.subplots(3, 2, figsize=(13, 15))

    prev = _load(PO / "exp1_prevalence.csv")
    xd = _load(XD / "xd_behavioral_summary.csv")
    # A: planning propensity
    ax = axes[0, 0]
    cloud = prev["p_plan"].to_numpy()
    emu = xd[xd["source"] == "emu"]["p_plan"].to_numpy()
    ax.scatter(np.zeros(len(cloud)) + np.random.uniform(-.08, .08, len(cloud)),
               cloud, color=C_CLOUD, s=28, label="cloud")
    ax.scatter(np.ones(len(emu)) + np.random.uniform(-.08, .08, len(emu)),
               emu, color=C_EMU, s=55, marker="D", label="EMU")
    ax.axhline(0.5, color="k", ls=":", lw=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["cloud", "EMU"])
    ax.set_ylabel("P(plan | conflict)")
    ax.set_title("A. How often do people plan?")
    ax.legend(fontsize=8)

    # B: environment sensitivity
    ax = axes[0, 1]
    env = _load(XD / "xd_environment.csv")
    for i, (coh, col) in enumerate([("cloud", C_CLOUD), ("emu", C_EMU)]):
        sub = env[env["source"] == coh]["slope"]
        ax.scatter(np.full(len(sub), i) + np.random.uniform(-.08, .08, len(sub)),
                   sub, color=col, s=28)
        ax.hlines(sub.mean(), i - .25, i + .25, color=col, lw=2.5)
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["cloud", "EMU"])
    ax.set_ylabel("slope planning rate ~ block conflict rate")
    ax.set_title("B. Sensitivity to the environment")

    # C: threat drift contrast
    ax = axes[1, 0]
    drift = _load(PO / "exp3_drift_contrast.csv")
    piv = drift.pivot_table(index="participant", columns="block_drift",
                            values="p_plan")
    for pid, r in piv.iterrows():
        ax.plot([0, 1], [r.get(0, np.nan), r.get(1, np.nan)], color="0.7",
                lw=0.8)
    ax.plot([0, 1], [piv[0].mean(), piv[1].mean()], color=C_PLAN, lw=3,
            marker="o")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["follow\n(no threat)", "drift\n(threat)"])
    ax.set_ylabel("P(plan | conflict)")
    ax.set_title("C. Planning under threat (cloud)")

    # D: timing phases
    ax = axes[1, 1]
    tsum = _load(PO / "timing_phase_summary.csv")
    sub = tsum[tsum["cohort"] == "online"]
    labels = [f"{i}\n{c}" for i in ["dec", "exec"] for c in ["greedy", "planning"]]
    x = np.arange(4)
    bottoms = np.zeros(4)
    k = 0
    for i in ["dec", "exec"]:
        for c in ["greedy", "planning"]:
            r = sub[(sub["interval"] == i) & (sub["choice"] == c)].iloc[0]
            ax.bar(k, r["fall_ms"], color="#7fb3d5", edgecolor="k", lw=.4)
            ax.bar(k, r["roll_ms"], bottom=r["fall_ms"], color="#f4b942",
                   edgecolor="k", lw=.4, label="roll" if k == 0 else None)
            bottoms[k] = r["fall_ms"] + r["roll_ms"]
            k += 1
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("time (ms)")
    ax.set_title("D. RT is transport: fixed fall, varying roll")

    # E: neural effect sizes
    ax = axes[2, 0]
    ss = _load(AGG / "aggregate_sign_consistency.csv")
    keys = ["lda:planning_vs_greedy rate post", "lda:planning_vs_greedy pca post",
            "lda:planning_vs_greedy_pre rate pre",
            "cont:conflict_mag pca post both", "threat:rel_y pca post both"]
    labels = ["plan/greed rate post", "plan/greed pca post", "plan/greed pre",
              "conflict_mag (both)", "threat (both)"]
    rates = ss[(ss["metric"] == "lda:planning_vs_greedy")
               & (ss["window"] == "post")].set_index("rep")["mean"]
    vals = [float(rates["rate"]), float(rates["pca"]),
            float(ss[ss["metric"] == "lda:planning_vs_greedy_pre"]["mean"].iloc[0]),
            float(ss[(ss["metric"] == "cont:conflict_mag") & (ss["control"] == "both")]["mean"].iloc[0]),
            float(ss[(ss["metric"] == "threat:rel_y") & (ss["control"] == "both")]["mean"].iloc[0])]
    colors = [C_PLAN, C_PLAN, "0.7", "#2a9d8f", "0.7"]
    ax.barh(np.arange(5), vals, color=colors, edgecolor="k")
    ax.set_yticks(np.arange(5)); ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0.5, color="k", ls="--", lw=1)
    ax.set_xlabel("effect size (acc for LDA, r for ridge)")
    ax.set_title("E. Neural effects (4/4 sessions, same sign)")

    # F: neural adds choice info
    ax = axes[2, 1]
    link = _load(XD / "xd_neural_linkage.csv")
    x = np.arange(len(link))
    ax.bar(x - .2, link["acc_geometry"], width=.4, color="0.6",
           label="geometry only")
    ax.bar(x + .2, link["acc_geometry_neural"], width=.4, color=C_PLAN,
           label="+ neural score")
    ax.set_xticks(x); ax.set_xticklabels(link["run_id"], fontsize=8)
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.set_ylabel("choice accuracy")
    ax.set_title("F. Neural adds choice info beyond geometry")
    ax.legend(fontsize=8)

    fig.suptitle("Falldown planning: cloud study x EMU — master results",
                 fontsize=15, y=0.995)
    fig.tight_layout()
    out = PO / "figures" / "fig_full_results_master.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    master = build_master()
    master.to_csv(PO / "full_results_master.csv", index=False)
    fig = make_master_figure()
    print(f"wrote {PO / 'full_results_master.csv'} ({len(master)} rows)")
    print(f"wrote {fig}")
    print(master.to_string(index=False))


if __name__ == "__main__":
    main()
