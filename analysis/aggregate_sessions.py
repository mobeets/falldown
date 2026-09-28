# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Cross-run aggregation and replication
#
# Reads the per-run result CSVs (no re-fitting) for every registered EMU run
# and asks whether the headline neural effects replicate across sessions.
#
# Headline effects aggregated:
#   cont:conflict_mag            ridge CV corr (post window; none/target/both)
#   cont:chosen_planning_cost    ridge CV corr (post; shows the confound collapse)
#   lda:planning_vs_greedy       LDA balanced accuracy (post; rate + pca)
#   lda:condition                LDA balanced accuracy (post; pca)
#   lda:agree                    LDA balanced accuracy (post; pca)
#   lda:planning_vs_greedy_pre   LDA balanced accuracy (pre; rate) - the null
#   sel:death_vs_normal_nsig     number of FDR-significant units (post)
#   bursty:conflict_mag_no_bursty  conflict_mag pca/both after dropping CV>1.3 units
#
# Statistics per metric:
#   - mean / sd across sessions
#   - sign-consistency (sessions matching the group mean sign) + binomial p
#   - session-level sign-flip permutation p (2^n sign assignments; honest with
#     few sessions)
#   - within-participant (YFZ x2) vs across-participant (YGA x2) means kept
#     separate (units are not matched across runs; per-session decoders are the
#     replication unit)
#
# Outputs (analysis/neural_outputs/aggregate/):
#   aggregate_headline_effects.csv   long per-session table
#   aggregate_sign_consistency.csv   per-metric summary
#   aggregate_forest.png             forest plot of the main effects
#
# Run with:
#   C:\Users\manik\AppData\Local\Programs\Python\Python311\python.exe analysis\aggregate_sessions.py

# %%
import warnings
warnings.filterwarnings("ignore")

from itertools import product

import numpy as np
import pandas as pd
from scipy import stats

from neural_common import RUN_ORDER, RUNS, AGG_DIR, load_run_results

# -----------------------------------------------------------------------


# %%
def collect_effects():
    """Long per-session table of headline effects from each run's CSVs."""
    rows = []

    def add(metric, run, pid, value, perm_p=np.nan, rep="", window="",
            control=""):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return
        rows.append({"metric": metric, "rep": rep, "window": window,
                     "control": control, "run_id": run, "participant": pid,
                     "value": float(value), "perm_p": perm_p})

    for run in RUN_ORDER:
        pid = RUNS[run].participant

        cd = load_run_results(run, "continuous_decoding_results.csv")
        if cd is not None:
            specs = [
                ("cont:conflict_mag", "conflict_mag", "pca", "post", "none"),
                ("cont:conflict_mag", "conflict_mag", "pca", "post", "target"),
                ("cont:conflict_mag", "conflict_mag", "pca", "post", "both"),
                ("cont:conflict_mag", "conflict_mag", "rate", "post", "both"),
                ("cont:chosen_planning_cost", "chosen_planning_cost", "pca",
                 "post", "none"),
            ]
            for metric, reg, rep, win, ctrl in specs:
                sub = cd[(cd["regressor"] == reg) & (cd["rep"] == rep)
                         & (cd["window"] == win) & (cd["control"] == ctrl)]
                if len(sub):
                    add(metric, run, pid, sub["corr_mean"].iloc[0],
                        float(sub["perm_p"].iloc[0]), rep, win, ctrl)

        ld = load_run_results(run, "neural_lda_decoding_results.csv")
        if ld is not None:
            specs = [
                ("lda:planning_vs_greedy", "planning_vs_greedy", "rate", "post"),
                ("lda:planning_vs_greedy", "planning_vs_greedy", "pca", "post"),
                ("lda:condition", "condition", "pca", "post"),
                ("lda:agree", "agree", "pca", "post"),
                ("lda:planning_vs_greedy_pre", "planning_vs_greedy", "rate",
                 "pre"),
            ]
            for metric, hyp, rep, win in specs:
                sub = ld[(ld["hypothesis"] == hyp) & (ld["rep"] == rep)
                         & (ld["window"] == win)]
                if len(sub):
                    add(metric, run, pid, sub["acc_mean"].iloc[0],
                        float(sub["perm_p"].iloc[0]), rep, win, "")

        sel = load_run_results(run, "selectivity_results.csv")
        if sel is not None and "contrast" in sel.columns:
            sub = sel[(sel["contrast"] == "death_vs_normal")
                      & (sel["window"] == "post")]
            if len(sub):
                add("sel:death_vs_normal_nsig", run, pid,
                    float((sub["q_fdr"] < 0.05).sum()), np.nan, "", "post", "")

        bs = load_run_results(run, "bursty_sensitivity_decoding.csv")
        if bs is not None:
            sub = bs[(bs["regressor"] == "conflict_mag") & (bs["rep"] == "pca")
                     & (bs["control"] == "both")
                     & (bs["unit_set"] == "no_bursty")]
            if len(sub):
                add("bursty:conflict_mag_no_bursty", run, pid,
                    sub["corr_mean"].iloc[0], float(sub["perm_p"].iloc[0]),
                    "pca", "post", "both")

        dd = load_run_results(run, "death_decoding.csv")
        if dd is not None:
            sub = dd[np.isclose(dd["window_lo_ms"], -500.0)]
            if len(sub):
                add("death:acc", run, pid, sub["acc_mean"].iloc[0],
                    float(sub["perm_p"].iloc[0]), "rate", "death-500", "")

        tc = load_run_results(run, "threat_continuous_decoding.csv")
        if tc is not None:
            for rep, win, ctrl in [("pca", "post", "both"),
                                   ("rate", "pre", "none")]:
                sub = tc[(tc["rep"] == rep) & (tc["window"] == win)
                         & (tc["control"] == ctrl)]
                if len(sub):
                    add("threat:rel_y", run, pid, sub["corr_mean"].iloc[0],
                        float(sub["perm_p"].iloc[0]), rep, win, ctrl)

        ep = load_run_results(run, "entry_locked_lda_results_pca.csv")
        if ep is not None:
            sub = ep[(ep["hypothesis"] == "planning_vs_greedy")
                     & (ep["window"] == "entry+250")]
            if len(sub):
                add("entry_pca:planning_vs_greedy", run, pid,
                    sub["acc_mean"].iloc[0], float(sub["perm_p"].iloc[0]),
                    "pca", "entry+250", "")

    return pd.DataFrame(rows)


# %%
def signflip_p(values):
    """Exact session-level sign-flip permutation p for the mean effect."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return np.nan
    observed = values.mean()
    signs = np.array(list(product([-1.0, 1.0], repeat=n)))
    null_means = (signs * values).mean(axis=1)
    return float(np.mean(np.abs(null_means) >= abs(observed) - 1e-12))


def summarize(effects):
    rows = []
    for key, sub in effects.groupby(["metric", "rep", "window", "control"],
                                    dropna=False):
        metric, rep, win, ctrl = key
        vals = sub["value"].to_numpy(float)
        parts = sub["participant"].to_numpy()
        n = len(vals)
        if n == 0:
            continue
        mean = float(vals.mean())
        sd = float(vals.std(ddof=1)) if n > 1 else 0.0
        n_same = int(np.sum(np.sign(vals) == np.sign(mean))) if mean != 0 else 0
        try:
            binom_p = float(stats.binomtest(n_same, n, 0.5,
                                            alternative="greater").pvalue)
        except Exception:
            binom_p = np.nan
        within = vals[parts == "YFZ"]
        across = vals[parts == "YGA"]
        rows.append({
            "metric": metric, "rep": rep, "window": win, "control": ctrl,
            "n_sessions": n,
            "runs": ",".join(sub["run_id"]),
            "mean": mean, "sd": sd,
            "n_same_sign": n_same, "binomial_p": binom_p,
            "signflip_p": signflip_p(vals),
            "within_participant_mean": float(within.mean()) if len(within) else np.nan,
            "across_participant_mean": float(across.mean()) if len(across) else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["metric", "rep", "control"])


# %%
def forest_plot(summary, effects, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    headline = [
        ("cont:conflict_mag", "pca", "both", "conflict_mag (pca, both)"),
        ("cont:chosen_planning_cost", "pca", "none",
         "chosen_planning_cost (pca, raw)"),
        ("lda:planning_vs_greedy", "rate", "post",
         "planning_vs_greedy (rate, post)"),
        ("lda:planning_vs_greedy", "pca", "post",
         "planning_vs_greedy (pca, post)"),
        ("lda:planning_vs_greedy_pre", "rate", "pre",
         "planning_vs_greedy (rate, pre)"),
        ("entry_pca:planning_vs_greedy", "pca", "entry+250",
         "planning_vs_greedy (pca, entry+250)"),
        ("death:acc", "rate", "death-500", "death vs normal (rate, -500)"),
        ("threat:rel_y", "pca", "post", "threat rel_y (pca, post, both)"),
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    yt, yl = [], []
    for i, (metric, rep, ctrl, label) in enumerate(headline):
        sub = effects[(effects["metric"] == metric) & (effects["rep"] == rep)
                      & (effects["control"] == ctrl)]
        if sub.empty:
            continue
        y = len(headline) - 1 - i
        ax.scatter(sub["value"], [y] * len(sub), s=45, zorder=3,
                   color="0.4", edgecolor="k")
        row = summary[(summary["metric"] == metric) & (summary["rep"] == rep)
                      & (summary["control"] == ctrl)]
        if len(row):
            m, sd, n = row["mean"].iloc[0], row["sd"].iloc[0], row["n_sessions"].iloc[0]
            se = sd / np.sqrt(n) if n else 0
            ax.errorbar([m], [y], xerr=[se], fmt="D", color="crimson",
                        ms=10, capsize=5, zorder=4)
        yt.append(y)
        yl.append(label)
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set_yticks(yt)
    ax.set_yticklabels(yl, fontsize=9)
    ax.set_xlabel("effect size (corr or balanced accuracy)")
    ax.set_title("Cross-run replication of headline neural effects\n"
                 "(dots = sessions, diamond = mean \u00b1 SEM)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# %%
def main():
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    print("Collecting per-run headline effects ...")
    effects = collect_effects()
    if effects.empty:
        print("No per-run results found.")
        return
    effects.to_csv(AGG_DIR / "aggregate_headline_effects.csv", index=False)

    summary = summarize(effects)
    summary.to_csv(AGG_DIR / "aggregate_sign_consistency.csv", index=False)

    print("\nPer-metric summary:")
    cols = ["metric", "rep", "control", "n_sessions", "mean", "n_same_sign",
            "signflip_p", "binomial_p"]
    print(summary[cols].round(3).to_string(index=False))

    forest_plot(summary, effects, AGG_DIR / "aggregate_forest.png")
    print(f"\nSaved aggregate_headline_effects.csv, "
          f"aggregate_sign_consistency.csv, aggregate_forest.png to {AGG_DIR}")


if __name__ == "__main__":
    main()
