# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#     jupytext_version: 1.19.4
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Neural figures: replication, confounds, death/threat, regions
#
# Reads the per-run result CSVs and the confound-audit outputs and renders one
# figure per conclusion (each showing the effect and its control). No
# re-fitting: run `neural_confound_audit.py` first.
#
# Output: analysis/neural_outputs/figures/fig_neural_*.png
#
# Run with:
#   C:\Users\manik\AppData\Local\Programs\Python\Python311\python.exe analysis\neural_figures.py

# %%
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neural_common import RUN_ORDER, RUNS, AGG_DIR, load_run_results

FIG_DIR = AGG_DIR.parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RUN_COLORS = {"yfz_1": "#1f6fb4", "yfz_2": "#5aa9e6",
              "yga_1": "#e07b39", "yga_2": "#f0b27a"}


def _agg(name):
    p = AGG_DIR / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def _save(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {name}")


def _sem(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0


# %% [markdown]
# ## 1. Cross-run replication

# %%
def fig_replication():
    eff = _agg("aggregate_headline_effects.csv")
    summ = _agg("aggregate_sign_consistency.csv")
    specs = [
        ("cont:conflict_mag", "pca", "both", "conflict_mag (pca, both)"),
        ("cont:chosen_planning_cost", "pca", "none",
         "chosen_planning_cost (pca, raw)"),
        ("lda:planning_vs_greedy", "rate", "", "planning_vs_greedy (rate, post)"),
        ("lda:planning_vs_greedy", "pca", "", "planning_vs_greedy (pca, post)"),
        ("lda:planning_vs_greedy_pre", "rate", "",
         "planning_vs_greedy (rate, pre)"),
        ("lda:condition", "pca", "", "condition (pca, post)"),
        ("lda:agree", "pca", "", "agree (pca, post)"),
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = []
    for i, (metric, rep, ctrl, label) in enumerate(specs):
        y = len(specs) - 1 - i
        sub = eff[(eff["metric"] == metric) & (eff["rep"] == rep)
                  & (eff["control"].fillna("") == ctrl)]
        if sub.empty:
            continue
        for r in sub.itertuples():
            ax.scatter(r.value, y, s=55, color=RUN_COLORS.get(r.run_id, "0.5"),
                       edgecolor="k", zorder=3)
        m = sub["value"].mean()
        ax.errorbar([m], [y], xerr=[_sem(sub["value"])], fmt="D", color="k",
                    ms=9, capsize=5, zorder=4)
        row = summ[(summ["metric"] == metric) & (summ["rep"] == rep)
                   & (summ["control"].fillna("") == ctrl)]
        if len(row):
            ax.text(1.02, y, f"{int(row['n_same_sign'].iloc[0])}/{int(row['n_sessions'].iloc[0])} "
                             f"same sign, p={row['signflip_p'].iloc[0]:.3f}",
                    transform=ax.get_yaxis_transform(), va="center", fontsize=8)
        labels.append((y, label))
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.axvline(0.5, color="0.6", ls=":", lw=1)
    ax.set_yticks([y for y, _ in labels])
    ax.set_yticklabels([l for _, l in labels], fontsize=9)
    ax.set_xlabel("effect size (corr, or balanced accuracy vs chance 0.5)")
    ax.set_title("Cross-run replication of headline neural effects\n"
                 "(dots = sessions, diamond = mean \u00b1 SEM, colour = run)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=r,
                          markeredgecolor="k") for r, c in RUN_COLORS.items()]
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    _save(fig, "fig_neural_replication.png")


# %% [markdown]
# ## 2. Window profile (pre vs post vs entry-anchored)

# %%
def _window_profile_rows():
    rows = []
    for run in RUN_ORDER:
        ld = load_run_results(run, "neural_lda_decoding_results.csv")
        if ld is not None:
            for win, wlabel in [("pre", "pre [-1000,0]"), ("post", "post [0,1000]")]:
                s = ld[(ld["hypothesis"] == "planning_vs_greedy")
                       & (ld["rep"] == "rate") & (ld["window"] == win)]
                if len(s):
                    rows.append({"run_id": run, "window": wlabel,
                                 "acc": s["acc_mean"].iloc[0]})
        ep = load_run_results(run, "entry_locked_lda_results.csv")
        if ep is not None:
            s = ep[(ep["hypothesis"] == "planning_vs_greedy")
                   & (ep["window"] == "entry+250")]
            if len(s):
                rows.append({"run_id": run, "window": "entry+250",
                             "acc": s["acc_mean"].iloc[0]})
            s = ep[(ep["hypothesis"] == "planning_vs_greedy")
                   & (ep["window"] == "approach")]
            if len(s):
                rows.append({"run_id": run, "window": "approach",
                             "acc": s["acc_mean"].iloc[0]})
    return pd.DataFrame(rows)


def fig_window_profile():
    d = _window_profile_rows()
    order = ["pre [-1000,0]", "entry+250", "approach", "post [0,1000]"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharey=True)
    for ax, run in zip(axes, RUN_ORDER):
        s = d[d["run_id"] == run].set_index("window").reindex(order)
        ax.bar(range(len(order)), s["acc"].values, color=RUN_COLORS[run],
               edgecolor="k")
        ax.axhline(0.5, color="k", ls="--", lw=1)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(["pre", "entry+250", "approach", "post"],
                           rotation=30, ha="right")
        ax.set_title(f"{run} ({RUNS[run].participant})")
        ax.set_ylim(0.4, 1.0)
    axes[0].set_ylabel("planning_vs_greedy balanced accuracy (rate)")
    fig.suptitle("Fixed pre-decision windows (pre, entry+250) are at chance; the "
                 "effect appears post-choice. The variable-length 'approach' "
                 "window is RT-confounded (see neural_decoding_confounds.md).",
                 y=1.05)
    _save(fig, "fig_neural_window_profile.png")


# %% [markdown]
# ## 3. Confound collapse (raw -> residualized)

# %%
def fig_confound_collapse():
    regs = ["conflict_mag", "chosen_planning_cost"]
    controls = ["none", "target", "both"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    for ax, reg in zip(axes, regs):
        width = 0.2
        for j, run in enumerate(RUN_ORDER):
            cd = load_run_results(run, "continuous_decoding_results.csv")
            if cd is None:
                continue
            vals = []
            for c in controls:
                s = cd[(cd["regressor"] == reg) & (cd["rep"] == "pca")
                       & (cd["window"] == "post") & (cd["control"] == c)]
                vals.append(s["corr_mean"].iloc[0] if len(s) else np.nan)
            ax.bar(np.arange(len(controls)) + (j - 1.5) * width, vals,
                   width=width, color=RUN_COLORS[run], edgecolor="k",
                   label=run if reg == regs[0] else None)
        ax.axhline(0, color="k", ls="--", lw=1)
        ax.set_xticks(range(len(controls)))
        ax.set_xticklabels(controls)
        ax.set_title(reg)
    axes[0].set_ylabel("CV correlation (pca, post)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Nuisance residualization: conflict_mag survives, "
                 "chosen_planning_cost collapses", y=1.03)
    _save(fig, "fig_neural_confound_collapse.png")


# %% [markdown]
# ## 4. Leakage control (shuffled vs leave-block-out CV)

# %%
def fig_leakage():
    lk = _agg("leakage_control.csv")
    lda = lk[lk["metric"].str.startswith("lda:")]
    ridge = lk[lk["metric"] == "cont:conflict_mag"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    for r in lda.itertuples():
        ax.scatter(r.shuffled, r.leave_block_out,
                   color=RUN_COLORS.get(r.run_id, "0.5"), s=60, edgecolor="k",
                   zorder=3)
    lims = [0.25, 1.0]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlim(*lims); ax.set_ylim(*lims)
    ax.set_xlabel("shuffled-CV balanced accuracy")
    ax.set_ylabel("leave-block-out balanced accuracy")
    ax.set_title("LDA decodings")

    ax = axes[1]
    for r in ridge.itertuples():
        ax.scatter(r.shuffled, r.leave_block_out,
                   color=RUN_COLORS.get(r.run_id, "0.5"), s=70, edgecolor="k",
                   zorder=3)
        ax.annotate(r.run_id, (r.shuffled, r.leave_block_out),
                    fontsize=7, xytext=(4, 3), textcoords="offset points")
    lims = [0.0, 0.55]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlim(*lims); ax.set_ylim(*lims)
    ax.set_xlabel("shuffled-CV correlation")
    ax.set_ylabel("leave-block-out correlation")
    ax.set_title("conflict_mag (ridge, pca)")
    fig.suptitle("Temporal-leakage control: block-grouped CV matches shuffled CV",
                 y=1.02)
    _save(fig, "fig_neural_leakage_control.png")


# %% [markdown]
# ## 5. Nuisance associations

# %%
def fig_nuisance_associations():
    frames = []
    for run in RUN_ORDER:
        a = load_run_results(run, "decoding_label_association.csv")
        if a is None:
            continue
        a = a[a["hypothesis"].isin(
            ["planning_vs_greedy", "condition", "agree", "side"])].copy()
        a["run_id"] = run
        frames.append(a)
    if not frames:
        return
    all_a = pd.concat(frames, ignore_index=True)
    hyps = ["planning_vs_greedy", "condition", "agree", "side"]
    nuis = sorted(all_a["nuisance"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for ax, run in zip(axes, ["yfz_1", "yga_1"]):
        sub = all_a[all_a["run_id"] == run]
        M = np.full((len(hyps), len(nuis)), np.nan)
        for i, h in enumerate(hyps):
            for j, n in enumerate(nuis):
                s = sub[(sub["hypothesis"] == h) & (sub["nuisance"] == n)]
                if len(s):
                    M[i, j] = s["stat"].iloc[0]
        im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(nuis)))
        ax.set_xticklabels(nuis, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(hyps)))
        ax.set_yticklabels(hyps, fontsize=9)
        ax.set_title(f"label x nuisance association ({run})")
        fig.colorbar(im, ax=ax, fraction=0.046, label="stat (r / eta)")
    fig.suptitle("Confound associations: planning_vs_greedy is strongly tied "
                 "to ball time; condition to RT", y=1.03)
    _save(fig, "fig_neural_nuisance_associations.png")


# %% [markdown]
# ## 6. Matched re-decode

# %%
def fig_matched_redecode():
    frames = []
    for run in RUN_ORDER:
        m = load_run_results(run, "decoding_matched_redecode.csv")
        if m is None:
            continue
        m["run_id"] = run
        frames.append(m)
    if not frames:
        return
    all_m = pd.concat(frames, ignore_index=True)
    hyps = ["planning_vs_greedy", "planning_optimal", "agree"]
    schemes = ["unmatched", "matched_side", "matched_side_rt_block",
               "matched_side_rt_ball"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for ax, h in zip(axes, hyps):
        for j, sch in enumerate(schemes):
            s = all_m[(all_m["hypothesis"] == h) & (all_m["scheme"] == sch)
                      & (all_m["rep"] == "pca")]
            vals = s["acc_mean"].dropna().values
            if len(vals) == 0:
                continue
            ax.bar(j, np.mean(vals), color="steelblue", edgecolor="k",
                   yerr=_sem(vals), capsize=4)
            ax.text(j, np.nanmax(vals) + 0.02, f"n={int(s['n_trials'].max())}",
                    ha="center", fontsize=7)
        ax.axhline(0.5, color="k", ls="--", lw=1)
        ax.set_xticks(range(len(schemes)))
        ax.set_xticklabels(["unmatched", "+side", "+side\nRT block",
                            "+side\nRT ball"], fontsize=8)
        ax.set_title(h)
        ax.set_ylim(0.4, 1.0)
    axes[0].set_ylabel("balanced accuracy (pca, post)")
    fig.suptitle("Matched re-decoding: planning_vs_greedy survives side/RT/block "
                 "matching (pca)", y=1.03)
    _save(fig, "fig_neural_matched_redecode.png")


# %% [markdown]
# ## 7. Death (exploratory)

# %%
def fig_death():
    dm = _agg("death_matched.csv")
    if dm.empty:
        print("  no death data")
        return
    runs = sorted(dm["run_id"].unique())
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    width = 0.35
    for j, c in enumerate(["all_normal", "block_matched"]):
        s = dm[dm["control"] == c].set_index("run_id").reindex(runs)
        ax.bar(np.arange(len(runs)) + (j - 0.5) * width, s["acc_mean"],
               width=width, label=c, edgecolor="k")
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels([f"{r}\n(n={int(dm[dm['run_id']==r]['n_deaths'].iloc[0])})"
                        for r in runs])
    ax.set_ylabel("death vs normal balanced accuracy")
    ax.set_ylim(0.4, 0.65)
    ax.legend(fontsize=8)
    ax.set_title("Death decoding (rate, [-500,0] ms)\n"
                 "exploratory: significant in 1 of 2 sessions; block-matching "
                 "does not change it")
    _save(fig, "fig_neural_death.png")


# %% [markdown]
# ## 8. Threat

# %%
def fig_threat():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    ax = axes[0]
    specs = [("rate", "pre", "none"), ("rate", "pre", "target"),
             ("rate", "pre", "both"), ("pca", "post", "none"),
             ("pca", "post", "target"), ("pca", "post", "both")]
    xl = [f"{r}/{w}/{c}" for r, w, c in specs]
    width = 0.2
    for j, run in enumerate(RUN_ORDER):
        tc = load_run_results(run, "threat_continuous_decoding.csv")
        if tc is None:
            continue
        vals = []
        for rep, w, c in specs:
            s = tc[(tc["rep"] == rep) & (tc["window"] == w) & (tc["control"] == c)]
            vals.append(s["corr_mean"].iloc[0] if len(s) else np.nan)
        ax.bar(np.arange(len(specs)) + (j - 1.5) * width, vals, width=width,
               color=RUN_COLORS[run], edgecolor="k", label=run)
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_xticks(range(len(specs)))
    ax.set_xticklabels(xl, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("CV correlation (threat rel_y)")
    ax.set_title("Threat decoding collapses under residualization")
    ax.legend(fontsize=8)

    ax = axes[1]
    ta = _agg("threat_associations.csv")
    if not ta.empty:
        mean_abs = ta.groupby("nuisance")["pearson_r"].apply(
            lambda s: np.abs(s).mean()).sort_values(ascending=False)
        mean_abs = mean_abs[mean_abs.index != "rel_y"]
        ax.barh(range(len(mean_abs)), mean_abs.values, color="indianred",
                edgecolor="k")
        ax.set_yticks(range(len(mean_abs)))
        ax.set_yticklabels(mean_abs.index, fontsize=9)
        ax.set_xlabel("mean |Pearson r| with threat regressor")
        ax.set_title("Threat is confounded with RT/duration")
    fig.suptitle("Continuous proximity-to-death: raw signal is RT/kinematic "
                 "confounded", y=1.03)
    _save(fig, "fig_neural_threat.png")


# %% [markdown]
# ## 9. Regions

# %%
def fig_region():
    rc = _agg("region_count_matched.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    ax = axes[0]
    regions = ["CA", "amygdala", "anterior_hippocampus", "hippocampal_body"]
    runs = RUN_ORDER
    width = 0.2
    s = rc[rc["hypothesis"] == "planning_vs_greedy"]
    for j, run in enumerate(runs):
        sr = s[s["run_id"] == run].set_index("region").reindex(regions)
        ax.bar(np.arange(len(regions)) + (j - 1.5) * width, sr["acc_mean"],
               width=width, color=RUN_COLORS[run], edgecolor="k", label=run)
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.set_xticks(range(len(regions)))
    ax.set_xticklabels(["CA", "amygdala", "ant HC", "HC body"], fontsize=9)
    ax.set_ylabel("balanced accuracy (unit-count matched)")
    ax.set_title("planning_vs_greedy decodes in every region")
    ax.legend(fontsize=8)

    # cross-region score correlation (mean across runs)
    ax = axes[1]
    mats = []
    for run in runs:
        rd = load_run_results(run, "region_decoding.csv")
        if rd is None:
            continue
        sc = rd[(rd["analysis"] == "score_correlation")
                & (rd["hypothesis"] == "planning_vs_greedy")]
        if sc.empty:
            continue
        M = pd.DataFrame(index=regions, columns=regions, dtype=float)
        for r in sc.itertuples():
            a, b = r.region.split("|")
            M.loc[a, b] = M.loc[b, a] = r.acc_mean
        np.fill_diagonal(M.values, 1.0)
        mats.append(M.astype(float))
    if mats:
        arr = np.stack([m.values for m in mats])          # (n_runs, R, R)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Mv = np.nanmean(arr, axis=0)
        im = ax.imshow(Mv, cmap="viridis", vmin=-0.1, vmax=0.6)
        ax.set_xticks(range(len(regions)))
        ax.set_xticklabels(["CA", "amyg", "antHC", "body"], fontsize=9)
        ax.set_yticks(range(len(regions)))
        ax.set_yticklabels(["CA", "amyg", "antHC", "body"], fontsize=9)
        for i in range(len(regions)):
            for j in range(len(regions)):
                if np.isfinite(Mv[i, j]):
                    ax.text(j, i, f"{Mv[i, j]:.2f}", ha="center",
                            va="center", color="white", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, label="score correlation")
        ax.set_title("Cross-region decision-score correlation\n"
                     "(planning_vs_greedy, mean over runs)")
    fig.suptitle("Region-resolved decoding and cross-region agreement", y=1.03)
    _save(fig, "fig_neural_region.png")


# %% [markdown]
# ## 10. Bursty-unit control

# %%
def fig_bursty():
    rows = []
    for run in RUN_ORDER:
        b = load_run_results(run, "bursty_sensitivity_decoding.csv")
        if b is None:
            continue
        s = b[(b["regressor"] == "conflict_mag") & (b["rep"] == "pca")
              & (b["control"] == "both")]
        for r in s.itertuples():
            rows.append({"run_id": run, "unit_set": r.unit_set,
                         "corr_mean": r.corr_mean})
    d = pd.DataFrame(rows)
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    sets = ["all", "no_bursty"]
    width = 0.35
    for j, us in enumerate(sets):
        s = d[d["unit_set"] == us].set_index("run_id").reindex(RUN_ORDER)
        ax.bar(np.arange(len(RUN_ORDER)) + (j - 0.5) * width, s["corr_mean"],
               width=width, label=us, edgecolor="k")
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_xticks(range(len(RUN_ORDER)))
    ax.set_xticklabels(RUN_ORDER)
    ax.set_ylabel("conflict_mag CV correlation (pca, both)")
    ax.legend(fontsize=8)
    ax.set_title("Dropping bursty units (ISI CV>1.3) leaves conflict_mag intact")
    _save(fig, "fig_neural_bursty.png")


# %%
def main():
    print("Making neural figures ...")
    fig_replication()
    fig_window_profile()
    fig_confound_collapse()
    fig_leakage()
    fig_nuisance_associations()
    fig_matched_redecode()
    fig_death()
    fig_threat()
    fig_region()
    fig_bursty()
    print(f"Done. Figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
