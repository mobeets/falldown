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
# # Entry-anchored single 250 ms bin sweep
#
# Ask how informative **one 250 ms window** is about each comparison, when the
# window is placed at different positions relative to the **entry pass** (and,
# as a dashed overlay, relative to the **choice pass**).
#
# * width  W = 250 ms (one window = 10 native 25 ms bins)
# * centers c in [-300, +1000] ms, step 50 ms  (27 windows)
# * per-window eligibility: keep trials whose valid span contains the window
# * labels: planning_vs_greedy, agree, agree_optimal_vs_lapse, planning_optimal,
#   side (left/right), move_dir, condition (4-class)
# * decoder: LDA (svd), balanced accuracy, repeated stratified 5-fold CV
# * a phase-composition strip marks, per entry-anchored center, the fraction of
#   eligible trials that are pre-choice / straddling / post-choice (rt varies)
#
# Outputs (aggregate/):
#   entry_250ms_sweep.csv, choice_250ms_sweep.csv, entry_250ms_phase.csv
#   figures/fig_entry_250ms_sweep.png, fig_choice_250ms_sweep.png
#
# Run:  .venv-analysis/bin/python analysis/neural_entry_250ms_sweep.py

# %%
import argparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from neural_common import RUN_ORDER, AGG_DIR, out_dir
from neural_lda_decoding import rate_features, build_label_vectors, RNG_SEED

# %%
LABELS = [
    "planning_vs_greedy", "agree", "agree_optimal_vs_lapse",
    "planning_optimal", "side", "move_dir", "condition",
]
LABEL_COLORS = {
    "planning_vs_greedy": "tab:red",
    "agree": "tab:orange",
    "agree_optimal_vs_lapse": "tab:purple",
    "planning_optimal": "tab:brown",
    "side": "tab:blue",
    "move_dir": "tab:cyan",
    "condition": "tab:green",
}
CHANCE_BINARY = 0.5
CHANCE_CONDITION = 0.25

WIN = 250.0
STEP = 50.0
LO, HI = -300.0, 1000.0
N_FOLDS = 5
N_REPEATS = 2
MIN_TRIALS = 40

# %%
def decode_cv(y, X, n_folds=N_FOLDS, n_repeats=N_REPEATS, seed=RNG_SEED):
    """Repeated stratified k-fold LDA balanced accuracy."""
    accs = []
    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed + rep)
        for tr, te in skf.split(X, y):
            sc = StandardScaler().fit(X[tr])
            lda = LinearDiscriminantAnalysis(solver="svd").fit(
                sc.transform(X[tr]), y[tr])
            accs.append(balanced_accuracy_score(y[te], lda.predict(sc.transform(X[te]))))
    return float(np.mean(accs))


def load_anchor(run_id, anchor):
    out = out_dir(run_id)
    fname = ("segmented_spikes_entrylocked_binned.npz" if anchor == "entry"
             else "segmented_spikes_binned.npz")
    z = np.load(out / fname, allow_pickle=True)
    tids = z["trial_ids"]
    lab = pd.read_csv(out / "trial_labels.csv").set_index("trial_id").loc[tids].reset_index()
    tt = pd.read_csv(out / "trial_table.csv").set_index("trial_id").loc[tids].reset_index()
    binned, bc = z["binned"], z["bin_centers"]
    valid = ~np.isnan(binned).all(axis=0)                 # (n_trials, n_bins)
    first = np.array([bc[np.flatnonzero(valid[i])[0]] if valid[i].any() else np.nan
                      for i in range(len(valid))])
    last = np.array([bc[np.flatnonzero(valid[i])[-1]] if valid[i].any() else np.nan
                     for i in range(len(valid))])
    rt = (tt["choice_time_ms"] - tt["entry_time_ms"]).to_numpy(float)
    rt_exec = (tt["exit_time_ms"] - tt["choice_time_ms"]).to_numpy(float)
    return dict(binned=binned, bc=bc, lab=lab, tt=tt, first=first, last=last,
                rt=rt, rt_exec=rt_exec)


def label_vectors(lab):
    lab = lab.copy()
    lab["move_dir_vx"] = 0            # avoid reading the behavior JSON
    return build_label_vectors(lab)


def sweep(data, anchor):
    """Decode every label at every center; return (rows, phase_rows)."""
    binned, bc = data["binned"], data["bc"]
    first, last, rt = data["first"], data["last"], data["rt"]
    lv = label_vectors(data["lab"])
    centers = np.arange(LO, HI + 1e-9, STEP)
    rows, phase_rows = [], []
    for c in centers:
        lo, hi = c - WIN / 2, c + WIN / 2
        elig = np.isfinite(first) & (first <= lo) & (last >= hi)
        n_elig = int(elig.sum())
        if n_elig >= MIN_TRIALS:
            X = rate_features(binned, bc, lo, hi)
        for name in LABELS:
            y, idx = lv[name]
            m = elig.copy()
            if idx is not None:
                sel = np.zeros(len(m), bool)
                sel[idx] = True
                m = m & sel
            if m.sum() < MIN_TRIALS:
                continue
            ys, Xs = y[m], X[m]
            if len(np.unique(ys)) < 2:
                continue
            acc = decode_cv(ys, Xs)
            rows.append({"anchor": anchor, "center_ms": float(c), "label": name,
                         "n": int(m.sum()), "acc": acc})
        # phase composition of the eligible trials (rt per trial)
        if n_elig:
            pre = int((hi <= rt[elig]).sum())
            post = int((lo >= rt[elig]).sum())
            straddle = n_elig - pre - post
            phase_rows.append({"anchor": anchor, "center_ms": float(c),
                               "n": n_elig, "frac_pre": pre / n_elig,
                               "frac_post": post / n_elig,
                               "frac_straddle": straddle / n_elig,
                               "median_rt": float(np.median(rt[elig]))})
    return pd.DataFrame(rows), pd.DataFrame(phase_rows)


# %%
def make_sweep_figure(entry, choice, phase, path, anchor_label, overlay=True):
    runs = list(RUN_ORDER)
    fig, axes = plt.subplots(len(runs), 1, figsize=(11, 3.0 * len(runs)),
                             sharex=True)
    if len(runs) == 1:
        axes = [axes]
    handles = [Line2D([0], [0], color=LABEL_COLORS[l], lw=2, label=l) for l in LABELS]
    handles += [Line2D([0], [0], color="k", ls="--", lw=1.2, label="chance (binary)"),
                Line2D([0], [0], color="gray", ls=":", lw=1.2, label="chance (4-class)")]
    for ax, run in zip(axes, runs):
        e = entry[entry["run_id"] == run]
        ch = choice[choice["run_id"] == run]
        for lab in LABELS:
            ge = e[e["label"] == lab].sort_values("center_ms")
            if len(ge):
                ax.plot(ge["center_ms"], ge["acc"], "-o", ms=3,
                        color=LABEL_COLORS[lab], lw=1.6)
            gc = ch[ch["label"] == lab].sort_values("center_ms")
            if overlay and len(gc):
                # shift choice-anchored centers by the run's median rt (approx)
                ph = phase[phase["run_id"] == run]
                shift = float(ph["median_rt"].median()) if len(ph) else 0.0
                ax.plot(gc["center_ms"] + shift, gc["acc"], "--", lw=1.1,
                        color=LABEL_COLORS[lab], alpha=0.7)
        ax.axhline(CHANCE_BINARY, color="k", ls="--", lw=1.0)
        ax.axhline(CHANCE_CONDITION, color="gray", ls=":", lw=1.0)
        ax.axvline(0, color="green", lw=1.0, alpha=0.6)
        ph = phase[phase["run_id"] == run].sort_values("center_ms")
        if len(ph):
            ax.fill_between(ph["center_ms"], 0, 1, where=ph["frac_post"] > 0.5,
                            transform=ax.get_xaxis_transform(), color="gray",
                            alpha=0.10, label="mostly post-choice")
        ax.set_ylim(0.30, 0.80)
        ax.set_ylabel("balanced acc")
        note = (f"; dashed = choice-anchored, shifted by median rt"
                if overlay else "")
        ax.set_title(f"{run}  ({anchor_label}{note})", fontsize=10)
    axes[-1].set_xlabel(f"window center (ms rel. {anchor_label.split()[0]})")
    axes[0].legend(handles=handles, ncol=3, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def make_phase_figure(phase, path):
    runs = list(RUN_ORDER)
    fig, axes = plt.subplots(1, len(runs), figsize=(4.0 * len(runs), 3.0),
                             sharey=True)
    for ax, run in zip(axes, runs):
        ph = phase[phase["run_id"] == run].sort_values("center_ms")
        ax.fill_between(ph["center_ms"], 0, ph["frac_pre"],
                        color="tab:blue", alpha=0.6, label="pre-choice")
        ax.fill_between(ph["center_ms"], ph["frac_pre"],
                        ph["frac_pre"] + ph["frac_straddle"],
                        color="tab:orange", alpha=0.6, label="straddle")
        ax.fill_between(ph["center_ms"], ph["frac_pre"] + ph["frac_straddle"], 1,
                        color="tab:red", alpha=0.6, label="post-choice")
        ax.axvline(0, color="green", lw=1.0, alpha=0.6)
        ax.set_title(run, fontsize=10)
        ax.set_xlabel("center (ms rel. entry)")
    axes[0].set_ylabel("fraction of eligible trials")
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Phase composition of each entry-anchored 250 ms window", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# %%
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=RUN_ORDER)
    args = ap.parse_args()

    entry_rows, choice_rows, phase_frames = [], [], []
    for run in args.runs:
        print(f"[{run}] entry-anchored sweep")
        e, ph = sweep(load_anchor(run, "entry"), "entry")
        e["run_id"] = run
        ph["run_id"] = run
        print(f"[{run}] choice-anchored sweep")
        c, _ = sweep(load_anchor(run, "choice"), "choice")
        c["run_id"] = run
        entry_rows.append(e)
        choice_rows.append(c)
        phase_frames.append(ph)

    entry = pd.concat(entry_rows, ignore_index=True)
    choice = pd.concat(choice_rows, ignore_index=True)
    phase = pd.concat(phase_frames, ignore_index=True)

    AGG_DIR.mkdir(parents=True, exist_ok=True)
    (AGG_DIR / "figures").mkdir(parents=True, exist_ok=True)
    entry.to_csv(AGG_DIR / "entry_250ms_sweep.csv", index=False)
    choice.to_csv(AGG_DIR / "choice_250ms_sweep.csv", index=False)
    phase.to_csv(AGG_DIR / "entry_250ms_phase.csv", index=False)

    make_sweep_figure(entry, choice, phase,
                      AGG_DIR / "figures" / "fig_entry_250ms_sweep.png", "entry")
    make_sweep_figure(choice, choice.iloc[0:0], phase.iloc[0:0],
                      AGG_DIR / "figures" / "fig_choice_250ms_sweep.png",
                      "choice", overlay=False)
    make_phase_figure(phase, AGG_DIR / "figures" / "fig_entry_250ms_phase.png")

    print("\nPeak accuracy per label (entry-anchored, mean over runs):")
    m = entry.groupby(["label", "center_ms"])["acc"].mean().reset_index()
    for lab in LABELS:
        s = m[m["label"] == lab]
        if len(s):
            r = s.loc[s["acc"].idxmax()]
            print(f"  {lab:26s} peak {r['acc']:.3f} @ {r['center_ms']:+.0f} ms")
    print(f"\nwrote CSVs + figures to {AGG_DIR}")


if __name__ == "__main__":
    main()
