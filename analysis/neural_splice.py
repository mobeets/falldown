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
# # Part 10 - Splicing the next trial into the leftover (vs zero-padding)
#
# The truncated binned arrays are NaN outside each trial's own span, and
# `pca_features` pads NaN->0, which leaks the coverage/duration cue (Part 8).
# Here we rebuild a *continuous* representation from `segmented_spikes_raw.pkl`
# (spikes relative to each trial's choice), extract choice-anchored windows from
# the global spike train (so post-exit bins contain the next trial's activity =
# "splice"), and re-run the headline decodes with leakage controls:
#
#   truncated      - original npz (NaN outside span -> zero-padded)
#   spliced        - continuous global extraction (next trial in the leftover)
#   ablation       - spliced but leftover bins set to NaN (discard the splice)
#   missplice      - leftover bins taken from a random other trial
#   shift          - leftover bins taken from trial t+2
#
# Outputs: splice.csv, figures/fig_splice.png

# %%
import json
import pickle
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score

from neural_common import RUN_ORDER, AGG_DIR, out_dir
from neural_lda_decoding import pca_features, RNG_SEED
from neural_pass_timing import ridge_cv
from neural_decoder_geometry import load, targets

# %%
BIN_MS = 25.0
LO, HI = -2000.0, 2000.0
TARGETS = ["conflict_mag", "chosen_cost", "step2", "rt_exec"]


def build_global(run):
    d = load(run)
    tt = d["tt"]
    raw = pickle.load(open(out_dir(run) / "segmented_spikes_raw.pkl", "rb"))
    npz = np.load(out_dir(run) / "segmented_spikes_binned.npz", allow_pickle=True)
    unit_ids = npz["unit_ids"]
    uid2row = {int(u): i for i, u in enumerate(unit_ids)}
    n_units = len(unit_ids)
    choice = tt["choice_time_ms"].to_numpy(float)
    t_start = tt["trial_start_ms"].to_numpy(float)
    t_exit = tt["exit_time_ms"].to_numpy(float)
    gmin = choice.min() + LO - BIN_MS
    gmax = choice.max() + HI + BIN_MS
    n_bins = int(np.ceil((gmax - gmin) / BIN_MS))
    G = np.zeros((n_units, n_bins))
    for (u, t), rel in raw.items():
        r = uid2row.get(int(u))
        if r is None or t >= len(choice):
            continue
        abs_t = choice[t] + rel
        idx = ((abs_t - gmin) / BIN_MS).astype(int)
        idx = idx[(idx >= 0) & (idx < n_bins)]
        if len(idx):
            G[r] += np.bincount(idx, minlength=n_bins)
    centers = gmin + (np.arange(n_bins) + 0.5) * BIN_MS
    return d, G, centers, gmin, choice, t_start, t_exit


def extract(G, gmin, choice, rel_centers):
    """(units, trials, bins) window for each trial, from the global grid."""
    n_units = G.shape[0]
    n_trials = len(choice)
    n_bins = len(rel_centers)
    W = np.zeros((n_units, n_trials, n_bins))
    for t in range(n_trials):
        idx = ((choice[t] + rel_centers - gmin) / BIN_MS).astype(int)
        ok = (idx >= 0) & (idx < G.shape[1])
        W[:, t, ok] = G[:, idx[ok]]
    return W


def main():
    rel_centers = np.arange(LO + BIN_MS / 2, HI, BIN_MS)   # 160 centers
    rows = []
    for run in RUN_ORDER:
        print(f"[{run}] splice")
        d, G, centers, gmin, choice, t_start, t_exit = build_global(run)
        T = targets(d)
        g = d["tt"]["block_index"].to_numpy()
        W = extract(G, gmin, choice, rel_centers)
        # own-span mask per (trial, bin)
        own = np.zeros((len(choice), len(rel_centers)), bool)
        for t in range(len(choice)):
            own[t] = (choice[t] + rel_centers >= t_start[t]) & \
                     (choice[t] + rel_centers <= t_exit[t])
        # build variants
        variants = {"spliced": W}
        Wc = W.copy(); Wc[:, ~own] = np.nan; variants["ablation"] = Wc
        rng = np.random.default_rng(RNG_SEED)
        Wm = W.copy()
        perm = rng.permutation(len(choice))
        for t in range(len(choice)):
            Wm[:, t, ~own[t]] = W[:, perm[t], ~own[t]]
        variants["missplice"] = Wm
        Ws = W.copy()
        for t in range(len(choice)):
            Ws[:, t, ~own[t]] = W[:, (t + 2) % len(choice), ~own[t]]
        variants["shift"] = Ws
        # truncated reference: the existing npz (same 160x25ms grid, NaN outside span)
        if d["binned"].shape[2] == len(rel_centers):
            variants = {"truncated": d["binned"], **variants}

        for vname, V in variants.items():
            Xp = pca_features(V, rel_centers, -1000.0, 1000.0)
            for tname in TARGETS:
                y = T[tname].to_numpy(float)
                rows.append({"run_id": run, "variant": vname, "target": tname,
                             "corr": ridge_cv(Xp, y, g),
                             "cover": float(np.isfinite(V).all(axis=0).mean())})
    df = pd.DataFrame(rows)
    df.to_csv(AGG_DIR / "splice.csv", index=False)

    piv = df.groupby(["target", "variant"])["corr"].mean().unstack()
    print(piv.round(3).to_string())
    print("\nmean valid-bin fraction by variant:")
    print(df.groupby("variant")["cover"].mean().round(3).to_string())

    fig, axes = plt.subplots(1, len(RUN_ORDER), figsize=(5.0 * len(RUN_ORDER), 3.8), sharey=True)
    order = [v for v in ["truncated", "spliced", "ablation", "missplice", "shift"] if v in set(df.variant)]
    x = np.arange(len(TARGETS)); w = 0.16
    for ax, run in zip(axes, RUN_ORDER):
        s = df[df.run_id == run]
        for i, v in enumerate(order):
            vals = [s[(s.variant == v) & (s.target == t)]["corr"].mean() for t in TARGETS]
            ax.bar(x + (i - len(order) / 2) * w, vals, w, label=(v if run == RUN_ORDER[0] else None))
        ax.axhline(0, color="k", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(TARGETS, rotation=45, ha="right", fontsize=7)
        ax.set_title(run, fontsize=10)
    axes[0].set_ylabel("ridge CV corr (pca)"); axes[0].legend(fontsize=6)
    fig.suptitle("Part 10: spliced vs truncated (+ leakage controls)", fontsize=12)
    fig.tight_layout()
    fig.savefig(AGG_DIR / "figures" / "fig_splice.png",
                dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote splice.csv + fig_splice.png")


if __name__ == "__main__":
    main()
