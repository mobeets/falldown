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
# # Entry-anchored LDA decoding: does a pre-decision planning signal exist?
#
# The choice-anchored pipeline (`neural_lda_decoding.py`) decodes trial labels
# from windows anchored at the **choice pass** (t = 0 = passing the 2-hole
# level) and finds `planning_vs_greedy` / `planning_optimal` / `condition` are
# at/below chance in the `pre [-1000, 0]` window and only decodable in the late
# post-choice window — the basis for the "post-choice decoding is consequence,
# not computation" conclusion.
#
# That conclusion rests on how the pre-choice window is defined. Choice-anchored
# `pre [-1000, 0]` is a **heterogeneous** window: for slow-approach trials
# (planning: rt median ~950 ms) it covers the whole approach interval, but for
# fast trials (agree_optimal: rt median ~267 ms) it extends *before the entry
# pass* — into a period where the trial has not really begun. This script
# re-anchors everything to the **entry pass** (t = 0 = passing the 1-hole entry
# level of the 1-2-1 sequence) so each trial's decision/approach interval
# [entry, choice) is measured cleanly, and **windows never run past the choice
# pass** (that would re-import the post-choice kinematic confound).
#
# Data: `segmented_spikes_entrylocked_binned.npz` (built by
# `neural_entry_locked.py`, parity-checked against the choice-anchored npz).
#
# Windows (entry-anchored, all strictly before the choice pass):
#   approach    [entry, choice) per trial — variable length (rt_ms); the whole
#               pre-decision interval, the primary window of interest.
#   entry+250   [0, 250] ms after entry (trials with rt_ms >= 250)
#   entry+500   [0, 500] ms (trials with rt_ms >= 500)
#   entry+750   [0, 750] ms (trials with rt_ms >= 750)
#   pre-entry   [-500, 0] ms before entry (control: no task structure yet)
#
# Hypotheses: same trial labels as neural_lda_decoding.py (side, agree,
# planning_optimal, move_dir, move_dir_vx, condition, planning_vs_greedy,
# agree_optimal_vs_lapse).
#
# Controls (the interpretation guard): the participant has already begun
# steering during the approach, so a "planning" signal found here could be a
# direction/movement readout. We therefore also decode `move_dir`/`move_dir_vx`
# in the same windows (the strongest possible steering proxy) and run a
# matched re-decode equalizing rt x ball x side strata for the key binary
# labels.
#
# Statistics mirror neural_lda_decoding.py exactly: LDA (Ledoit-Wolf shrunk
# covariance, solver='eigen'), repeated stratified 10-fold CV (x5), balanced
# accuracy, chance = 1/n_classes, empirical two-sided permutation p (500 label
# shuffles through a 5-fold CV). `n` per (window, hypothesis) is reported
# because window eligibility is condition-dependent (fast trials drop out of
# the wide windows).
#
# Outputs (analysis/neural_outputs/<run>/):
#   entry_locked_lda_results.csv     fixed-window + approach decoding table
#   entry_locked_sliding.csv         sliding 150 ms windows rel entry
#   entry_locked_direction_control.csv  move_dir/vx decode per window (proxy
#                                       that bounds what the value labels can
#                                       be riding on)
#
# Run with:
#   C:\Users\manik\AppData\Local\Programs\Python\Python311\python.exe analysis\neural_entry_decoding.py

# %%
import numpy as np
import pandas as pd
from pathlib import Path

from neural_entry_locked import load_entry_binned, rate_features_entry, \
    approach_rate_features

from neural_lda_decoding import (
    RNG_SEED, PERM_FOLDS, build_label_vectors,
    decode_balanced_accuracy, permutation_p, make_transform,
)
from neural_decoding_confounds import stratify_match_indices, match_schemes
from neural_common import get_run, out_dir

# ---------------------------- Configuration ----------------------------
_RUN = get_run()
OUT_DIR = out_dir(_RUN.run_id)
TRIAL_TABLE = OUT_DIR / "trial_table.csv"
TRIAL_LABELS = OUT_DIR / "trial_labels.csv"

# fixed post-entry windows (ms after entry), plus the per-trial approach
FIXED_WINDOWS_MS = [250.0, 500.0, 750.0]
PRE_ENTRY_WINDOW = (-500.0, 0.0)     # control: before the trial's entry pass
N_FOLDS = 10
N_REPEATS = 5
N_PERM = 300
PERM_N_FOLDS = 5

# sliding trace uses lighter CV (mirrors neural_temporal_decoding.py)
SLIDE_FOLDS = 5
SLIDE_REPEATS = 2

DECODE_HYPOTHESES = [
    "planning_vs_greedy", "planning_optimal", "condition",
    "agree", "side", "agree_optimal_vs_lapse",
]
DIRECTION_CONTROLS = ["move_dir", "move_dir_vx"]

# matched re-decode strata (same as neural_decoding_confounds.py)
MATCH_N_FOLDS = 5
MATCH_N_REPEATS = 3
MATCH_N_PERM = 200
MIN_MATCH_N = 40

# sliding trace config
SLIDE_WIN_MS = 150.0
SLIDE_STEP_MS = 50.0
SLIDE_LO, SLIDE_HI = -500.0, 2000.0   # rel entry
# -----------------------------------------------------------------------


# %%
def load_trial_data():
    z = load_entry_binned()
    tt = pd.read_csv(TRIAL_TABLE)
    labels = pd.read_csv(TRIAL_LABELS)
    # attach entry-anchored timing to each label row. NOTE: do NOT add a
    # choice_time_ms column here — build_label_vectors/add_move_dir_vx merge
    # trial_table's choice_time_ms themselves and a duplicate would collide.
    timing = tt[["trial_id", "trial_start_ms", "entry_time_ms"]].copy()
    timing["rt_ms"] = tt["choice_time_ms"] - tt["entry_time_ms"]
    timing["ball_time_ms"] = tt["exit_time_ms"] - tt["choice_time_ms"]
    lab = labels.merge(timing, on="trial_id", how="left")
    return z, lab


# %%
def label_pack(lv, hname):
    """(full-length y, index array/bool mask or None)."""
    y, idx = lv[hname]
    return y, idx


def idx_mask(idx, n_trials):
    """Convert an index array / bool mask / None into a bool mask over all
    trials."""
    if idx is None:
        return np.ones(n_trials, dtype=bool)
    idx = np.asarray(idx)
    if idx.dtype == bool:
        return idx
    m = np.zeros(n_trials, dtype=bool)
    m[idx] = True
    return m


def approach_rate_matrix(binned, bin_centers, rt_ms):
    """(n_trials, n_units) sqrt-Hz over each trial's full [entry, choice)
    interval. Every trial is eligible (rt_ms > 0 always)."""
    return approach_rate_features(binned, bin_centers, np.asarray(rt_ms))


def fixed_rate_matrix(binned, bin_centers, lo, hi):
    return rate_features_entry(binned, bin_centers, lo, hi)


# %%
def decode_hypothesis(y, X, transform):
    """Balanced accuracy + permutation p for one label set."""
    acc_m, acc_s = decode_balanced_accuracy(y, X, transform, N_FOLDS,
                                            N_REPEATS, RNG_SEED)
    chance = 1.0 / len(np.unique(y))
    p = permutation_p(y, X, transform, acc_m, n_perm=N_PERM,
                      n_folds=PERM_N_FOLDS, seed=RNG_SEED)
    return {"acc_mean": acc_m, "acc_std": acc_s, "chance": chance, "perm_p": p}


# %%
def main():
    z, lab = load_trial_data()
    binned = z["binned"]
    bin_centers = z["bin_centers"]
    rt = lab["rt_ms"].to_numpy(float)
    n_trials = len(lab)

    lv = build_label_vectors(lab)
    transform = make_transform("rate")
    rows = []

    # ---------------- fixed post-entry windows ----------------
    for W in FIXED_WINDOWS_MS:
        elig = rt >= W
        X = fixed_rate_matrix(binned, bin_centers, 0.0, W)
        for hname in DECODE_HYPOTHESES + DIRECTION_CONTROLS:
            y, idx = label_pack(lv, hname)
            keep = elig & idx_mask(idx, n_trials)
            if keep.sum() < 20:
                rows.append({"window": f"entry+{W:.0f}",
                             "hypothesis": hname, "n_trials": int(keep.sum()),
                             "acc_mean": np.nan, "acc_std": np.nan,
                             "chance": np.nan, "perm_p": np.nan,
                             "note": "too few eligible trials"})
                continue
            r = decode_hypothesis(y[keep], X[keep], transform)
            rows.append({"window": f"entry+{W:.0f}", "hypothesis": hname,
                         "n_trials": int(keep.sum()), "note": None, **r})
            print(f"  entry+{W:.0f}/{hname}: acc {r['acc_mean']:.3f} "
                  f"(chance {r['chance']:.2f}, p {r['perm_p']:.3f}, "
                  f"n={keep.sum()})")

    # ---------------- approach window [entry, choice) ----------------
    X_ap = approach_rate_matrix(binned, bin_centers, rt)
    for hname in DECODE_HYPOTHESES + DIRECTION_CONTROLS:
        y, idx = label_pack(lv, hname)
        keep = idx_mask(idx, n_trials)
        r = decode_hypothesis(y[keep], X_ap[keep], transform)
        rows.append({"window": "approach", "hypothesis": hname,
                     "n_trials": int(keep.sum()), "note": None, **r})
        print(f"  approach/{hname}: acc {r['acc_mean']:.3f} "
              f"(chance {r['chance']:.2f}, p {r['perm_p']:.3f}, "
              f"n={keep.sum()})")

    # ---------------- pre-entry control window ----------------
    X_pre = fixed_rate_matrix(binned, bin_centers, *PRE_ENTRY_WINDOW)
    fullcov = (lab["entry_time_ms"].to_numpy(float)
               - lab["trial_start_ms"].to_numpy(float)) >= 500.0
    for hname in DECODE_HYPOTHESES:
        y, idx = label_pack(lv, hname)
        keep = idx_mask(idx, n_trials) & fullcov
        if keep.sum() < 20:
            continue
        r = decode_hypothesis(y[keep], X_pre[keep], transform)
        rows.append({"window": "pre-entry[-500,0]", "hypothesis": hname,
                     "n_trials": int(keep.sum()), "note": None, **r})
        print(f"  pre-entry/{hname}: acc {r['acc_mean']:.3f} "
              f"(chance {r['chance']:.2f}, p {r['perm_p']:.3f}, "
              f"n={keep.sum()})")

    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / "entry_locked_lda_results.csv", index=False)
    print(f"\nSaved entry_locked_lda_results.csv ({len(res)} rows)")
    print(res[["window", "hypothesis", "n_trials", "acc_mean", "chance",
               "perm_p"]].to_string(index=False))

    # ---------------- sliding trace (rate rep, rel entry) ----------------
    # Each sliding window is decoded only on trials whose rt_ms >= window end
    # (window stays entirely before the choice pass — the decision interval),
    # so windows never mix in post-choice kinematics. n shrinks as the window
    # moves later; past ~+800 ms only slow-approach (planning/lapse) trials
    # remain.
    sliding = []
    centers = np.arange(SLIDE_LO, SLIDE_HI + SLIDE_STEP_MS, SLIDE_STEP_MS)
    half = SLIDE_WIN_MS / 2.0
    for c in centers:
        win_hi = c + half
        if win_hi <= 0:
            # fully pre-entry windows use all trials (coverage-filtered below)
            elig = np.ones(n_trials, dtype=bool)
        else:
            elig = rt >= win_hi
        X = fixed_rate_matrix(binned, bin_centers, c - half, c + half)
        for hname in DECODE_HYPOTHESES:
            y, idx = label_pack(lv, hname)
            keep = elig & idx_mask(idx, n_trials)
            if keep.sum() < 20:
                continue
            acc_m, acc_s = decode_balanced_accuracy(
                y[keep], X[keep], transform, SLIDE_FOLDS, SLIDE_REPEATS,
                RNG_SEED)
            sliding.append({"window_center_ms": float(c),
                            "hypothesis": hname, "n_trials": int(keep.sum()),
                            "acc_mean": acc_m, "acc_std": acc_s,
                            "chance": 1.0 / len(np.unique(y[keep])),
                            "perm_p": np.nan})
    slid = pd.DataFrame(sliding)
    slid.to_csv(OUT_DIR / "entry_locked_sliding.csv", index=False)
    print(f"Saved entry_locked_sliding.csv ({len(slid)} rows)")

    # ---------------- matched re-decode (key binary labels) -------------
    nuis = pd.read_csv(OUT_DIR / "decoding_nuisance_table.csv")
    matched_rows = []
    rng = np.random.default_rng(RNG_SEED)
    schemes = match_schemes(nuis)
    for hname in ["planning_vs_greedy", "planning_optimal", "agree",
                  "agree_optimal_vs_lapse"]:
        y, idx = label_pack(lv, hname)
        base = np.flatnonzero(idx_mask(idx, n_trials))
        for scheme_name, strata in [("unmatched", None)] + list(schemes.items()):
            if scheme_name == "unmatched":
                sel = base
            else:
                matched_pos = stratify_match_indices(
                    y[base], strata.iloc[base].reset_index(drop=True), rng)
                sel = base[matched_pos]
            if len(sel) < MIN_MATCH_N:
                matched_rows.append({"hypothesis": hname, "scheme": scheme_name,
                                     "n_trials": int(len(sel)),
                                     "acc_mean": np.nan, "chance": np.nan,
                                     "perm_p": np.nan,
                                     "note": "insufficient overlap"})
                continue
            r = decode_hypothesis(y[sel], X_ap[sel], transform)
            matched_rows.append({"hypothesis": hname, "scheme": scheme_name,
                                 "n_trials": int(len(sel)), "note": None, **r})
            print(f"  matched/{hname}/{scheme_name}: acc {r['acc_mean']:.3f} "
                  f"(p {r['perm_p']:.3f}, n={len(sel)})")
    mt = pd.DataFrame(matched_rows)
    mt.to_csv(OUT_DIR / "entry_locked_matched_redecode.csv", index=False)
    print(f"Saved entry_locked_matched_redecode.csv ({len(mt)} rows)")


# %%
if __name__ == "__main__":
    main()

# %%
