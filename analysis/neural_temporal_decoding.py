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
# # Temporal generalization & sliding-window decoding
#
# neural_lda_decoding.py collapses each trial into three fixed windows
# (pre/post/whole). This script asks WHEN in the trial the population can
# linearly decode each hypothesis, which is the direct test of the central
# confound concern: a label like planning_vs_greedy decodes strongly in the
# POST-choice window but at/below chance PRE-choice. If the population were
# computing "planning", the signal should exist before/during the choice, not
# only in its motor/outcome aftermath.
#
# Part 1 — sliding-window accuracy trace.
#   Rate-rep LDA (repeated stratified 5-fold CV) on a 150 ms window stepping
#   50 ms across [-2000, +2000] ms around choice (t = 0). One trace per
#   hypothesis (planning_vs_greedy, planning_optimal, condition, agree, side,
#   move_dir_vx). Reports balanced accuracy + chance.
#
# Part 2 — a-priori window permutation tests.
#   For four windows chosen before seeing the data (pre [-1000,0], pre
#   [-500,0], post [0,+500], post [0,+1000]) the full permutation null (label
#   shuffle through CV). This is a small, pre-registered set, so per-window p
#   is interpretable without a sliding-window multiple-comparison penalty.
#
# Part 3 — temporal generalization matrix (King & Dehaene 2014).
#   Train in window A, test in window B, for a coarse 6-window grid on the key
#   hypotheses. A diagonal band means the representation is stable across time;
#   off-diagonal mass confined to the post-choice region means the signal is a
#   post-decision transient (confound signature).
#
# Statistics mirror neural_lda_decoding.py (balanced accuracy, empirical
# permutation p). All decoding is on the 'rate' representation.
#
# Outputs:
#   temporal_decoding_sliding.csv         per-window accuracy trace
#   temporal_decoding_summary.csv         a-priori window permutation tests
#   temporal_decoding_generalization.csv  train-window x test-window matrix
#
# Run with:
#   C:\Users\manik\AppData\Local\Programs\Python\Python311\python.exe analysis\neural_temporal_decoding.py

# %%
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from neural_lda_decoding import (
    RNG_SEED, PERM_FOLDS, load_binned, rate_features,
    make_transform, build_label_vectors, permutation_p,
)

# ---------------------------- Configuration ----------------------------
from neural_common import get_run, out_dir
_RUN = get_run()
OUT_DIR = out_dir(_RUN.run_id)
BINNED = OUT_DIR / "segmented_spikes_binned.npz"
TRIAL_LABELS = OUT_DIR / "trial_labels.csv"

SLIDE_WIN_MS = 150.0
SLIDE_STEP_MS = 50.0
SLIDE_LO, SLIDE_HI = -1900.0, 1900.0

TRACE_HYPOTHESES = [
    "planning_vs_greedy", "planning_optimal", "condition", "agree",
    "side", "move_dir_vx",
]
A_PRIORI_WINDOWS = {
    "pre1k": (-1000.0, 0.0),
    "pre0k5": (-500.0, 0.0),
    "post0k5": (0.0, 500.0),
    "post1k": (0.0, 1000.0),
}
TG_WINDOWS = {
    "pre_1500_-500": (-1500.0, -500.0),
    "pre_1000_0": (-1000.0, 0.0),
    "pre_500_500": (-500.0, 500.0),
    "post_0_1000": (0.0, 1000.0),
    "post_500_1500": (500.0, 1500.0),
    "post_1000_2000": (1000.0, 2000.0),
}
TG_HYPOTHESES = ["planning_vs_greedy", "planning_optimal", "condition"]

TRACE_FOLDS = 5
TRACE_REPEATS = 2
N_PERM = 300            # a-priori window permutation null
TG_FOLDS = 5
TG_REPEATS = 3
# -----------------------------------------------------------------------


# %%
def decode_balanced_accuracy(y, X, transform, n_folds, n_repeats, seed):
    """Repeated stratified k-fold CV balanced accuracy (per-fold values)."""
    accs = []
    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                              random_state=seed + rep)
        for tr, te in skf.split(X, y):
            Xtr, Xte = transform(X[tr], y[tr], X[te])
            lda = LinearDiscriminantAnalysis(
                solver="eigen", shrinkage="auto").fit(Xtr, y[tr])
            accs.append(balanced_accuracy_score(y[te], lda.predict(Xte)))
    accs = np.asarray(accs)
    return float(accs.mean()), float(accs.std())


# %%
def sliding_centers():
    centers = np.arange(SLIDE_LO, SLIDE_HI + SLIDE_STEP_MS / 2, SLIDE_STEP_MS)
    return np.clip(centers, SLIDE_LO, SLIDE_HI)


def sliding_trace(binned, bin_centers, label_dict, transforms):
    """Per-window balanced accuracy for every hypothesis (rate rep)."""
    half = SLIDE_WIN_MS / 2.0
    centers = sliding_centers()
    rows = []
    for hname, (y, idx) in label_dict.items():
        if hname not in TRACE_HYPOTHESES:
            continue
        y_sub = y if idx is None else y[idx]
        n = len(y_sub)
        chance = 1.0 / len(np.unique(y_sub))
        for c in centers:
            X = rate_features(binned, bin_centers, c - half, c + half)
            X_sub = X if idx is None else X[idx]
            acc_m, acc_s = decode_balanced_accuracy(
                y_sub, X_sub, transforms["rate"], TRACE_FOLDS, TRACE_REPEATS,
                RNG_SEED)
            rows.append({"hypothesis": hname, "window_center_ms": float(c),
                         "acc_mean": acc_m, "acc_std": acc_s,
                         "chance": chance, "n_trials": n})
            if c == centers[0] or c == centers[-1]:
                print(f"  {hname} @ {c:+.0f} ms: acc {acc_m:.3f}")
    return pd.DataFrame(rows)


# %%
def a_priori_permutation(binned, bin_centers, label_dict, transforms):
    """Full permutation p at the four a-priori windows."""
    rows = []
    for wname, (lo, hi) in A_PRIORI_WINDOWS.items():
        X = rate_features(binned, bin_centers, lo, hi)
        for hname, (y, idx) in label_dict.items():
            if hname not in TRACE_HYPOTHESES:
                continue
            y_sub = y if idx is None else y[idx]
            X_sub = X if idx is None else X[idx]
            acc_m, acc_s = decode_balanced_accuracy(
                y_sub, X_sub, transforms["rate"], TRACE_FOLDS, TRACE_REPEATS,
                RNG_SEED)
            chance = 1.0 / len(np.unique(y_sub))
            p = permutation_p(y_sub, X_sub, transforms["rate"], acc_m,
                              n_perm=N_PERM, n_folds=PERM_FOLDS, seed=RNG_SEED)
            rows.append({"hypothesis": hname, "window": wname,
                         "lo_ms": lo, "hi_ms": hi,
                         "acc_mean": acc_m, "acc_std": acc_s,
                         "chance": chance, "perm_p": p,
                         "n_trials": len(y_sub)})
            print(f"  {hname}/{wname}: acc {acc_m:.3f} "
                  f"(chance {chance:.2f}, p {p:.3f})")
    return pd.DataFrame(rows)


# %%
def temporal_generalization(binned, bin_centers, label_dict, transforms):
    """Train-window x test-window balanced accuracy (shared CV splits)."""
    Xw = {wname: rate_features(binned, bin_centers, lo, hi)
          for wname, (lo, hi) in TG_WINDOWS.items()}
    names = list(TG_WINDOWS)
    rows = []
    for hname, (y, idx) in label_dict.items():
        if hname not in TG_HYPOTHESES:
            continue
        y_sub = y if idx is None else y[idx]
        accs = np.zeros((len(names), len(names), TG_FOLDS * TG_REPEATS))
        k = 0
        for rep in range(TG_REPEATS):
            skf = StratifiedKFold(n_splits=TG_FOLDS, shuffle=True,
                                  random_state=RNG_SEED + rep)
            splits = list(skf.split(np.zeros((len(y_sub), 1)), y_sub))
            for tr, te in splits:
                for i, wa in enumerate(names):
                    Xa = Xw[wa]
                    Xa_sub = Xa if idx is None else Xa[idx]
                    for j, wb in enumerate(names):
                        Xb = Xw[wb]
                        Xb_sub = Xb if idx is None else Xb[idx]
                        sc = StandardScaler().fit(Xa_sub[tr])
                        lda = LinearDiscriminantAnalysis(
                            solver="eigen", shrinkage="auto").fit(
                                sc.transform(Xa_sub[tr]), y_sub[tr])
                        accs[i, j, k] = balanced_accuracy_score(
                            y_sub[te], lda.predict(sc.transform(Xb_sub[te])))
                k += 1
        for i, wa in enumerate(names):
            for j, wb in enumerate(names):
                rows.append({"hypothesis": hname,
                             "train_window": wa, "test_window": wb,
                             "acc_mean": float(accs[i, j].mean()),
                             "acc_std": float(accs[i, j].std())})
        print(f"  {hname} TG done")
    return pd.DataFrame(rows)


# %%
def main():
    print("Loading binned spikes ...")
    binned, unit_ids, bin_centers = load_binned()
    labels = pd.read_csv(TRIAL_LABELS)
    label_dict = build_label_vectors(labels)
    transforms = {"rate": make_transform("rate")}

    print("Sliding-window accuracy trace ...")
    trace = sliding_trace(binned, bin_centers, label_dict, transforms)
    trace.to_csv(OUT_DIR / "temporal_decoding_sliding.csv", index=False)
    print(f"  saved temporal_decoding_sliding.csv ({len(trace)} rows)")

    print("A-priori window permutation tests ...")
    summ = a_priori_permutation(binned, bin_centers, label_dict, transforms)
    summ.to_csv(OUT_DIR / "temporal_decoding_summary.csv", index=False)
    print(f"  saved temporal_decoding_summary.csv ({len(summ)} rows)")

    print("Temporal generalization matrix ...")
    tg = temporal_generalization(binned, bin_centers, label_dict, transforms)
    tg.to_csv(OUT_DIR / "temporal_decoding_generalization.csv", index=False)
    print(f"  saved temporal_decoding_generalization.csv ({len(tg)} rows)")

    print("\nSummary (a-priori windows):")
    print(summ[["hypothesis", "window", "acc_mean", "acc_std", "chance",
                "perm_p"]].to_string(index=False))


# %%
# ----------------------- Plot functions (not saved) ---------------------

def load_trace():
    return pd.read_csv(OUT_DIR / "temporal_decoding_sliding.csv")


def plot_sliding_trace(hypothesis, ax=None):
    """Accuracy trace over time with chance line."""
    import matplotlib.pyplot as plt
    tr = load_trace()
    sub = tr[tr["hypothesis"] == hypothesis]
    if ax is None:
        ax = plt.gca()
    ax.plot(sub["window_center_ms"], sub["acc_mean"], "o-", ms=3)
    ax.axhline(sub["chance"].iloc[0], color="k", ls="--",
               label=f"chance = {sub['chance'].iloc[0]:.2f}")
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("window center (ms, relative to choice)")
    ax.set_ylabel("balanced accuracy")
    ax.set_title(hypothesis)
    ax.legend()
    return ax


def plot_tg_matrix(hypothesis, ax=None):
    """Train x test generalization matrix."""
    import matplotlib.pyplot as plt
    tg = pd.read_csv(OUT_DIR / "temporal_decoding_generalization.csv")
    sub = tg[tg["hypothesis"] == hypothesis]
    names = list(TG_WINDOWS)
    piv = sub.pivot_table(index="train_window", columns="test_window",
                          values="acc_mean")
    piv = piv.reindex(index=names, columns=names)
    if ax is None:
        ax = plt.gca()
    im = ax.imshow(piv.values, cmap="RdBu_r", vmin=0.4, vmax=0.7,
                   aspect="auto")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("test window")
    ax.set_ylabel("train window")
    ax.set_title(f"{hypothesis} — temporal generalization")
    ax.figure.colorbar(im, ax=ax)
    return ax


# %%
if __name__ == "__main__":
    main()
