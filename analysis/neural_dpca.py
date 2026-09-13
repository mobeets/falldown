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
# # Demixed PCA (dPCA) of the mesial-temporal population
#
# neural_lda_decoding.py shows the population can decode condition / planning
# labels in the post-choice window, but an LDA decoder is agnostic to *which
# population axes* carry the signal. If the planning-vs-greedy decoding rides
# on the condition-independent temporal axis (trial-locked kinematics), the
# "planning" interpretation collapses into a motor/timing confound (Kobak et
# al. 2016: 65-90% of signal variance in higher-order areas is condition-
# independent time). dPCA demixes the population activity into terms
#   't'  time (condition-independent),  'c'  condition,  's'  side,
#   and interactions ('cs', 'ct', 'st', 'cst'),
# and we then ask which demixed subspace supports decoding of each hypothesis.
#
# Part 1 — variance decomposition.
#   Explained variance of each demixed component, per term. The 't' term's
#   dominance quantifies how much of the population variance is condition-
#   independent temporal structure.
#
# Part 2 — demixed-axis decoding (the confound test).
#   Single trials are projected onto each term's decoder axes and the usual
#   LDA (balanced accuracy, permutation p) is run. Decoding from the 't'
#   (condition-independent) axes is the key control: if planning_vs_greedy
#   decodes there, the signal is carried by generic trial-locked dynamics.
#   Rows whose term contains the hypothesis' own defining parameter are
#   flagged `circular=True` (dPCA was given those labels).
#
# Part 3 — axis geometry.
#   Cosine similarity between each term's leading decoder axis (Ruff et al.
#   2025-style formatting metric: are condition and time axes separable?).
#
# Uses the machenslab/dPCA package (`pip install dPCA`).
#
# Outputs:
#   dpca_variance_decomposition.csv
#   dpca_decoding.csv
#   dpca_axis_similarity.csv
#   dpca_components.csv
#
# Run with:
#   C:\Users\manik\AppData\Local\Programs\Python\Python311\python.exe analysis\neural_dpca.py

# %%
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd
from pathlib import Path

from dPCA.dPCA import dPCA

from neural_lda_decoding import (
    RNG_SEED, PERM_FOLDS, load_binned, build_label_vectors,
    decode_balanced_accuracy, permutation_p, make_transform,
)

# ---------------------------- Configuration ----------------------------
from neural_common import get_run, out_dir
_RUN = get_run()
OUT_DIR = out_dir(_RUN.run_id)
BINNED = OUT_DIR / "segmented_spikes_binned.npz"
TRIAL_LABELS = OUT_DIR / "trial_labels.csv"

DPCA_WINDOWS = {"post": (0.0, 1000.0), "whole": (-2000.0, 2000.0)}
MIN_CELL_TRIALS = 3        # bins where a cell covers fewer trials are dropped
N_COMPONENTS = {"t": 20, "c": 6, "s": 4, "cs": 3, "ct": 6, "st": 4, "cst": 4}
REGULARIZER = 0            # unregularized: samples (>8x time bins) > neurons
DECODE_TERMS = ["t", "c", "s", "cs", "ct", "st", "cst"]
DECODE_HYPOTHESES = ["planning_vs_greedy", "planning_optimal", "condition",
                     "side", "agree", "agree_optimal_vs_lapse"]
MAX_COMP_DECODE = 3        # top dims per term used for decoding
N_PERM = 150               # label shuffles per (hypothesis, term) decode
DECODE_FOLDS = 5
DECODE_REPEATS = 3

CONDITION_ORDER = {"planning": 0, "greedy": 1, "agree_optimal": 2, "lapse": 3}
# -----------------------------------------------------------------------


# %%
def build_psths(binned, bin_centers, lo, hi, cell_trial_ids, n_units, n_bins):
    """(n_units, n_cond, n_side, n_time) NaN-aware PSTH of per-bin firing rates.

    rate per bin = count / bin_width (Hz). Cells are indexed by the returned
    (cond_idx, side_idx) per trial; cell_trial_ids is a list of (trial_id,
    cond_idx, side_idx).
    """
    mask = np.ones(n_bins, dtype=bool) if lo is None else (
        (bin_centers >= lo) & (bin_centers <= hi))
    bin_idx = np.flatnonzero(mask)
    # include side even if it ends up empty
    psth = np.full((n_units, 4, 2, len(bin_idx)), np.nan)
    trials_by_cell = {}
    for tid, c, s in cell_trial_ids:
        trials_by_cell.setdefault((c, s), []).append(tid)
    for (c, s), tids in trials_by_cell.items():
        sel = binned[:, tids, :][:, :, bin_idx]      # (n_units, n_trials, n_t)
        valid = ~np.isnan(sel)
        cnt = np.where(valid, sel, 0.0)
        nv = valid.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = cnt.sum(axis=1) / np.maximum(nv, 1) / 0.025
        rate = np.where(nv > 0, rate, np.nan)
        psth[:, c, s, :] = rate
    return psth, bin_idx


def usable_bin_mask(psth, min_trials=MIN_CELL_TRIALS):
    """Bins kept where every (unit x cell) has enough covered trials."""
    covered = ~np.isnan(psth)                        # (units, 4, 2, n_time)
    frac = covered.mean(axis=(0, 1, 2))              # per-bin coverage
    return frac > 0.5


# %%
def run_dpca_window(binned, bin_centers, lo, hi, labels, cell_trial_ids):
    """Fit dPCA on one window, return (dpca, psth, bin_idx, mu)."""
    n_units, n_trials, n_bins = binned.shape
    psth, bin_idx = build_psths(binned, bin_centers, lo, hi, cell_trial_ids,
                                n_units, n_bins)
    keep = usable_bin_mask(psth)
    psth = psth[:, :, :, keep]
    bin_idx = bin_idx[keep]
    # residual NaNs come from cells whose trials did not extend to a bin
    # (window truncation). Impute with the cell's temporal mean rate.
    for u in range(n_units):
        for c in range(psth.shape[1]):
            for s in range(psth.shape[2]):
                row = psth[u, c, s]
                mrow = np.nanmean(row)
                if not np.isnan(mrow):
                    row[np.isnan(row)] = mrow
    assert not np.isnan(psth).any(), "PSTH still contains NaN after imputation"
    X = psth
    dpca = dPCA(labels=labels, n_components=N_COMPONENTS,
                regularizer=REGULARIZER)
    dpca.fit(X)
    mu = np.nanmean(X.reshape((n_units, -1)), axis=1)
    # explained variance via transform on the fit data
    dpca.transform(X)
    return dpca, X, bin_idx, mu


# %%
def raw_window_rate(binned, bin_centers, bin_idx, lo, hi):
    """(n_trials, n_units) mean per-bin rate over the coverage-trimmed bins."""
    mask = np.ones(len(bin_centers), dtype=bool) if lo is None else (
        (bin_centers >= lo) & (bin_centers <= hi))
    sel = binned[:, :, bin_idx]                       # (units, trials, n_bins)
    valid = ~np.isnan(sel)
    cnt = np.where(valid, sel, 0.0)
    nv = valid.sum(axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = cnt.sum(axis=2) / np.maximum(nv, 1) / 0.025
    rate = np.where(nv > 0, rate, 0.0)
    return rate.T                                      # (n_trials, n_units)


def decode_from_term(dpca, term, X_trial, y, idx, mu, transform):
    """Balanced accuracy decoding a hypothesis from a demixed term's axes."""
    D = dpca.D[term]
    k = min(D.shape[1], MAX_COMP_DECODE)
    Z = (X_trial - mu) @ D[:, :k]
    y_sub = y if idx is None else y[idx]
    Z_sub = Z if idx is None else Z[idx]
    acc_m, acc_s = decode_balanced_accuracy(
        y_sub, Z_sub, transform, DECODE_FOLDS, DECODE_REPEATS, RNG_SEED)
    chance = 1.0 / len(np.unique(y_sub))
    p = permutation_p(y_sub, Z_sub, transform, acc_m,
                      n_perm=N_PERM, n_folds=PERM_FOLDS, seed=RNG_SEED)
    return acc_m, acc_s, chance, p


# %%
def main():
    print("Loading binned spikes ...")
    binned, unit_ids, bin_centers = load_binned()
    labels = pd.read_csv(TRIAL_LABELS)
    label_dict = build_label_vectors(labels)
    transform = make_transform("rate")

    cond_idx = labels["condition"].map(CONDITION_ORDER).to_numpy()
    side_idx = (labels["choice_hole"] >= 6).astype(int).to_numpy()
    cell_trial_ids = [(t, cond_idx[t], side_idx[t])
                      for t in range(len(labels))]

    var_rows, dec_rows, sim_rows, comp_rows = [], [], [], []
    for wname, (lo, hi) in DPCA_WINDOWS.items():
        print(f"Fitting dPCA on {wname} window ...")
        dpca, X, bin_idx, mu = run_dpca_window(
            binned, bin_centers, lo, hi, labels="cst", cell_trial_ids=cell_trial_ids)
        t_centers = bin_centers[bin_idx]

        for term, ev in dpca.explained_variance_ratio_.items():
            for k, v in enumerate(ev):
                var_rows.append({"window": wname, "term": term,
                                 "component": k, "explained_variance": float(v)})

        # demixed component time series for plotting
        V = dpca.transform(X)
        for term, Vt in V.items():
            for k in range(Vt.shape[0]):
                vals = Vt[k]                     # (n_cond, n_side, n_time)
                flat = vals.mean(axis=(0, 1))
                for b, t in enumerate(t_centers):
                    comp_rows.append({"window": wname, "term": term,
                                      "component": k,
                                      "bin_center_ms": float(t),
                                      "value": float(flat[b])})

        # axis similarity between leading axes of each term pair
        terms = list(dpca.D)
        for i, a in enumerate(terms):
            for b in terms[i + 1:]:
                u = dpca.D[a][:, 0]
                v = dpca.D[b][:, 0]
                cos = float(np.dot(u, v) / (np.linalg.norm(u) *
                                            np.linalg.norm(v) + 1e-12))
                sim_rows.append({"window": wname, "term_A": a, "term_B": b,
                                 "cosine_sim": cos})

        # demixed-axis decoding
        X_trial = raw_window_rate(binned, bin_centers, bin_idx, lo, hi)
        for hname, (y, idx) in label_dict.items():
            if hname not in DECODE_HYPOTHESES:
                continue
            # dPCA parameter char that defines the hypothesis: condition-based
            # labels (condition, planning_*, agree, lapse) -> 'c'; side -> 's'.
            defining = "s" if hname == "side" else "c"
            for term in DECODE_TERMS:
                acc_m, acc_s, chance, p = decode_from_term(
                    dpca, term, X_trial, y, idx, mu, transform)
                circ = defining in term
                dec_rows.append({
                    "window": wname, "hypothesis": hname, "term": term,
                    "n_components": min(dpca.D[term].shape[1], MAX_COMP_DECODE),
                    "acc_mean": acc_m, "acc_std": acc_s, "chance": chance,
                    "perm_p": p, "circular": circ,
                })
                print(f"  {hname}/{term}: acc {acc_m:.3f} "
                      f"(chance {chance:.2f}, p {p:.3f}, circ={circ})")

    pd.DataFrame(var_rows).to_csv(
        OUT_DIR / "dpca_variance_decomposition.csv", index=False)
    pd.DataFrame(dec_rows).to_csv(OUT_DIR / "dpca_decoding.csv", index=False)
    pd.DataFrame(sim_rows).to_csv(OUT_DIR / "dpca_axis_similarity.csv",
                                  index=False)
    pd.DataFrame(comp_rows).to_csv(OUT_DIR / "dpca_components.csv",
                                   index=False)
    print("\nSaved dpca_variance_decomposition.csv, dpca_decoding.csv, "
          "dpca_axis_similarity.csv, dpca_components.csv")

    print("\nSummary (variance decomposition, post window):")
    var = pd.read_csv(OUT_DIR / "dpca_variance_decomposition.csv")
    sub = var[var["window"] == "post"].groupby("term")[
        "explained_variance"].sum().sort_values(ascending=False)
    print(sub.to_string())
    print("\nSummary (non-circular demixed decoding, post window):")
    dec = pd.read_csv(OUT_DIR / "dpca_decoding.csv")
    sub = dec[(dec["window"] == "post") & (~dec["circular"])]
    print(sub[["hypothesis", "term", "acc_mean", "perm_p"]].to_string(
        index=False))


# %%
# ----------------------- Plot functions (not saved) ---------------------

def load_dpca_variance():
    return pd.read_csv(OUT_DIR / "dpca_variance_decomposition.csv")


def load_dpca_decoding():
    return pd.read_csv(OUT_DIR / "dpca_decoding.csv")


def load_dpca_components():
    return pd.read_csv(OUT_DIR / "dpca_components.csv")


def plot_variance_bars(window="post", ax=None):
    """Explained variance by term (stacked by component)."""
    import matplotlib.pyplot as plt
    var = load_dpca_variance()
    sub = var[var["window"] == window]
    terms = sub["term"].unique()
    if ax is None:
        ax = plt.gca()
    for term in terms:
        s = sub[sub["term"] == term]
        ax.bar(term, s["explained_variance"].sum())
    ax.set_ylabel("explained variance (fraction)")
    ax.set_title(f"dPCA variance decomposition ({window})")
    return ax


def plot_demixed_axis_components(term="t", window="post", ax=None):
    """Component time series for one term."""
    import matplotlib.pyplot as plt
    comp = load_dpca_components()
    sub = comp[(comp["term"] == term) & (comp["window"] == window)]
    if ax is None:
        ax = plt.gca()
    for k in sub["component"].unique():
        s = sub[sub["component"] == k]
        ax.plot(s["bin_center_ms"], s["value"], label=f"comp {k}")
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("ms relative to choice")
    ax.set_title(f"{term} demixed components ({window})")
    ax.legend(fontsize=8)
    return ax


# %%
if __name__ == "__main__":
    main()
