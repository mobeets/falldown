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
# # Entry-anchored (t=0 = entry-hole pass) segmentation and loader
#
# The standard pipeline (`segment_trials.py`) anchors every trial's spike data
# to the **choice pass** (t = 0 = passing the 2-hole level of the 1-2-1
# sequence). Choice-anchored windows split each trial into pre/post but the
# "pre" period mixes two behaviorally different things: the decision/approach
# interval (entry pass -> choice pass, when the participant picks a direction)
# and, for fast trials, time before the entry pass. This module re-bins the
# same spikes anchored to the **entry pass** (t = 0 = passing the 1-hole entry
# level) so the entry->choice decision interval can be measured cleanly
# without running past the choice pass (which would re-import the post-choice
# kinematic confound the rest of the pipeline controls).
#
# It deliberately does NOT modify the existing choice-anchored artifacts: it
# writes a new per-run npz (`segmented_spikes_entrylocked_binned.npz`) with
# identical binning conventions, coverage (NaN) semantics, and unit QC, so
# results from the two anchors are directly comparable side by side.
#
# Builder source of truth: per-unit spike times are re-derived from the same
# `spikes.mat` (`segment_trials.load_units`) and segmented per trial exactly as
# `segment_trials.py` does, but with `anchor = entry_time_ms`.
#
# Parity check: `--check-parity` rebuilds the *choice*-anchored binned array
# with the same code and compares it to the stored `segmented_spikes_binned.npz`
# (bit-for-bit on covered-bin counts). If it does not reproduce, the entry
# dataset is not trustworthy and the script fails loudly.
#
# Output: analysis/neural_outputs/<run>/segmented_spikes_entrylocked_binned.npz
#
# Run with:
#   C:\Users\manik\AppData\Local\Programs\Python\Python311\python.exe analysis\neural_entry_locked.py
#   ... analysis\neural_entry_locked.py --check-parity

# %%
import numpy as np
import pandas as pd
import scipy.io as sio
from pathlib import Path

from neural_common import get_run, out_dir
from segment_trials import load_units

# ---------------------------- Configuration ----------------------------
_RUN = get_run()
OUT_DIR = out_dir(_RUN.run_id)
SPIKES_PATH = _RUN.spikes_mat
NEURON_DATA_PATH = _RUN.neuron_data_json
TRIAL_TABLE = OUT_DIR / "trial_table.csv"
REF_BINNED = OUT_DIR / "segmented_spikes_binned.npz"
ENTRY_BINNED = OUT_DIR / "segmented_spikes_entrylocked_binned.npz"

BIN_WIDTH_MS = 25.0
WINDOW_MS = (-2000.0, 2000.0)
MODE = "truncated"
# -----------------------------------------------------------------------


# %%
def _window_bin_mask(centers, lo, hi):
    """Boolean mask over bins whose center lies in [lo, hi]."""
    if lo is None:
        return np.ones(len(centers), dtype=bool)
    return (centers >= lo) & (centers <= hi)


# %%
def build_binned(spike_times_by_unit, unit_ids, trial_table, anchor_times,
                 window_ms=WINDOW_MS, bin_width_ms=BIN_WIDTH_MS):
    """Segment per-unit spike trains into a binned (units, trials, bins)
    array anchored at `anchor_times[t]` (per-trial ms), truncated to each
    trial's [trial_start_ms, exit_time_ms] span.

    Replicates segment_trials.segment_trials exactly (same bin edges, same
    NaN = "bin outside this trial's actual span" semantics) with the anchor
    as the free parameter.

    Returns dict with binned, bin_centers, window_ms, bin_width_ms, mode.
    """
    lo, hi = window_ms
    n_bins = int(np.ceil((hi - lo) / bin_width_ms))
    edges = lo + np.arange(n_bins + 1) * bin_width_ms
    bin_centers = (edges[:-1] + edges[1:]) / 2.0

    t_start = trial_table["trial_start_ms"].to_numpy()
    t_exit = trial_table["exit_time_ms"].to_numpy()

    n_units = len(spike_times_by_unit)
    n_trials = len(trial_table)
    binned = np.full((n_units, n_trials, n_bins), np.nan)
    raw = {}

    for u, spikes in enumerate(spike_times_by_unit):
        unit_id = int(unit_ids[u])
        spikes = np.asarray(spikes, dtype=np.float64)
        if spikes.size == 0:
            continue
        for t in range(n_trials):
            c = anchor_times[t]
            win_lo = max(c + lo, t_start[t])
            win_hi = min(c + hi, t_exit[t])
            if win_hi <= win_lo:
                continue
            rel_lo = win_lo - c
            rel_hi = win_hi - c
            cover_lo = int(np.searchsorted(edges, rel_lo, side="right") - 1)
            cover_hi = int(np.searchsorted(edges, rel_hi, side="right"))
            cover_lo = max(0, cover_lo)
            binned[u, t, cover_lo:cover_hi] = 0.0
            a = np.searchsorted(spikes, win_lo, side="left")
            b = np.searchsorted(spikes, win_hi, side="left")
            if a >= b:
                continue
            sel = spikes[a:b]
            rel = sel - c
            raw[(unit_id, int(t))] = rel
            idx = np.searchsorted(edges, rel, side="right") - 1
            idx = idx[(idx >= 0) & (idx < n_bins)]
            if len(idx) > 0:
                counts = np.bincount(idx, minlength=n_bins).astype(np.float64)
                binned[u, t] += counts
    return {"binned": binned, "bin_centers": bin_centers,
            "window_ms": window_ms, "bin_width_ms": bin_width_ms,
            "mode": MODE}


# %%
def load_reference_npz():
    """Reference (choice-anchored) binned array + embedded tables."""
    z = np.load(REF_BINNED, allow_pickle=True)
    return z


def load_spikes_by_unit():
    """{unit_id: sorted spike times} for QC-passed units, from the same
    spikes.mat + firing-rate QC that segment_trials.load_units applies.
    Unit ordering follows the reference npz's unit_ids when it exists."""
    unit_ids, spike_times, meta = load_units(SPIKES_PATH, NEURON_DATA_PATH)
    out = {}
    for uid, ts in zip(unit_ids, spike_times):
        out[int(uid)] = np.sort(np.asarray(ts, dtype=np.float64))

    # preserve reference unit ordering + membership exactly
    if REF_BINNED.exists():
        ref = np.load(REF_BINNED, allow_pickle=True)
        ref_ids = [int(u) for u in np.asarray(ref["unit_ids"])]
        ref_units = set(ref_ids)
        out = {u: out[u] for u in ref_ids if u in out}
    return out


# %%
def _build_for_anchor(anchor_times_key):
    """Return (result_dict, unit_ids array, trial_table)."""
    trial_table = pd.read_csv(TRIAL_TABLE)
    by_unit = load_spikes_by_unit()

    kept_ids = np.array(sorted(by_unit.keys()), dtype=np.int64)
    times_by_unit = [by_unit[int(u)] for u in kept_ids]
    anchors = trial_table[anchor_times_key].to_numpy()
    return (build_binned(times_by_unit, kept_ids, trial_table, anchors),
            kept_ids, trial_table)


# %%
def entry_locked_npz_exists():
    return ENTRY_BINNED.exists()


def build_entry_locked(force=False):
    """Build and save the entry-anchored npz (no-op if it already exists
    unless force=True). Returns True if (re)built, False if cached."""
    if ENTRY_BINNED.exists() and not force:
        print(f"Entry-locked npz exists ({ENTRY_BINNED.name}); loading. "
              f"Use force=True / --force to rebuild.")
        return False

    res, unit_ids, trial_table = _build_for_anchor("entry_time_ms")

    unit_meta = pd.read_csv(OUT_DIR / "unit_metadata.csv")
    trial_table["run_id"] = _RUN.run_id
    trial_table["participant_id"] = _RUN.participant
    unit_meta["run_id"] = _RUN.run_id
    unit_meta["participant_id"] = _RUN.participant

    # embedded eligibility + timing helpers for the entry-anchored analyses
    tt = trial_table.copy()
    rt = (tt["choice_time_ms"] - tt["entry_time_ms"]).to_numpy()
    ball = (tt["exit_time_ms"] - tt["choice_time_ms"]).to_numpy()
    tt["rt_ms"] = rt
    tt["ball_time_ms"] = ball
    tt["full_coverage_pre500"] = (tt["entry_time_ms"]
                                  - tt["trial_start_ms"]) >= 500.0

    np.savez_compressed(
        ENTRY_BINNED,
        binned=res["binned"],
        unit_ids=unit_ids,
        trial_ids=trial_table["trial_id"].to_numpy(),
        bin_centers=res["bin_centers"],
        trial_table=tt.to_records(index=False),
        unit_metadata=unit_meta.to_records(index=False),
        window_ms=np.asarray(res["window_ms"]),
        bin_width_ms=np.asarray(res["bin_width_ms"]),
        mode=res["mode"],
        anchor="entry",
        ref_anchor="choice",
    )
    n_spikes = int(np.nansum(res["binned"]))
    n_nonempty = int(np.count_nonzero(~np.isnan(res["binned"])))
    print(f"Saved {ENTRY_BINNED.name}: {res['binned'].shape}, "
          f"{n_spikes:,} segmented spikes")
    return True


# %%
def parity_check():
    """Rebuild the choice-anchored binned array with this code and compare to
    the stored reference npz. Must match on every covered bin."""
    res, unit_ids, trial_table = _build_for_anchor("choice_time_ms")
    ref = load_reference_npz()
    rb = np.asarray(ref["binned"])
    nb = res["binned"]
    assert rb.shape == nb.shape, (rb.shape, nb.shape)

    # counts must match on every covered (non-NaN) bin
    covered = ~np.isnan(rb)
    diff = np.abs(rb[covered] - nb[covered])
    max_diff = float(diff.max()) if diff.size else 0.0
    n_mismatch = int(np.sum(diff > 0))
    n_covered = int(covered.sum())

    # coverage pattern must match too
    pat_diff = int(np.sum(np.isnan(rb) != np.isnan(nb)))
    total_ref = float(np.nansum(rb))
    total_new = float(np.nansum(nb))
    print(f"Parity vs {REF_BINNED.name}:")
    print(f"  covered bins      : {n_covered}")
    print(f"  bin-count mismatches (>0 diff): {n_mismatch}")
    print(f"  max abs diff on covered bins  : {max_diff:.6f}")
    print(f"  coverage-pattern mismatches   : {pat_diff}")
    print(f"  total segmented spikes ref/new: {total_ref:.0f} / {total_new:.0f}")
    ok = (n_mismatch == 0) and (pat_diff == 0) and abs(total_ref - total_new) < 1
    print("PARITY " + ("PASS" if ok else "FAIL"))
    if not ok:
        raise SystemExit("Parity check failed: entry-locked builder does not "
                         "reproduce the stored choice-anchored npz.")


# %%
def load_entry_binned():
    """Load the entry-anchored npz (building it first if absent)."""
    build_entry_locked(force=False)
    return np.load(ENTRY_BINNED, allow_pickle=True)


# %%
# ----------------------- Feature helpers --------------------------------

def rate_features_entry(binned, bin_centers, lo, hi):
    """(n_trials, n_units) sqrt-transformed mean firing rate (Hz) over the
    bins with centers in [lo, hi]. NaN bins (outside a trial's actual span)
    are excluded from the mean, exactly like neural_lda_decoding.rate_features
    but on the entry-anchored grid."""
    mask = _window_bin_mask(bin_centers, lo, hi)
    sel = binned[:, :, mask]
    valid = ~np.isnan(sel)
    counts = np.where(valid, sel, 0.0)
    n_valid = valid.sum(axis=2)
    rate = counts.sum(axis=2) / (n_valid * (BIN_WIDTH_MS / 1000.0))
    rate = np.where(n_valid > 0, rate, 0.0)
    return np.sqrt(np.maximum(rate, 0.0)).T


def approach_rate_features(binned, bin_centers, rt_ms):
    """Per-trial mean firing rate (Hz, sqrt-transformed) over each trial's
    full approach interval [entry, choice) = bins with 0 <= center < rt_ms[t].

    The interval is variable per trial (rt_ms), which is exactly the quantity
    the entry anchor makes measurable: the whole pre-decision window, never
    crossing the choice pass. Returns (n_trials, n_units)."""
    sel = binned                                   # (units, trials, bins)
    valid = ~np.isnan(sel)
    counts = np.where(valid, sel, 0.0)
    centers = bin_centers[None, None, :]
    # bin eligible for a trial iff its center is in [0, rt) and covered
    in_window = (centers >= 0.0) & (centers < rt_ms[None, :, None])
    eligible = valid & in_window
    n_valid = eligible.sum(axis=2)
    sums = counts.sum(axis=2)
    rate = np.zeros_like(sums, dtype=float)
    m = n_valid > 0
    rate[m] = sums[m] / (n_valid[m] * (BIN_WIDTH_MS / 1000.0))
    return np.sqrt(np.maximum(rate, 0.0)).T


def eligibility_mask(rt_ms, window_ms, min_rt_ms=0.0):
    """Trials whose approach interval can contain the window. For a window
    [0, W] the trial is eligible iff rt_ms >= W (window stays before the
    choice pass). window_ms=(lo, hi) here with lo=0 by construction."""
    W = window_ms[1]
    return rt_ms >= max(W, min_rt_ms)


# %%
if __name__ == "__main__":
    import sys
    if "--check-parity" in sys.argv:
        parity_check()
    else:
        build_entry_locked(force=("--force" in sys.argv))

# %%
