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
# # Segment unit-level spikes into per-trial windows
#
# Aligns every unit's spike times to the **choice moment** (t=0 = the moment the
# ball passes through the 2-hole level of a 1-2-1 sequence), then segments spikes
# into per-trial windows. No PSTHs are saved here — this script only produces the
# segmented data structures for later analysis.
#
# A "trial" is the 1-2-1 hole sequence: entry (1 hole) → choice (2 holes) →
# exit (1 hole). Each trial spans from the previous sequence's exit pass-through
# (or spawn) to the current sequence's exit pass-through.
#
# Run with the same interpreter that has scipy/pandas:
#   C:\Users\manik\AppData\Local\Programs\Python\Python311\python.exe analysis\segment_trials.py

# %%
import json
import pickle
import numpy as np
import pandas as pd
import scipy.io as sio
from pathlib import Path

# ---------------------------- Configuration ----------------------------
BEHAVIOR_PATH = Path(r"C:\Users\manik\Desktop\Obsidian\General Thoughts\Z Images and Files\Hennig Lab Project\falldown\analysis\emu data\YFZ-2026-07-29T21-37-47-781Z-kdyd.json")
SPIKES_PATH = Path(r"C:\Users\manik\Desktop\Spike Sorting For Hennig Project\spikesort_results\cluster_viewer_results\spikes.mat")
NEURON_DATA_PATH = Path(r"C:\Users\manik\Desktop\Spike Sorting For Hennig Project\spikesort_results\cluster_viewer_results\neuron_data.json")
OUT_DIR = Path(r"C:\Users\manik\Desktop\Obsidian\General Thoughts\Z Images and Files\Hennig Lab Project\falldown\analysis\spike_data_alignment_output")

BIN_WIDTH_MS = 25.0           # default bin width (parameter of segment_trials)
WINDOW_MS = (-2000.0, 2000.0) # window around choice time (t=0)
MIN_FIRING_RATE_HZ = 0.5      # units below this are dropped
MIN_EXPERIMENT_BLOCK = 4      # blocks 0-3 are practice/instruction
# -----------------------------------------------------------------------


# %%
# ------------------------- 1. Build the trial table --------------------
def build_trial_table(path, min_block=MIN_EXPERIMENT_BLOCK):
    """Dedup by (block_index, trial_index) keeping the last occurrence, keep
    experiment blocks only, then segment the flat event list into 1-2-1
    sequences. Returns a DataFrame, one row per sequence:
      trial_id, block_index, sequence_index, trial_start_ms, entry_time_ms,
      choice_time_ms, exit_time_ms, choice_hole, hole_locations
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    last = {}
    for block in data["blocks"]:
        bi = block.get("block_index")
        if bi is None or bi < min_block:
            continue
        for trial in block.get("trials", []):
            key = (bi, trial.get("index"))
            last[key] = trial

    keys = sorted(last.keys())
    flat = []
    for bi, ti in keys:
        trial = last[(bi, ti)]
        events = trial.get("events", [])
        if not events:
            continue
        ev = events[0]
        holes = ev.get("hole_locations", trial.get("hole_locations"))
        flat.append({
            "block_index": bi,
            "trial_index": ti,
            "time_ms": float(ev["time"]),
            "n_holes": len(holes) if holes else 0,
            "hole_used": ev.get("holeUsed"),
            "hole_locations": holes,
        })

    rows = []
    trial_id = 0
    for i in range(0, len(flat) - 2, 3):
        entry, choice, exit_ = flat[i], flat[i + 1], flat[i + 2]
        if (entry["n_holes"], choice["n_holes"], exit_["n_holes"]) != (1, 2, 1):
            raise ValueError(
                f"Sequence at index {i} is not 1-2-1: "
                f"{entry['n_holes']},{choice['n_holes']},{exit_['n_holes']}"
            )
        prev_exit_time = flat[i - 1]["time_ms"] if i > 0 else flat[0]["time_ms"]
        rows.append({
            "trial_id": trial_id,
            "block_index": choice["block_index"],
            "sequence_index": i // 3,
            "trial_start_ms": prev_exit_time,
            "entry_time_ms": entry["time_ms"],
            "choice_time_ms": choice["time_ms"],
            "exit_time_ms": exit_["time_ms"],
            "choice_hole": choice["hole_used"],
            "hole_locations": json.dumps(choice["hole_locations"]),
        })
        trial_id += 1

    return pd.DataFrame(rows)


# %%
# --------------------- 2. Load and QC units ----------------------------
def load_units(spikes_path, neuron_data_path, min_firing_rate=MIN_FIRING_RATE_HZ):
    """Load unit-level spike times from spikes.mat, reconcile firing-rate QC
    from neuron_data.json (matched by filename+cluster_id), drop low-rate
    units. Returns (unit_ids, spike_times_by_unit, unit_metadata).
    """
    d = sio.loadmat(str(spikes_path), variable_names=[
        "spikes", "chan", "cluster_ids", "channel_file_names"])
    spikes = d["spikes"].tocsr()
    chan = np.asarray(d["chan"]).flatten().astype(int)
    cids = np.asarray(d["cluster_ids"]).flatten()
    fnames = [str(f).strip() for f in np.asarray(d["channel_file_names"]).flatten()]

    with open(neuron_data_path, encoding="utf-8") as fh:
        nd = json.load(fh)
    nd_by_key = {(str(u["filename"]).strip(), int(u["cluster_id"])): u for u in nd}

    unit_ids, times_by_unit, meta_rows = [], [], []
    for i in range(spikes.shape[0]):
        cols = spikes[i].indices
        if len(cols) == 0:
            continue
        key = (fnames[i], int(cids[i])) if not np.isnan(cids[i]) else (fnames[i], None)
        unit_id = i
        if key in nd_by_key:
            firing_rate = float(nd_by_key[key]["firing_rate_hz"])
        else:
            firing_rate = np.nan
        if not np.isnan(firing_rate) and firing_rate < min_firing_rate:
            continue
        unit_ids.append(unit_id)
        times_by_unit.append(cols.astype(np.float64))
        meta_rows.append({
            "unit_id": unit_id,
            "channel": int(chan[i]),
            "cluster_id": int(cids[i]) if not np.isnan(cids[i]) else None,
            "source_file": fnames[i],
            "firing_rate_hz": firing_rate,
        })
    return np.array(unit_ids), times_by_unit, pd.DataFrame(meta_rows)


# %%
# ----------------- 3. Segment spikes into per-trial windows -------------
def segment_trials(spike_times_by_unit, trial_table, window_ms=WINDOW_MS,
                   mode="truncated", bin_width_ms=BIN_WIDTH_MS,
                   unit_ids=None):
    """Segment spikes into per-trial windows aligned to choice time (t=0).

    Parameters
    ----------
    spike_times_by_unit : list[np.ndarray]
        Spike times (behavioral ms) per unit, in the same order as unit_ids.
    trial_table : pd.DataFrame
        One row per trial with choice_time_ms, trial_start_ms, exit_time_ms.
    window_ms : tuple (lo, hi)
        Fixed window around choice time for naive mode.
    mode : "truncated" | "naive"
        "truncated": clip each trial's window to the neighboring choice/exit
            boundaries (no double-counting; per-trial length varies).
        "naive": use the fixed window_ms on every trial (may double-count).
    bin_width_ms : float
        Bin width for the binned count matrix.
    unit_ids : array-like, optional
        Unit id for each element of spike_times_by_unit. Defaults to 0..n-1.

    Returns
    -------
    dict with:
      raw    : dict {(unit_id, trial_id): np.ndarray} of spike times relative
               to choice time (ms), sorted.
      binned : np.ndarray (n_units, n_trials, n_bins) spike counts. Bins with
               no time coverage (out-of-window) are NaN.
      bin_centers : np.ndarray (n_bins,)
      window_ms, bin_width_ms, mode : the inputs echoed back for provenance.
    """
    choice = trial_table["choice_time_ms"].to_numpy()
    t_start = trial_table["trial_start_ms"].to_numpy()
    t_exit = trial_table["exit_time_ms"].to_numpy()
    lo, hi = window_ms
    if unit_ids is None:
        unit_ids = np.arange(len(spike_times_by_unit))
    unit_ids = np.asarray(unit_ids)

    # bin edges centered so t=0 sits inside a bin
    n_bins = int(np.ceil((hi - lo) / bin_width_ms))
    edges = lo + np.arange(n_bins + 1) * bin_width_ms
    bin_centers = (edges[:-1] + edges[1:]) / 2.0

    n_units = len(spike_times_by_unit)
    n_trials = len(trial_table)
    binned = np.full((n_units, n_trials, n_bins), np.nan)
    raw = {}

    for u, spikes in enumerate(spike_times_by_unit):
        unit_id = int(unit_ids[u])
        for t in range(n_trials):
            c = choice[t]
            if mode == "naive":
                win_lo, win_hi = c + lo, c + hi
            else:  # truncated
                win_lo = max(c + lo, t_start[t])
                win_hi = min(c + hi, t_exit[t])
            if win_hi <= win_lo:
                continue
            # bins inside the trial's window are covered (0-count valid);
            # bins outside the window stay NaN. Edges/bin centers are
            # relative to choice time, so map the absolute window to rel.
            rel_lo = win_lo - c
            rel_hi = win_hi - c
            cover_lo = int(np.searchsorted(edges, rel_lo, side="right") - 1)
            cover_hi = int(np.searchsorted(edges, rel_hi, side="right"))
            cover_lo = max(0, cover_lo)
            binned[u, t, cover_lo:cover_hi] = 0.0
            sel = spikes[(spikes >= win_lo) & (spikes < win_hi)]
            if len(sel) == 0:
                continue
            rel = sel - c
            raw[(unit_id, int(t))] = rel
            idx = np.searchsorted(edges, rel, side="right") - 1
            idx = idx[(idx >= 0) & (idx < n_bins)]
            if len(idx) > 0:
                counts = np.bincount(idx, minlength=n_bins).astype(np.float64)
                binned[u, t] += counts
    return {"raw": raw, "binned": binned, "bin_centers": bin_centers,
            "window_ms": window_ms, "bin_width_ms": bin_width_ms,
            "mode": mode}


# %%
# ------------------------------- Main ----------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building trial table ...")
    trials = build_trial_table(BEHAVIOR_PATH)
    print(f"  {len(trials)} sequences (trials) in experiment blocks")
    print(f"  median trial duration: "
          f"{(trials['exit_time_ms'] - trials['trial_start_ms']).median():.0f} ms")

    print("Loading units ...")
    unit_ids, spike_times, unit_meta = load_units(SPIKES_PATH, NEURON_DATA_PATH)
    print(f"  {len(unit_ids)} units pass QC (firing rate >= "
          f"{MIN_FIRING_RATE_HZ} Hz)")
    unit_meta.to_csv(OUT_DIR / "unit_metadata.csv", index=False)

    print("Segmenting spikes into per-trial windows ...")
    result = segment_trials(spike_times, trials, mode="truncated",
                            unit_ids=unit_ids)
    result["unit_ids"] = unit_ids
    result["trial_ids"] = trials["trial_id"].to_numpy()

    with open(OUT_DIR / "segmented_spikes_raw.pkl", "wb") as fh:
        pickle.dump(result["raw"], fh)
    np.savez_compressed(
        OUT_DIR / "segmented_spikes_binned.npz",
        binned=result["binned"],
        unit_ids=unit_ids,
        trial_ids=result["trial_ids"],
        bin_centers=result["bin_centers"],
        trial_table=trials.to_records(index=False),
        unit_metadata=unit_meta.to_records(index=False),
        window_ms=result.get("window_ms", WINDOW_MS),
        bin_width_ms=result.get("bin_width_ms", BIN_WIDTH_MS),
        mode=result.get("mode", "truncated"),
    )
    trials.to_csv(OUT_DIR / "trial_table.csv", index=False)

    n_raw = len(result["raw"])
    print(f"\nOutputs in {OUT_DIR}")
    print("  trial_table.csv")
    print("  unit_metadata.csv")
    print("  segmented_spikes_raw.pkl")
    print("  segmented_spikes_binned.npz")
    print(f"\n{len(trials)} trials x {len(unit_ids)} units, "
          f"{n_raw} non-empty (unit,trial) segments")
    total_seg = sum(len(v) for v in result["raw"].values())
    print(f"  segmented spikes: {total_seg:,}")


if __name__ == "__main__":
    main()
