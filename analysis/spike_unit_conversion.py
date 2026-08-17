"""Convert unit-level spike times (spikes.mat) to behavioral time and link
spikes to behavioral trials.

Uses ONLY the already-computed alignment from spike_data_alignment.py:
- offset model coefficients from alignment_report.json (no NS5 re-read)
- photodiode flash times from photodiode_flashes.csv (no NS5 re-read)

Outputs (in analysis/spike_data_alignment_output/):
  spikes_units.csv            single merged file, one row per spike
                              (unit_id, channel, cluster_id, source_file,
                               spike_time_ns5_ms, spike_time_behavioral_ms)
  flash_event_linkage.csv     flash index -> behavioral event (block/trial)
  spike_trial_assignments.csv every spike -> the trial it falls in
"""

import json
import numpy as np
import pandas as pd
import scipy.io as sio
from pathlib import Path

from spike_data_alignment import load_events, align_intervals, offset_at

SPIKES_PATH = Path(r"C:\Users\manik\Desktop\Spike Sorting For Hennig Project\spikesort_results\cluster_viewer_results\spikes.mat")
BEHAVIOR_PATH = Path(r"C:\Users\manik\Desktop\Obsidian\General Thoughts\Z Images and Files\Hennig Lab Project\falldown\analysis\emu data\YFZ-2026-07-29T21-37-47-781Z-kdyd.json")
OUT_DIR = Path(r"C:\Users\manik\Desktop\Obsidian\General Thoughts\Z Images and Files\Hennig Lab Project\falldown\analysis\spike_data_alignment_output")


def load_event_records(path):
    """Events in file order with metadata: (time_ms, block, trial, hole_locations)."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    records = []
    for block in data["blocks"]:
        for trial in block.get("trials", []):
            for ev in trial.get("events", []):
                t = ev.get("time")
                if t is None:
                    continue
                records.append({
                    "time_ms": float(t),
                    "block_index": ev.get("block_index", trial.get("block_index")),
                    "trial_index": ev.get("trial_index", trial.get("index")),
                    "event_index": ev.get("index"),
                    "hole_locations": trial.get("hole_locations"),
                })
    return records


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = json.loads((OUT_DIR / "alignment_report.json").read_text(encoding="utf-8"))
    coeff = report["coeff"]
    print(f"Offset model from alignment_report.json: coeff={coeff}")

    # ------------------------- 1. Unit-level spike times ------------------
    d = sio.loadmat(str(SPIKES_PATH), variable_names=["spikes", "chan", "cluster_ids", "channel_file_names"])
    spikes = d["spikes"].tocsr()
    chan = np.asarray(d["chan"]).flatten().astype(int)
    cids = np.asarray(d["cluster_ids"]).flatten()
    fnames = [str(f).strip() for f in np.asarray(d["channel_file_names"]).flatten()]

    unit_rows = []
    for i in range(spikes.shape[0]):
        cols = spikes[i].indices
        if len(cols) == 0:
            continue
        ns5_ms = cols.astype(np.float64)
        behav_ms = ns5_ms + offset_at(coeff, ns5_ms)
        unit_rows.append(pd.DataFrame({
            "unit_id": i,
            "channel": int(chan[i]),
            "cluster_id": int(cids[i]) if not np.isnan(cids[i]) else None,
            "source_file": fnames[i],
            "spike_time_ns5_ms": ns5_ms,
            "spike_time_behavioral_ms": behav_ms,
        }))
    units = pd.concat(unit_rows, ignore_index=True)
    units.to_csv(OUT_DIR / "spikes_units.csv", index=False)
    print(f"Unit-level spikes: {len(units):,} rows ({units['unit_id'].nunique()} units with spikes)")
    total_channel = report.get("n_spikes_converted")
    print(f"  matches channel-level total ({total_channel:,})? {len(units) == total_channel}")

    # ----------------- 2. Reproduce DTW anchors (no NS5 read) ------------
    flashes = pd.read_csv(OUT_DIR / "photodiode_flashes.csv")
    flash_ms = flashes["flash_time_ns5_ms"].to_numpy()
    event_ms = load_events(BEHAVIOR_PATH)
    events = load_event_records(BEHAVIOR_PATH)
    assert len(event_ms) == len(events), "event records/time mismatch"

    fi, ei = align_intervals(flash_ms, event_ms)
    linkage = pd.DataFrame({
        "flash_index": flashes["flash_index"].to_numpy()[fi],
        "event_index": ei,
        "event_time_behavioral_ms": event_ms[ei],
        "block_index": [events[j]["block_index"] for j in ei],
        "trial_index": [events[j]["trial_index"] for j in ei],
        "hole_locations": [json.dumps(events[j]["hole_locations"]) if events[j]["hole_locations"] is not None else None for j in ei],
    })
    linkage.to_csv(OUT_DIR / "flash_event_linkage.csv", index=False)
    print(f"Flash<->event anchors: {len(linkage):,}")

    # ------------ 3. Assign every spike to the trial it falls in -----------
    ev_t = event_ms
    ev_idx = np.arange(len(ev_t))
    st = units["spike_time_behavioral_ms"].to_numpy()

    next_i = np.searchsorted(ev_t, st, side="left")  # first event at or after spike
    next_i = np.clip(next_i, 0, len(ev_t) - 1)
    prev_i = np.clip(next_i - 1, 0, len(ev_t) - 1)

    def ev_meta(idx, key):
        return np.array([events[j][key] if events[j][key] is not None else None for j in idx])

    assignments = pd.DataFrame({
        "unit_id": units["unit_id"],
        "channel": units["channel"],
        "cluster_id": units["cluster_id"],
        "source_file": units["source_file"],
        "spike_time_behavioral_ms": st,
        "next_event_index": ev_idx[next_i],
        "dt_to_next_event_ms": ev_t[next_i] - st,
        "next_block_index": ev_meta(next_i, "block_index"),
        "next_trial_index": ev_meta(next_i, "trial_index"),
        "next_hole_locations": [json.dumps(events[j]["hole_locations"]) if events[j]["hole_locations"] is not None else None for j in next_i],
        "prev_event_index": ev_idx[prev_i],
        "dt_since_prev_event_ms": st - ev_t[prev_i],
    })
    assignments.to_csv(OUT_DIR / "spike_trial_assignments.csv", index=False)
    print(f"Spike->trial assignments: {len(assignments):,}")

    # ----------------------------- 4. Summary -----------------------------
    print(f"\nOutputs in {OUT_DIR}")
    print("  spikes_units.csv")
    print("  flash_event_linkage.csv")
    print("  spike_trial_assignments.csv")


if __name__ == "__main__":
    main()
