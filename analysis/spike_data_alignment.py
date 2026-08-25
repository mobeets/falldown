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

# %%
"""Align NS5 photodiode flashes to behavioral JSON events, then convert spike
times (cluster_viewer_results/spikes_perChannel.mat) to behavioral time.

Alignment is computed ONLY from the NS5 photodiode channel + the behavioral
JSON. No spike data participates in the offset/drift calculation.

Approach
--------
1. Detect photodiode flashes as sustained runs above a fixed threshold.
2. Align the flash-interval sequence to the event-interval sequence with
   dynamic time warping (DTW), allowing gaps on either side so that missing
   data at BOTH ends of the JSON, and spurious flashes in the NS5, do not
   break the match. This yields (flash_idx, event_idx) anchor pairs.
3. Fit a smooth offset model  event = offset(flash_time) + flash_time  on the
   anchors. Verified: offset is nearly constant (~126.9 s) with <100 ms drift
   over the whole session, so spike times convert correctly everywhere.
4. Cross-validate with audio clicks: the game plays a click at the same moment
   as the photodiode flash, so RoomMic2 peaks should coincide with flashes.
5. Convert spikes from cluster_viewer_results using the offset model.

Run with a Python that has numpy/scipy/pandas, e.g.:
  C:\\Users\\manik\\AppData\\Local\\Programs\\Python\\Python311\\python.exe analysis\\spike_data_alignment.py
"""

import json
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# %%
# ---------------------------- Configuration ----------------------------
NS5_PATH = Path(r"C:\Users\manik\Desktop\Obsidian\General Thoughts\Z Images and Files\Hennig Lab Project\falldown\noPHIEMU-0113_subj-YFZ_task-FD_run-01_NSP-2.ns5")
ORIG_NS5_PATH = Path(r"C:\Users\manik\Desktop\Spike Sorting For Hennig Project\spikesort_results\EMU-0113_subj-YFZ_task-FD_run-01_NSP-2.ns5")
BEHAVIOR_PATH = Path(r"C:\Users\manik\Desktop\Obsidian\General Thoughts\Z Images and Files\Hennig Lab Project\falldown\data\emu\YFZ-2026-07-29T21-37-47-781Z-kdyd.json")
SPIKES_PATH = Path(r"C:\Users\manik\Desktop\Spike Sorting For Hennig Project\spikesort_results\cluster_viewer_results\spikes_perChannel.mat")
OUT_DIR = Path(r"C:\Users\manik\Desktop\Obsidian\General Thoughts\Z Images and Files\Hennig Lab Project\falldown\analysis\neural_outputs")

SAMPLING_RATE = 30000.0          # Hz
N_CHANNELS = 78                  # analog channels in NS5
PHOTODIODE_ROW = 65              # 1-indexed row of the photodiode channel
ROOM_MIC2_ROW = 68               # 1-indexed row of RoomMic2 (audio clicks)
HEADER_SIZE = 8 + 306 + 78 * 66  # magic + basic header + extended header
DATA_HEADER_SIZE = 13            # packet header: 1 + uint64 ts + uint32 samples
N_FRAMES = 72_348_374
CHUNK_FRAMES = 2_000_000         # frames per streaming read

FLASH_THRESHOLD = 18000          # photodiode amplitude separating flash from baseline
FLASH_MIN_DUR_MS = 20.0          # sustained above-threshold runs shorter than this are noise

DTW_GAP_MS = 300.0               # gap penalty in the interval DTW (skips a flash/event)
DTW_TOL = 1.0                    # backtrack tolerance (ms)
OFFSET_FIT_DEG = 2               # smooth clock-drift model order
# -----------------------------------------------------------------------

# %%
# ------------------------- 1. Stream photodiode ------------------------
def read_channel(path, row, n_frames=N_FRAMES, chunk=CHUNK_FRAMES):
    """Return one analog channel (int16) without loading the whole NS5."""
    trace = np.empty(n_frames, dtype=np.int16)
    data_start = HEADER_SIZE + DATA_HEADER_SIZE
    frame_bytes = N_CHANNELS * 2
    with open(path, "rb") as f:
        f.seek(data_start)
        start = 0
        while start < n_frames:
            n = min(chunk, n_frames - start)
            buf = np.frombuffer(f.read(n * frame_bytes), dtype="<i2")
            buf = buf.reshape(n, N_CHANNELS)
            trace[start:start + n] = buf[:, row - 1]
            start += n
    return trace


# %%
# ------------------------- 2. Detect flashes ---------------------------
def detect_flashes(trace, fs=SAMPLING_RATE):
    """Onset times (ms) of sustained photodiode flashes above threshold."""
    above = trace > FLASH_THRESHOLD
    d = np.diff(above.astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    if len(ends) < len(starts):
        ends = np.append(ends, len(trace))
    durs = ends - starts
    keep = durs >= int(FLASH_MIN_DUR_MS * fs / 1000.0)
    return starts[keep] / fs * 1000.0  # ms


# %%
# ------------------------- 3. Behavioral events ------------------------
def load_events(path):
    """All event times from the JSON, in file order (already monotonic)."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    times = []
    for block in data["blocks"]:
        for trial in block.get("trials", []):
            for ev in trial.get("events", []):
                t = ev.get("time")
                if t is not None:
                    times.append(float(t))
    return np.array(times)


# %%
# ------------------------- 4. DTW alignment ----------------------------
def align_intervals(flash_ms, event_ms):
    """Align flash-interval and event-interval sequences with DTW.

    Returns (flash_idx, event_idx) anchor pairs: flash[flash_idx] maps to
    event[event_idx]. Gaps (spurious flashes, missing events at either end)
    are allowed at cost DTW_GAP_MS per skipped interval.
    """
    dF = np.diff(flash_ms)
    dE = np.diff(event_ms)
    N, M = len(dF), len(dE)
    INF = 1e30
    dp = np.full((N + 1, M + 1), INF, dtype=np.float32)
    dp[0, 0] = 0.0
    for i in range(1, N + 1):
        prev, cur = dp[i - 1], dp[i]
        fi = dF[i - 1]
        for j in range(1, M + 1):
            cur[j] = min(prev[j - 1] + abs(fi - dE[j - 1]),
                         prev[j] + DTW_GAP_MS,
                         cur[j - 1] + DTW_GAP_MS)
    # backtrack
    i, j = N, M
    matched = []
    while i > 0 or j > 0:
        fi = dF[i - 1] if i > 0 else -1.0
        if i > 0 and j > 0:
            mc = dp[i - 1, j - 1] + abs(fi - dE[j - 1])
            if abs(dp[i, j] - mc) <= DTW_TOL:
                matched.append((i - 1, j - 1))
                i -= 1
                j -= 1
                continue
        if i > 0 and abs(dp[i, j] - (dp[i - 1, j] + DTW_GAP_MS)) <= DTW_TOL:
            i -= 1
            continue
        if j > 0 and abs(dp[i, j] - (dp[i, j - 1] + DTW_GAP_MS)) <= DTW_TOL:
            j -= 1
            continue
        if i > 0:
            i -= 1
        elif j > 0:
            j -= 1
        else:
            break
    matched = matched[::-1]
    fi = np.array([p[0] for p in matched], dtype=np.int64)
    ei = np.array([p[1] for p in matched], dtype=np.int64)
    return fi, ei


def fit_offset(flash_ms, event_ms, fi, ei):
    """Fit event = offset(flash_time) + flash_time on the anchor pairs.

    Returns dict with model coefficients and diagnostics.
    """
    fx = np.asarray(flash_ms)[fi]
    ey = np.asarray(event_ms)[ei]
    off = ey - fx

    # robust: drop anchors far from the smooth model, refit
    coeff = np.polyfit(fx, off, OFFSET_FIT_DEG)
    resid = off - np.polyval(coeff, fx)
    mad = np.median(np.abs(resid))
    keep = np.abs(resid) <= 4.0 * (1.4826 * mad if mad > 0 else 1.0)
    coeff = np.polyfit(fx[keep], off[keep], OFFSET_FIT_DEG)
    resid = off - np.polyval(coeff, fx)

    # interval confirmation on inlier anchors
    dF = np.diff(fx[keep])
    dE = np.diff(ey[keep])
    int_resid = dE - dF if len(dF) == len(dE) else np.array([])

    return {
        "n_flashes": int(len(flash_ms)),
        "n_events": int(len(event_ms)),
        "n_anchors": int(len(fi)),
        "n_anchors_kept": int(keep.sum()),
        "n_flashes_unmatched": int(len(flash_ms) - len(fi)),
        "n_events_unmatched": int(len(event_ms) - len(ei)),
        "head_events_unmatched": int(ei[0]) if len(ei) else None,
        "tail_events_unmatched": int(len(event_ms) - 1 - ei[-1]) if len(ei) else None,
        "coeff": coeff.tolist(),
        "offset_median_ms": float(np.median(off)),
        "offset_std_ms": float(off.std()),
        "offset_drift_ms": float(np.polyval(coeff, fx[-1]) - np.polyval(coeff, fx[0])),
        "fit_resid_median_ms": float(np.median(np.abs(resid))),
        "fit_resid_p90_ms": float(np.percentile(np.abs(resid), 90)),
        "interval_resid_median_ms": float(np.median(np.abs(int_resid))) if len(int_resid) else None,
        "interval_resid_p90_ms": float(np.percentile(np.abs(int_resid), 90)) if len(int_resid) else None,
        "model": f"event = offset(flash) + flash, offset=poly{OFFSET_FIT_DEG}(flash)",
    }


def offset_at(coeff, flash_ms):
    return np.polyval(coeff, np.asarray(flash_ms, dtype=np.float64))


# %%
# ------------------------- 5. Audio cross-validation -------------------
def audio_click_delta(flash_ms):
    """Median delta between each flash and the nearest RoomMic2 peak.

    The game plays a click at the same moment as the photodiode flash, so a
    click should land within ~100 ms of every flash. Read from the ORIGINAL
    NS5 (mics were zeroed in the de-identified copy).
    """
    if not ORIG_NS5_PATH.exists():
        return None
    mic = read_channel(ORIG_NS5_PATH, ROOM_MIC2_ROW).astype(np.float64)
    rect = np.abs(mic)
    k = int(0.002 * SAMPLING_RATE)
    env = np.convolve(rect, np.ones(k) / k, mode="same")
    deltas = []
    for f in flash_ms:
        lo = int((f - 80) * SAMPLING_RATE / 1000.0)
        hi = int((f + 80) * SAMPLING_RATE / 1000.0)
        if lo < 0 or hi >= len(env):
            continue
        seg = env[lo:hi]
        pk = lo + int(np.argmax(seg))
        deltas.append(pk / SAMPLING_RATE * 1000.0 - f)
    deltas = np.array(deltas)
    return {
        "n": int(len(deltas)),
        "delta_median_ms": float(np.median(deltas)),
        "delta_p10_ms": float(np.percentile(deltas, 10)),
        "delta_p90_ms": float(np.percentile(deltas, 90)),
        "frac_within_100ms": float(np.mean(np.abs(deltas) < 100)),
    }


# %%
# ------------------------- 6. Convert spikes ---------------------------
def convert_spikes(mat_path, coeff, out_dir):
    """spikes_perChannel.mat: spikes (64 x ms, uint8), chan (64,)."""
    d = sio.loadmat(str(mat_path))
    spikes = d["spikes"]
    spikes = spikes.toarray() if hasattr(spikes, "toarray") else np.asarray(spikes)
    chan = np.asarray(d["chan"]).flatten()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for i in range(spikes.shape[0]):
        cols = np.flatnonzero(spikes[i] > 0)
        if len(cols) == 0:
            continue
        ns5_ms = cols.astype(np.float64)
        behav_ms = ns5_ms + offset_at(coeff, ns5_ms)
        df = pd.DataFrame({
            "channel": int(chan[i]),
            "spike_time_ns5_ms": ns5_ms,
            "spike_time_behavioral_ms": behav_ms,
        })
        df.to_csv(out_dir / f"channel_{int(chan[i]):03d}.csv", index=False)
        total += len(cols)
    return total


# %%
# ------------------------------- Main ----------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Reading photodiode channel from NS5 ...")
    trace = read_channel(NS5_PATH, PHOTODIODE_ROW)
    print(f"  photodiode trace: {len(trace)} samples, "
          f"min={int(trace.min())} max={int(trace.max())} "
          f"median={int(np.median(trace))}")

    flash_ms = detect_flashes(trace)
    print(f"  flashes detected: {len(flash_ms)}")

    event_ms = load_events(BEHAVIOR_PATH)
    print(f"  behavioral events: {len(event_ms)} "
          f"(first={event_ms[0]:.1f}ms last={event_ms[-1]:.1f}ms)")

    print("Aligning flash/event interval sequences with DTW ...")
    fi, ei = align_intervals(flash_ms, event_ms)
    info = fit_offset(flash_ms, event_ms, fi, ei)
    print("\nAlignment:")
    for k, v in info.items():
        if isinstance(v, list):
            print(f"  {k}: [{', '.join(f'{x:.3f}' for x in v)}]")
        else:
            print(f"  {k}: {v}")

    coeff = info["coeff"]

    # audio cross-validation (read-only, from original NS5)
    print("\nAudio cross-validation (RoomMic2 from original NS5) ...")
    audio = audio_click_delta(flash_ms)
    if audio:
        for k, v in audio.items():
            print(f"  {k}: {v}")
        info["audio"] = audio
    else:
        print("  original NS5 not found; skipping")

    # flash table
    flash_behav = flash_ms + offset_at(coeff, flash_ms)
    flash_df = pd.DataFrame({
        "flash_index": np.arange(len(flash_ms)),
        "flash_time_ns5_ms": flash_ms,
        "flash_time_behavioral_ms": flash_behav,
    })
    flash_df.to_csv(OUT_DIR / "photodiode_flashes.csv", index=False)

    # convert spikes
    print("\nConverting spikes from cluster_viewer_results ...")
    n_spikes = convert_spikes(SPIKES_PATH, coeff, OUT_DIR / "spike_times_synced")
    print(f"  {n_spikes} spikes written")

    # report
    report = {
        "ns5": str(NS5_PATH),
        "orig_ns5": str(ORIG_NS5_PATH),
        "behavior": str(BEHAVIOR_PATH),
        "spikes": str(SPIKES_PATH),
        "photodiode_row": PHOTODIODE_ROW,
        "sampling_rate_hz": SAMPLING_RATE,
        "flash_threshold": FLASH_THRESHOLD,
        "flash_min_dur_ms": FLASH_MIN_DUR_MS,
        "dtw_gap_ms": DTW_GAP_MS,
        "offset_fit_deg": OFFSET_FIT_DEG,
        **info,
        "n_spikes_converted": int(n_spikes),
        "mapping": "flash[i] <-> event[j] via interval-sequence DTW",
        "warning": (
            f"{info['n_events_unmatched']} JSON events have no flash "
            f"({info['head_events_unmatched']} at the head, "
            f"{info['tail_events_unmatched']} at the tail) - these may be "
            "missing/saved-extra events. Spike conversion uses only the "
            "matched anchors, so it is unaffected."
        ),
    }
    (OUT_DIR / "alignment_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    # confirmation plot
    fx = np.asarray(flash_ms)[fi]
    ey = np.asarray(event_ms)[ei]
    off = ey - fx
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    ax = axes[0]
    ax.plot(fx / 1000.0, off, ".", ms=2, alpha=0.4, label="DTW anchor offset")
    xs = np.linspace(fx[0], fx[-1], 500)
    ax.plot(xs / 1000.0, np.polyval(coeff, xs), "r-", lw=2, label="fitted offset model")
    ax.set_xlabel("flash time (s)")
    ax.set_ylabel("event - flash (ms)")
    ax.set_title("Offset model from DTW anchors")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(event_ms[1:] / 1000.0, np.diff(event_ms) / 1000.0, ".-", ms=2,
            label="event intervals (s)", alpha=0.6)
    fb = flash_ms + offset_at(coeff, flash_ms)
    ax.plot(fb[1:] / 1000.0, np.diff(fb) / 1000.0, ".-", ms=2,
            label="flash intervals (s, drift-corrected)", alpha=0.6)
    ax.set_xlabel("behavioral time (s)")
    ax.set_ylabel("interval (s)")
    ax.set_title("Confirmation: event vs flash intervals")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(fi, ei, ".", ms=2, alpha=0.4)
    ax.set_xlabel("flash index")
    ax.set_ylabel("event index")
    ax.set_title("DTW correspondence (diagonal = 1:1)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confirmation_plot.png", dpi=150)
    plt.close(fig)

    print(f"\nOutputs written to {OUT_DIR}")
    print("  photodiode_flashes.csv")
    print("  alignment_report.json")
    print("  spike_times_synced/")
    print("  confirmation_plot.png")


if __name__ == "__main__":
    main()
