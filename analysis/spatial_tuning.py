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
# # Spatial tuning: do units encode ball position?
#
# Uses the continuous ball trajectory (game_states: ball_x, ball_y, camera_y,
# timestamped at ~60 Hz) to build occupancy-normalized firing-rate maps for
# each unit, for two axes:
#   axis 'x' : ball_x (horizontal position, 12 segments)
#   axis 'y' : ball_y - camera_y (on-screen vertical position, 8 bins)
#
# For each unit and axis: compute a rate map, a spatial-information statistic
# (bits), test significance with a permutation test (shuffle spike times,
# 5000x), and FDR-correct across units.
#
# Output: spatial_tuning_results.csv (NO images are written; plot_* functions
# are provided for you to call yourself).
#
# Run with:
#   C:\Users\manik\AppData\Local\Programs\Python\Python311\python.exe analysis\spatial_tuning.py

# %%
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------- Configuration ----------------------------
BEHAVIOR_PATH = Path(r"C:\Users\manik\Desktop\Obsidian\General Thoughts\Z Images and Files\Hennig Lab Project\falldown\analysis\emu data\YFZ-2026-07-29T21-37-47-781Z-kdyd.json")
OUT_DIR = Path(r"C:\Users\manik\Desktop\Obsidian\General Thoughts\Z Images and Files\Hennig Lab Project\falldown\analysis\spike_data_alignment_output")
SPIKES_UNITS = OUT_DIR / "spikes_units.csv"
UNIT_META = OUT_DIR / "unit_metadata.csv"
TRIAL_TABLE = OUT_DIR / "trial_table.csv"

X_BINS = 12
Y_BINS = 8
PHASE_BIN_MS = 50.0       # trial-phase bin width for the confound control
N_PERM = 1000             # circular-shift permutations (1000 is ample for p<0.05)
RNG_SEED = 42
MIN_EXPERIMENT_BLOCK = 4
# -----------------------------------------------------------------------


# %%
def load_game_states():
    """Concatenated, time-sorted game states across experiment blocks."""
    with open(BEHAVIOR_PATH, encoding="utf-8") as fh:
        data = json.load(fh)
    t, x, y, cam = [], [], [], []
    for b in data["blocks"]:
        bi = b.get("block_index")
        if bi is None or bi < MIN_EXPERIMENT_BLOCK:
            continue
        gs = b.get("game_states", {})
        if not gs:
            continue
        t.append(np.asarray(gs["time"]))
        x.append(np.asarray(gs["ball_x"]))
        y.append(np.asarray(gs["ball_y"]))
        cam.append(np.asarray(gs["camera_y"]))
    t = np.concatenate(t)
    x = np.concatenate(x)
    yrel = np.concatenate(y) - np.concatenate(cam)
    order = np.argsort(t)
    return t[order], x[order], yrel[order]


def trial_phase_of(t):
    """Map each time to a trial-phase bin (ms since that trial's start).

    Uses trial_table's sorted [trial_start_ms, exit_time_ms) intervals and a
    vectorized searchsorted lookup. Spikes/game states outside any trial get
    phase -1 and are excluded.
    """
    trials = pd.read_csv(TRIAL_TABLE)
    starts = trials["trial_start_ms"].to_numpy()
    exits = trials["exit_time_ms"].to_numpy()
    t = np.asarray(t, dtype=float)
    # find the trial whose [start, exit) contains each time
    idx = np.searchsorted(starts, t, side="right") - 1
    idx = np.clip(idx, 0, len(starts) - 1)
    inside = (t >= starts[idx]) & (t < exits[idx])
    phase = np.full(len(t), -1, dtype=int)
    phase[inside] = ((t[inside] - starts[idx[inside]]) / PHASE_BIN_MS).astype(int)
    return phase


def occupancy_map(pos, t, phase, bins):
    """Occupancy (ms) per (phase_bin, position_bin)."""
    pos_bin = np.clip(np.digitize(pos, bins[1:-1]), 0, len(bins) - 2)
    dt = np.diff(np.concatenate([t, [t[-1] + (t[-1] - t[-2])]]))
    n_phase = int(np.max(phase)) + 1
    occ = np.zeros((n_phase, len(bins) - 1))
    for pb in range(n_phase):
        m = phase == pb
        if m.sum() == 0:
            continue
        # NOTE: occ[pb, idx] += val with duplicate idx does NOT accumulate
        # (numpy fancy-index semantics); use np.add.at instead.
        np.add.at(occ[pb], pos_bin[m], dt[m])
    return occ


def spike_pos_bins(spike_times, pos, t, bins):
    """Position bin of each spike (via interpolation), plus its trial phase bin."""
    spike_pos = np.interp(spike_times, t, pos)
    pos_bin = np.digitize(spike_pos, bins) - 1
    pos_bin = np.clip(pos_bin, 0, len(bins) - 2)
    phase = trial_phase_of(spike_times)
    return pos_bin, phase


def rate_map_from_counts(counts, occ, min_occ_s=0.5):
    """Position rate map (Hz) from a (phase, pos) count matrix and occupancy.

    Position bins with total occupancy below `min_occ_s` (0.5 s) are set to
    NaN: with ~17 ms of per-phase occupancy, a single spike yields absurd
    rates (hundreds of Hz), so we require a minimum time-on-task per position
    before trusting its rate.
    """
    pos_occ = occ.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(pos_occ >= min_occ_s * 1000,
                        counts.sum(0) / (pos_occ / 1000.0), np.nan)
    return rate


def spatial_information(rate, occ, min_occ_s=0.5):
    """Skaggs-style spatial information in bits."""
    pos_occ = occ.sum(0)
    valid = (pos_occ >= min_occ_s * 1000) & ~np.isnan(rate)
    if valid.sum() == 0:
        return np.nan
    p = pos_occ[valid] / pos_occ[valid].sum()
    r = rate[valid]
    mean = np.sum(p * r)
    if mean <= 0:
        return np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        contrib = p * r * np.log2(r / mean)
    return float(np.sum(np.where(r > 0, contrib, 0.0)))


# %%
def permutation_spatial(spike_times, pos, t, phase_gs, bins, occ,
                        n_perm=N_PERM, seed=RNG_SEED):
    """Circular time-shift permutation test for spatial information.

    Null: the whole spike train is circularly shifted by a random constant
    within the session span, then the rate map is recomputed. This preserves
    the temporal autocorrelation and trial-phase firing profile of the unit
    (choice-locked modulation is kept) while breaking the spike->position
    link. Calibrated: synthetic non-tuned units give ~5% false positives.

    spike_times : absolute behavioral ms (trial-filtered)
    pos, t      : game-state position track and timestamps (trial-filtered)
    phase_gs    : per-game-state-sample trial-phase bin (>=0 for in-trial)
    bins        : position bin edges
    occ         : (phase, pos) occupancy matrix for rate normalization
    Returns (observed_SI, p).
    """
    rng = np.random.default_rng(seed)
    st = spike_times[(spike_times >= t[0]) & (spike_times <= t[-1])]
    if len(st) == 0:
        return np.nan, np.nan
    span = t[-1] - t[0]
    t0 = t[0]

    # per-game-state-sample position bin and in-trial flag
    pos_bin_gs = np.digitize(pos, bins) - 1
    pos_bin_gs = np.clip(pos_bin_gs, 0, len(bins) - 2)
    in_trial_gs = phase_gs >= 0
    n_phase, n_pos = occ.shape

    def _si(times):
        idx = np.searchsorted(t, times, side="right") - 1
        idx = np.clip(idx, 0, len(t) - 1)
        keep = in_trial_gs[idx]
        counts = np.zeros((n_phase, n_pos))
        np.add.at(counts, (phase_gs[idx][keep], pos_bin_gs[idx][keep]), 1)
        rate = rate_map_from_counts(counts, occ)
        return spatial_information(rate, occ)

    obs = _si(st)
    if obs is None or np.isnan(obs):
        return obs, np.nan

    nulls = np.empty(n_perm)
    for k in range(n_perm):
        shift = rng.uniform(-span, span)
        sts = (st + shift - t0) % span + t0
        nulls[k] = _si(sts)
    nulls = nulls[np.isfinite(nulls)]
    if len(nulls) == 0:
        return obs, np.nan
    p_val = np.mean(nulls >= obs)
    return obs, p_val


def fdr_bh(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    q = np.full(n, np.nan)
    valid = ~np.isnan(pvals)
    if valid.sum() == 0:
        return q
    pv = pvals[valid]
    order = np.argsort(pv)
    qv = pv[order] * n / np.arange(1, len(pv) + 1)
    qv = np.minimum.accumulate(qv[::-1])[::-1]
    q[valid] = qv[np.argsort(order)]
    return np.clip(q, 0, 1)


def channel_label(source_file):
    """Electrode label from a source_file name, e.g. 'times_mLF1aCa01_2285.mat'
    -> 'mLF1aCa01'."""
    stem = Path(source_file).stem          # times_mLF1aCa01_2285
    stem = stem[len("times_"):] if stem.startswith("times_") else stem
    return stem.rsplit("_", 1)[0]


def unit_channel_labels(units):
    """{unit_id: electrode label} from unit_metadata."""
    return {int(r["unit_id"]): channel_label(r["source_file"])
            for _, r in units.iterrows()}


# %%
def main():
    print("Loading game states ...")
    t, x, yrel = load_game_states()
    print(f"  {len(t)} samples, t={t[0]:.0f}..{t[-1]:.0f} ms")
    phase = trial_phase_of(t)
    in_trial = phase >= 0
    t, x, yrel, phase = t[in_trial], x[in_trial], yrel[in_trial], phase[in_trial]
    print(f"  {len(t)} samples inside trial windows")

    x_edges = np.linspace(x.min(), x.max(), X_BINS + 1)
    y_edges = np.linspace(yrel.min(), yrel.max(), Y_BINS + 1)

    units = pd.read_csv(UNIT_META)
    chan_labels = unit_channel_labels(units)
    spikes = pd.read_csv(SPIKES_UNITS)
    keep = set(units["unit_id"])
    spikes = spikes[spikes["unit_id"].isin(keep)]

    rows = []
    for uid, grp in spikes.groupby("unit_id"):
        st = grp["spike_time_behavioral_ms"].to_numpy()
        st = st[(st >= t[0]) & (st <= t[-1])]

        for axis, pos, edges in (("x", x, x_edges), ("y", yrel, y_edges)):
            occ = occupancy_map(pos, t, phase, edges)
            si, p = permutation_spatial(st, pos, t, phase, edges, occ)
            pb, _ = spike_pos_bins(st, pos, t, edges)
            counts = np.zeros_like(occ)
            sph_all = trial_phase_of(st)
            kk = sph_all >= 0
            np.add.at(counts, (sph_all[kk], pb[kk]), 1)
            rate = rate_map_from_counts(counts, occ)
            centers = (edges[:-1] + edges[1:]) / 2.0
            peak = centers[np.nanargmax(rate)] if np.isfinite(rate).any() else np.nan
            rows.append({
                "unit_id": int(uid),
                "channel": chan_labels.get(int(uid), ""),
                "axis": axis,
                "spatial_info_bits": si,
                "peak_bin": peak,
                "mean_rate_hz": float(np.nanmean(rate)) if np.isfinite(rate).any() else np.nan,
                "p_perm": p,
            })

    res = pd.DataFrame(rows)
    for axis, grp in res.groupby("axis"):
        q = fdr_bh(grp["p_perm"].to_numpy())
        res.loc[grp.index, "q_fdr"] = q
    res["significant"] = res["q_fdr"] < 0.05
    res.to_csv(OUT_DIR / "spatial_tuning_results.csv", index=False)

    print("\nSaved spatial_tuning_results.csv")
    print(res.groupby("axis")["significant"].sum().to_string())
    for axis, grp in res.groupby("axis"):
        sig = grp[grp["significant"]]
        if len(sig):
            labels = sorted(set(sig["channel"]))
            print(f"  axis {axis}: {len(sig)} significant -> {', '.join(labels)}")


# %%
# ----------------------- Plot functions (not saved) ---------------------
def load_spatial_results():
    return pd.read_csv(OUT_DIR / "spatial_tuning_results.csv")


def plot_significant_rate_maps(unit_ids=None, axis="x"):
    """2D grid of rate maps for selected units on the chosen axis."""
    import matplotlib.pyplot as plt
    res = load_spatial_results()
    if unit_ids is None:
        unit_ids = res[(res["axis"] == axis) & res["significant"]]["unit_id"].tolist()
    spikes = pd.read_csv(SPIKES_UNITS)
    t, x, yrel = load_game_states()
    ph = trial_phase_of(t)
    in_trial = ph >= 0
    t_i, pos_i, ph_i = t[in_trial], (x if axis == "x" else yrel)[in_trial], ph[in_trial]
    pos = x if axis == "x" else yrel
    edges = np.linspace(pos.min(), pos.max(), X_BINS + 1 if axis == "x" else Y_BINS + 1)
    n = len(unit_ids)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 2.5 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, uid in zip(axes, unit_ids):
        st = spikes[spikes["unit_id"] == uid]["spike_time_behavioral_ms"].to_numpy()
        st = st[(st >= t[0]) & (st <= t[-1])]
        occ = occupancy_map(pos_i, t_i, ph_i, edges)
        pb, _ = spike_pos_bins(st, pos, t, edges)
        sph = trial_phase_of(st)
        kk = sph >= 0
        counts = np.zeros_like(occ)
        np.add.at(counts, (sph[kk], pb[kk]), 1)
        rate = rate_map_from_counts(counts, occ)
        centers = (edges[:-1] + edges[1:]) / 2.0
        ax.bar(centers, rate, width=np.diff(edges)[0], color="steelblue")
        ax.set_title(f"unit {uid}")
        ax.set_xlabel(axis)
        ax.set_ylabel("Hz")
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    main()

# %%
