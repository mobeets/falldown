# Spike–Trial Segmentation Data Structure

This file documents the outputs produced by `analysis/segment_trials.py` for the
YFZ EMU run
(`analysis/emu data/YFZ-2026-07-29T21-37-47-781Z-kdyd.json` +
`spikesort_results/cluster_viewer_results/spikes.mat`). Read this before doing
any analysis on the segmented spike data so you don't have to reverse-engineer
the structure.

All outputs live in `analysis/spike_data_alignment_output/`.

## Recording site

All 122 spike-sorted units come from bilateral **mesial temporal lobe depth
electrodes** (Blackrock NSP, 30 kHz). The 64 channels map to 8 leads × 8
contacts, named `m{lead}{elec}` in the NS5 header / `times_*.mat` files:

| Leads | Region |
|---|---|
| `LF1aCa`, `RF1aCa` | Cornu ammonis (CA fields of the hippocampus) |
| `LT2aA`, `RT2aA` | Amygdala |
| `LT2bHa`, `RT2bHa` | Anterior hippocampus (head) |
| `LT2cHB`, `RT2cHb` | Hippocampal body |

`unit_metadata.csv` `channel` is the 1-based NS5 channel. Channel→lead mapping
(from the NS5 header labels): 1–8 `LF1aCa`, 9–16 `LT2cHB`, 17–24 `LT2aA`,
25–32 `LT2bHa`, 33–40 `RF1aCa`, 41–48 `RT2cHb`, 49–56 `RT2aA`, 57–64
`RT2bHa`; within a lead, contact 01–08 maps to channel `start + contact − 1`.

---

## What a "trial" means here

A trial is one **1-2-1 hole sequence**:

```
[1 hole]  →  [2 holes]  →  [1 hole]
 entry        choice       exit/goal
```

- **entry** = the pass-through of the single-hole level (starts the sequence)
- **choice** = the pass-through of the 2-hole level — **this is t = 0 for all spike data**
- **exit** = the pass-through of the final single-hole level (ends the sequence)

Trial boundaries:
- trial N **starts** when the participant passes the **exit hole of trial N−1**
  (the first trial starts at the first entry event; there is no spawn marker
  in the JSON)
- trial N **ends** when the participant passes the **exit hole of trial N**

A "trial" in the *behavioral JSON* is **one level** (one pass-through event).
The JSON's `trial_index` counts levels, not sequences. `segment_trials.py`
rebuilds the sequences from the canonical `block_config.levels` rhythm.

## Source files

| File | Role |
|---|---|
| `analysis/segment_trials.py` | Produces everything below |
| `spikes.mat` | Sorted unit-level spike times (sparse, 147 unit rows, 30 kHz) |
| `neuron_data.json` | Per-unit QC (firing rate, ISI, waveform) |
| behavioral JSON | Trial timing, hole positions, choices |

## Outputs

### `trial_table.csv` — one row per trial (910 rows)

| Column | Meaning |
|---|---|
| `trial_id` | 0-based sequence index (this run: 0–909) |
| `block_index` | Behavioral block (≥ 4; blocks 0–3 are practice/instruction, excluded) |
| `sequence_index` | Position of the sequence within the block |
| `trial_start_ms` | Pass-through time of previous trial's exit hole (behavioral clock, ms) |
| `entry_time_ms` | Pass-through of the 1-hole entry level |
| `choice_time_ms` | Pass-through of the 2-hole choice level (**t = 0**) |
| `exit_time_ms` | Pass-through of the 1-hole exit level |
| `choice_hole` | The hole index the participant chose on the choice level |
| `hole_locations` | JSON string of the choice level's two hole indices |

### `unit_metadata.csv` — one row per unit (137 rows, after QC)

| Column | Meaning |
|---|---|
| `unit_id` | Row index in `spikes.mat` (0-based) — the stable unit identifier |
| `channel` | Recording channel |
| `cluster_id` | Sorted cluster id within the channel (1–8) |
| `source_file` | Original `times_*.mat` file the unit came from |
| `firing_rate_hz` | Firing rate from `neuron_data.json` (units < 0.1 Hz dropped) |

### `segmented_spikes_raw.pkl` — lossless per-trial spike times

A pickled dict:

```python
raw = {(unit_id, trial_id): np.ndarray}   # float64, ms relative to choice (t=0)
```

- Keys are `(unit_id, trial_id)`; `trial_id` is the `trial_table` index.
- Values are the spike times **relative to that trial's `choice_time_ms`**,
  i.e. choice = 0, positive = after choice, negative = before.
- Only non-empty (unit, trial) pairs appear (102,107 of them).
- 1,521,368 spikes total (this run).

### `segmented_spikes_binned.npz` — binned counts + embedded metadata

```
binned          (137, 910, 160)  float64   counts per (unit, trial, bin)
unit_ids        (137,)   int64
trial_ids       (910,)   int64
bin_centers     (160,)   float64  bin center time in ms, relative to choice
trial_table     (910,)   numpy.record   same columns as trial_table.csv
unit_metadata   (137,)   numpy.record   same columns as unit_metadata.csv
window_ms       [-2000., 2000.]
bin_width_ms    25.0
mode            "truncated"
```

**Axis meaning:** `binned[u, t, b]` = spike count of unit `unit_ids[u]` in
trial `trial_ids[t]`, bin centered at `bin_centers[b]` ms relative to that
trial's choice. Bin 0 is at −1987.5 ms (first bin), t=0 sits at bin index 80.

**NaN semantics (important):** a NaN entry means the bin is **outside that
trial's actual span** (there is no time coverage there). Bins inside the
trial's window always hold a real count, including 0. A whole trial row is
never all-NaN (minimum coverage ≈ 21% of the window).

**Load it like this** (embedded tables are object arrays because of string
columns, so `allow_pickle=True` is required):

```python
import numpy as np
z = np.load("analysis/spike_data_alignment_output/segmented_spikes_binned.npz",
            allow_pickle=True)
binned = z["binned"]            # (137, 910, 160)
unit_ids = z["unit_ids"]        # axis-0 index
trial_ids = z["trial_ids"]      # axis-1 index
bin_centers = z["bin_centers"]  # axis-2 time (ms)
trials = z["trial_table"]       # records; use trials[z["trial_ids"][t]]
units  = z["unit_metadata"]
```

Example lookups:

```python
# Choice time of trial 5:
choice_ms = trials["choice_time_ms"][5]

# Spike counts of unit 3 in trial 5, as a function of time:
counts = binned[np.where(unit_ids == 3)[0][0], 5, :]

# Which hole was chosen in trial 5:
chosen = trials["choice_hole"][5]   # 0–11
```

## Alignment & preprocessing rules (how it was built)

- **Clock**: all times are on the **behavioral clock** (ms). Spike times were
  converted from the NS5 recording clock to behavioral time in
  `spike_data_alignment.py` using a photodiode-flash ↔ JSON-event DTW offset
  model (offset ≈ 126.9 s, drift < 100 ms, median fit residual ≈ 4.5 ms).
- **Dedup**: behavioral blocks/trials are deduplicated by
  `(block_index, trial_index)`, keeping the **last** occurrence (handles the
  block-8 snapshot split and replayed "death" trials).
- **Blocks 0–3** (practice/instruction) are excluded.
- **Unit QC**: units with `firing_rate_hz < 0.5` are dropped (122 of 142
  spike-bearing units remain). The floor is deliberately conservative: below
  0.5 Hz there are so few spikes that per-trial PSTHs are statistically
  unreliable. Firing rates come from `neuron_data.json` (matched by
  filename + cluster_id). NOTE: this is a firing-rate floor only — it does
  **not** screen on ISI violations or waveform stability (those fields exist
  in `neuron_data.json` but are not used here).
- **Window**: `[-2000, +2000]` ms around choice time.
- **Mode = truncated**: each trial's window is clipped to `[trial_start, exit]`
  so no spike is double-counted across adjacent trials. (The code also
  supports `mode="naive"`, a fixed window with overlap, via the `segment_trials`
  function's `mode` argument; the saved `.npz` uses `truncated`.)

## Key numbers (this run)

- 910 trials, 122 units, 160 bins, 25 ms bin width
- 1,514,339 segmented spikes
- Median trial duration ≈ 2250 ms; median window coverage ≈ 57%
- Raw (`raw.pkl`) and binned (`npz`) spike totals match exactly

---

# Neural analysis pipeline (selectivity, spatial tuning)

Downstream analyses of the segmented data. All run with
`C:\Users\manik\AppData\Local\Programs\Python\Python311\python.exe` and write
**CSV results only** (no images; each script ships `plot_*` functions you can
call yourself in Jupyter).

## `classify_trials.py` → `trial_labels.csv`

Rebuilds per-trial geometry and condition labels for the 910 trials:

| Column | Meaning |
|---|---|
| `entry_hole`, `goal_hole` | Single-hole positions of level 1 and level 3, from `block_config.levels` (`s_local = sequence_index % 35`) |
| `greedy_cost_L/R` | `\|entry − hole\|` for each of the 2 choice holes |
| `planning_cost_L/R` | `\|entry − hole\| + \|hole − goal\|` for each choice hole |
| `greedy_optimal_hole`, `planning_optimal_hole` | Hole minimizing each cost |
| `agree` | `greedy_optimal == planning_optimal` |
| `condition` | `planning` (125), `greedy` (306), `agree_optimal` (412), `lapse` (67) |

**Death definition:** a "death" is a genuine death moment detected from the
ball trajectory — the ball falls to the bottom of the screen and the camera
catches up (`ball.y − cameraY → 0`). `find_deaths()` detects these episodes in
`game_states`. This session has **one** death (~758.9 s in block 8); the 10
"empty trials" in the raw JSON are levels that scrolled past during that single
fatal fall, not 10 separate deaths. See `death_times.csv`.

## `neural_selectivity.py` → `selectivity_results.csv`

For each of 122 units, per-trial firing rate in two windows around choice time
(reported separately): **pre** `[-1000, 0]` ms and **post** `[0, +1000]` ms.
Four contrasts:

1. `planning_vs_agree_optimal` — conflict-chose-planning vs no-conflict-optimal
2. `planning_vs_greedy` — conflict, chose planning vs conflict, chose greedy
3. `lapse_vs_agree_optimal` — agree, chose worst vs agree, chose best
4. `death_vs_normal` — death-anchored firing vs all choice-anchored firing

**Effect size:** modulation index `MI = (mean_A − mean_B)/(mean_A + mean_B)`
(−1…+1, 0 = no preference).

**Significance:** two-sided **permutation test** — the pooled condition labels
are shuffled 5,000 times, MI recomputed each time, and the observed MI is
placed in the resulting null distribution; p = fraction of `|null MI| ≥
|observed MI|`. Shuffle indexes are precomputed once per contrast, so all 122
units share the same permutation scheme (deterministic, `seed=42`).

**Multiple comparisons:** p-values FDR-corrected across all 122 units with the
**Benjamini–Hochberg** procedure. `significant = q_fdr < 0.05`.

## `spatial_tuning.py` → `spatial_tuning_results.csv`

For each unit, occupancy-normalized firing-rate maps over:
- `axis='x'` — `ball_x`, 12 bins (horizontal position)
- `axis='y'` — `ball_y − camera_y`, 8 bins (on-screen vertical position)

Spikes are assigned to position bins by interpolating `ball_x`/`ball_y−camera_y`
onto the game-state timestamps (~60 Hz). Occupancy = time each sample owns
(interval to the next sample). Rate map = `counts / (occupancy / 1000)` Hz,
with position bins below 0.5 s of total occupancy set to NaN (they would
otherwise produce absurd rates from a single spike).

**Statistic:** Skaggs spatial information (bits):
`Σ pᵢ rᵢ log₂(rᵢ / mean r)`, `pᵢ` = time fraction in bin.

**Significance:** circular **time-shift permutation** — the whole spike train
is circularly shifted by a random constant within the session span (1,000
shifts), the rate map recomputed, and spatial information recomputed. This
preserves the unit's temporal autocorrelation and trial-phase firing profile
(choice-locked modulation is kept) while breaking the spike→position link.
**Calibration verified:** synthetic non-tuned units give ~5% false positives,
so a significant result means genuine position selectivity above and beyond
trial-phase-locked firing. FDR (BH) correction across units per axis.

**Note on the y axis:** because `ball_y − camera_y` is correlated with trial
phase (choices occur at a concentrated on-screen y), the y-axis is the more
conservative test — significant y-tuning means position selectivity survives
the phase control. The x axis (`ball_x`) has no such phase confound.

### Known caveats (read before trusting results)

1. **Death analysis.** The raw JSON lists 10 "death trials" (block 8, trials
   54–63, empty events), but the trajectory shows these are **one** death
   event: the ball fell to the bottom of the screen once (~758.9 s) and 10
   levels scrolled past during the fatal fall. `classify_trials.py` now detects
   genuine death moments from `game_states` (`ball.y − cameraY` approaching 0
   with the ball frozen at the bottom). With **N=1 death** in this session,
   `neural_selectivity.py` **skips** the `death_vs_normal` permutation contrast
   (needs N≥5) and instead writes a descriptive-only `death_locked_rates.csv`.
   A death-attuned analysis is not statistically feasible on this session.
2. **Spatial results are now calibration-checked.** The earlier "114/122
   y-significant" result was a bug: (a) occupancy was computed with numpy
   fancy-index accumulation that silently dropped ~99% of time, and (b) the
   within-phase multinomial null was structurally biased. Both are fixed; the
   current circular time-shift null is verified to give ~5% false positives on
   synthetic non-tuned data.
3. **Lapse group is small** (67 trials) → low power.

