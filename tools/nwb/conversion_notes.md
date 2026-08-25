# NWB Conversion Notes — Falldown Experiment

This document describes how falldown behavioral JSON session files are mapped to NWB 2.0 (Neurodata Without Borders) format. Use this as a reference when loading or analyzing `.nwb` files.

---

## Quick Start — Reading an NWB File

```python
from pynwb import NWBHDF5IO

with NWBHDF5IO("session.nwb", "r") as io:
    nwb = io.read()

    # Basic metadata
    print(nwb.session_description)
    print(nwb.subject.subject_id)
    print(nwb.session_start_time)

    # Trials table (pandas DataFrame)
    df = nwb.trials.to_dataframe()

    # Ball position (Nx2 array, pixels)
    pos = nwb.processing["behavior"]["ball_position"]["ball_position"].data[:]

    # Velocity (1D arrays)
    vx = nwb.processing["behavior"]["BehavioralTimeSeries"]["ball_velocity_x"].data[:]

    # Directional input events
    input_table = nwb.get_events_table("directional_input").to_dataframe()
```

---

## How to Read Each Component

All data access follows a single pattern: `nwb` is the root object. Everything branches from it. Here is an map of the exact access paths.

### 1. Session-level metadata

`nwb` attributes set at the file level.

| What | How to read | Type |
|---|---|---|
| Subject ID | `nwb.subject.subject_id` | string |
| Session start | `nwb.session_start_time` | `datetime` (UTC) |
| Description | `nwb.session_description` | string |
| Experiment description | `nwb.experiment_description` | string |
| Institution | `nwb.institution` | string (`"Baylor College of Medicine"`) |
| Lab | `nwb.lab` | string (`"Hennig Lab"`) |
| Experimenter | `nwb.experimenter` | string (`"Jay Hennig"`) |
| Session ID | `nwb.session_id` | string (same as subject_id) |
| Keywords | `nwb.keywords` | list of strings |
| Identifier | `nwb.identifier` | string (unique per file) |
| Notes | `nwb.notes` | JSON string — contains game params, block configs, game info |

The `notes` field is a JSON-encoded string with three keys:

```python
import json
meta = json.loads(nwb.notes)
meta["params"]          # game physics parameters (scrollSpeed, gravity, FPS, etc.)
meta["game_info"]       # window dimensions, level geometry (may be absent)
meta["block_configs"]   # full experiment block definitions, keyed by block index (may be absent)
```

### 2. Continuous data — ball and camera tracking

`nwb.processing["behavior"]` — the `behavior` processing module. Contains ordered data logged every frame at ~60 Hz.

All continuous signals share identical timestamps (the union of all blocks' `game_states.time` arrays, merged, sorted, and converted to seconds). Each sub-component id stored inside the module.

**Ball position** — the (x,y) position of the ball every frame.

```python
nwb.processing["behavior"]["ball_position"]["ball_position"]
# .data → shape (N, 2) float64, unit = "pixels"
# .timestamps → shape (N,) float64, seconds since session start
# .reference_frame → "(0,0) is top-left of the game canvas. x increases right, y increases down."
```

**Ball velocity** — computed by the game engine (not derived from position). Two separate 1D arrays sharing the same timestamps as position.

```python
nwb.processing["behavior"]["BehavioralTimeSeries"]["ball_velocity_x"]
# .data → shape (N,) float64, unit = "pixels/second"

nwb.processing["behavior"]["BehavioralTimeSeries"]["ball_velocity_y"]
# .data → shape (N,) float64, unit = "pixels/second"
```

**Camera** — the vertical offset of the scrolling camera and the scroll speed in pixels per second.

```python
nwb.processing["behavior"]["BehavioralTimeSeries"]["camera_offset_y"]
# .data → shape (N,) float64, unit = "pixels"

nwb.processing["behavior"]["BehavioralTimeSeries"]["scroll_speed"]
# .data → shape (N,) float64, unit = "pixels/second"
```

### 3. Directional input — player key presses

`nwb.events` — NWB events list. Player directional input is logged only when it changes (not every frame).

```python
nwb.get_events_table("directional_input").to_dataframe()
# columns: timestamp (seconds), direction (int)
# direction: -1 = moving left, 0 = released, 1 = moving right
```

### 4. Trials — per-level event data

`nwb.trials` — one primary row per completed trial (a trial corresponds to one level the ball passed through). Rows where the ball died (no pass-through event) are not included.

```python
df = nwb.trials.to_dataframe()
```

| Column | Type | Description |
|---|---|---|
| `start_time` | float | Timestamp of the pass-through event (seconds) |
| `stop_time` | float | start_time + 0.001 (instantaneous events) |
| `block_index` | int | Which block the trial belongs to (0-based, config index) |
| `trial_index` | int | Trial number within the block (0-based) |
| `hole_locations` | string | JSON-encoded list of hole segment indices (0-11), e.g. `"[3]"` or `"[2, 10]"` |
| `hole_chosen` | int | The segment index the ball actually passed through. `NaN` if unknown |
| `is_decision_trial` | bool | `True` if the trial had 2 holes (a choice), `False` if 1 hole (no choice) |
| `camera_mode` | int | Camera mode at event time: `0` = follow mode, `1` = drift mode |
| `is_mode_switch` | bool | Whether the camera mode switched on this level |
| `event_level_y` | float | Vertical pixel position of the level |
| `event_ball_x` | float | Ball x position at pass-through (pixels) |
| `event_ball_y` | float | Ball y position at pass-through (pixels) |
| `event_camera_y` | float | Camera y offset at pass-through (pixels) |
| `event_scroll_speed` | float | Scroll speed at pass-through (pixels/second) |

**Parsing `hole_locations`** — it is stored as a JSON string because NWB columns cannot hold variable-length arrays.

```python
import json
df["hole_locations_parsed"] = df["hole_locations"].apply(json.loads)
```

### 5. Epochs — block intervals

`nwb.epochs` — one epoch per block. Each epoch spans from the block's first frame to its last frame.

```python
nwb.epochs.to_dataframe()
# columns: start_time, stop_time, tags
# tags example: ["block_0"], ["block_1"], ...
```

Block epoch numbering is sequential (0, 1, 2, ...) based on iteration order through the blocks array. This may differ from the `block_index` in the trials table when snapshots of the same block appear at different times.

### 6. Invalid times — pause intervals

`nwb.invalid_times` — intervals when the game was paused. Data during these intervals should be excluded from analysis.

```python
nwb.invalid_times.to_dataframe()
# columns: start_time, stop_time
```

---

## Time Handling

### Source timestamps

All raw timestamps in the JSON data come from the browser's `performance.now()` — a monotonic clock reporting milliseconds since page load. These are **not** wall-clock times.

### Conversion to NWB

The conversion script:

1. Finds the earliest timestamp across all data sources in the session (first frame, first input, first block start, or first unpause) — this becomes `t = 0`.
2. Converts all raw millisecond values to **seconds, relative to that session start**:

   ```
   t_nwb = (t_performance_now_ms - session_start_ms) / 1000.0
   ```

3. Sets `nwb.session_start_time` to a real-world `datetime` extracted from the filename's ISO timestamp, or (if unavailable) the file's modification time. This datetime is **only metadata** — it is not linked to the relative timestamps inside the file.

### What this means for analysis

- All times inside the NWB file are in **seconds**, measured from the first recorded event in that session.
- To compute reaction time (RT): subtract the `start_time` of consecutive trial rows.

  ```python
  df = nwb.trials.to_dataframe()
  df["rt"] = df["start_time"].diff()
  ```

  The first trial's RT will be `NaN` (no previous trial).
- Pauses are marked in `nwb.invalid_times`. If you need trial RTs that exclude pause durations, subtract pause intervals that fall between consecutive trials.
- **IMPORTANT**: Because timestamps are offset to start at 0 for each file, you cannot directly compare absolute timestamps across different NWB files. Each file has its own `t = 0`.

### Multi-snapshot sessions

The JSON data files contain redundant snapshots — the game saves the entire experiment state after every block transition. This means a session with N blocks produces N snapshots in the JSON, each containing all prior blocks plus the current one.

The NWB conversion includes **all blocks present in the final (latest) snapshot only** — because each snapshot is a complete, self-contained JSON file. If you are converting a directory containing multiple snapshots from the same participant (e.g., `EA4EE5B9...-session1.json`, `EA4EE5B9...-session2.json`), each snapshot becomes a separate `.nwb` file. The merge/dedup logic from `analysis/checking_data_validity.py` is **not** applied during conversion — each NWB file reflects exactly one serialized state.

To get a single clean session from multiple snapshots, either:
- Use the `data/cloud_study/` pre-merged files as input, or
- Run `merge_participant_files()` from `analysis/checking_data_validity.py` on the raw logs first, then convert the merged result.

---

## Data Coverage — What IS and IS NOT Included

### Included

- Ball position and velocity (every frame)
- Camera offset and scroll speed (every frame)
- Player directional input (on-change events)
- Per-trial decision data (hole offered, hole chosen, camera state at pass-through)
- Block epoch intervals
- Pause intervals (invalid times)
- Experiment parameters and block configs (in `nwb.notes` as JSON)
- Subject ID

### Not included (stays in raw JSON only)

- **`block_config.params.instructions`** and **`pre_instructions`** — the instructional text shown to participants between blocks. These are in `nwb.notes` inside `block_configs` but are not structured into the NWB trial/epoch model.
- **Trials with no events** — trials where the ball died before passing through a level. The raw JSON has these (`trial.events = []`), but they produce no NWB trial row.
- **`trial_block.pause_times`** — the full pause-start and pause-end arrays are converted to `nwb.invalid_times` intervals, but the raw arrays are not preserved verbatim. Only the paired (start, end) intervals survive.
- **Older-format fields** — legacy JSON keys like `trial["holeUsed"]` (top-level, pre-`events[]` format) and `trial["timePassedThru"]`. The converter reads from the `events[]` array only. Files in the old format should be migrated first.

---

## Derived Measures You Can Compute from the NWB

These are not stored directly but can be calculated from what is stored.

| Measure | How to compute |
|---|---|
| **Reaction time (RT)** | `df["rt"] = df["start_time"].diff()` — time between consecutive trial pass-through events |
| **1-step (greedy) distance** | For decision trial `i`: `abs(hole_locs[0] - df.hole_chosen[i-1])` and `abs(hole_locs[1] - df.hole_chosen[i-1])` |
| **2-step (planning) distance** | Greedy distance + distance from each choice hole to the next trial's hole |
| **Conflict** | `abs(abs(dist_L1 - dist_R1) - abs(dist_L2 - dist_R2))` |
| **Choice** | On decision trials, which hole was chosen: `df["chose_left"] = df["hole_chosen"] == hole_locs[0]` |
| **Ball path** | The full trajectory is in `ball_position.data` at frame rate |
| **Input timing** | Directional input events are timestamped; align with trial `start_time` to see when the participant pressed left/right relative to level transitions |

---

## File Naming and Organization

- **Input**: One JSON file → one NWB file. The NWB filename matches the JSON stem: `session.json` → `session.nwb`.
- **Default output**: A subdirectory `nwb/` next to the input files, e.g. converting `logs/` produces `logs/nwb/*.nwb`.
- **Size**: An NWB file is roughly 10-20% larger than the source JSON due to HDF5 overhead and binary float storage. A typical 200K-trial session produces a 4-8 MB NWB file.

---

## Dependencies and Requirements

- **Python 3.9+**
- **pynwb >= 2.0** (tested on 4.0)
- **numpy**
- **h5py** (installed automatically with pynwb)

The NWB files are valid HDF5 and can be opened with any HDF5 tool (`h5py`, HDFView, MATLAB, etc.). The `pynwb` library provides the schema-aware API described above.
