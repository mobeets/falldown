# Data Schema — Falldown Experiment

This document describes every JSON data file in the falldown project. It is written so that a coding agent (Claude, Copilot, etc.) can understand the full data structure without reading the actual JSON files. Where applicable, commands to inspect real values are included — feel free to run them when you need ground-truth verification.

---

## File Inventory

### Behavioral session data (the data you analyze)

| Location | Description |
|---|---|
| `logs/*.json` | Raw per-session snapshots. One file = one serialized experiment state saved at a block boundary. Contains duplicated prior blocks (see Multi-Snapshot note below). |
| `data/cloud_study/*.json` | Merged, deduplicated per-participant files from batch 1 (June 2026). |
| `data/cloud_study/v2/*.json` | Merged/deduplicated files from batch 2 (July 2026). |
| `data/emu/*.json` | Pilot/local-testing data (smaller session files). |

To see the exact files available:

```bash
ls logs/*.json | head -5
ls "data/cloud_study"/*.json | head -5
ls "data/cloud_study/v2"/*.json | head -5
```

### Experiment configs (what the participant was asked to do)

| Location | Description |
|---|---|
| `app/configs/*.json` | Game physics params and pre-generated level/block definitions. |

### Level generation outputs (inputs to config creation)

| Location | Description |
|---|---|
| `data/generated_levels/trials_new.json` | Discrete-hole trial structures. |
| `data/generated_levels/trials_output.json` | Continuous-position trial structures with cost metadata. |
| `data/generated_levels/stripped_trials_output.json` | Just hole positions, flattened. |

---

## Behavioral JSON — Full Schema

Each behavioral JSON file is the serialized `Experiment` JavaScript object (`Experiment.toJSON()`), which recursively serializes all child objects. A single file represents one saved state of a session. The top-level type is `object`.

### Top-level fields

```
{
  "subject_id":        string,
  "params_path":       string,
  "params":            object (see Game Parameters below),
  "experiment_path":   string,
  "block_configs":     object (see Experiment Config below),
  "block_index":       int,
  "block_count":       int,
  "blocks":            array of Block objects,
  "gameInfo":          object (may be absent, see Game Info below)
}
```

| Field | Type | Description |
|---|---|---|
| `subject_id` | string | Participant identifier. For cloud studies: `{participantId}-{assignmentId}-{projectId}`. For local: `"YFX"`, `"RAH"`, etc. `"unknown"` for early test runs. |
| `params_path` | string | Path to the params config, e.g. `"configs/cloudresearch_params.json"` |
| `params` | object | Game physics and display parameters (see below) |
| `experiment_path` | string | Path to the block/level config, e.g. `"configs/short_trials_experiment-7-10.json"` |
| `block_configs` | array of BlockConfig | Pre-loaded experiment block definitions from the config file. Same object loaded from `experiment_path`. |
| `block_index` | int | Index of the most recently started block (0-based into `block_configs` array) |
| `block_count` | int | Monotonically increasing counter of total blocks played (includes restarts) |
| `blocks` | array of Block | All blocks played so far (see Block below) |
| `gameInfo` | object | Game rendering dimensions (may be absent; see Game Info below) |

To inspect these fields on any real file:

```bash
python -c "
import json
with open('logs/<filename>.json') as f:
    d = json.load(f)
print('subject_id:', d['subject_id'])
print('experiment_path:', d['experiment_path'])
print('block_count:', d['block_count'])
print('number of block snapshots in this file:', len(d['blocks']))
print('top-level keys:', list(d.keys()))
"
```

### Block object

Each element in `blocks[]` is an object:

```
{
  "block_index":           int,
  "block_count":           int,
  "block_config":          BlockConfig object,
  "trial_index":           int,
  "last_trial_completed":  int,
  "start_time":            float,
  "pause_times":           { "starts": [float,...], "ends": [float,...] },
  "game_states":           FrameData object,
  "user_inputs":           InputData object,
  "trials":                array of Trial objects
}
```

| Field | Type | Description |
|---|---|---|
| `block_index` | int | Index of this block in `block_configs` (0-based). Stays constant on restart. |
| `block_count` | int | Ordinal counter for this block (1-based, incremented on every new-block event including restarts). |
| `block_config` | BlockConfig | The pre-defined config for this block (see below). |
| `trial_index` | int | Index of the last trial spawned from `block_config.levels` (0-based). |
| `last_trial_completed` | int | Number of trials the ball successfully passed through. `is_complete()` checks `last_trial_completed >= block_config.levels.length`. |
| `start_time` | float | `performance.now()` timestamp (ms) when the block started (or first unpaused). |
| `pause_times` | object | Pause event arrays (see below). |
| `game_states` | FrameData | Frame-by-frame ball and camera state (see below). |
| `user_inputs` | InputData | Directional input change events (see below). |
| `trials` | array of Trial | Trial objects spawned so far (see below). |

### BlockConfig object (inside `block_config` and `block_configs[]`)

```
{
  "params": {
    "instructions":      [string, ...],
    "pre_instructions":  [string, ...],
    "startCameraMode":   int (may be absent)
  },
  "levels": [ [int], [int,int], [int], ... ]
}
```

| Field | Type | Description |
|---|---|---|
| `params.instructions` | array of string | Text shown during block play. |
| `params.pre_instructions` | array of string | Text shown as an overlay before the block starts. |
| `params.startCameraMode` | int | Override for initial camera mode: `0` = follow, `1` = drift. Falls back to top-level `params.startCameraMode` if absent. |
| `levels` | array of array of int | Pre-generated hole locations for every level in this block. Each element is an array of segment indices (all in 0-11). Length 1 = no-choice level. Length 2 = decision level. |

### Trial object

```
{
  "index":           int,
  "block_index":     int,
  "hole_locations":  [int] or [int, int],
  "events":          array of Event objects
}
```

| Field | Type | Description |
|---|---|---|
| `index` | int | Trial number within the block (0-based). |
| `block_index` | int | Block index this trial belongs to. |
| `hole_locations` | array of int | Segment indices (0-11) of the holes in this level. Length 1 = no choice. Length 2 = decision trial. |
| `events` | array of Event | Decision events. Length 1 if the ball passed through. Length 0 if the ball died before passing through (no data for that trial). |

### Event object (inside `trial.events[]`)

Created by merging `Level.toJSON()` + `decisionEvent()` + `Trial.logEvent()`:

```
{
  "index":                  int,
  "levelY":                 float,
  "modeIndex":              int,
  "isModeSwitch":           bool,
  "holeUsed":               int,
  "ballTouched":            bool,
  "timeBallFirstTouched":   float,
  "cameraMode":             int,
  "ballX":                  float,
  "ballY":                  float,
  "cameraY":                float,
  "scrollSpeed":            float,
  "trial_index":            int,
  "hole_locations":         [int],
  "block_index":            int,
  "time":                   float
}
```

| Field | Type | Description |
|---|---|---|
| `index` | int | Level index (1-based sequential number within the block). |
| `levelY` | float | Y-coordinate (pixels) of this level in the game world. |
| `modeIndex` | int | Camera mode at event time: `0` = follow, `1` = drift. |
| `isModeSwitch` | bool | Whether a camera mode switch occurred on this level. |
| `holeUsed` | int | **The participant's choice** — segment index (0-11) the ball passed through. |
| `ballTouched` | bool | Whether the ball contacted this level's platform. |
| `timeBallFirstTouched` | float | `performance.now()` (ms) of first ball-platform contact. |
| `cameraMode` | int | Same as `modeIndex` (redundant). |
| `ballX` | float | Ball x position (pixels) at event time. |
| `ballY` | float | Ball y position (pixels) at event time. |
| `cameraY` | float | Camera y offset (pixels) at event time. |
| `scrollSpeed` | float | Current scroll speed (pixels/second) at event time. |
| `trial_index` | int | Trial index (redundant with parent Trial's `index`). |
| `hole_locations` | array of int | Hole locations (redundant with parent Trial). |
| `block_index` | int | Block index (redundant with parent Trial). |
| `time` | float | **Primary timestamp** — `performance.now()` (ms) when the ball passed through. This is the key timestamp for computing reaction times. |

### FrameData object (`game_states`)

Logged every frame (~60 Hz). Stored at the block level. All arrays are parallel (same index = same frame).

```
{
  "time":          [float, ...],
  "ball_x":        [float, ...],
  "ball_y":        [float, ...],
  "ball_vx":       [float, ...],
  "ball_vy":       [float, ...],
  "camera_y":      [float, ...],
  "scroll_speeds": [float, ...]
}
```

| Field | Type | Unit | Description |
|---|---|---|---|
| `time` | array of float | ms | `performance.now()` timestamp per frame |
| `ball_x` | array of float | pixels | Ball horizontal position |
| `ball_y` | array of float | pixels | Ball vertical position |
| `ball_vx` | array of float | pixels/sec | Ball horizontal velocity (as computed by game engine) |
| `ball_vy` | array of float | pixels/sec | Ball vertical velocity |
| `camera_y` | array of float | pixels | Camera vertical offset |
| `scroll_speeds` | array of float | pixels/sec | Current camera scroll speed in drift mode |

### InputData object (`user_inputs`)

Logged only when the player's directional input CHANGES (not every frame). Stored at the block level.

```
{
  "time":  [float, ...],
  "input": [int, ...]
}
```

| Field | Type | Description |
|---|---|---|
| `time` | array of float | `performance.now()` (ms) timestamps of input changes |
| `input` | array of int | Direction: `-1` = moving left, `0` = released (no key), `1` = moving right |

### Pause times

```
{
  "starts": [float, ...],
  "ends":   [float, ...]
}
```

Both arrays hold `performance.now()` millisecond timestamps. `starts[i]` and `ends[i]` form a pair — the i-th pause interval. Note: `ends` may have one extra entry at the beginning marking the initial unpause (to establish `start_time`).

---

## Game Parameters (`params` object)

The full params object with typical values from `configs/cloudresearch_params.json`:

| Field | Type | Typical | Description |
|---|---|---|---|
| `scrollSpeed` | float | 1.5 | Initial camera scroll speed in drift mode |
| `maxScrollSpeed` | float | 2.5 | Maximum camera scroll speed |
| `scrollSpeedSecsToMax` | int | 120 | Seconds to reach max scroll speed |
| `relativeGravity` | float | 0.75 | Gravity multiplier (scaled to level width) |
| `relativeBallAccel` | float | 0.6 | Ball horizontal acceleration from input |
| `maxBallAccelScale` | int | 10 | Cap on ball horizontal velocity |
| `friction` | float | 0.92 | Ball horizontal momentum friction per frame |
| `isMomentum` | bool | false | If true, input is acceleration; if false, input is velocity |
| `levelWidthProportion` | float | 0.8 | Fraction of window width used for the maze |
| `nSegments` | int | 12 | Number of segments (columns) per level — **holes are always 0-11** |
| `nLevelsVisible` | int | 7 | How many levels visible on screen |
| `levelHeight` | int | 10 | Vertical pixel height of each platform |
| `FPS` | int | 60 | Target frames per second |
| `photodiode.size` | int | 120 | Size of photodiode flash square |
| `startCameraMode` | int | 0 | Initial camera mode: `0` = follow, `1` = drift |
| `modeSwitchRates` | [float,float] | [0, 0] | Probabilities per level of mode switch for [follow, drift] |
| `minLevelsPerMode` | int | 10 | Minimum levels between mode switches |
| `modeRectColors` | [string,string] | ["gray","gray"] | Colors for platform segments in [follow, drift] |
| `redirectUrl` | string | cloudresearch URL | Where to send the participant on completion |
| `isCloudStudy` | bool | true/false | Whether this is a cloud-based study |
| `showFewerLevels` | bool | false | Whether to render fewer future levels |

To see the exact params for any file:

```bash
python -c "
import json
with open('logs/<filename>.json') as f:
    d = json.load(f)
print(json.dumps(d['params'], indent=2))
"
```

---

## Game Info object (`gameInfo`)

May be absent from older files. Contains rendering dimensions computed from the window and params:

| Field | Description |
|---|---|
| `width`, `height` | Canvas dimensions (pixels) |
| `levelWidth`, `levelHeight` | Computed level geometry |
| `levelStartX`, `levelEndX` | Horizontal bounds of level within canvas |
| `ballRadius` | Ball circle radius (pixels) |
| `ballAccel` | Scaled ball acceleration value |
| `gravity` | Scaled gravity value |
| `initScrollSpeed`, `maxScrollSpeed` | Scaled scroll speed values |
| `levelSpacing` | Vertical distance between consecutive levels (pixels) |

---

## Experiment Config Files

The config files in `app/configs/` are arrays of BlockConfig objects (one per block). The available configs and their block counts:

```bash
python -c "
import json, os
for f in sorted(os.listdir('app/configs')):
    if f.endswith('.json'):
        with open(f'app/configs/{f}') as fh:
            d = json.load(fh)
        print(f'{f}: {len(d)} blocks')
"
```

Each config is an array where `config[i]["levels"]` is the pre-generated sequence of hole locations for block `i`. Level generation happens offline in `tools/level_generation/` scripts, not during the experiment.

---

## Timestamp Conventions

**All timestamps in behavioral JSON files are `performance.now()` values in milliseconds.**

`performance.now()` is a browser monotonic clock that counts milliseconds since the page loaded. Key implications:

1. **Timestamps are relative to page load**, not to wall-clock time.
2. **Within a single JSON file**, all timestamps share the same origin (`performance.now()` from one page load). You can safely subtract them to get elapsed times in milliseconds.
3. **Across different JSON files** (different page loads), timestamps have different origins. You cannot directly compare absolute timestamp values between files.
4. **The true session start** for analysis purposes is `min(all timestamps in the file)` — the first recorded event. This is NOT the same as the filename's ISO timestamp (which comes from the server, not the browser).
5. **Reaction time** between trial `i` and trial `i-1` is computed as:
   ```
   RT[i] = trial[i].events[0].time - trial[i-1].events[0].time   (ms)
   ```
   Or if you prefer seconds: divide by 1000.

---

## Multi-Snapshot Behavior (Critical for Analysis)

The game calls `wsLogger.saveJson(E)` on **every block transition** (advance, restart, or go-back). Each call writes the ENTIRE experiment state — including all previous blocks — into a `.json` file, **overwriting the previous save**.

This means:

1. A session with N block transitions produces N snapshots, each a complete-write of the experiment state.
2. Every snapshot contains `blocks[0..k]` where `k` is the current block. Earlier blocks appear in every subsequent snapshot.
3. The **final snapshot** (the last `.json` file written) is the most complete.

The `merge_participant_files()` function in `analysis/checking_data_validity.py` handles deduplication by grouping by `(block_index, trial_index)` and keeping the last occurrence. The merged files in `data/cloud_study/` have already been processed this way.

### When a participant reconnects (WebSocket drops and re-establishes)

A new `.json` file is created with the current state. The complete session may be split across multiple JSON files (e.g., the `EA4EE5B9...` participant has 8 separate files). Each represents a partial snapshot from a different reconnect. They must be merged to reconstruct the full session.

---

## Common Analysis Patterns

### Load a session and get trial-level data

```python
import json

with open("logs/session.json") as f:
    data = json.load(f)

trials = []
for block in data["blocks"]:
    for trial in block["trials"]:
        if trial["events"]:
            e = trial["events"][0]
            trials.append({
                "block_index": block["block_index"],
                "trial_index": trial["index"],
                "hole_locations": trial["hole_locations"],
                "hole_chosen": e["holeUsed"],
                "timestamp_ms": e["time"],
                "ball_x": e["ballX"],
                "ball_y": e["ballY"],
                "camera_y": e["cameraY"],
                "camera_mode": e["cameraMode"],
            })

# Compute RTs
for i in range(1, len(trials)):
    trials[i]["rt_ms"] = trials[i]["timestamp_ms"] - trials[i-1]["timestamp_ms"]
```

### Get frame-level ball trajectory for a single block

```python
import numpy as np

block = data["blocks"][0]
gs = block["game_states"]

times = np.array(gs["time"])
x = np.array(gs["ball_x"])
y = np.array(gs["ball_y"])

# Convert to seconds from block start
times_sec = (times - times[0]) / 1000.0
```

### Check which config a participant used

```python
print(data["experiment_path"])   # e.g. "configs/short_trials_experiment-7-10.json"
```

---

## Older Format (Legacy)

Some older files use a different schema. The analysis code in `behavior.py` handles both. Key differences:

| Old field | New equivalent |
|---|---|
| `trial["holes"]["hole_locations"]` | `trial["hole_locations"]` |
| `trial["holeUsed"]` (top-level) | `trial["events"][0]["holeUsed"]` |
| `trial["timePassedThru"]` (top-level) | `trial["events"][0]["time"]` |
| `block["user_inputs"]` may be absent | Always present in current format |

To check if a file uses the old format:

```python
sample_trial = data["blocks"][0]["trials"][0]
if "holes" in sample_trial and "hole_locations" not in sample_trial:
    print("OLD FORMAT")
else:
    print("Current format")
```

---

## Edge Cases and Gotchas

1. **Decision trials have 2 holes, no-choice trials have 1.** A trial with `len(hole_locations) == 2` is a decision trial. The participant's choice is `event.holeUsed`.

2. **First trial has no RT.** `RT[0]` is undefined because there is no previous trial to subtract from.

3. **Trials with `events: []`** mean the ball died before passing through a level. No choice was recorded. These trials are excluded from most analyses.

4. **`block_index` vs `block_count`**: `block_index` (0-based, config index, doesn't change on restart) is the correct field for grouping trials by experimental condition. `block_count` (1-based counter, increments on restarts) is for tracking total play time.

5. **Training blocks**: Blocks 0-3 in all configs are training/practice blocks. Block 0 has no restrictions. Blocks 1-3 enforce specific hole choices and repeat if the participant chooses the wrong hole. The experimental blocks start at block_index 4.

6. **Hole indices are 0-11.** Each level has 12 segments (columns). A hole at index 5 means the 6th segment from the left is open. The ball passes through whichever hole segment it is closest to.

7. **The `pause_times.ends` array may have one extra entry.** The first entry in `ends` marks the initial unpause (to set `start_time`). When pairing starts and ends, skip this extra entry if `len(ends) > len(starts)`.

8. **Some files have `block_configs` as an empty list `[]`.** These are corrupt/incomplete saves. The config is present but empty. Discard these files.

9. **File sizes vary enormously.** A session with 1 block of 4 trials is ~2 KB. A session with 200+ blocks of 105 trials each can be >1 MB. The `game_states` arrays (frame data) dominate the file size.
