# Repeated Trials Analysis

## How the merge works

`merge_participant_files` in `analysis/checking_data_validity.py` groups raw JSON files by the first 30 characters of the filename (the participant ID prefix), then merges all blocks and trials from every file belonging to the same participant. When two files — or two block entries within the same file — contain a trial with the same `(block_index, trial_index)` pair, the later occurrence overwrites the earlier one.

During this process, every overwrite is counted as a **repeat** and classified as either **cloned** or **updated** based on timestamp comparison.

## Cloned vs updated

| Label | Meaning | Likely cause |
|---|---|---|
| **cloned** (same timestamp) | The incoming trial's timestamp (`timePassedThru` or `events[0].time`) equals the existing one (within 1 µs). | The trial belongs to a block the participant had already finished. A subsequent save snapshot dumped the same completed data again without any new play. |
| **updated** (different timestamp) | The timestamps differ. | The block was being actively played between saves (each trial got a new real-time clock value), or the data came from a different session after a WebSocket reconnect / page reload. |

## Why the save mechanism produces many repeats

The game calls `wsLogger.saveJson(this)` on every block transition (`experiment.js:98`). Each call writes the *entire* current experiment state (all blocks played so far) to a JSON file.

### Within a single WebSocket connection

Each `saveJson` overwrites the same file on the server (`server.py:108-109`), so a normal session with N blocks produces exactly 1 file. The resulting file contains N block entries — one per block transition, each a full snapshot at that point. Block 0 appears in all N entries, block 1 in entries 2–N, etc. The merge function processes each entry in order, so trials from earlier block entries get overwritten by the same trials in later entries. These within-file repeats are the primary source of **cloned** counts: the participant wasn't replaying block 0 on every save — the snapshot was just dumping it again unchanged.

### Across multiple files (WebSocket reconnections)

Each time the WebSocket reconnects, `_generateFilename()` in `ws_logger.js:33` creates a new filename with a fresh timestamp, producing a separate `.json` file. Trials that appear in multiple files (same block + same trial index but from different reconnections) are counted as repeats during merge. If the timestamps differ, they're **updated**; if identical (e.g., blocked finished before the disconnect), they're **cloned**.

## The EA4EE5 case: 8 files, 21690 repeats

### File list and timing

| File | Block indices | Play time elapsed |
|------|--------------|-------------------|
| 1 | 0–14 | ~ start |
| *gap ~22 min* | | likely tab closed / laptop slept |
| 2 | 0–18 | resumed |
| 3–8 | 0–27 | ~16 min |

File 1 has 97 block entries and only 15 unique block indices — that's ~82 duplicate entries from within-file snapshots alone. Blocks 0 and 1 appear once (finished quickly), while blocks 4+ appear 7–10 times each. Across 8 files the repetition compounds.

### Why the time-on-task was normal

The 21690 repeat count is **not** 21690 unique gameplay trials. The participant actually played **2350 unique trials** — slightly *fewer* than the other four participants (2533–3478). The play time (~37 min from first to last event time) is also comparable. The inflated repeat count comes entirely from the same trials being saved over and over across file snapshots and WebSocket reconnections.

### Most likely root cause

The participant's WebSocket disconnected and reconnected 7 times, likely due to:
- Network instability (intermittent WiFi)
- The browser tab being backgrounded or the laptop going to sleep
- Each reconnect → new filename → new file → existing blocks re-saved with the same data

The 22-minute gap between file 1 and file 2 supports the "tab closed / laptop slept" theory.

### Diagnostic output

The `report_repeated_trials` function in `checking_data_validity.py` shows the raw timestamps for any repeated `(block, trial)` pair across files. Running it on the EA4EE5 files confirms that the "cloned" repeats are blocks 0–3 (finished before any reconnect) while "updated" repeats are blocks 4+ where the ongoing gameplay produced new timestamps between saves.

## Implications

1. The merge function correctly deduplicates — the final merged output contains exactly 2350 unique trials from 27 unique blocks, which is a reasonable session length.
2. The "repeated trials" count alone is misleading if interpreted as actual replay attempts. Most repeats are cost-free snapshots.
3. An optimization would skip earlier block entries within the same file (keep only the last occurrence of each `block_index` per file), which would eliminate most within-file "cloned" repeats before they reach the cross-file merge.
