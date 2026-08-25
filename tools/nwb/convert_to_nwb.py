"""
convert_to_nwb.py - Convert falldown behavioral JSON data to NWB 2.0 format.

Usage:
    python convert_to_nwb.py logs/                        # convert all JSONs in a directory
    python convert_to_nwb.py logs/session.json            # convert a single JSON file
    python convert_to_nwb.py logs/ -o output/             # specify output directory

Each JSON session file maps to one .nwb file.
"""

import json
import os
import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
from pynwb import NWBFile, NWBHDF5IO, TimeSeries
from pynwb.behavior import (
    SpatialSeries,
    Position,
    BehavioralTimeSeries,
    BehavioralEpochs,
)
from pynwb.epoch import TimeIntervals
from pynwb.event import EventsTable
from pynwb.misc import IntervalSeries


def find_session_start(block):
    """Find the earliest timestamp in a block's data to use as t=0 reference."""
    candidates = []
    if block.get("start_time"):
        candidates.append(block["start_time"])
    gs = block.get("game_states", {})
    if gs.get("time") and len(gs["time"]) > 0:
        candidates.append(gs["time"][0])
    ui = block.get("user_inputs", {})
    if ui.get("time") and len(ui["time"]) > 0:
        candidates.append(ui["time"][0])
    pt = block.get("pause_times", {})
    if pt.get("ends") and len(pt["ends"]) > 0:
        candidates.append(pt["ends"][0])
    return min(candidates) if candidates else 0


def timestamp_to_relative(t_ms, session_start_ms):
    """Convert a performance.now() timestamp in ms to relative seconds."""
    return (t_ms - session_start_ms) / 1000.0


def convert_to_ts(ts_array_ms, session_start_ms):
    """Convert an array of performance.now() timestamps (ms) to relative seconds."""
    return (np.array(ts_array_ms, dtype=np.float64) - session_start_ms) / 1000.0


def extract_filename_timestamp(filepath):
    """Attempt to extract an ISO timestamp from the filename."""
    stem = Path(filepath).stem
    parts = stem.split("-")
    for part in parts:
        if "T" in part and len(part) >= 20:
            try:
                ts_str = part.replace("Z", "")
                if len(ts_str) == 23:
                    ts_str += "000"
                return datetime.strptime(ts_str, "%Y-%m-%dT%H-%M-%S-%f")
            except ValueError:
                continue
    return None


def convert_json_to_nwb(json_path, output_dir=None):
    """Convert a single falldown JSON session file to an NWB file."""
    json_path = Path(json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data.get("blocks"):
        print(f"  WARNING: {json_path.name} has no blocks - skipping")
        return None

    # --- Session metadata ---
    subject_id = data.get("subject_id", "unknown")
    params_path = data.get("params_path", "")
    experiment_path = data.get("experiment_path", "")
    params = data.get("params", {})
    game_info = data.get("gameInfo", {}) or {}

    # Determine session start time
    session_start_ms = None
    for block in data["blocks"]:
        ss = find_session_start(block)
        if session_start_ms is None or ss < session_start_ms:
            session_start_ms = ss

    if session_start_ms is None:
        session_start_ms = 0

    # Try to get the real-world session time from the filename
    file_ts = extract_filename_timestamp(json_path)
    if file_ts:
        session_start_time = file_ts.replace(tzinfo=timezone.utc)
    else:
        mtime = json_path.stat().st_mtime
        session_start_time = datetime.fromtimestamp(mtime, tz=timezone.utc)

    # Build experiment description with params info included
    cfg_summary = f"Params path: {params_path}. Experiment path: {experiment_path}."
    full_description = (
        f"Ball-falling maze decision-making task. "
        f"Participants guide a falling ball through holes in descending levels, "
        f"making left/right choices on 2-hole decision trials. "
        f"Measures planning depth (greedy vs rollout strategies) and reaction times. "
        f"{cfg_summary}"
    )

    # Build extra metadata for notes field
    meta_for_notes = {"params": params}
    if game_info:
        meta_for_notes["game_info"] = game_info
    block_configs = data.get("block_configs", {})
    if block_configs:
        meta_for_notes["block_configs"] = block_configs

    # Subject
    try:
        from pynwb.file import Subject
        subject_obj = Subject(subject_id=subject_id)
    except Exception:
        subject_obj = None

    # --- Build NWBFile ---
    nwbfile = NWBFile(
        session_description=f"Falldown behavioral experiment - {len(data['blocks'])} blocks",
        identifier=f"{subject_id}_{uuid4().hex[:8]}",
        session_start_time=session_start_time,
        experimenter="Jay Hennig",
        lab="Hennig Lab",
        institution="Baylor College of Medicine",
        experiment_description=full_description,
        session_id=subject_id,
        keywords=[
            "falldown",
            "decision-making",
            "planning",
            "behavioral",
            "two-step task",
            "reaction time",
        ],
        notes=json.dumps(meta_for_notes),
        subject=subject_obj,
    )

    # --- Processing module ---
    behavior_module = nwbfile.create_processing_module(
        name="behavior",
        description="Processed behavioral data from the Falldown experiment",
    )

    # =====================================================================
    # PER-BLOCK: SpatialSeries, TimeSeries, Trials, Epochs, Invalid Times
    # =====================================================================

    # Collect all blocks' data into unified arrays where possible
    all_ball_ts = []
    all_ball_x = []
    all_ball_y = []
    all_ball_vx = []
    all_ball_vy = []
    all_camera_y = []
    all_scroll_speed = []

    all_input_ts = []
    all_input_val = []

    for block in data["blocks"]:
        gs = block.get("game_states", {})
        if gs.get("time"):
            t = convert_to_ts(gs["time"], session_start_ms)
            all_ball_ts.append(t)
            all_ball_x.append(np.array(gs.get("ball_x", []), dtype=np.float64))
            all_ball_y.append(np.array(gs.get("ball_y", []), dtype=np.float64))
            all_ball_vx.append(np.array(gs.get("ball_vx", []), dtype=np.float64))
            all_ball_vy.append(np.array(gs.get("ball_vy", []), dtype=np.float64))
            all_camera_y.append(np.array(gs.get("camera_y", []), dtype=np.float64))
            all_scroll_speed.append(np.array(gs.get("scroll_speeds", []), dtype=np.float64))

        ui = block.get("user_inputs", {})
        if ui.get("time"):
            t = convert_to_ts(ui["time"], session_start_ms)
            all_input_ts.append(t)
            all_input_val.append(np.array(ui["input"], dtype=np.int8))

    # --- Position: ball_x, ball_y ---
    if all_ball_ts:
        ts_cat = np.concatenate(all_ball_ts)
        x_cat = np.concatenate(all_ball_x)
        y_cat = np.concatenate(all_ball_y)

        sort_idx = np.argsort(ts_cat)
        position_data = np.column_stack([x_cat[sort_idx], y_cat[sort_idx]])

        ball_spatial = SpatialSeries(
            name="ball_position",
            description="Ball (x, y) position in game-window pixel coordinates.",
            data=position_data,
            timestamps=ts_cat[sort_idx],
            reference_frame="(0,0) is top-left of the game canvas. x increases right, y increases down.",
            unit="pixels",
        )
        ball_position = Position(spatial_series=ball_spatial, name="ball_position")
        behavior_module.add(ball_position)

    # --- BehavioralTimeSeries: velocity, camera, scroll ---
    bt_series_list = []

    if all_ball_ts:
        ts_cat = np.concatenate(all_ball_ts)
        vx_cat = np.concatenate(all_ball_vx)
        vy_cat = np.concatenate(all_ball_vy)
        sort_idx = np.argsort(ts_cat)

        ball_vx_ts = TimeSeries(
            name="ball_velocity_x",
            data=vx_cat[sort_idx],
            timestamps=ts_cat[sort_idx],
            unit="pixels/second",
            description="Ball horizontal velocity (x component).",
        )
        bt_series_list.append(ball_vx_ts)

        ball_vy_ts = TimeSeries(
            name="ball_velocity_y",
            data=vy_cat[sort_idx],
            timestamps=ts_cat[sort_idx],
            unit="pixels/second",
            description="Ball vertical velocity (y component).",
        )
        bt_series_list.append(ball_vy_ts)

        camera_ts = TimeSeries(
            name="camera_offset_y",
            data=np.concatenate(all_camera_y)[sort_idx],
            timestamps=ts_cat[sort_idx],
            unit="pixels",
            description="Camera vertical offset (how far down the world has scrolled).",
        )
        bt_series_list.append(camera_ts)

        scroll_ts = TimeSeries(
            name="scroll_speed",
            data=np.concatenate(all_scroll_speed)[sort_idx],
            timestamps=ts_cat[sort_idx],
            unit="pixels/second",
            description="Camera scroll speed in drift mode.",
        )
        bt_series_list.append(scroll_ts)

    if bt_series_list:
        bt = BehavioralTimeSeries(time_series=bt_series_list[0])
        for ts in bt_series_list[1:]:
            bt.add_timeseries(ts)
        behavior_module.add(bt)

    # --- User inputs as EventsTable ---
    if all_input_ts:
        input_events = EventsTable(
            name="directional_input",
            description="Timestamped directional input events from the participant.",
        )
        input_events.add_column(
            name="direction",
            description="-1 = moving left, 0 = released, 1 = moving right",
        )

        ts_cat = np.concatenate(all_input_ts)
        val_cat = np.concatenate(all_input_val)
        sort_idx = np.argsort(ts_cat)

        for t, v in zip(ts_cat[sort_idx], val_cat[sort_idx]):
            input_events.add_event(timestamp=float(t), direction=int(v))

        nwbfile.add_events_table(input_events)

    # --- Trials table ---
    nwbfile.add_trial_column(
        name="block_index",
        description="Index of the block (0-based) this trial belongs to.",
    )
    nwbfile.add_trial_column(
        name="trial_index",
        description="Index of the trial within its block (0-based).",
    )
    nwbfile.add_trial_column(
        name="hole_locations",
        description="Segment indices (0-11) of available holes. Length 1 = no-choice, 2 = decision trial.",
    )
    nwbfile.add_trial_column(
        name="hole_chosen",
        description="Segment index (0-11) the ball actually passed through.",
    )
    nwbfile.add_trial_column(
        name="is_decision_trial",
        description="Whether this trial had 2 holes (decision trial) vs 1 hole (no-choice).",
    )
    nwbfile.add_trial_column(
        name="camera_mode",
        description="Camera mode at time of level pass-through: 0 = follow, 1 = drift.",
    )
    nwbfile.add_trial_column(
        name="is_mode_switch",
        description="Whether a camera mode switch occurred on this level.",
    )
    nwbfile.add_trial_column(
        name="event_level_y",
        description="Y-coordinate (pixels) of the level at event time.",
    )
    nwbfile.add_trial_column(
        name="event_ball_x",
        description="Ball x position (pixels) at level pass-through.",
    )
    nwbfile.add_trial_column(
        name="event_ball_y",
        description="Ball y position (pixels) at level pass-through.",
    )
    nwbfile.add_trial_column(
        name="event_camera_y",
        description="Camera y offset (pixels) at level pass-through.",
    )
    nwbfile.add_trial_column(
        name="event_scroll_speed",
        description="Scroll speed at level pass-through.",
    )

    # Flatten all trials across blocks
    for block in data["blocks"]:
        block_idx = block.get("block_index", 0)
        for trial in block.get("trials", []):
            events = trial.get("events", [])
            hole_locs = trial.get("hole_locations", [])

            if events:
                # Map event timestamps relative to session start
                e = events[0]
                t_rel = timestamp_to_relative(e.get("time", 0), session_start_ms)

                start_time = t_rel
                stop_time = t_rel + 0.001  # instantaneous event, tiny duration

                hole_chosen = e.get("holeUsed", -1)
                if hole_chosen == -1:
                    hole_chosen = None

                nwbfile.add_trial(
                    start_time=start_time,
                    stop_time=stop_time,
                    block_index=block_idx,
                    trial_index=trial.get("index", 0),
                    hole_locations=str(hole_locs),
                    hole_chosen=hole_chosen,
                    is_decision_trial=len(hole_locs) == 2,
                    camera_mode=e.get("cameraMode", e.get("modeIndex", None)),
                    is_mode_switch=e.get("isModeSwitch", False),
                    event_level_y=e.get("levelY", None),
                    event_ball_x=e.get("ballX", None),
                    event_ball_y=e.get("ballY", None),
                    event_camera_y=e.get("cameraY", None),
                    event_scroll_speed=e.get("scrollSpeed", None),
                )

    # --- Epochs: one per block ---
    for block_idx, block in enumerate(data["blocks"]):
        gs = block.get("game_states", {})

        if gs.get("time") and len(gs["time"]) > 0:
            t_start = timestamp_to_relative(gs["time"][0], session_start_ms)
            t_end = timestamp_to_relative(gs["time"][-1], session_start_ms)
        elif block.get("start_time"):
            t_start = timestamp_to_relative(block["start_time"], session_start_ms)
            trials = block.get("trials", [])
            if trials and trials[-1].get("events"):
                t_end = timestamp_to_relative(
                    trials[-1]["events"][0].get("time", 0), session_start_ms
                )
            else:
                t_end = t_start + 0.001
        else:
            continue

        tags = [f"block_{block_idx}"]
        nwbfile.add_epoch(start_time=t_start, stop_time=t_end, tags=tags)

    # --- Invalid times: pause intervals ---
    invalid_intervals = []
    for block in data["blocks"]:
        pt = block.get("pause_times", {})
        starts = pt.get("starts", [])
        ends = pt.get("ends", [])
        for s, e in zip(starts, ends):
            invalid_intervals.append(
                (
                    timestamp_to_relative(s, session_start_ms),
                    timestamp_to_relative(e, session_start_ms),
                )
            )

    for t_start, t_end in invalid_intervals:
        nwbfile.add_invalid_time_interval(start_time=t_start, stop_time=t_end)

    # --- Write to disk ---
    if output_dir:
        out_dir = Path(output_dir)
    else:
        out_dir = json_path.parent / "nwb"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{json_path.stem}.nwb"

    with NWBHDF5IO(str(out_path), "w") as io:
        io.write(nwbfile)

    total_trials = sum(len(b.get("trials", [])) for b in data["blocks"])
    print(f"  -> {out_path}  ({len(data['blocks'])} blocks, {total_trials} trials)")

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert falldown behavioral JSON data to NWB format."
    )
    parser.add_argument(
        "input",
        nargs="+",
        help="JSON file(s) or directory(ies) of JSON files to convert",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output directory for .nwb files (default: <input_dir>/nwb/)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search input directories for .json files",
    )
    args = parser.parse_args()

    json_files = []

    for inp in args.input:
        p = Path(inp)
        if p.is_dir():
            pattern = "**/*.json" if args.recursive else "*.json"
            for f in p.glob(pattern):
                json_files.append(f)
        elif p.is_file() and p.suffix == ".json":
            json_files.append(p)
        else:
            print(f"WARNING: {inp} is not a .json file or directory - skipping")

    if not json_files:
        print("No JSON files found.")
        return

    print(f"Found {len(json_files)} JSON file(s). Converting...\n")

    success = 0
    skipped = 0
    for jf in sorted(json_files):
        try:
            result = convert_json_to_nwb(jf, args.output)
            if result:
                success += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"  ERROR: {jf.name}: {e}")
            skipped += 1

    print(f"\nDone. {success} converted, {skipped} skipped.")


if __name__ == "__main__":
    main()
