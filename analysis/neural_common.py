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
# # Neural pipeline run registry and path resolution
#
# Everything in this module exists to take the neural analysis pipeline from a
# single hardcoded session to **N runs across M participants**. Every script
# in the pipeline resolves its inputs/outputs through this module instead of
# embedding absolute paths, so adding a new session is one entry in `RUNS`.
#
# Conventions
# -----------
# - Each run maps to `analysis/neural_outputs/<run_id>/`. All per-run outputs
#   (segmented spikes, trial tables/labels, decoding results, ...) live there,
#   so running the pipeline for a second session never clobbers the first.
# - `run_id` is short and stable (e.g. `yfz_1`, `yfz_2`, `newA`, `newB`) and is
#   recorded in every output CSV / embedded record, so cross-run aggregation
#   can key on `(run_id, trial_id)` without collision.
# - Pass `--run <run_id>` on the command line, or set `NEURAL_RUN` env var.
#   Default is `yfz_1` (the first analyzed session), so old invocations with no
#   flag keep working.
#
# Units are per-run (channel x cluster_id); this project deliberately does NOT
# match units across runs. Per-session decoders are the unit of replication.

# %%
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_RUN = "yfz_1"


def _resolve(p: str) -> Path:
    """Absolute paths are returned unchanged; relative paths resolve against
    the repo root. Empty strings stay empty so unset runs fail loudly."""
    if not p:
        return Path("")
    q = Path(p)
    return q if q.is_absolute() else (_REPO_ROOT / q)


# %%
@dataclass(frozen=True)
class Run:
    """A recording session: participant + its raw inputs."""

    run_id: str
    participant: str
    behavior: str                # repo-relative path to the behavioral JSON
    ns5: str = ""                # de-identified NS5 (absolute or repo-relative)
    orig_ns5: str = ""           # original NS5 with mics (absolute or repo-relative)
    spikesort: str = ""          # path to this run's spikesort_results dir
    notes: str = ""

    # NS5 binary geometry for the raw photodiode reader in spike_data_alignment.py
    n_channels: int = 78             # analog channels in the NS5
    n_frames: int = 72_348_374       # samples per channel (header DataPoints)
    photodiode_row: int = 65         # 1-indexed analog row of the photodiode
    room_mic2_row: int | None = 68   # 1-indexed row of RoomMic2; None if absent

    @property
    def behavior_path(self) -> Path:
        return _resolve(self.behavior)

    @property
    def ns5_path(self) -> Path:
        return _resolve(self.ns5)

    @property
    def orig_ns5_path(self) -> Path:
        return _resolve(self.orig_ns5)

    @property
    def spikesort_dir(self) -> Path:
        return _resolve(self.spikesort)

    @property
    def spikes_mat(self) -> Path:
        return self.spikesort_dir / "cluster_viewer_results" / "spikes.mat"

    @property
    def neuron_data_json(self) -> Path:
        return self.spikesort_dir / "cluster_viewer_results" / "neuron_data.json"

    @property
    def spikes_perchannel_mat(self) -> Path:
        return self.spikesort_dir / "cluster_viewer_results" / "spikes_perChannel.mat"


# %%
RUNS: dict[str, Run] = {
    "yfz_1": Run(
        run_id="yfz_1",
        participant="YFZ",
        behavior="data/emu/YFZ-2026-07-29T21-37-47-781Z-kdyd.json",
        ns5=r"C:\Users\manik\Desktop\Obsidian\General Thoughts\Z Images and Files\Hennig Lab Project\falldown\noPHIEMU-0113_subj-YFZ_task-FD_run-01_NSP-2.ns5",
        orig_ns5=r"C:\Users\manik\Desktop\Spike Sorting For Hennig Project\spikesort_results\EMU-0113_subj-YFZ_task-FD_run-01_NSP-2.ns5",
        spikesort=r"C:\Users\manik\Desktop\Spike Sorting For Hennig Project\spikesort_results",
        n_channels=78,
        n_frames=72_348_374,
        photodiode_row=65,
        room_mic2_row=68,
        notes="first analyzed session (this is the one all published numbers come from)",
    ),
    "yfz_2": Run(
        run_id="yfz_2",
        participant="YFZ",
        behavior="data/emu/YFZ-2026-07-30T19-27-27-747Z-b7de.json",
        ns5=r"C:\Users\manik\Desktop\DEPHI_EMU-0122_subj-YFZ_task-FD_run-02_NSP-2.ns5",
        orig_ns5="",  # no mic channel in EMU-0122 -> audio cross-check skipped
        spikesort="data/spikesort/yfz_2",
        # EMU-0122 NS5 header: 73 analog channels, 62,368,991 samples (vs 78 /
        # 72,348,374 for yfz_1). Photodiode is still row 65; this run has no
        # RoomMic2 channel, so the audio cross-check is skipped.
        n_channels=73,
        n_frames=62_368_991,
        photodiode_row=65,
        room_mic2_row=None,
        notes="same participant, second run (EMU-0122 task-FD run-02); spikesort in data/spikesort/yfz_2",
    ),
    # YGA participant, two runs. Both NS5s are 89-channel BRSMPGRP group files
    # (vs 73/78 for YFZ); the photodiode sits at analog row 81 and there is no
    # RoomMic2 channel, so the audio cross-check is skipped.
    "yga_1": Run(
        run_id="yga_1",
        participant="YGA",
        behavior="data/emu/YGA-2026-08-22T20-49-57-915Z-fr0w.json",
        ns5=r"C:\Users\manik\Desktop\Spike Sorting For Hennig Project\YGA_0172_FD\DEPHI_EMU-0172_subj=YGA_task-FD_run01_NSP-2.ns5",
        orig_ns5="",  # no mic channel -> audio cross-check skipped
        spikesort=r"C:\Users\manik\Desktop\Spike Sorting For Hennig Project\YGA_0172_FD",
        n_channels=89,
        n_frames=74_205_368,
        photodiode_row=81,
        room_mic2_row=None,
        notes="YGA first run (EMU-0172 task-FD run01)",
    ),
    "yga_2": Run(
        run_id="yga_2",
        participant="YGA",
        behavior="data/emu/YGA-2026-08-23T18-57-12-341Z-g70g.json",
        ns5=r"C:\Users\manik\Desktop\Spike Sorting For Hennig Project\YGA_0192_FD\DEPHI_EMU-0192_subj-YGA_task-FD_run02_NSP-2.ns5",
        orig_ns5="",  # no mic channel -> audio cross-check skipped
        spikesort=r"C:\Users\manik\Desktop\Spike Sorting For Hennig Project\YGA_0192_FD",
        n_channels=89,
        n_frames=77_492_623,
        photodiode_row=81,
        room_mic2_row=None,
        notes="YGA second run (EMU-0192 task-FD run02)",
    ),
}


# %%
def parse_run_arg() -> str:
    """Resolve the target run_id: `--run <id>` on the CLI > `NEURAL_RUN` env
    > DEFAULT_RUN. Reads sys.argv at import time, which works because these
    scripts parse flags before their `__main__` guard."""
    argv = sys.argv[1:]
    if "--run" in argv:
        i = argv.index("--run")
        if i + 1 < len(argv):
            rid = argv[i + 1]
            if rid in RUNS:
                return rid
            raise SystemExit(f"Unknown run {rid!r}. Known: {sorted(RUNS)}")
    env = os.environ.get("NEURAL_RUN")
    if env in RUNS:
        return env
    return DEFAULT_RUN


def get_run(run_id: str | None = None) -> Run:
    rid = run_id or parse_run_arg()
    return RUNS[rid]


def out_dir(run_id: str | None = None) -> Path:
    """Per-run output directory (created on demand)."""
    rid = run_id or parse_run_arg()
    d = _REPO_ROOT / "analysis" / "neural_outputs" / rid
    d.mkdir(parents=True, exist_ok=True)
    return d


def out_path(name: str, run_id: str | None = None) -> Path:
    return out_dir(run_id) / name
