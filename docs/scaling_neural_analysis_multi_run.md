# Scaling the neural analysis to multiple runs and participants

Status: plan + Phase 0 (refactor) implemented. Phases 1–3 pending data.

## Why this exists

The entire neural decoding pipeline (`analysis/neural_*.py`,
`spike_data_alignment.py`, `spike_unit_conversion.py`, `segment_trials.py`,
`classify_trials.py`) was built for a single recording: the YFZ-2026-07-29
session, one participant, 910 trials. Every script hardcoded that session's
paths and wrote results into one flat folder
(`analysis/neural_outputs/`), so a second run would silently overwrite the
first.

We now have more data coming:
- **YFZ participant, run 2** (`data/emu/YFZ-2026-07-30T19-27-27-747Z-b7de.json`)
  — same participant, different day.
- **A second participant, 2 runs** — behavior JSONs not downloaded yet.

This doc records how the pipeline is being made run- and participant-aware so
the existing single-session findings can be replicated across sessions and,
eventually, across participants.

## Scientific question

The first session was n=1 participant. To draw conclusions from multiple runs:

1. **Per-session replication** — run the full pipeline identically for each
   session; treat each session's decoder as an independent estimate.
2. **Meta-analysis across sessions** — test sign-consistency of the headline
   effects (`conflict_mag` surviving residualization; `planning_vs_greedy`
   decoding) and a session-level permutation test (shuffle session labels,
   recompute the mean effect) to get an honest cross-session p.

Units are **not** matched across runs (no unit-level cross-run comparisons by
decision — unit identity is channel × cluster_id per session). Per-session
decoders are the unit of replication.

## Design

### Run registry — `analysis/neural_common.py`

Single source of truth. One `Run` entry per recording:

```python
RUNS = {
  "yfz_1": Run(participant="YFZ", behavior="data/emu/YFZ-2026-07-29T…kdyd.json", ns5=…, orig_ns5=…, spikesort=…, notes=…),
  "yfz_2": Run(participant="YFZ", behavior="data/emu/YFZ-2026-07-30T…b7de.json", …),
  "newA":  Run(participant="TBD", …),
  "newB":  Run(participant="TBD", …),
}
```

Helpers:
- `get_run()` → the active `Run` (from `--run <id>`, `NEURAL_RUN` env, else `yfz_1`).
- `out_dir()` → `analysis/neural_outputs/<run_id>/` (created on demand).
- `out_path(name)` → that directory + a filename.

Adding a new session = adding one dict entry (fill in `behavior`, `ns5`,
`orig_ns5`, `spikesort` when the data lands) — no script edits.

### Per-run output directories

Every pipeline script now resolves `OUT_DIR` (and its input paths) through
`neural_common`, so all results for a run land under
`analysis/neural_outputs/<run_id>/`. Existing scripts were changed only in
their configuration block — analysis logic is untouched.

| Script | Changed |
|---|---|
| `spike_data_alignment.py` | config → registry (NS5, behavior, spikesort) |
| `spike_unit_conversion.py` | config → registry |
| `segment_trials.py` | config → registry; stamps `run_id`, `participant_id` into trial_table / unit_metadata / npz records |
| `classify_trials.py` | config → registry; stamps `run_id`, `participant_id` into trial_labels / death_times |
| `neural_selectivity.py`, `spatial_tuning.py`, `left_right_selectivity.py` | config → registry |
| `neural_lda_decoding.py`, `neural_temporal_decoding.py`, `neural_dpca.py` | config → registry |
| `neural_continuous_decoding.py`, `neural_decoding_confounds.py` | config → registry |
| `neural_bursty_unit_sensitivity.py` | config → registry |

### Identity in artifacts

- `trial_id` restarts at 0 for every run; the stamped `run_id` / `participant_id`
  columns make `(run_id, trial_id)` the unique key for cross-run aggregation.
- The embedded records in `segmented_spikes_binned.npz` (trial_table,
  unit_metadata) carry the same stamps.

### Migration of the first run

The existing flat outputs were moved verbatim into
`analysis/neural_outputs/yfz_1/` (no recompute needed). `DATA_STRUCTURE.md`
now documents the per-run layout. Validation: `neural_bursty_unit_sensitivity.py
--run yfz_1` reproduces the published `conflict_mag` numbers exactly
(PCA 0.269/0.241/0.301 all-units).

### Aggregate step (Phase 3, not yet written)

Planned `analysis/aggregate_sessions.py`:
- Loop over `RUNS`; read each run's per-session result CSVs (no re-fit).
- Per-session table of headline effects (`conflict_mag` none/target/both,
  `planning_vs_greedy`, `condition`).
- Sign-consistency + replication count across sessions.
- Session-level permutation test (shuffle session labels) for a cross-session p.
- Keep within-participant (YFZ x2) and across-participant comparisons separate.

## How to add a new run

1. Put the behavioral JSON in `data/emu/`.
2. Add (or fill in) an entry in `RUNS` in `analysis/neural_common.py` with the
   run's behavior path, NS5 paths, and spike-sort directory.
3. Run the pipeline per session:
   ```
   python analysis/spike_data_alignment.py --run <run_id>
   python analysis/spike_unit_conversion.py --run <run_id>
   python analysis/segment_trials.py --run <run_id>
   python analysis/classify_trials.py --run <run_id>
   python analysis/neural_selectivity.py --run <run_id>
   python analysis/spatial_tuning.py --run <run_id>
   python analysis/left_right_selectivity.py --run <run_id>
   python analysis/neural_lda_decoding.py --run <run_id>
   python analysis/neural_temporal_decoding.py --run <run_id>
   python analysis/neural_dpca.py --run <run_id>
   python analysis/neural_continuous_decoding.py --run <run_id>
   python analysis/neural_decoding_confounds.py --run <run_id>
   python analysis/neural_bursty_unit_sensitivity.py --run <run_id>
   ```
   (No `--run` → defaults to `yfz_1`.)
4. Run the aggregation step once data from ≥2 runs exists.

## Decisions / non-decisions

- **No unit-level cross-run matching** (deliberate). Per-session decoding is the
  replication unit. A future unit-matching module (electrode channel+cluster
  across YFZ sessions) would be opt-in if ever needed.
- **Pooling option deferred.** A pooled-trial decoder with per-session
  z-scoring is a possible Phase-4 robustness check, but it risks treating
  session/participant as a confound; per-session replication is the headline.
