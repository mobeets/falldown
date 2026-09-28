# Trajectory-phase decomposition: is the planning/greedy RT effect transport or deliberation?

Status: implemented by `analysis/planning_timing.py`; outputs in
`analysis/planning_outputs/` (online) and `analysis/neural_outputs/aggregate/`
(EMU). This is the mechanistic backbone of the "RT is a kinematic confound"
claim in the planning synthesis.

## Motivation

The game's ball **falls continuously**. A level is "passed" when the ball drops
past it, and the hole is whichever hole is nearest the ball's horizontal
position (`app/static/objects.js:161-181`). There is no stop, no button press, no
pause at the choice level. So:

- `rt_decision = t(choice) - t(entry)` is the **travel time to the chosen hole**;
- `rt_exec = t(exit) - t(choice)` is the **travel time from the chosen hole to the goal**.

On conflict trials the planning-optimal hole is necessarily the *farther* 1-step
hole and the *shorter* 2-step path, so the raw RT trade-off (decision +425 ms,
execution −815 ms, total −390 ms) is expected to be geometry. This analysis
tests that by decomposing each interval frame-by-frame and asking whether
planning differs in **transport** or in **dwell/hesitation** (a decision pause).

## Method

**Phase detection (model-free, displacement plateau).** For each consecutive
frame pair inside an interval, with `dy`, `dx` the ball displacements:

| phase | condition | physical meaning |
|---|---|---|
| **fall** | `|dy| > 0.05 px` | airborne / free vertical motion |
| **roll** | `|dy| <= 0.05` and `|dx| > 0.05` | in platform contact, moving |
| **hold** | `|dy| <= 0.05` and `|dx| <= 0.05` | in contact, stationary |

Thresholds are a small fraction of a pixel; because resting contact sets
`ball.y` exactly to the platform top, `dy == 0` on contact, so the scheme is
insensitive to the exact epsilon. The detected phases were validated by
overlaying them on example trajectories
(`figures/fig_timing_example_trajectories.png`): flat `ball_y` while `ball_x`
moves is roll, steep `ball_y` drops are fall.

**Coverage.** Some online blocks only logged `game_states` for the tail of the
block, so a trial can straddle the buffer edge. Trials whose interval is not
fully spanned by the block's frame clock are skipped rather than decomposed on a
truncated window (`covers()`). This retains 2,529 / 3,890 online conflict trials
(65%; 13 drift/alt-drift participants) and 1,478 / 1,478 EMU conflict trials
(4 runs, 97–98% block coverage).

**Unit of analysis.** Online = participant; EMU = run (`yfz_1/2`, `yga_1/2`).

## Findings

Phase means (ms) from `timing_phase_summary.csv`:

| cohort | interval | choice | total | fall | roll | hold | x-path (px) |
|---|---|---|---|---|---|---|---|
| online | decision | greedy | 566 | 213 | 294 | 59 | 304 |
| online | decision | planning | 958 | 212 | 639 | 107 | 520 |
| online | execution | greedy | 1178 | 219 | 924 | 35 | 685 |
| online | execution | planning | 443 | 210 | 202 | 31 | 242 |
| EMU | decision | greedy | 515 | 202 | 266 | 47 | 419 |
| EMU | decision | planning | 964 | 202 | 695 | 66 | 811 |
| EMU | execution | greedy | 1433 | 205 | 1118 | 109 | 1172 |
| EMU | execution | planning | 479 | 202 | 192 | 85 | 314 |

1. **Fall time is constant.** Across every cohort, interval and choice, the
   airborne component is 202–219 ms. The vertical fall between levels does not
   depend on the choice — as it should not, since the physics is identical.
2. **The entire RT trade-off is roll/transport.** Planning decision intervals
   are longer because the ball rolls ~520–811 px vs ~304–419 px; planning
   execution intervals are shorter because it rolls only ~192–242 px vs
   ~685–1172 px. Raw differences are almost fully accounted for by horizontal
   path length.
3. **No dwell/deliberation pause.** Hold time is small (31–109 ms) and the
   planning−greedy difference is not significant: online decision +30 ms
   (p = 0.35, 7/13), online execution −16 ms (p = 0.13, 6/13); EMU decision
   +19 ms (p = 0.11, 4/4), EMU execution −28 ms (p = 0.11, 1/4). Planning does
   not pause to compute; under an exact sign test n = 4 cannot be significant,
   so EMU is directionally suggestive but unpowered.
4. **The decision/execution trade-off is between strategies, not within a
   trial.** Within a choice, `rt_decision` and `rt_exec` are essentially
   uncorrelated (participant-level r ≈ −0.1 to +0.4), while pooled across both
   choices the correlation is strongly negative (r ≈ −0.2 to −0.6). This is a
   Simpson's paradox: the "trade-off" is produced by comparing two geometrically
   disjoint choice sets, not by participants reallocating time within a trial.
5. **The post-path residual is specification-dependent, not momentum.** A nested
   GEE battery (`timing_carryover_models.csv`, figure `fig_timing_carryover.png`)
   adds controls to the rolling-time model. For the execution interval online:
   raw −690 ms → +path length −96 ms (86% explained) → adding entry velocity
   `vx`, `|vx|` and `vy` leaves −91 ms (velocity controls add ~nothing, so the
   residual is **not** momentum carryover) → a natural-spline in path length
   gives **+29 ms, p = 0.16 (n.s.)**; decision-roll control gives n.s. On EMU the
   sign of the residual flips with the control set (spline −61 ms, overshoot
   +3 ms) and there are only four clusters. The decision interval behaves the
   same way (online +30 ms → +7 ms n.s. under a spline). **No non-kinematic
   planning transport effect is robust across plausible specifications.**
6. **Geometry is (almost) perfectly separated.** There are 0/48 (online) and
   0/35 (EMU) exact `(1-step, 2-step)` cells containing both choices; even under
   coarse 2-px binning only 2/17 (online) and 0/14 (EMU) cells overlap. A
   matched planning-vs-greedy comparison is therefore not identified from these
   data; every adjustment is an extrapolation — which is exactly why the
   residual in finding 5 moves with the functional form.

## Interpretation

The "planning is slower to commit / faster to execute" effect is a statement
about *path geometry*, not about decision time. The clean cross-dataset
statement is that fall time is fixed and the whole RT difference is horizontal
transport; dwell is negligible. This means the raw RT effect must not be read as
evidence of a time-consuming planning computation.

For the neural story this is the mechanistic constraint that explains why the
post-choice `planning_vs_greedy` decode (`[0, +1000]` ms) is dominated by
post-commitment kinematics and why the pre-decision window is null: the physical
"decision interval" is itself just transport, and the label is nearly disjoint in
traversal time. The confound-resistant neural target remains the
choice-independent geometry variable (`conflict_mag` / `diff_planning`), not the
RT.

## Outputs

- `timing_phases_online.csv`, `timing_phases_emu.csv` — trial-level phases
- `timing_phase_summary.csv` — cohort x interval x choice means/fractions
- `timing_participant_means.csv`, `timing_dwell_tests.csv` — dwell tests
- `timing_speed_model.csv` — GEE rolling-time models
- `timing_carryover_models.csv` — nested-control battery (M0–M7)
- `timing_bin_overlap.csv` — within-path-decile overlap
- `timing_within_trial_corr.csv` — dec/exec correlation structure
- `timing_geometry_separation.csv` — overlap / non-identifiability
- figures: `fig_timing_phases_online.png`, `fig_timing_phases_emu.png`,
  `fig_timing_phases_both.png`, `fig_timing_diagnostics.png`,
  `fig_timing_example_trajectories.png`, `fig_timing_carryover.png`

## Caveats

- Online trajectory coverage is 65% (a data-logging artifact); covered trials
  are disproportionately later in a block, so early-block/learning trials are
  under-sampled.
- EMU has four runs from two patients; the dwell test is unpowered (n = 4) and
  its GEE standard errors are unreliable.
- Phase thresholds are principled but sensitivity to `EPS_DY`/`EPS_DX` has not
  been swept; the contact-set-to-zero property makes the qualitative result
  robust.
- The residual in finding 5 is not robust to functional form; it should be
  reported as non-significant, not as a transport effect.

## Reproduce

```
.venv-analysis\Scripts\python.exe analysis\planning_timing.py
.venv-analysis\Scripts\python.exe analysis\planning_timing.py --from-cache
```

