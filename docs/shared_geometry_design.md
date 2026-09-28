# Shared-geometry task design (Tier 1.1)

Goal: remove the structural confound that makes planning and greedy choices
geometrically distinct. In the current 1-2-1 task the two policies never choose
paths of the same length, so choice is collinear with post-choice kinematics
(RT and neural). This document specifies candidate designs, reports what the
validation harness (`analysis/shared_geometry_validation.py`) found, and gives a
recommendation.

## The confound (measured)

`analysis/shared_geometry_validation.py` computes, for every conflict trial, the
`(1-step, 2-step)` distance profile of the greedy-optimal and planning-optimal
choices, and counts geometry cells shared by both policies.

| pool | n conflict | greedy cells | planning cells | shared cells | greedy matched to some planning |
|---|---|---|---|---|---|
| current `trials_new.json` | 5000 | 22 | 16 | **0** | 0.00 |
| `matched_pairs` | 0 | – | – | – | – |
| `symmetric1step` | 196 | 38 | 31 | 13 | 0.37 |
| `three_hop` | 400 | 30 | 38 | 5 | 0.15 |
| `cue_goal` | 222 | 33 | 25 | 11 | 0.53 |

## Key structural result: role-swap is impossible as-is

A matched-geometry **pair** would need one level where the greedy-optimal hole
has profile P and the planning-optimal hole has Q, and another with the roles
reversed. This is **impossible under the current cost rules**, because greedy is
defined as the *shorter 1-step* hole and planning as the *shorter total* hole:
the greedy-optimal hole therefore always has the minimal 1-step distance, so
`greedy_profile` can never equal another trial's `planning_profile`. The
`matched_pairs` generator returns 0 trials — the confound cannot be removed by
**level selection** alone. A rule or structure change is required.

## Variants

1. **`symmetric1step`** — entry equidistant to both holes; conflict comes from
   the 2-step geometry. The 1-step distance is matched for both options, but
   greedy becomes a tie and needs a documented tie-break (leftmost), so "greedy"
   is no longer a genuine optimum. Partial cross-trial matching (0.37).
2. **`three_hop`** — an extra hop so the policies can prescribe equal-length
   paths. Still a 1-step-greedy rule, so the same profile asymmetry persists;
   weakest matching (0.15).
3. **`cue_goal`** — the two holes are geometrically identical and the target is
   signalled by a **cue** (colour), so the chosen paths are matched for both
   policies and the confound is removed by construction. Cost: the decision
   variable changes from geometry to cue, which is a different experiment.
4. **`matched_pairs`** — exact role-swapped pairs; **not achievable** under the
   1-step greedy rule (see above). Kept as the formal target for a future rule
   change (e.g., a reward at the 2-step level that decouples greedy from
   1-step distance).

## Acceptance criteria for a shared-geometry pool

A pool qualifies if, among conflict trials:
- the greedy-option and planning-option profile sets have substantial overlap
  (target `frac_shared_of_union` ≳ 0.5), and
- for a matched subset, every included geometry cell contains both a
  greedy-optimal and a planning-optimal instance, so RT/neural comparisons can
  be made at fixed `(1-step, 2-step)`.
Only `cue_goal` currently approaches this (0.53); `symmetric1step` is partial.

## Recommendation

- **If the scientific goal is to keep a geometric planning rule:** adopt the
  `symmetric1step` variant (matched 1-step, conflict from the 2-step) and
  pre-register the tie-break, accepting that "greedy" is then a defined
  tie-break rather than a cost minimum.
- **If the goal is to definitively separate policy from kinematics:** adopt the
  `cue_goal` variant. It is the only design that makes the two options
  geometrically identical; it changes the decision variable to a cue, which must
  be stated as a limitation.
- **A true matched-pair design requires a rule change** (not level selection):
  reward/penalty at the second step, or a symmetric maze where 1-step distance
  does not determine the greedy choice.

## Implementation

- Generator: `tools/level_generation/generating_levels_sharedgeom.py`
  (`--variant {matched_pairs,symmetric1step,three_hop,cue_goal}`), emitted in
  the app's `levels` schema.
- Validation: `analysis/shared_geometry_validation.py`
  (`shared_geometry_validation.csv`, `fig_shared_geometry_validation.png`).
- App support for `cue_goal` (coloured holes / cue rendering) is **not yet
  implemented**; it is a prerequisite for data collection. No participants were
  run for this document (design + validation only).
