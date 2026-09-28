# Tier 1 findings — closing the central question

Companion to `planning_synthesis.md` §8 (Tier 1). Three directions were run:
test the value/geometry account directly (A), re-examine the pre-decision window
at finer resolution (B), and break the geometry–policy entanglement by design
(C). Headline: **MTL shows no pre-decision planning computation at finer
resolution, its value/geometry code is dissociated from place/successor
geometry, and the design confound cannot be removed by level selection alone.**

Data: 4 EMU runs (`yfz_1/2`, `yga_1/2`), 418 conflict trials each, choice- and
entry-anchored 25 ms spike bins in-repo. No LFP (theta/ripple out of scope).

---

## A. Value/geometry account (Tier 1.2)

**A1. Successor / place RSA** (`neural_successor_geometry.py`). Built an empirical
12-segment transition matrix `T` (entry→chosen→goal), successor `M=(I−γT)⁻¹`
(γ=0.5/0.9), and model RDMs (SR, place, entry-distance, value). The neural RDM
(post window) is essentially uncorrelated with **every** model RDM:
|Spearman r| < 0.01 on all four runs, conflicted and all trials, with partial
correlations ≈ 0. The MTL population geometry is **not** a successor/place code
at the entry-state level.

**A2. dPCA by geometry** (`neural_dpca_geometry.py`). Demixing by conflict-magnitude
quartile vs by policy: the variance sits in **time-interaction terms** (`ct`, `cst`),
not a static condition axis (`c`≈0.016–0.017, `t`≈0.045 for both). The leading
geometry axis and policy axis are **near-orthogonal** (cosines −0.18…+0.12 across
runs). Geometry and policy do not share a demixed subspace.

**A3. Value vs place/SR axis alignment** (`neural_axis_alignment.py`).
OOF ridge prediction of `conflict_mag` is uncorrelated with OOF predictions of
place (`entry_hole`, r≈−0.04) and SR (r≈−0.02); fitted-axis cosines ≈ 0. SR and
place predictions are strongly related (r≈−0.73), as they should be. The value
axis is **not** the place/successor axis.

**A4. Choice beyond geometry, out-of-fold** (`neural_choice_increment.py`).
Nested-CV logistic on conflict trials (planning vs greedy):

| window | folds | geometry | geometry+neural | neural only |
|---|---|---|---|---|
| post | stratified | 0.50 | **0.69** | 0.69 |
| post | block-held-out | 0.50 | **0.69** | 0.70 |
| entry | stratified | 0.50 | 0.48 | 0.48 |

A strong **post-choice** neural choice signal (~0.69 balanced accuracy) that is
absent pre-decision. Geometry alone is at chance (the policy is stochastic, not
determined by the analytic gaps). **Audit:** `xd_neural_linkage.csv` reports a
constant `n_conflict = 431`; the true conflict count is **418** for every run
(`n_conflict_audit.csv`) — a denominator artifact to fix downstream.

## B. Fine pre-decision window (Tier 1.3)

`neural_entry_sliding_fine.py` — entry-anchored 50 ms window / 50 ms step (to
450 ms), rate features, LDA / ridge, permutation p with BH-FDR:

- binary `planning_vs_greedy`: peak mean balanced accuracy **0.517** (chance 0.5),
  **0/36** bins significant after FDR;
- continuous `conflict_mag`: mean CV r ≈ 0 (1/36 nominal);
- entry-frame temporal generalization: mean accuracy **0.499** (max 0.591; 3/100
  cells > 0.55).

No brief early MTL planning computation is visible at this resolution.

## C. Shared-geometry design (Tier 1.1)

`docs/shared_geometry_design.md`, `tools/level_generation/generating_levels_sharedgeom.py`,
`analysis/shared_geometry_validation.py`:

- the current pool has **0 shared `(1-step, 2-step)` geometry cells** between the
  two policies' chosen paths;
- **exact role-swapped matched pairs are impossible** under the 1-step greedy
  rule (greedy always takes the smaller 1-step), so the confound cannot be fixed
  by level *selection*;
- variants: `cue_goal` reaches 0.53 cross-trial greedy↔planning profile matching,
  `symmetric1step` 0.37, `three_hop` 0.15;
- recommendation: `cue_goal` (removes the confound by construction, changes the
  decision variable to a cue) or `symmetric1step` with a pre-registered tie-break.

## What would change the story

None of these moved it toward a "computation" account:

- no pre-decision read-out at finer resolution (A4 entry, B);
- the value/geometry code is not a place/successor code (A1–A3);
- the only route to a genuine non-kinematic test is a task redesign (C).

The positive, robust findings stand: `conflict_mag`/value is decodable
post-choice, and a design change is required to separate it from the committed
trajectory.
