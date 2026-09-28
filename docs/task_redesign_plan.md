# Task redesign plan — future experiment portfolio

Status: planning document. This collects candidate task designs that would give
**new** information about planning in the brain and overcome the current
task's structural limitations. It builds on `docs/shared_geometry_design.md`,
`docs/tier1_findings.md`, and `docs/planning_synthesis.md` §7–8.

---

## 1. Why none of the shared-geometry variants is enough

The Tier 1.1 variants in `shared_geometry_design.md` are **confound controls,
not new information**:

- `symmetric1step` — matches the 1-step distance; isolates the 2-step planning
  effect from immediate kinematics, but "greedy" becomes an arbitrary tie-break
  (no longer an optimum), and matching is partial (0.37).
- `three_hop` — weakest matching (0.15); minor gain.
- `cue_goal` — fully removes the geometry confound, but only by **replacing the
  geometric planning variable with a cue/reward variable**. New information
  only if reframed as a value-based choice task (see D1).
- `matched_pairs` — **impossible** under the 1-step greedy rule (greedy always
  takes the smaller 1-step, so roles can never swap); motivates a rule change.

Key measured facts motivating this document:
- current pool has **0 shared `(1-step, 2-step)` geometry cells**; choice is
  collinear with the committed trajectory;
- no pre-decision neural signal at fine resolution (entry sliding 0/36 bins
  significant; temporal generalization ≈ chance);
- dPCA: condition-independent time ≈ 4.5%, conflict (policy) ≈ 24.6%, geometry
  (conflict magnitude) ≈ 17.4% of the condition-averaged PSTH variance, and
  geometry/policy axes are near-orthogonal — i.e., the population is dominated
  by time-varying, not stationary, structure.

Conclusion: to learn something new we must change the **dimensions** of the
task — depth, time (deliberation), value, uncertainty, and where/when the plan
is read out — not just the level geometry.

---

## 2. Limitations to overcome

| # | Limitation | Consequence today |
|---|---|---|
| L1 | Plan ≡ kinematics (0 shared geometry cells) | choice collinear with trajectory |
| L2 | No independent deliberation-time measure (continuous fall) | RT = transport, not decision |
| L3 | Fixed, shallow horizon (1-step lookahead) | can't measure planning depth |
| L4 | No reward/value — speed only | the "value" variable is path length |
| L5 | Deterministic transitions | can't separate model-based from model-free |
| L6 | Threat = camera mode, coupled to ball height | threat and height confounded |
| L7 | No non-geometric choice-independent variable | neural code is only geometry/value |
| L8 | No pre-commitment/decision epoch | pre-decision null may be a window limit |
| L9 | MTL only | no PFC/ACC conflict or planning contrast |
| L10 | No confidence/gaze | deliberation vs execution unidentifiable |

---

## 3. Design portfolio

### D1. Fixed-kinematics value choice ("two gates") — addresses L1, L4, L7
Extend `cue_goal`: two **geometrically identical** holes; reward (or goal)
signalled by a cue colour; the ball's path is identical for both options.
- **New info:** a value-based choice with zero kinematic confound; the
  definitive value-account test for MTL.
- **Feasibility:** app cue rendering + reward bookkeeping; generator already
  emits cue metadata.
- **Predicts:** a true value axis in MTL, versus the near-orthogonal
  geometry/policy axes observed.

### D2. Planning-depth titration (1-/2-/3-step horizon) — addresses L3
A longer track/tree where the optimal hole requires N-step lookahead, with N
varied per trial (block-wise or interleaved).
- **New info:** planning depth and its RT/neural scaling; a behavioural "depth"
  regressor; links to Mattar et al. 2025 (few rollouts) and Keramati et al.
  2016 (depth-limited planning).
- **Feasibility:** level-generator extension (pluggable cost functions already
  exist in `tools/level_generation/agentic_decision_making.py`); moderate app
  change to show deeper levels.

### D3. Explicit commit / decision epoch (pause-and-release) — addresses L2, L8
Add a phase where the maze is visible and the ball is held; the participant
commits (input), and only then the ball is released.
- **New info:** (a) a time-locked commitment moment → a real pre-decision
  window; (b) deliberation time separated from execution; (c) enables a
  "hold-the-plan" read-out.
- **Feasibility:** moderate `sketch.js` change + event logging (aligns with the
  "Decision-Locked Epoch Markers" roadmap item).

### D4. Prospective / anticipatory probe — addresses L1, L8
Reveal the goal at variable delay while the ball waits; measure prospective
coding (ramping, successor/replay) before movement.
- **New info:** whether MTL encodes the **future** path during planning, not
  just the committed path — the cleanest computation-vs-consequence test.
- **Feasibility:** moderate; pairs naturally with D3.

### D5. Reward + stochastic transitions (spatial two-step task) — addresses L4, L5
Add a probabilistic second step and reward; classic model-based vs model-free
dissociation.
- **New info:** model-based control, eligibility traces, and whether
  hippocampus carries model-based value.
- **Feasibility:** larger app change; strong payoff.

### D6. Non-geometric choice-independent variable — addresses L7
A hidden-goal or reward-magnitude variable orthogonal to path geometry.
- **New info:** directly targets the synthesis's "what would change the story"
  (a richer planning code than value-of-chosen-path).
- **Feasibility:** design-dependent; combine with D1/D5.

### D7. Threat decoupled from camera — addresses L6
Explicit, position-independent threat (deadline/penalty/timer or a death-line
cue).
- **New info:** a clean threat effect for the EMU cohort, removing the
  camera/height confound.
- **Feasibility:** modest.

### D8. Plan report / metacognition — addresses L2, L10
Post-choice probe ("which goal did you plan for?") plus a confidence slider.
- **New info:** whether the plan is accessible independent of the action;
  deliberation vs execution; metacognitive sensitivity.
- **Feasibility:** low (`possible-future-changes.md`: confidence UI + logging).

### D9. Information-demand / Horizon-task previews — addresses L3, individual differences
Intersperse "preview" trials where participants choose to reveal more of the
upcoming maze at a cost (Horizon Task adaptation).
- **New info:** individual differences in information demand, tied to strategy
  clusters.
- **Feasibility:** moderate (already scoped in the roadmap).

### D10. Region coverage (PFC/ACC) — addresses L9
Not a task design, but required to test whether conflict/planning computation
lives outside mesial temporal lobe.

---

## 4. Recommended core redesign

A single variant combining **D3 (pause/commit) + D4 (prospective probe) + D2
(depth titration)** is the highest-yield: it simultaneously provides a true
pre-decision window, separates deliberation from execution, and measures
planning depth — the three things the current design cannot do.

Cheapest decisive confound fix: **D1**. Strong follow-ups: **D5** (model-based
control) and **D7** (clean threat).

---

## 5. Implementation and analysis plan

- `docs/task_redesign_plan.md` (this file) — design spec for D1–D9 with
  manipulations, predictions, sample-size/power notes, preregistration hooks.
- Generator extensions under `tools/level_generation/`:
  - depth-titrated trees (reuse the cost functions in
    `agentic_decision_making.py`),
  - stochastic transitions (transition-probability metadata),
  - cue/reward metadata for D1/D6.
- App prototypes under `app/static/` behind new configs (`app/configs/`):
  - cue rendering (D1/D6),
  - pause/commit phase + decision-locked event logging (D3),
  - goal-preview delay (D4),
  - stochastic/reward transitions (D5),
  - position-independent threat (D7),
  - confidence slider / plan-report probe (D8),
  - preview trials (D9).
- Validation scripts: geometry matching and reward balance (extend
  `analysis/shared_geometry_validation.py`), timing/logging checks.
- Analysis plans: planning-depth estimation, drift-diffusion on trajectories
  (`analysis/diffusion_model.py`), successor/model-based fits, and
  pre-commitment neural decoding using the new commit epoch.

---

## 6. Open questions to resolve before implementation

1. **Cohort.** EMU patients (must be patient-safe/simple) or the online cloud
   cohort (more design freedom)?
2. **App-change appetite.** Level-generation only, or open to `sketch.js`
   changes (pause/commit, cues, rewards, logging)?
3. **Spatial vs abstract.** Keep the ball-fall metaphor or allow button/abstract
   choices?
4. **Scope.** Design document + generator prototypes + validation (design-only),
   or full app implementation?
5. **Priority.** Which limitations matter most — value separation (D1/D5/D6),
   deliberation timing / prospective coding (D3/D4), depth (D2), or
   threat/region (D7/D10)?

---

## Related documents

- `docs/shared_geometry_design.md` — the Tier 1.1 variants and the
  role-swap impossibility result.
- `docs/tier1_findings.md` — value/geometry RSA, dPCA by geometry, axis
  alignment, choice increment, fine pre-decision results.
- `docs/planning_synthesis.md` §7–8 — limitations, next directions, and
  "what would change the story".
- `docs/possible-future-changes.md` — instrumentation roadmap (confidence,
  decision-locked markers, gaze proxy, questionnaires).
