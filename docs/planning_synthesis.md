# Planning in the brain: synthesis of the Falldown results and next directions

Status: synthesis. This document ties the behavioural, timing, neural, and
cross-dataset results into one argument, states what is and is not confounded,
and sets the next research directions. It is the top-level companion to
`cross_dataset_findings.md`, `neural_multi_run_findings.md`,
`neural_decoding_confounds.md`, and `timing_decomposition.md`.

---

## The story in one paragraph

In this task **planning is a graded choice over two-step path geometry, not a
slow deliberative state.** People plan on a minority of conflict trials (~34%),
more when the block's conflict rate is high (a real but modest block-level
effect, +0.30 not +0.57), with no latent strategy states — though a mild negative
within-block sequential dependence (alternation) — and no RT
cost once path geometry is controlled — the entire planning/greedy RT trade-off
is horizontal transport, with a fixed ~210 ms fall time and no dwell. The
mesial-temporal population strongly distinguishes planning from greedy
*post-commitment*, but that signal is largely the kinematic consequence of the
chosen path and cannot be matched away (the two choices never share geometry).
What survives every control is the **choice-independent geometry variable**
(`conflict_mag`): the population tracks how much the two policies disagree
*before* the choice. There is **no pre-decision planning signal**. So the neural
read-out is best described as representing the value/geometry of the committed
choice — a genuine decision variable — rather than executing a time-consuming
plan; any planning computation is fast, and its trace in mesial temporal lobe is
the selected path, not a held plan.

---

## 1. What planning is in this task (behaviour)

- **Minority, graded, individual — but mostly within-person.** `P(plan | conflict)`
  = 0.13–0.46 (mean 0.34) across 18 online participants; EMU runs 0.25–0.29,
  inside the cloud distribution. `P(plan)` and the planning weight are reliable
  participant summaries (split-half Spearman-Brown 0.93), but the choice
  ICC(participant) is only **0.02**: most choice variance is trial/within-person,
  not a stable trait (`planning_reanalysis_reliability.py`, `fig_claim_audit.png`).
- **No latent strategy state, but not strictly memoryless.** GLM-HMM BIC selects
  K* = 1 for every online participant and every EMU run, and adding a lag input
  does not improve the HMM BIC. However, the old stay test was one-sided for
  excess *stay* and so missed alternation: the within-block lag-1 effect is small
  and **negative** (LPM β = −0.09, within-participant permutation p < 0.001),
  surviving uncentered block fixed effects (`planning_reanalysis_sequential.py`;
  the old `fig_planning_temporal` is superseded).
- **Context-sensitive (block-level).** Within a participant, planning rises with
  the block's conflict rate: within-participant GEE **+0.30** (p = 0.03), robust
  to block-height and threat controls. The previously quoted **+0.57** is an
  unweighted mean of noisy per-participant slopes (split-half SB = 0.52) and
  overstates it; the pooled cross-block GEE is n.s. EMU shows no such
  within-session adaptation (mean −0.01).
- **Threat-sensitive (cloud), coupled with camera height.** Drift blocks have
  less planning than follow: univariate Δ = −0.051 (p = 0.0008, block-level GEE),
  attenuating to **−0.036 (p = 0.09)** once block-mean ball height is controlled.
  In EMU the threat result is inconclusive (rate vs coefficient disagree;
  all-drift so no follow contrast).
- **Ball height in 5 bins: a mid-screen peak, but block-confounded.** Pooled
  quintiles give an inverted-U (planning highest mid-screen; quadratic p = 0.041,
  drift-only p = 0.0002) that **does not survive within-block centring**
  (p = 0.87; drift-only p = 0.55). The drift-only, *non-centred* fit overlaid in
  `fig_threat_bins5.png` is even stronger than the pooled one (b2 = −5.5e-6,
  p = 0.0002, peak ≈ 270 px) — but it realizes only 4 quantile bins (trials tie
  at 100 px, q0 n = 996) and is still between-block: drift block-mean height
  predicts block planning rate (slope = −7.8e-5, p = 0.0024, per-participant
  r ≈ +0.03). It is a between-block/camera-position effect, not a trial-level
  height effect (`threat_bins_analysis.py`, `threat_bins5_drift_pooled.csv`).
  The neural 5-bin decode mirrors the pooled shape but the near/far difference
  is n.s. after matching.
- **No shared per-participant "best bin".** Allowing each participant their own
  quantiles, the preferred (argmax) ball-height bin scatters: raw peaks are
  bimodal at bins 1 and 4 (4/11 each), interior-bin peaks are at chance (6/11
  raw, permutation p = 0.77; 11/18 within-block, p = 0.60), per-participant
  quadratic curvature is ~50/50 (7/11 raw, 9/18 within-block inverted-U), and
  peak locations are widely dispersed (SD ≈ 131 px raw). So "people plan best in
  the middle" is not a generally true individual rule
  (`participant_peak_location.py`, `fig_participant_peak_location.png`).
  Drift-only cannot be tested per participant (only 1 participant has 5
  populated bins).
- **Model-based summary agrees and is reliable.** Planning weight
  `|b_plan|/(|b_plan|+|b_1step|)` from the choice logistic correlates r = +0.90
  with `P(plan|conflict)`, has split-half SB = 0.93, and is stable across levels
  (within 0.18 / block 0.21 / participant 0.16); EMU sits inside the cloud range
  (`xd_planning_weight.csv`).
- **Geometry is value, not a bare "conflict magnitude".**
  `conflict_mag = plan_advantage + greedy_advantage` exactly (design rank 2 of 3),
  so the magnitude is the sum of the two policy regrets; the value framing fits
  better (AIC 7091 vs 7394) and the three must never be co-fitted. This reframes
  the neural "conflict_mag" code as consistent with value coding
  (`level_decomposition.py`, `planning_reanalysis_block.py`).

**Reading:** planning is a flexible, geometry-driven policy — a point on a
continuum of look-ahead — not a discrete cognitive mode.

## 2. The RT confound, resolved (timing)

The raw planning/greedy RT trade-off (decision **+425 ms**, execution
**−815 ms**, total **−390 ms**) is a kinematic identity, not deliberation:

- The ball falls continuously and a level is passed by the nearest hole when the
  ball drops past it (`app/static/objects.js:161-181`). `rt_decision` is travel
  to the chosen hole; `rt_exec` is travel to the goal.
- Frame-by-frame decomposition (`planning_timing.py`,
  `timing_decomposition.md`) shows the airborne component is **constant
  (~202–219 ms) under every condition**; the entire difference is horizontal
  **roll/transport**.
- **No dwell/hesitation:** planning−greedy hold-time difference is not
  significant (online decision +30 ms, p = 0.35; execution −16 ms, p = 0.13;
  EMU n = 4 unpowered).
- The trade-off is a **Simpson's paradox**: decision and execution time are
  uncorrelated *within* a choice (r ≈ −0.1…+0.4) and only negatively correlated
  when the two disjoint choice sets are pooled.
- **Non-identifiable by construction:** 0/48 exact `(1-step, 2-step)` geometry
  cells (online) contain both choices (0/35 EMU). The post-path residual is
  **specification-dependent and becomes n.s. under a spline** (online execution
  +29 ms, p = 0.16); velocity controls do not attenuate it, so it is not
  momentum.

**Rule:** the raw RT effect must not be interpreted as a planning-time cost.

## 3. What the mesial-temporal population says (neural)

Four EMU sessions (2 patients, 81–124 units, 910 trials each; all-drift).

**Robust positives**

- **Choice-independent geometry is encoded and survives controls.**
  `conflict_mag = |planning_gap − greedy_gap|` (a property of the maze) is
  decodable in all four sessions (r = 0.09–0.46) and survives target+feature
  residualization, position controls, leave-block-out CV, and bursty-unit
  removal (`fig_neural_confound_collapse`, `neural_multi_run_findings.md`).
  This is the strongest, cleanest result.
- **Policy information is distributed.** Every recorded region (CA, amygdala,
  anterior hippocampus, hippocampal body) decodes `planning_vs_greedy` above
  chance at matched unit count (`region_count_matched.csv`).
- **The cloud decision rule transfers to EMU.** A geometry-only logistic fit on
  the cloud all-drift cohort predicts EMU choices as well as an EMU-fitted
  model (accuracy 0.738 both; `xd_transfer.csv`).

**Robust cautions**

- **Post-choice policy decoding is largely consequence.** `planning_vs_greedy`
  decodes at 0.63–0.78 (rate) / 0.89–0.98 (PCA) post-choice in all four
  sessions, but `planning_vs_greedy × ball_time` r = 0.87 and the classes share
  only 16 trials in a traversal-time stratum — the signal cannot be separated
  from the trajectory the choice produces.
- **No pre-decision signal.** Entry-anchored windows (`entry+250 ms`) are at
  chance in all four sessions in both rate and PCA representations
  (`fig_neural_window_profile`, `entry_locked_lda_results_pca.csv`). This is the
  clean "computation vs consequence" test, and it is null.
- **Model-based outcome variables collapse.** `chosen_planning_cost` is large
  raw (0.50–0.68) but collapses under residualization (≈0.11–0.20) — its
  representation was the post-choice kinematics.
- **Continuous threat is a timing signal.** `ball_y − camera_y` decoding
  collapses under residualization; it correlates with RT (|r| ≈ 0.40) and trial
  duration (0.25).
- **Ball-height modulation of the policy decode is itself confounded.** Near vs
  far splits give a large difference unmatched (e.g. yfz_1 0.72 vs 0.54,
  p = 0.005) that shrinks to 0.05 after side/RT/block matching
  (`xd_neural_planning_controlled.csv`); cluster tests across ball-height bins
  are n.s. (`xd_neural_planning_clusters.csv`). Treat as inconclusive.
- **Death decoding is exploratory** (one of two YGA sessions; block-matching
  survives but n is small).

**Connecting behaviour to neurons.** The bridge that survives is
**geometry, not RT**: an out-of-fold neural planning score adds choice
information beyond `diff_planning` (+0.06 accuracy; `xd_neural_linkage.csv`),
while the pre-decision score adds nothing, and neural `conflict_mag` does not
track the kinematic-adjusted RT residual (|r| ≤ 0.06).

## 4. The synthesis

Two interpretations are compatible with the data; the evidence favours the
second as the positive claim:

- **(A) Consequence, not computation.** The post-choice binary decode reflects
  the selected trajectory; no planning computation is visible in MTL. *Supported
  by the pre-decision null and the n=16 ball-time non-overlap.*
- **(B) Value/geometry representation.** MTL encodes a *choice-independent*
  decision variable — the conflict/value of the option (`conflict_mag`),
  consistent with hippocampal/amygdala roles in value and cognitive maps. The
  binary "planning vs greedy" decode is the post-commitment read-out of that
  choice. *Supported by conflict_mag surviving every control.*

The task's geometry makes A and B hard to separate: planning and greedy choices
never share a path, so the policy label and the kinematics are collinear. The
honest summary is: **MTL represents the geometry/value of the chosen path; there
is no evidence for a time-consuming planning computation, and the only
choice-independent planning-relevant code is the conflict/geometry signal.**

## 5. Claim ledger

| # | Claim | Evidence | Confound status |
|---|---|---|---|
| 1 | Planning is graded/minority | prevalence 0.34; P(plan) split-half SB=.93 | clean |
| 1b | Planning is an individual trait | ICC(participant)=.02 | **relabeled**: mostly within-person |
| 2 | No latent strategy states (K*=1) | GLM-HMM BIC 18/18, 4/4; lag input worsens BIC | clean |
| 2b | Strategy choices are memoryless | within-block lag-1 β=−.09, perm p<.001 | **qualified**: mild alternation; one-sided stay test missed it |
| 3b | Planning rises with block conflict rate (+0.57) | within-participant GEE +.30 (p=.03), robust to controls | **revised**: +.57 is a noisy unweighted mean (SB=.52); pooled GEE n.s. |
| 3 | RT trade-off is kinematic transport | timing phases; constant fall; no dwell | resolved: confound |
| 4 | Post-path RT residual is a real effect | log-GEE −0.28 | **rejected**: n.s. under spline; not momentum |
| 5 | Geometry is non-identifiable across choices | 0/48 exact, 0/35 EMU cells | structural |
| 6 | Post-choice policy decode (4/4) | rate 0.63–0.78; PCA 0.89–0.98 | consequence (r=.87 ball time) |
| 7 | No pre-decision planning signal | entry+250 ≈ chance, 4/4 | clean |
| 8 | `conflict_mag` (choice-independent) encoded | r=.09–.46, survives all controls | **robust** |
| 9 | Neural score adds choice info over geometry | +0.06 accuracy, OOF | supported |
| 10 | Distributed across MTL regions | unit-count matched | supported |
| 11 | Cloud model transfers to EMU | acc .738 = | supported |
| 12 | Threat shifts choice (cloud) | univariate −.051 (p=.0008); with block height −.036 (p=.09) | **qualified**: threat and camera height coupled; EMU inconclusive |
| 12e | `conflict_mag` is choice-independent geometry | `conflict_mag = plan_advantage + greedy_advantage` exactly | **relabeled**: a value sum; value framing fits better (AIC 7091 vs 7394) |
| 12b | Planning peaks mid-screen (5 bins) | pooled p=.041; drift-only non-centred p=.0002 | **block-confounded**; n.s. within block (p=.87; drift p=.55) |
| 12c | Per-participant "best bin" is interior | peaks scatter; interior at chance (perm p=.77) | **rejected** (no shared rule) |
| 12d | Geometry acts trial-level; context is block/participant | Mundlak within p<1e‑6; trial 72% of explained variance | **robust** (`level_decomposition.py`) |
| 13 | Continuous threat neural code | — | negative (timing confound) |
| 14 | Death code | yga_2 0.55 | exploratory (1/2) |

## 6. Confound protocol (adopt going forward)

1. **Never interpret `rt_decision`/`rt_exec` as deliberation.** Report only the
   model-free total RT and phase decomposition; state the fixed fall time.
2. **Prefer choice-independent geometry variables** (`conflict_mag`,
   `diff_planning`, `greedy_gap`, `planning_gap`) as neural targets.
3. **Use the pre-decision window as the only clean test of a planning
   computation**, and report its null.
4. **For any post-choice decoding, report the label × traversal-time
   association** and the overlap count; label it a consequence unless matched.
5. **Any residual after kinematic adjustment must survive a functional-form
   sweep** (raw, log, spline, nonparametric) before being called real.
6. **Decompose predictors by level** (within-block / block / participant;
   `level_decomposition.py`) before interpreting them. Only a within-block
   coefficient licenses a trial-level claim. The ball-height U is the worked
   example: within p = 0.51, i.e. block/participant-level. Also check algebraic
   redundancy first — `conflict_mag = plan_advantage + greedy_advantage`
   exactly, so these three must never be fitted together.
7. **Report split-half reliability for any participant-level summary**, and test
   sequential structure **two-sided**. The one-sided "excess stay" test missed a
   real negative lag-1 dependence (alternation) in this dataset.

## 7. Open questions and limitations

- n = 2 EMU patients, all-drift; sign-flip tests floor at p ≈ 0.125.
- The pre-decision null could be a window, representation, or *region* limit
  (MTL only; no PFC/ACC).
- Online `game_states` coverage is 65% (a logging artifact); early-block trials
  are under-sampled.
- Threat has no clean EMU manipulation; death events are few.
- No confidence reports, no gaze, no explicitly time-locked decision epoch.

## 8. Next directions (prioritized)

**Tier 1 — close the central question** (run; see `docs/tier1_findings.md`)

1. **Break the geometry–policy entanglement by design.** [done: design +
   validation] Exact matched pairs are **impossible** under the 1-step greedy
   rule; the current pool has 0 shared `(1-step, 2-step)` cells. `cue_goal` is
   the only variant that removes the confound by construction
   (`shared_geometry_design.md`, `shared_geometry_validation.py`).
2. **Test the value/geometry account directly.** [done] The MTL value/geometry
   code is **not** a successor/place code (RSA |r|<0.01; geometry vs policy dPCA
   axes near-orthogonal; value-vs-place/SR axis alignment ≈ 0). Post-choice
   choice decoding is 0.69 OOF, entry 0.48. Audit: `xd_neural_linkage.csv`'s
   fixed `n_conflict=431` should be 418.
3. **Re-examine the pre-decision window at higher temporal resolution.** [done
   for spikes] Fine entry sliding (50 ms, FDR over time) shows **0/36**
   significant bins; temporal generalization ≈ chance. Theta/ripple remains
   blocked (no LFP in-repo).

**Tier 2 — extend the evidence base**

4. **More EMU sessions and non-MTL coverage** (PFC/ACC) to test whether conflict
   computation lives outside mesial temporal lobe.
5. **Planning-depth estimation** (1-/2-/3-step tree search) per participant, and
   a bounded drift-diffusion fit to characterise the choice process.
6. **Instrumentation:** confidence ratings, decision-locked epoch markers, gaze
   proxy — to separate deliberation from execution and test metacognition.

**Tier 3 — generalize and communicate**

7. **Cross-dataset/cross-species transfer** using the cloud model as a prior;
   report per-participant planning weights.
8. **Publish the confound protocol** as a reusable checklist for
   decision-making neuroscience (the geometry-separation non-identifiability
   result is broadly relevant).

### What would change the story

- A reliable pre-decision MTL read-out of the plan → "computation", not just
  consequence.
- A design where planning/greedy share geometry and an RT or neural difference
  remains → a genuine non-kinematic planning signal.
- A choice-independent variable other than geometry that predicts choices and
  is neurally encoded → a richer planning code than "value of the chosen path".

Concrete task designs that would test these are collected in
`docs/task_redesign_plan.md` (pause/commit epoch, prospective probe, planning-depth
titration, fixed-kinematics value choice, stochastic two-step reward, decoupled
threat).

---

## Artifact index

- Behaviour: `analysis/planning_outputs/` (+ `exp3_*`, `exp1_*`, `exp2_*`);
  ball-height bins `threat_bins_analysis.py` (incl. `threat_bins5_drift_pooled.csv`),
  per-participant peaks `participant_peak_location.py`, level decomposition
  `level_decomposition.py` (`level_decomposition_*.csv`,
  `fig_level_decomposition.png`), and the claim re-analysis
  `planning_reanalysis_{reliability,sequential,block}.py` + `claim_audit.py`
  (`claim_audit.csv`, `fig_claim_audit.png`). Superseded figures:
  `analysis/superseded_figures/`.
- Timing: `analysis/planning_outputs/timing_*.csv`,
  `figures/fig_timing_*.png`; write-up `docs/timing_decomposition.md`.
- Neural: `analysis/neural_outputs/<run>/` and `aggregate/`; write-up
  `docs/neural_multi_run_findings.md`, `docs/neural_decoding_confounds.md`.
- Tier 1: `docs/tier1_findings.md`; `analysis/neural_successor_geometry.py`,
  `neural_dpca_geometry.py`, `neural_axis_alignment.py`,
  `neural_choice_increment.py`, `neural_entry_sliding_fine.py`;
  `docs/shared_geometry_design.md` +
  `tools/level_generation/generating_levels_sharedgeom.py` +
  `analysis/shared_geometry_validation.py`; future task designs
  `docs/task_redesign_plan.md`.
- Cross-dataset: `analysis/cross_dataset_outputs/`; write-up
  `docs/cross_dataset_findings.md`.
- Confound audits: `analysis/neural_confound_audit.py`,
  `analysis/neural_decoding_confounds.py`.
