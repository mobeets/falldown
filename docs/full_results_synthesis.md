# Full results synthesis: the Falldown cloud study and the EMU intracranial data

Status: comprehensive results account. This is the exhaustive companion to
`planning_synthesis.md` (which is the argument). It covers every headline
result across the cloud study and the four EMU sessions, how and when people
plan, what the experiment showed about behaviour, and how behaviour links to the
mesial-temporal (MTL) neural data. Machine-readable version:
`analysis/planning_outputs/full_results_master.csv` (86 rows), produced by
`analysis/full_results_summary.py`. Master figure:
`figures/fig_full_results_master.png`.

---

## 0. Datasets and design

**Cloud study** (online, `data/cloud_study`, `data/logs_sorted`). 18 retained
participants across three configs (Definition A completion filter):

| dataset | threat | environment | participants |
|---|---|---|---|
| `exp1_nodrift` | follow (no threat) | balanced | 5 |
| `exp3_altdrift` | alternating drift/follow | high agreement | 9 |
| `exp3_alldrift` | drift | balanced | 4 |

Total ~17,721 experimental 1-2-1 sequences. **EMU**: four sessions from two
epilepsy patients (`yfz_1/2`, `yga_1/2`), 910 trials each (3,640 sequences),
all-drift, 77–124 sorted MTL units (CA, amygdala, anterior hippocampus,
hippocampal body).

**Task.** Three-level sequences: entry → choice (two holes) → goal. Two rules:
`greedy_cost(h) = |entry − h|` (1-step) and `planning_cost(h) = |entry − h| +
|h − goal|` (2-step). On **conflict** trials the rules prescribe different
holes; `chose_planning` marks the 2-step-optimal pick.

---

## 1. How often and when do people plan? (behaviour)

**Prevalence.** `P(plan | conflict)` cloud mean **0.336** (range 0.130–0.464);
EMU mean **0.262** (0.246–0.290). People are near-optimal when the rules agree:
`P(optimal | agreement)` = **0.835**, lapse rate **0.165**. So participants use
both strategies but plan on a minority of conflict trials. `P(plan)` is a
reliable participant summary (**split-half Spearman-Brown 0.93**), but the choice
ICC(participant) is only **0.02**: most choice variance is within-person/trial,
not a stable trait (`planning_reanalysis_reliability.py`).

**Model-based corroboration.** The canonical choice logistic
`logit P(left) = b_1step·z(diff_1step) + b_plan·z(diff_planning) + incoming`
gives a **planning weight** `|b_plan|/(|b_plan|+|b_1step|)` that agrees with the
raw rate at r = **+0.90**. Cloud mean weight 0.270; EMU 0.151 (inside the cloud
range).

**No latent strategy states — but not strictly memoryless.** GLM-HMM BIC selects
**K\* = 1** for all 18 cloud participants and all 4 EMU runs; forced K=2 produces
only weak, non-BIC-supported state separation, and adding a lag input does not
improve the HMM BIC. However, the old stay test was one-sided for excess *stay*
and so missed **alternation**: the within-block lag-1 effect is small and
negative (LPM β = **−0.09**, within-participant permutation p < 0.001), surviving
uncentered block fixed effects (`planning_reanalysis_sequential.py`).

**When: environment.** Within a participant, planning rises with the block's
conflict rate — but the robust estimate is the within-participant GEE **+0.30**
(p = 0.03), robust to block-height and threat controls. The quoted **+0.565** is
an unweighted mean of noisy per-participant slopes (split-half SB = 0.52) and
overstates it; the pooled cross-block GEE is n.s. **EMU shows no such
adaptation** (mean slope **−0.013**, range −0.17…+0.16) — the one clear
cross-dataset divergence.

**When: threat (cloud).** Planning is lower in drift than follow blocks
(univariate **−0.051**, p = 0.0008, block-level GEE; paired t p = 0.010, N = 9),
but this **attenuates to −0.036 (p = 0.09)** once block-mean ball height is
controlled — threat and camera height are coupled. The continuous `ball_y` threat
index is weak/block-level (see caveats below).

**When: ball height, in 5 bins (the mid-screen peak).** Splitting
`ball_y_at_top` into quintiles (`analysis/threat_bins_analysis.py`,
`fig_threat_bins5.png`) gives a **pooled inverted-U**: planning is highest
mid-screen and lower at both extremes.

| cohort / binning | q0 (near death) | q1 | q2 | q3 | q4 (safe/far) | quadratic p |
|---|---|---|---|---|---|---|
| cloud, pooled (drift+alt) | 0.279 | 0.304 | **0.379** | 0.366 | 0.319 | **0.041** |
| cloud, drift only | 0.255 | 0.282 | **0.301** | 0.228 | — | **0.0002** |
| cloud, alternating | 0.349 | 0.403 | **0.425** | 0.406 | 0.365 | 0.202 |
| cloud, within-block-centred | 0.296 | 0.325 | 0.338 | 0.338 | 0.326 | **0.873** |
| EMU, pooled | 0.223 | 0.287 | 0.288 | 0.206 | 0.304 | 0.226 |

- The pooled inverted-U is significant in the cloud data (quadratic term
  negative, p = 0.041; peak at `ball_y ≈ 378 px`) and strongest nominally in the
  drift cohort (p = 0.0002; only 4 realized quantile bins because trials tie at
  `ball_y_at_top = 100`).
- **It does not survive within-block centring** (p = 0.87; drift-only p = 0.55),
  and the per-participant middle-vs-extreme difference is only +0.011 (p = 0.58).
  A drift-only, *non-centred* fit is overlaid in `fig_threat_bins5.png` (panel A)
  and is stronger than the pooled one (b2 = −5.5e-6, p = 0.0002, peak ≈ 270 px),
  so the mid-screen peak is not merely the drift-vs-follow camera contrast — but
  it is still a between-block effect: drift block-mean ball height predicts block
  planning rate (slope = −7.8e-5, p = 0.0024, per-participant r ≈ +0.03). The
  trial-level inverted-U is a shallow, non-significant trend.
- **No shared per-participant "best bin".** Giving each participant their own
  quintiles (`analysis/participant_peak_location.py`,
  `fig_participant_peak_location.png`), the preferred bin scatters — raw peaks
  are bimodal at bins 1 and 4 (4/11 each) and interior peaks are at chance
  (6/11 raw, permutation p = 0.77; 11/18 within-block, p = 0.60). Per-participant
  quadratic curvature is ~50/50 (7/11 raw inverted-U, 9/18 within-block) and peak
  locations are dispersed (SD ≈ 131 px raw, 61 px within-block). The pooled
  mid-screen peak is not a shared individual rule. Drift-only cannot be tested
  per participant (only 1 participant has all 5 bins populated).
- The EMU (all-drift) profile is noisy (4 runs, p = 0.23).
- The neural counterpart mirrors the pooled shape: the 5-bin
  `planning_vs_greedy` decode (`xd_neural_planning_bally.csv`) is highest in the
  middle bins (rate 0.64/0.68 vs 0.56/0.57 at the extremes), while the
  near-vs-far difference becomes n.s. after side/RT/block matching (see §3.2).

**When: individual differences / model.** A threat-aware greedy/planning mixture
exists but is **unstable** (inverse temperatures hit the 20 bound, `p_plan_base`
saturates at 0.98, `bias_dir` at ±8) in both samples; model-free quantities are
used instead.

**Where: level decomposition.** Splitting predictors into within-block / block /
participant components (`analysis/level_decomposition.py`,
`fig_level_decomposition.png`) locates each effect. Geometry is **trial-level**:
value-framing within coefficients `plan_advantage` **+0.108** and
`greedy_advantage` **−0.273** (both p < 1e‑6), and trial-level variance is
**72%** of the (small) explained variance; `ball_y_at_top` has no within-block
effect (p = 0.51) and context is ~**83% participant-level** — the ball-height
confound made quantitative. Note `conflict_mag = plan_advantage +
greedy_advantage` exactly (rank 2 of 3), so those three are never co-fitted;
`conflict_mag` is kept as the standalone choice-independent variable. The
canonical planning weight is stable across levels (within 0.18, block 0.21,
participant 0.16).

**Cross-dataset behavioural comparability.** EMU prevalence, planning weight,
and lapse rate all sit inside the cloud distributions. A geometry-only logistic
fit on the cloud all-drift cohort predicts EMU conflict choices **as well as an
EMU-fitted model** (accuracy **0.738** both; mean log-likelihood −0.568 vs
−0.565) — strong task-level generalization.

---

## 2. What the RTs mean: transport, not deliberation

Raw planning−greedy on conflict trials: decision **+425 ms**, execution
**−815 ms**, total **−390 ms**. This is a kinematic identity:

- The ball falls continuously; a level is passed by the nearest hole when the
  ball drops past it (`objects.js:161-181`). `rt_decision` = travel entry→choice;
  `rt_exec` = travel choice→goal.
- Frame-by-frame phase decomposition (displacement-plateau; `planning_timing.py`)
  splits each interval into **fall / roll / hold**:

| cohort | interval | choice | total | fall | roll | hold |
|---|---|---|---|---|---|---|
| online | decision | greedy | 566 | 213 | 294 | 59 |
| online | decision | planning | 958 | 212 | 639 | 107 |
| online | execution | greedy | 1178 | 219 | 924 | 35 |
| online | execution | planning | 443 | 210 | 202 | 31 |
| EMU | decision | greedy | 515 | 202 | 266 | 47 |
| EMU | decision | planning | 964 | 202 | 695 | 66 |
| EMU | execution | greedy | 1433 | 205 | 1118 | 109 |
| EMU | execution | planning | 479 | 202 | 192 | 85 |

- **Fall time is constant (~201–219 ms) in every condition**: the vertical
  physics does not depend on the choice. The entire RT trade-off is horizontal
  **roll**.
- **No dwell/hesitation:** planning−greedy hold difference is non-significant
  (online decision +30 ms, p = 0.35; execution −16 ms, p = 0.13; EMU n = 4
  unpowered).
- **Simpson's paradox:** decision and execution time are uncorrelated *within* a
  choice (r ≈ −0.1…+0.4) and only negatively correlated when the geometrically
  disjoint choice sets are pooled.
- **Non-identifiable:** 0/48 (online) and 0/35 (EMU) exact `(1-step, 2-step)`
  cells contain both choices. A nested-control battery shows the post-path
  "residual" is **specification-dependent and not momentum**: adding entry
  velocity controls changes nothing, and a spline in path length makes the
  online residual non-significant (+29 ms, p = 0.16; decision +7 ms, p = 0.16).

**Rule:** no planning-time effect can be inferred from these RTs.

---

## 3. Neural results (four EMU sessions)

### 3.1 Headline effects (mean over sessions, same-sign count)

| effect | representation | mean | sign | note |
|---|---|---|---|---|
| `planning_vs_greedy` post | rate | 0.691 | 4/4 | ~0.94 PCA |
| `planning_vs_greedy` post | pca | 0.939 | 4/4 | near-deterministic |
| `planning_vs_greedy` pre | rate | 0.494 | — | chance |
| `entry+250` planning-vs-greedy | pca | 0.508 | 4/4 (n.s.) | no pre-decision signal |
| `condition` post | pca | 0.414 | 4/4 | chance 0.25 |
| `agree` post | pca | 0.590 | 4/4 | |
| `conflict_mag` post, both | pca | 0.242 | 4/4 | survives residualization |
| `conflict_mag` post, both | rate | 0.062 | 4/4 | |
| `chosen_planning_cost` post | pca | 0.473 | 4/4 | **collapses** under residualization |
| `threat:rel_y` post, both | pca | 0.043 | 4/4 | n.s. |
| bursty-removed `conflict_mag` | pca | 0.238 | 4/4 | not burst-driven |
| death vs normal | rate | 0.524 | 2/4 | exploratory |

### 3.2 Confound audits

- **Temporal leakage:** leave-block-out CV reproduces shuffled CV for every
  headline metric (largest |Δ| ≈ 0.06; lag-1 autocorrelation of planning ≈ 0;
  condition × block Cramér's V ≈ 0.17).
- **Position:** adding `ball_x/y`, `camera_y`, `rel_y` leaves `conflict_mag`
  unchanged (e.g. yfz_1 0.301 → 0.301).
- **Region unit-count matched:** every region still decodes
  `planning_vs_greedy` above chance at matched N (0.49–0.68).
- **Threat is a timing signal:** `rel_y` correlates with RT (|r| = 0.13–0.48),
  duration (0.12–0.37), `camera_y` (0.08–0.22); its decoding collapses under
  residualization.
- **Matched re-decoding (yfz_1):** `planning_vs_greedy` survives side/RT/block
  matching (rate 0.68–0.72; pca 0.95–0.97) but **cannot be matched on ball
  time** — only 16/431 trials share a traversal-time stratum. By contrast
  `planning_optimal` and `agree` fall to chance once ball time is matched
  (pca 0.51/0.50): a clean dissociation between a robust policy signal and a
  purely kinematic one.
- **Ball-height modulation of the policy decode is itself confounded:** near/far
  splits give a large unmatched difference (yfz_1 0.72 vs 0.54, p = 0.005) that
  shrinks to 0.05 after side/RT/block matching; cluster tests across ball-height
  bins are n.s.

### 3.3 Regions

Per-region `planning_vs_greedy` (post, rate, unit-count matched) is above chance
in every region with enough units — CA, amygdala, anterior hippocampus,
hippocampal body — with no single dominant region. Policy information is
distributed across MTL. RSA of population geometry against model RDMs
(`greedy_gap`, `planning_gap`, `conflict_mag`) is weak (~0) — an exploratory
negative.

---

## 4. Linking behaviour to neurons

- **Neural adds choice information beyond geometry.** An out-of-fold neural
  planning score improves a cross-validated geometry-only logistic in all four
  runs: accuracy 0.710→0.789, 0.754→0.875, 0.753→0.783, 0.767→0.783
  (mean +0.06). The **pre-decision** neural score adds **nothing**
  (Δ ≤ 0.005).
- **Neural conflict code is not timing.** The out-of-fold `conflict_mag`
  prediction does not track the kinematic-adjusted RT residual (|r| ≤ 0.06).
- **Model transfer:** cloud-fitted geometry predicts EMU choices as well as
  EMU-fitted (0.738).
- **Behavioural–neural convergence:** the choice-independent quantity the
  behavioural model uses (`diff_planning`) and the quantity the population
  encodes (`conflict_mag`) are two views of the same policy-conflict geometry;
  the binary planning/greedy label is the committed-choice consequence.

---

## 5. Claim ledger (what is and is not confounded)

| # | Claim | Status |
|---|---|---|
| 1 | Planning is graded, minority, environment-sensitive | **solid** (cloud) |
| 2 | No latent strategy states (K*=1) | **solid** (18/18, 4/4) |
| 3 | Environment adaptation | cloud **solid**; EMU **absent** |
| 4 | Threat reduces planning | cloud **moderate** (p=.010, N=9); EMU **inconclusive** |
| 5 | Planning/greedy RT trade-off | **kinematic** (fixed fall, roll only) |
| 6 | Post-path RT residual | **rejected** (spec-dependent; not momentum) |
| 7 | Post-choice policy decode (4/4) | **real but consequence-dominated** |
| 8 | Pre-decision planning signal | **absent** (4/4) |
| 9 | `conflict_mag` (choice-independent) encoded | **robust** (all controls) |
| 10 | Neural adds choice info over geometry | **solid** (+0.06, 4/4) |
| 11 | Cloud model transfers to EMU | **solid** |
| 12 | Distributed across MTL regions | **solid** |
| 13 | Continuous threat neural code | **negative** (timing confound) |
| 14 | Death code | **exploratory** (1/2 sessions) |
| 15 | Ball-height modulation of policy decode | **confounded** (n.s. after matching) |

---

## 6. What the experiment shows

**About behaviour.** Planning is not a mode or a state: it is a graded,
context-sensitive policy — used on ~1/3 of decisions, increasing when the
environment makes it relevant, decreasing under time pressure/threat, stable
within a person, and not associated with time spent deciding once the physics is
accounted for. People are near-optimal when the choice is easy and mix strategies
when it is not, consistent with a value comparison over 1-step vs 2-step paths
rather than an effortful deliberative process.

**About the brain (MTL).** The population encodes a **choice-independent value/
geometry signal** (`conflict_mag`) that survives every confound control. It also
separates the committed policies almost perfectly *after* the choice — but that
is the trajectory the choice produces, not a plan computed beforehand. There is
**no pre-decision plan read-out**. The most defensible synthesis: MTL represents
the value/geometry of the option/action that is chosen (consistent with
hippocampal/amygdala value and cognitive-map roles); a "planning computation", if
it exists, is fast and leaves no held-plan trace in these recordings.

**About method.** In a task where the two policies imply different paths, the
policy label and the kinematics are collinear (0/48 overlapping geometries). Raw
RTs and post-choice neural decodes are therefore confounded by construction, and
any adjustment is an extrapolation. Reporting choice-independent geometry
variables and pre-decision windows is the only clean route.

---

## 6b. Ancillary results

Beyond the main narrative, a systematic sweep produced a large set of secondary
results, now catalogued in **`docs/ancillary_results.md`**. Headlines that
qualify or constrain the story:

- **Transition structure** has no latent states (forced GLM-HMM K=2 not
  BIC-supported; 0/18 excess *persistence*), but is **not strictly memoryless**:
  the one-sided stay test missed a small negative within-block lag-1 dependence
  (alternation, LPM β = −0.09, perm p < 0.001).
- **Environment sensitivity** is confirmed within-subject (+0.312, p = 0.044;
  planning-advantage +0.144, p = 0.012) but the between-cohort comparison
  (0.39 vs 0.28) is confounded with the drift design.
- **Continuous threat** is weak (near−safe −0.03, p = 0.30) and the per-participant
  `ball_y` slopes are mixed; the mixture threat parameter is unidentified.
- **Per-unit selectivity for planning is sparse** (≤4 units per session) while
  **left/right motor selectivity is far more prevalent** (54–58 units in YFZ) —
  a key control. Spatial tuning is essentially absent (1 unit).
- **RSA/value/dPCA** are weak or negative; the one dPCA side effect is a flagged
  circular artifact.
- **Transfer scaling** is positive but modest (0.754→0.773; matched logistic
  0.780); the multi-task RT-loss bug that collapsed choice to chance is fixed.
- Data/QC notes (122 units, DTW 126.9 s / 4.5 ms residual; one genuine death in
  yfz_1) and the deprecated `exp2_shift` outputs are documented there.

---

## 7. Next directions

**Tier 1 — close the central question**
1. A task variant where the 1-step and 2-step rules share path lengths
   (cue-coloured holes / equi-distant choices) to break the geometry–policy
   entanglement.
2. Direct test of the value/geometry account: relate `conflict_mag`/
   `planning_gap` codes to model-based values and to place/successor structure;
   ask whether the code predicts choice beyond `diff_planning` out-of-fold.
3. Higher-temporal-resolution pre-decision analysis (theta/ripple-locked sliding
   decoders from the entry pass) to rule out a brief early computation.

**Tier 2 — extend the evidence base**
4. More EMU sessions and non-MTL (PFC/ACC) coverage.
5. Planning-depth estimation (1-/2-/3-step tree search) and bounded
   drift-diffusion fitting.
6. Instrumentation: confidence ratings, decision-locked epoch markers, gaze
   proxy.

**Tier 3 — generalize and communicate**
7. Cross-dataset/cross-species transfer with the cloud model as a prior.
8. Publish the confound protocol and the geometry non-identifiability result as
   a reusable checklist.

---

## 8. Artifact index

- Master: `analysis/planning_outputs/full_results_master.csv`,
  `figures/fig_full_results_master.png`, `analysis/full_results_summary.py`.
- Behaviour: `analysis/planning_outputs/` (`exp1_*`, `exp2_*`, `exp3_*`).
- Ball-height 5-bin / threat: `analysis/threat_bins_analysis.py`,
  `analysis/planning_outputs/threat_bins5_*.csv`, `figures/fig_threat_bins5.png`.
- Level decomposition (within/block/participant): `analysis/level_decomposition.py`,
  `analysis/planning_outputs/level_decomposition_*.csv`,
  `figures/fig_level_decomposition.png`.
- Per-participant peak: `analysis/participant_peak_location.py`,
  `figures/fig_participant_peak_location.png`.
- Claim re-analysis: `analysis/planning_reanalysis_{reliability,sequential,block}.py`,
  `analysis/claim_audit.py` → `planning_outputs/fig_claim_audit.png`,
  `claim_audit.csv`; superseded figures in `analysis/superseded_figures/`.
- Timing: `analysis/planning_outputs/timing_*.csv`, `figures/fig_timing_*.png`;
  `docs/timing_decomposition.md`.
- Neural: `analysis/neural_outputs/<run>/`, `aggregate/`;
  `docs/neural_multi_run_findings.md`, `docs/neural_decoding_confounds.md`.
- Cross-dataset: `analysis/cross_dataset_outputs/`;
  `docs/cross_dataset_findings.md`.
- Ancillary results: `docs/ancillary_results.md`.
- Scaling figures: `analysis/scaling_figures.py`,
  `analysis/figures/scaling_accuracy_*.png`.
- Argument-level synthesis: `docs/planning_synthesis.md`.

## Caveats

- n = 2 EMU patients (4 runs); sign-flip tests floor at p ≈ 0.125.
- Cloud trajectory coverage is 65% (logging artifact).
- Continuous threat in cloud is weaker than the block-level drift contrast;
  the 5-bin mid-screen peak is between-block/camera-confounded and does not
  survive within-block centring; EMU threat is inconclusive.
- Mixture parameters are unstable; model-free summaries are used.
- RSA/geometry comparisons are exploratory and weak.
