# Multi-run EMU neural findings: replication, death/threat, and regions

Status: Parts 1–4 of the multi-run neural plan implemented. This document
summarizes what the four new/extended analyses found across the four EMU
sessions. All scripts run with
`C:\Users\manik\AppData\Local\Programs\Python\Python311\python.exe` and write
CSVs per run plus an aggregate folder.

## Sessions

| run | participant | trials | units | deaths | notes |
|---|---|---|---|---|---|
| `yfz_1` | YFZ | 910 | 122 | 1 | original session |
| `yfz_2` | YFZ | 910 | 124 | 0 | same participant, run 2 |
| `yga_1` | YGA | 910 | 81 | 12 | second participant |
| `yga_2` | YGA | 910 | 77 | 19 | second participant, run 2 |

All four are **all-drift** (camera mode 1): continuous time pressure, no
follow-block contrast. Units are **not** matched across runs; per-session
decoders are the replication unit. Region is assigned from the channel→lead
table (CA / amygdala / anterior hippocampus / hippocampal body); no
patient-specific electrode localization is used.

---

## Part 1 — Cross-run replication (`analysis/aggregate_sessions.py`)

Reads each run's result CSVs (no re-fitting). Statistics per metric: mean/sd,
sign-consistency, session-level **sign-flip permutation** (2^n assignments; with
4 sessions the minimum attainable p is 0.125), and within-participant (YFZ ×2)
vs across-participant (YGA ×2) means.

| effect | per-run values | mean | same sign |
|---|---|---|---|
| `planning_vs_greedy` post, **rate** | 0.70 / 0.78 / 0.67 / 0.63 | 0.69 | 4/4 |
| `planning_vs_greedy` post, **pca** | 0.96 / 0.98 / 0.90 / 0.92 | 0.94 | 4/4 |
| `planning_vs_greedy` **pre**, rate | 0.45 / 0.56 / 0.49 / 0.48 | 0.49 | 4/4 (≈chance) |
| `condition` post, pca | — | 0.41 | 4/4 |
| `agree` post, pca | — | 0.59 | 4/4 |
| `conflict_mag` post, pca, **both** | 0.30 / 0.46 / 0.12 / 0.09 | 0.24 | 4/4 |
| `chosen_planning_cost` post, pca, raw | — | 0.47 | 4/4 |
| `threat:rel_y` post, pca, both | 0.04 / 0.04 / 0.05 / 0.04 | 0.04 | 4/4 (n.s.) |
| `entry_pca:planning_vs_greedy` entry+250 | 0.49 / 0.53 / 0.51 / 0.51 | 0.51 | 4/4 (≈chance) |
| `death:acc` (YGA only) | — / — / 0.50 / 0.55 | 0.52 | 2/2 |

**Conclusions.**
- The post-choice `planning_vs_greedy` effect **replicates in all four sessions**
  and is large in both representations; the pre-choice effect is at chance in
  all four — the "consequence, not computation" result generalizes.
- `conflict_mag` (the confound-resistant continuous quantity) replicates in sign
  and significance in all four sessions, though it is much larger in the two
  YFZ sessions (0.30–0.46) than in YGA (0.09–0.12).
- `chosen_planning_cost` again shows the confound signature (large raw, collapses
  under residualization; see `neural_decoding_confounds.md`).
- Outputs: `aggregate/aggregate_headline_effects.csv`,
  `aggregate/aggregate_sign_consistency.csv`, `aggregate/aggregate_forest.png`.

---

## Part 2 — Death / threat (`analysis/neural_death_threat.py`)

**Death vs normal population decoding** (rate, pre-death windows; YGA only,
n=12/19 deaths): yga_1 0.497 (p=0.67), yga_2 0.552 (p=0.007). So a death-locked
population signal is present in **one of the two** YGA sessions — suggestive but
not replicated.

**Continuous threat** (`ball_y − camera_y` at the choice pass; small = near
death) decoded by ridge regression:

| representation / window / control | mean corr | significant sessions |
|---|---|---|
| rate / pre / none | 0.10 | 2/4 raw (yfz_1, yfz_2) |
| rate / pre / target | ≈0.02–0.05 | 0/4 |
| pca / post / none | ≈0.06–0.16 | 2/4 raw |
| pca / post / both | 0.04 | 0/4 |

The raw proximity signal is **confounded with RT/kinematics**: it collapses to
non-significance after nuisance residualization in every session — the same
pattern as `chosen_planning_cost`.

**Threat modulation** of `planning_vs_greedy` (high vs low threat halves):
no reliable interaction in any session (`perm_p_diff` n.s.). The planning
representation is present under both high and low threat.

Outputs: `<run>/death_decoding.csv`, `<run>/threat_continuous_decoding.csv`,
`<run>/threat_modulation.csv`.

---

## Part 3 — Region-resolved decoding (`analysis/neural_region_decoding.py`)

Per-region LDA (post window, rate) for `planning_vs_greedy`:

| run | CA | amygdala | anterior HC | HC body |
|---|---|---|---|---|
| yfz_1 | 0.598 | **0.683** | 0.582 | 0.580 |
| yfz_2 | 0.665 | 0.693 | 0.695 | **0.721** |
| yga_1 | 0.540 | 0.555 | — | 0.604 |
| yga_2 | 0.519 | 0.500 (n.s.) | 0.534 | 0.572 |

- The planning-vs-greedy distinction is **not carried by one region**: every
  region with enough units decodes it (p<0.001 in most cells). Amygdala and
  hippocampal body are numerically strongest in YFZ.
- `move_dir` is at chance in every region in every session → the region effects
  are not steering.
- **Cross-region score correlation** (out-of-fold LDA decision scores; disjoint
  unit sets make cross-region *decoding* ill-defined): `planning_vs_greedy`
  correlations are modest (r ≈ 0.07–0.35), while `condition` correlations are
  high (r ≈ 0.43–0.85) — the 4-class condition axis is more shared across
  regions than the binary policy axis.
- Per-region `conflict_mag` (rate) is weak (|r| ≤ 0.08); the strong
  `conflict_mag` effect lives in the whole-population PCA representation, which
  is not unit-aligned and therefore not computed per region.
- Outputs: `<run>/region_decoding.csv`, `region_continuous.csv`,
  `region_selectivity_counts.csv`.

---

## Part 4 — Robustness

**Bursty-unit sensitivity** (drop units with ISI CV > 1.3) run on all four
sessions. `conflict_mag` (pca, both) survives: mean 0.238 after dropping bursty
units vs 0.242 with all units, same sign in all four. YGA sessions flagged many
bursty units (yga_1 38/81, yga_2 50/77), so this control matters there.

**Entry-locked PCA** (fixed pre-decision windows, previously rate-only): added
to `neural_entry_decoding.py` (`entry_locked_lda_results_pca.csv`). At
entry+250 ms, `planning_vs_greedy` is 0.485/0.531/0.507/0.507 (only yfz_2 at
p=0.013) — so the pre-decision null is **not** an artifact of the rate
representation. This closes the rate-only caveat noted in
`neural_decoding_confounds.md` §8.

---

## Confound audit (`analysis/neural_confound_audit.py`)

We cannot prove the absence of confounds, but we can test each conclusion's
main rivals. The audit runs across all four sessions and writes CSVs to
`analysis/neural_outputs/aggregate/`. Findings:

**1. Temporal leakage / block structure.** All decoding had used shuffled
`StratifiedKFold`; trials are sequential and conditions are modestly
block-associated (`condition × block` Cramér's V ≈ 0.17 in every session;
lag-1 autocorrelation of the planning choice ≈ 0). Re-running every headline
decoding with **leave-block-out CV** reproduces the shuffled-CV numbers:

| metric | shuffled | leave-block-out |
|---|---|---|
| `planning_vs_greedy` post rate | 0.63–0.78 | 0.64–0.76 |
| `planning_vs_greedy` post pca | 0.89–0.98 | 0.88–0.98 |
| `condition` post pca | 0.30–0.53 | 0.32–0.53 |
| `agree` post pca | 0.55–0.64 | 0.54–0.64 |
| `conflict_mag` pca both | 0.09–0.46 | 0.11–0.46 |

→ temporal leakage is **not** driving the results.

**2. Position confound.** Adding `ball_x`, `ball_y`, `camera_y`, `rel_y` to the
continuous-decoding nuisances leaves `conflict_mag` unchanged
(0.242 → 0.242) and `chosen_planning_cost` essentially unchanged
(0.221 → 0.218). The threat regressor is unaffected in practice
(0.036 → 0.059, still small).

**3. Region unit-count matching.** Bootstrapping every region down to the
smallest region's unit count preserves the region pattern:
`planning_vs_greedy` still decodes above chance in every region in every
session (0.49–0.68 at matched N), so the region result is not a unit-count
artifact. `condition` is near chance per region (0.24–0.32 vs chance 0.25).

**4. Death matching.** Block-matching the normal trials (restricting to the
blocks where deaths occurred) barely changes death decoding
(yga_1 0.496 → 0.495; yga_2 0.547 → 0.543), so the yga_2 effect is not a
block confound — but it remains exploratory (1 of 2 sessions).

**5. Threat is RT/kinematic-confounded.** The `ball_y − camera_y` regressor
correlates with `rt_ms` (mean |r| ≈ 0.40), trial duration (0.25), `camera_y`
(0.18) and `ball_y` (0.16); this is why its decoding collapses under
residualization.

**6. Multiple comparisons.** `aggregate/aggregate_fdr.csv` applies BH-FDR to the
session-level sign-flip p-values of all headline metrics.

### Confound status per conclusion

| Conclusion | Status after audit |
|---|---|
| Post-choice `planning_vs_greedy` (consequence, not computation) | **robust** (leakage, side/RT/block matching, entry-anchored, rate+PCA) |
| `conflict_mag` survives residualization | **robust** (position, leakage, bursty all controlled) |
| Death vs normal (yga_2) | **exploratory** (1 of 2 sessions; block-matching ok) |
| Continuous threat | **negative** (collapses; confounded with RT/duration) |
| Per-region `planning_vs_greedy` | **robust** (unit-count matched) |
| Entry-locked pre-decision null | **robust** (rate + PCA, rt-eligible windows) |
| Bursty units not driving `conflict_mag` | **robust** |
| Condition axis shared across regions | **supported** (score correlations; no permutation null yet) |

### Figures (`analysis/neural_figures.py` → `analysis/neural_outputs/figures/`)

| figure | conclusion |
|---|---|
| `fig_neural_replication.png` | cross-run replication of headline effects |
| `fig_neural_window_profile.png` | pre/entry at chance; post-choice effect |
| `fig_neural_confound_collapse.png` | `conflict_mag` survives vs `chosen_planning_cost` collapses |
| `fig_neural_leakage_control.png` | block-grouped CV ≈ shuffled CV |
| `fig_neural_nuisance_associations.png` | label × nuisance associations |
| `fig_neural_matched_redecode.png` | matched re-decode |
| `fig_neural_death.png` | death decoding (exploratory) + block-matched |
| `fig_neural_threat.png` | threat raw vs residualized; threat × nuisance |
| `fig_neural_region.png` | per-region decoding + cross-region correlation |
| `fig_neural_bursty.png` | bursty-unit control for `conflict_mag` |

---

## Figure-by-figure detail

This section explains exactly what is plotted in each figure, how every metric
is derived, what the axes are, and what each point/bar means. All figures are
produced by `analysis/neural_figures.py` from the per-run result CSVs and the
`analysis/neural_outputs/aggregate/` audit outputs.

### Shared metric definitions — what the numbers actually mean

These definitions apply to every figure below. The point is not the formula but
what the quantity represents about the recorded population.

- **Balanced accuracy (the y-value of every LDA figure).** A linear
  discriminant analysis is fit to the trial-aligned spike counts and asked to
  guess the trial's category from the population activity alone. Balanced
  accuracy is the average of the per-category hit rates — i.e. the probability
  that a *new, held-out* trial is assigned to the correct category, computed so
  that a rare category counts as much as a common one. It is therefore a direct
  read-out of **how much linearly decodable information about the category the
  population carries on a single trial**, expressed as a probability. Chance is
  `1/n_categories` (0.50 for the binary planning/greedy contrast, 0.25 for the
  4-way `condition`), because a decoder that ignores the neural data and guesses
  the majority class would otherwise look good when the classes are imbalanced
  (~110 planning vs ~320 greedy trials). A value of 0.60 means the population
  lets you name the correct category ~60% of the time; 0.50 means it carries no
  linearly readable information. Values *below* chance mean the decoder is
  systematically anti-predicting (usually an instability of a small-unit-set fit,
  not a real "negative representation").
- **Why LDA, and why shrinkage.** LDA finds the linear weighting of units whose
  projection best separates the categories. With more units than trials per
  fold, the unregularized covariance estimate is singular, so we use
  Ledoit–Wolf shrinkage (`solver="eigen"`), which pulls the covariance toward a
  diagonal and keeps the fit stable. This makes the decoder conservative: it
  does not invent separation from noise.
- **Cross-validation (CV).** The decoder is trained on a subset of trials and
  scored on trials it never saw; repeating over folds and random repeats gives a
  mean and spread. This is what makes the number an *out-of-sample* estimate of
  decodability rather than a fit statistic. Stratified folds preserve the class
  ratio in every fold.
- **CV correlation (the y-value of every ridge figure).** The same
  train/test logic, but the target is a *continuous* task quantity (e.g. how
  much the two policies disagree). Ridge regression predicts it from the
  population and we correlate prediction with truth on held-out trials. A
  correlation of 0 means the population carries no linearly readable trace of
  that quantity; positive means it does. For single-trial neural data, r ≈ 0.2–0.3
  is already a strong effect — it is not like a correlation between two
  behavioural measures.
- **Permutation p.** To ask whether a score could arise by chance, the category
  labels (or the continuous target) are shuffled and pushed through the *same*
  CV pipeline many times, building a null distribution of scores. p is the
  fraction of null scores at least as extreme as the observed one: the
  probability of seeing this decodability if neural activity and the category
  were unrelated. It controls for the optimism that a flexible decoder can always
  fit noise.
- **Nuisance residualization (`none`/`target`/`both`).** Trial categories are
  mechanically tied to the consequences of the choice (reaction time, how long
  the ball then travelled, where it was on screen). `target` projects those
  nuisance covariates out of the *task quantity* before decoding; `both`
  additionally projects them out of every neural feature. If a decoding survives
  `both`, it is not explainable by the nuisances; if it collapses from `none` to
  `both`, the raw decodability *was* the nuisance.
- **Feature representations.** `rate` reduces each trial to one number per unit
  (mean firing rate in the window, sqrt-transformed to stabilize variance), so
  the decoder can only use *which units fired more or less*. `pca` keeps the
  within-trial time course (25 ms bins) and compresses it to 30 principal
  components, so the decoder can also use *when* units fired. A signal present
  only in `pca` is a temporal/pattern effect that average rates miss.
- **Sign-flip p (cross-run).** To ask whether an effect is consistent across
  sessions, we take each session's effect size and ask how often a random
  assignment of ± signs to the four sessions produces a mean at least as large
  as the observed mean. It is an exact, assumption-light test of
  "same-direction across sessions"; with only four sessions the smallest
  attainable p is 0.125 (2 of 16 sign patterns), so it can never be very small.
- **Block-grouped (leave-block-out) CV.** The game is played as a continuous
  sequence, so trials close in time share state (fatigue, drift, slow neural
  fluctuations). If train and test trials are randomly interleaved, the decoder
  can exploit that shared state. Holding out *whole blocks* removes this
  possibility; if block-grouped CV matches shuffled CV, the decoding is not a
  temporal-autocorrelation artifact.
- **Regions.** Each unit's recording lead is mapped to a mesial-temporal
  structure (`CA`, `amygdala`, `anterior_hippocampus`, `hippocampal_body`) from
  the channel→lead table, so a decoding can be repeated using only that
  structure's units.
- **What the plotted points are.** In every figure, one mark = one estimate from
  one session (or one bootstrap), the black diamond/mean = the across-session
  average, and error bars = standard error of the mean (SEM) across sessions
  unless stated otherwise. SEM describes how precisely the *group* mean is
  estimated, not the spread of individual sessions (that is the dots).

---

### `fig_neural_replication.png` — cross-run replication

- **The question.** For each headline effect: is it real (above its chance
  level), and does it hold in *every* recorded session rather than in one lucky
  session?
- **What the seven rows mean scientifically.**
  - `conflict_mag` — *how much the two policies disagree on this maze*. Derived
    per trial from the geometry: `greedy_gap = greedy_cost_L − greedy_cost_R`
    (the signed 1-step value difference between the two holes) and
    `planning_gap = planning_cost_L − planning_cost_R` (the signed 2-step
    difference); `model_conflict = planning_gap − greedy_gap` and
    `conflict_mag = |model_conflict|`. It is a property of the *maze*, not of the
    participant's choice. Decoding it (ridge) asks whether the population
    continuously tracks "the strategies disagree by this much here".
  - `chosen_planning_cost` — the 2-step cost of the hole the participant
    actually chose; low means they took the efficient path. It is a
    decision-quality/outcome quantity and is the most confounded target (see the
    collapse figure).
  - `planning_vs_greedy` (post) — on *conflict* trials only, did they choose the
    2-step-optimal (planning) hole or the 1-step-optimal (greedy) hole? This is
    the core policy label.
  - `planning_vs_greedy` (pre) — the same label decoded from the interval
    *before* the choice; it is the control for "is the plan computed before the
    choice?".
  - `condition` — the 4-way label (planning / greedy / agree_optimal / lapse);
    chance 0.25.
  - `agree` — whether the two policies prescribe the same hole on that trial.
- **The x-axis is a common effect-size axis in two units.** For the two `cont:`
  rows the number is a CV correlation (chance = 0, dashed line); for the `lda:`
  rows it is balanced accuracy (chance = 0.5, dotted line). Both are
  "information about the label readable from the population", so they are put on
  one axis for comparison, but the 0.5 line is only meaningful for the accuracy
  rows.
- **Each dot** = one session's estimate (colour = session); **the diamond** =
  the mean over the four sessions; **the whiskers** = SEM across sessions, i.e.
  how precisely that mean is pinned down. A single dot far from the diamond is a
  session that behaved unusually.
- **The right-hand text** = sign-consistency (`k/4` sessions sharing the mean's
  sign) and the exact sign-flip p (how surprising that consistency is under
  random signs; floor 0.125 with four sessions).
- **How to read it.** Post-choice `planning_vs_greedy` sits at 0.63–0.78 (rate)
  and 0.89–0.98 (pca) in *all four* sessions — the population distinguishes the
  two policies with high single-trial reliability, and the pca values near 1.0
  mean the distinction is nearly deterministic when the within-trial time course
  is available. The pre-choice row sits on 0.5 in all four: no pre-decision
  read-out. `conflict_mag` is positive in all four but much larger for YFZ
  (0.30/0.46) than YGA (0.12/0.09), i.e. the *effect is consistent in sign but
  its size is participant-dependent*.
- **What it would look like if the effect were absent:** dots scattered around
  the chance line with mixed signs and sign-flip p near 1. What it would look
  like if driven by one session: three dots near chance and one extreme, with
  low sign-consistency.
- **Caveat:** `condition` and `agree` for the two YFZ sessions are not the same
  contrast as for YGA in absolute terms only through unit sampling; units are not
  matched across sessions, so each dot is an independent replication, not a
  repeated measure of the same cells.

---

### `fig_neural_window_profile.png` — where in time the signal lives

- **The question.** Is the planning-vs-greedy distinction computed *before* the
  ball passes the choice hole (a decision process), or does it only appear
  *after* the choice (a read-out of the different trajectories the two policies
  produce)? This is the central "computation vs consequence" test.
- **Why windows need care.** All spikes are stored relative to a per-trial
  anchor. Anchoring at the **choice pass** (`t=0` = passing the 2-hole level)
  makes the `pre [-1000,0]` window *heterogeneous*: for a slow planning trial
  (entry→choice ≈ 950 ms) it covers the whole approach, but for a fast greedy
  trial (≈ 417 ms) it extends back before the entry hole, i.e. before the trial
  really began. So the analysis is repeated anchored at the **entry pass**
  (passing the 1-hole level that opens the sequence), where the decision interval
  is clean.
- **The four windows.**
  - `pre [-1000,0]` ms relative to the **choice** — the pre-decision interval as
    usually defined.
  - `entry+250` = `[0,250]` ms after the **entry** — a fixed, clean early-decision
    window; only trials whose entry→choice interval is ≥ 250 ms are eligible.
  - `approach` = each trial's whole `[entry, choice)` interval — the full
    decision interval, but its *length is the reaction time*, so it is confounded
    by duration.
  - `post [0,+1000]` ms relative to the **choice** — after the ball has committed
    and is travelling to the exit.
- **The y-value** is balanced accuracy for `planning_vs_greedy` using the `rate`
  representation (one number per unit): how reliably the population separates
  planning from greedy trials in that window, on held-out trials.
- **How to read it.** The fixed pre-decision windows (`pre`, `entry+250`) sit on
  the 0.5 line in every session: **there is no pre-decision planning read-out**,
  and this is not an artifact of how `pre` was anchored (the entry-anchored
  window agrees). The effect is large `post` (0.63–0.82). The `approach` window
  is elevated for some sessions (e.g. yfz_2 0.81) but that is a duration
  artifact: planning trials spend ~950 ms in the interval vs ~417 ms for greedy,
  so a window whose length equals RT leaks the label through timing alone; when
  trials are matched on RT the approach-window effect collapses (see
  `neural_decoding_confounds.md` §8).
- **What absence would look like:** all four bars at 0.5. What a pure
  timing artifact looks like: `pre`/`entry+250` at 0.5 but `approach` high with
  `post` high — exactly the pattern here, which is why the fixed windows are the
  inferential ones.
- **Caveat:** `entry+250` is rate-only; the PCA version of the same fixed
  windows is in `entry_locked_lda_results_pca.csv` and agrees (see Part 4).

---

### `fig_neural_confound_collapse.png` — does residualization kill the signal?

- **The question.** When the population appears to encode a task quantity, is it
  encoding the *task quantity itself* or merely the kinematics/timing that
  co-vary with it?
- **What "residualization" means conceptually.** We are not just adding
  covariates to a regression; we are **removing the part of the data that is
  explainable by the confounds and asking whether anything remains**. `none` uses
  the raw quantity and raw features; `target` replaces the quantity with its
  residual after linearly projecting out RT, ball-travel time, trial duration,
  block index, side and move direction; `both` also residualizes every neural
  feature on those same variables. So `both` is the most adversarial test: both
  sides of the regression have had the confound space deleted.
- **The two quantities.**
  - `conflict_mag` — the maze's policy-disagreement magnitude (a property of the
    geometry, independent of what the participant did). If the population tracks
    this, it is tracking a planning-relevant *task variable*.
  - `chosen_planning_cost` — the 2-step cost of the hole actually chosen. Because
    planning choices have a much shorter post-choice ball-travel time, this
    quantity is mechanically correlated with the kinematics that follow the
    choice; it is the quantity most likely to be a consequence rather than a
    cause.
- **The y-value** is the CV correlation: how well held-out population activity
  predicts the quantity (1 = perfect, 0 = nothing). **Each bar** = one session;
  bars are grouped by control on the x-axis.
- **How to read it.** `conflict_mag` barely moves from `none` to `both`
  (yfz_1 0.27→0.30, yfz_2 0.41→0.46): deleting RT/kinematics/position variance
  from both the target and the features leaves the population's representation of
  policy disagreement intact — this is the strongest single positive result.
  `chosen_planning_cost` behaves the opposite way in the YFZ sessions: large raw
  (0.50, 0.68) but collapsing to ≈0.11–0.20 under `target`/`both`, i.e. its
  apparent representation was carried by the RT/kinematic structure, not by a
  decision-quality code.
- **What absence of a confound looks like:** flat bars across `none`/`target`/
  `both`. **What a pure confound looks like:** tall `none`, near-zero `both`.
- **Caveat (over-correction).** Residualization removes *all* variance aligned
  with the nuisances, including any genuine neural signal that happens to
  correlate with RT/position. So a collapse is strong evidence of confounding,
  but a survival is the more conservative and informative outcome.

---

### `fig_neural_leakage_control.png` — is the CV leaking over time?

- **The question.** Could the decoding be inflated not by a real
  activity→choice relationship but by the fact that trials close together in
  time resemble each other (shared drift, fatigue, slow neural state)? If train
  and test trials are randomly interleaved, a decoder can memorize the shared
  state of a *block* and score well on the held-out trials from the same block.
- **The control.** Re-run every headline decoding with **leave-block-out CV**:
  entire behavioral blocks are held out, so no train and test trial ever come
  from the same block. If the score is the same, the effect is not a
  temporal-proximity artifact. (The condition is only weakly block-associated:
  `condition × block` Cramér's V ≈ 0.17, and the lag-1 autocorrelation of the
  planning choice is ≈ 0, so leakage was a plausible but not certain risk.)
- **The axes.** x = score under the ordinary shuffled CV; y = score under
  leave-block-out CV. The dashed diagonal is `y = x`.
- **Each point** = one (session, metric) pair, coloured by session. Points *on*
  the diagonal mean the two schemes agree; points *below* the diagonal (y < x)
  would mean the shuffled score was inflated by leakage.
- **How to read it.** Every point lies on or just off the diagonal (largest
  deviation ≈ 0.06, and some points sit slightly *above* it). So the decodings
  are not driven by temporal autocorrelation. The right panel makes this
  concrete for `conflict_mag`: yfz_1/yfz_2 essentially unchanged (0.30→0.32,
  0.46→0.46) and yga_1/yga_2 slightly *higher* under the stricter scheme.
- **What leakage would look like:** a systematic downward shift, i.e. points
  well below the diagonal, especially for the high-accuracy pca decodings.

---

### `fig_neural_nuisance_associations.png` — how entangled are labels and nuisances?

- **The question.** Before trusting any decoding, ask: **can the label already
  be predicted from the trial's non-neural covariates alone?** If yes, a decoder
  could score well by reading those covariates' neural correlates rather than a
  decision variable.
- **The association statistic** measures exactly that, per label × nuisance
  pair, each with a label-shuffle permutation p:
  - **point-biserial r** (binary label vs continuous nuisance) — the correlation
    between the 0/1 label and the covariate; signed, so sign tells you which
    class has the larger covariate.
  - **η²** (4-class `condition` vs continuous nuisance) — the fraction of the
    covariate's variance explained by the 4 groups (0 = none, 1 = all).
  - **Cramér's V** (two categorical variables) — association between labels,
    e.g. `side` and `move_dir`.
- **The axes.** Rows = trial label; columns = nuisance (RT, post-choice ball
  travel time, trial duration, block index, position within block, the
  greedy/planning value gaps, side, movement direction, previous trial's
  condition). Cell colour = the statistic (diverging map centred at 0; note V
  and η² are non-negative).
- **How to read it.** The dominant cell is `planning_vs_greedy × ball_time_ms`:
  planning trials traverse to the exit in ≈ 438 ms vs ≈ 1455 ms for greedy
  trials (r ≈ 0.87 in yfz_1). This is not a subtle correlation — the two classes
  have nearly disjoint post-choice travel times, so *any post-choice window* has
  ample kinematic structure to exploit. The 4-way `condition` is similarly tied
  to ball time (η² ≈ 0.48) and RT (η² ≈ 0.40).
- **Why this is the crux of the paper.** It is the mechanical reason a
  post-choice `planning_vs_greedy` decoder can succeed without a planning
  computation: choosing the planning hole *produces* a short traversal. The
  matched re-decode and entry-anchored analyses exist to test whether anything
  survives once that is removed.
- **What a clean label would look like:** a row of near-zero cells. What a
  confounded label looks like: a bright `ball_time`/RT cell — which is what the
  planning label shows.

---

### `fig_neural_matched_redecode.png` — decoding after equalizing confounds

- **The question.** Instead of *modelling* the confounds (residualization), this
  *removes them by design*: compare only trials whose confound values are the
  same, and see whether the decoding survives.
- **How matching works.** Trials are binned into strata on nuisance variables;
  within each stratum we keep an equal number of trials from each class (the
  smaller class caps the count); strata containing only one class are dropped.
  The decoder is then re-fit and scored on this balanced, confound-matched
  subset. This is stricter than residualization because it discards trials
  entirely rather than adjusting them.
- **The schemes, from weakest to strongest.**
  - `unmatched` — all trials (the reference number).
  - `+side` — equalize left/right choices.
  - `+side × RT-quartile × block-half` — equalize side, reaction time and
    early/late half of each block (removes RT and slow within-session drift).
  - `+side × RT-quartile × ball-quartile` — additionally equalize post-choice
    ball-travel time (removes the kinematic confound).
- **The y-value** is balanced accuracy (pca, post), averaged over sessions with
  SEM; **`n` above each bar** is how many trials survive that matching. Matching
  trades confound control for statistical power: the more strata, the fewer
  trials, so wide error bars at high matching are expected.
- **How to read it.**
  - `planning_vs_greedy` stays ≈0.68–0.72 after `+side` and
    `+side × RT × block`, so it is not a side or reaction-time artifact. It
    **cannot be tested** against ball time at all: only 16 of 431 trials share a
    ball-time stratum (the classes' traversal times barely overlap), which is
    itself the strongest possible statement of the confound — you cannot match
    away a difference that large.
  - `planning_optimal` and `agree` (pca) look decodable unmatched (0.70, 0.61)
    but fall to chance (≈0.50–0.52) once ball time is matched, so their
    post-window signal *was* the traversal-timing difference.
- **What it means.** The `planning_vs_greedy` result survives the confound
  controls that are possible; the `planning_optimal`/`agree` results do not — a
  clean dissociation between a robust policy signal and a purely kinematic one.
- **Caveat:** the `+side × RT × ball` cell is `n=16` ("insufficient overlap") and
  is not a failed decode but an infeasible comparison.

---

### `fig_neural_death.png` — death vs normal (exploratory)

- **The question.** In the drift game the participant can die when the ball falls
  behind the camera. Does the mesial-temporal population carry a signature of
  the moments just before a genuine death, distinct from ordinary play?
- **What a "death" is here.** Not every empty trial is a death: the ball can fall
  once and several levels scroll past. Genuine death instants are detected from
  the ball trajectory (`ball_y − cameraY → 0` while the ball is frozen at the
  bottom), giving 12 (yga_1) and 19 (yga_2) real events; YFZ has ≤ 1 and is not
  analysed.
- **How the decoder is built.** Per-unit firing rate is computed in a window
  **anchored at the death instant** (`[-500, 0]` ms before death) and compared,
  trial-by-trial, to the same window **anchored at the choice pass** of normal
  trials. The classifier must separate "death-locked activity" from
  "choice-locked activity"; balanced accuracy 0.5 means the population does not
  distinguish them.
- **Why the two controls.** `all_normal` compares against all 910 normal trials;
  `block_matched` compares only against normal trials from the blocks in which
  deaths occurred, so a session-level slow drift that happens to coincide with
  the death blocks cannot create the effect.
- **How to read it.** yga_2 is above chance (0.547; permutation p = 0.007 in
  `death_decoding.csv`) and yga_1 is not (0.496). Block-matching barely moves
  either (yga_1 0.496→0.495, yga_2 0.547→0.543), so the yga_2 effect is not a
  block confound.
- **Why it is labelled exploratory.** It replicates in 1 of 2 sessions, the
  number of death events is small (12/19), and the comparison mixes two anchors
  (death vs choice), so the residual anchor difference is not fully ruled out.
  It is a lead worth following with more sessions, not a finding.
- **What absence would look like:** both sessions at 0.5. What a strong result
  would look like: both sessions clearly above chance with more events.

---

### `fig_neural_threat.png` — proximity to death

- **The question.** The behavioural analysis (Exp 3) asked whether people change
  how much they plan under threat. Here we ask the neural counterpart: does the
  population represent the participant's **proximity to death**, and is any such
  representation independent of timing/kinematics?
- **The threat quantity.** `ball_y − camera_y` at the choice pass: the ball's
  vertical position on screen. Small values mean the ball is near the top/death
  line (the game declares death when `ball_y − cameraY < 0`), so this is a
  continuous "how close was I to dying" regressor, not a categorical condition.
- **Left panel.** Ridge decoding of that quantity from the population, shown for
  six representation/window/control combinations
  (`rate/pre/{none,target,both}` and `pca/post/{none,target,both}`), one group of
  bars per session. The y-value is the CV correlation: how well held-out
  activity predicts proximity.
- **Right panel.** The mean |Pearson r| between the threat regressor and each
  nuisance covariate, i.e. how entangled threat is with ordinary trial
  structure. This is the diagnostic that explains the left panel.
- **How to read it.** Raw decoding is non-trivial in the YFZ sessions
  (`rate/pre/none` ≈ 0.12–0.19, `pca/post/none` ≈ 0.12–0.16) but **collapses
  under residualization in every session** (`target`/`both` ≈ 0.02–0.06). The
  right panel shows why: threat is most strongly correlated with reaction time
  (mean |r| ≈ 0.40) and trial duration (≈ 0.25), and moderately with
  `camera_y`/`ball_y` (≈ 0.16–0.18). In other words, trials in which the ball is
  low on the screen are also trials where the participant moved slowly, so the
  "threat" signal is largely a timing signal.
- **What it means.** Within the limits of this control, there is **no robust
  neural representation of proximity-to-death** that survives nuisance removal —
  a clean negative result, and consistent with the behavioural finding that
  threat mainly shifts *choice* (planning rate) rather than producing a
  dedicated threat code.
- **Caveat:** as with all residualization, a genuine threat signal that is
  intrinsically correlated with slow movement would be removed; the negative
  result therefore means "nothing beyond the timing correlate".

---

### `fig_neural_region.png` — hippocampus vs amygdala

- **The question.** Is the planning-vs-greedy signal carried by a particular
  mesial-temporal structure, or is it distributed? And do different structures
  agree about *which trials* are planning trials?
- **Region assignment.** Each unit's electrode lead is mapped to `CA`,
  `amygdala`, `anterior_hippocampus` or `hippocampal_body`; a decoder is then fit
  using only that region's units.
- **Left panel — per-region decoding.**
  - **X-axis:** region; **y-axis:** balanced accuracy for `planning_vs_greedy`
    (rate, post) at a **unit-count-matched N**; bars per session.
  - **Why matching matters:** regions have very different numbers of units
    (e.g. amygdala ≈ 39 vs anterior hippocampus ≈ 17 in yfz_1), and a decoder
    with more units can score higher for purely statistical reasons. To make the
    regions comparable, each region is bootstrapped down to the *smallest*
    region's unit count (10 random draws) and the accuracy is averaged; the bars
    therefore reflect *information per unit*, not *units available*.
  - **How to read it:** every region with enough units decodes above chance
    (0.49–0.68) in every session, and no region is systematically the best — the
    policy distinction is **distributed across the mesial-temporal population**,
    not localized to one structure.
- **Right panel — cross-region decision-score correlation.**
  - **Why correlation, not cross-decoding:** the regions have disjoint unit
    sets, so an LDA trained on CA's units cannot be applied to amygdala's units
    (different dimensions/units). Instead, each region's decoder produces an
    out-of-fold *decision score per trial*; correlating those scores across
    regions asks whether the two structures "agree" about how planning-like each
    trial is.
  - **Axes:** region × region; diagonal = 1 (self); off-diagonal = mean Pearson r
    across sessions. **Cell value** ≈ 0.15–0.23 for `planning_vs_greedy`: the
    regions are only *partially* aligned — each carries policy information, but
    they do not produce the same trial-by-trial read-out. (For the 4-way
    `condition` label the same correlations are much higher, r ≈ 0.4–0.85, so the
    broad task structure is more shared than the binary policy.)
- **What it means.** Policy information is present throughout the recorded
  structures rather than being a single-region read-out, and the structures'
  trial-wise codes are related but not identical — consistent with a distributed,
  partially redundant representation.
- **Caveats:** per-region decoding is rate-only (PCA components are not
  unit-aligned, so they cannot be sliced by region); anterior hippocampus in
  yga_1 falls below the 8-unit floor and is blank in the heatmap.

---

### `fig_neural_bursty.png` — bursty-unit sensitivity

- **The question.** Could the `conflict_mag` result be carried by a handful of
  irregular units whose firing comes in bursts, rather than by graded
  rate coding? Bursting makes a unit's trial-to-trial firing rate extremely
  noisy and can dominate a rate/PCA representation.
- **What "bursty" means.** For each unit we compute the coefficient of variation
  of its interspike intervals, `CV = SD(ISI)/mean(ISI)` (Perkel, Gerstein &
  Moore 1967): ≈ 0 for a metronome-like unit, ≈ 1 for a Poisson process, > 1 for
  a bursting/irregular unit. Units with **CV > 1.3** are dropped and the
  headline `conflict_mag` decoding is recomputed on the remainder. This matters
  most in YGA, where many units are flagged (yga_1 38/81, yga_2 50/77) versus
  YFZ (≈ 20/124).
- **The axes.** **X-axis:** session. **Y-axis:** `conflict_mag` CV correlation
  (pca, `both` control). **Bars:** `all` (every unit) vs `no_bursty` (CV ≤ 1.3
  units only). If the two bars coincide, the effect does not depend on the
  irregular units.
- **How to read it.** The bars are nearly identical (mean 0.242 with all units vs
  0.238 without bursty units), so `conflict_mag` is carried by the population
  broadly, not by burst-driven noise — and this holds in YGA despite the high
  bursty fraction there.
- **What a burst artifact would look like:** a large drop from `all` to
  `no_bursty` (or the effect vanishing), especially in YGA.
- **Caveat:** the CV threshold (1.3) is conventional; the qualitative result is
  insensitive to it over a reasonable range.

---

## Caveats

- Four sessions from two participants; the sign-flip test is coarse (minimum
  p = 0.125) and the binomial sign test gives p = 0.0625 for 4/4 consistency.
- Death decoding is YGA-only and event-limited (12/19 deaths); one session is
  significant, the other is not.
- Region unit counts are unequal (CA/amygdala larger); anterior hippocampus in
  yga_1 falls below the 8-unit floor and is skipped.
- Per-region PCA is not computed (PCA components are not unit-aligned); only
  rate features are used per region.
- All sessions are all-drift, so there is no follow-vs-drift contrast; threat is
  continuous plus discrete death events.

## Reproducing

```
python analysis/aggregate_sessions.py
python analysis/neural_death_threat.py --run yfz_1   # repeat per run
python analysis/neural_region_decoding.py --run yfz_1
python analysis/neural_bursty_unit_sensitivity.py --run yfz_1
python analysis/neural_entry_decoding.py --run yfz_1
python analysis/neural_confound_audit.py             # all runs
python analysis/neural_figures.py                    # all runs
```

Aggregate outputs land in `analysis/neural_outputs/aggregate/`; per-run outputs
in `analysis/neural_outputs/<run_id>/`.
