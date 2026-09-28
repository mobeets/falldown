# Cross-dataset findings: cloud study x EMU neural

Status: implemented. Combines the online cloud-study behavioural data (large N,
three environments) with the four EMU intracranial sessions (2 participants) to
test whether the neural findings are about the same behavioural phenomenon and
to link neural quantities to behavioural models.

Scripts (run under the pinned analysis env, numpy<2 + `ssm`):

```
.venv-analysis\Scripts\python.exe analysis\cross_dataset_planning.py   # analyses -> CSVs
.venv-analysis\Scripts\python.exe analysis\cross_dataset_figures.py    # 18 figures
```

Outputs: `analysis/cross_dataset_outputs/` (CSVs) and
`analysis/cross_dataset_outputs/figures/` (PNGs).

## Data and harmonization (Part 0)

One unified sequence table (`cross_dataset_trials.csv`) is built from the cloud
`sequence_table.csv` (all 18 retained online participants, 17,721 experimental
sequences) and the four EMU runs' `trial_labels.csv` + `trial_table.csv`
(3,640 sequences; 910 trials × 4). Both are 1-2-1 sequences, so geometry,
costs, choice, RTs, threat (`ball_y` at entry) and per-block environment
statistics map onto the same schema. **Baseline = cloud all-drift cohort**
(`short_trials_experiment-7-10`: 32FC87F1, 96CA2FB7, B0525260, C8C4C97C), the
closest task match to the all-drift EMU sessions.

Note on the two datasets: cloud = online participants, EMU = 2 epilepsy
patients. This is a **task-level generalization / case study**, not a joint
neural model (cloud has no neural data; EMU is all-drift only).

---

## Part 1 — Behavioural comparability

`xd_behavioral_summary.csv`, `xd_mixture.csv`, `xd_hmm_selection.csv`.

| | P(plan\|conflict) | P(optimal\|agree) |
|---|---|---|
| Cloud (18 participants) | 0.13–0.46 (mean 0.34) | 0.67–0.93 |
| Cloud all-drift (4) | 0.24–0.28 | 0.84–0.93 |
| EMU (4 runs) | 0.25–0.29 | 0.86–0.90 |

- **Planning prevalence in EMU sits inside the cloud range**, and matches the
  cloud all-drift cohort almost exactly.
- The **planning/greedy RT trade-off** is present in EMU: planning is slower to
  commit (951–1066 vs 512–608 ms) and faster to execute (456–575 vs 1337–1716
  ms), mirroring cloud.
- **GLM-HMM state selection: K* = 1 for all four EMU runs** (BIC), exactly as in
  the cloud data — no hidden strategy states in either sample.
- The mixture model's `p_plan_base` saturates at its 0.98 bound and `w1` hits the
  ±8 bound for several participants in *both* datasets, so mixture parameters are
  reported but treated as unstable; model-free quantities are used downstream.

**Figures:** `fig_xd_prevalence.png`, `fig_xd_rt_tradeoff.png`, `fig_xd_mixture.png`.

---

## Part 1b — Model-based planning weight (coefficient space)

`xd_planning_weight.csv` (also merged into `xd_behavioral_summary.csv`).

`P(plan | conflict)` treats every conflict trial as equivalent. A complementary,
geometry-aware measure is the **planning weight** from the canonical cloud
decision model, fit per participant on **all choice trials** (the
`evaluate_logistic_baseline` specification):

```
logit P(choose left) = b_1step · z(diff_1step)
                     + b_plan  · z(diff_planning)
                     + incoming-direction terms   (+ drift×incoming if drift varies)

planning_weight = |b_plan| / (|b_plan| + |b_1step|)
```

Choosing the left hole is favoured by *negative* coefficients (left closer →
`diff_1step < 0`), so magnitudes are used; the weight is in [0, 1] (0 = purely
greedy geometry, 1 = purely planning geometry). It answers "how much of the
geometry-driven choice is explained by the 2-step rule?" rather than "how often
did they pick the planning hole?".

- The two measures agree strongly: `corr(P(plan|conflict), planning_weight) =
  +0.90` across the 22 participants/runs.
- EMU weights (0.08–0.22) sit inside the cloud range (0.02–0.58; cloud mean
  0.27); EMU percentiles vs cloud are 0.17–0.44. Same comparability conclusion
  as Part 1, now in model space.

**Figure:** `fig_xd_prevalence_weight.png`.

---

## Part 2 — Neural–behavioural linkage within EMU

`xd_neural_linkage.csv`, `xd_neural_trials.csv`. Out-of-fold neural scores
(LDA for planning-vs-greedy, ridge for `conflict_mag`, entry-anchored LDA for
pre-decision) are added to a geometry-only logistic (cross-validated).

| run | choice acc geometry → + neural | choice acc pre geometry → + pre |
|---|---|---|
| yfz_1 | 0.710 → 0.789 | 0.710 → 0.710 |
| yfz_2 | 0.754 → 0.875 | 0.753 → 0.753 |
| yga_1 | 0.753 → 0.783 | 0.753 → 0.746 |
| yga_2 | 0.767 → 0.783 | 0.767 → 0.765 |

- The **post-choice neural planning score adds choice information beyond the
  geometry** in all four runs (+0.06 accuracy on average).
- The **pre-decision (entry-anchored) neural score adds nothing** in all four —
  the pre-decision null, now framed as a behavioural-prediction test.
- Neural `conflict_mag` does **not** track the kinematic-adjusted RT residual
  (|r| ≤ 0.06), so the conflict representation is not a decision-time signal.

**Figures:** `fig_xd_neural_choice.png`, `fig_xd_neural_rt.png`,
`fig_xd_predecision.png`.

---

## Part 3 — Cross-dataset model transfer

`xd_transfer.csv`, `xd_model_neural.csv`.

| model | accuracy predicting EMU choices | log-likelihood |
|---|---|---|
| logistic fit on **cloud all-drift** | 0.738 | −0.568 |
| logistic fit on **EMU (block-held-out)** | 0.738 | −0.565 |

- A behavioural model fit on the cloud all-drift cohort predicts EMU choices
  **as well as an EMU-fitted model** — strong task-level generalization.
- The out-of-fold neural `conflict_mag` prediction correlates with the analytic
  `conflict_mag` (YFZ: r = 0.26/0.41; YGA: r = 0.07/0.08), i.e. the neural
  representation matches the model quantity, strongly in YFZ and weakly in YGA.

**Figures:** `fig_xd_transfer.png`, `fig_xd_model_neural.png`.

---

## Part 4 — Threat

`xd_threat_behavior.csv`, `xd_threat_neural.csv`, `xd_threat_weight.csv`.

- Cloud: planning rate is lower in drift than follow blocks
  (0.367 vs 0.418, paired p = 0.010), reproducing the Exp 3 threat effect. The
  block-level GEE gives −0.051 (p = 0.0008) univariate, but it **attenuates to
  −0.036 (p = 0.09) once block-mean ball height is controlled** — threat and
  camera height are coupled and partly share the same variance
  (`planning_reanalysis_block.py`).
- EMU: the near-death vs safe tercile split of `ball_y` shows lower planning
  near death in 3 of 4 runs (deltas −0.015, −0.055, 0.000, −0.028); the linear
  slope is near zero because `ball_y` is in pixels (|slope| ≤ 2×10⁻⁴ per
  pixel). EMU is all-drift, so there is no follow contrast to mirror.
- **Coefficient space** (`xd_threat_weight.csv`): a per-trial choice logistic
  with a `ball_y_at_top × planning` interaction gives a mean interaction of
  **−0.10 for cloud** (8/18 participants positive; per-participant t-test
  p ≈ 0.05). Since planning corresponds to a negative `b_plan`, a negative
  interaction means the planning coefficient is *larger when the ball is lower
  on screen (safe)* — the same direction as the rate-based result. The
  normalized weight is noisier because it also depends on `b_1step`.
- **EMU is mixed in coefficient space**: the separate tercile fit gives raw
  `b_plan` = −0.10 near death vs +0.11 safe (3/4 runs lean toward *more*
  planning near death — the opposite of cloud), but with n = 4 runs and a
  narrow ball-height range this is not interpretable, and the interaction model
  disagrees with the separate fits in sign. Treat the EMU threat result as
  inconclusive.
- **Sign convention matters.** `ball_y_at_top` is screen-y from the top, so
  *small = ball high = near death* (see the metric definitions). A model using
  raw world `ball_y`, or the distance from the death line (`camera_y −
  ball_y`), flips the sign of any ball-height coefficient.
- **Block vs trial confound.** In cloud, ball height is tied to camera mode:
  drift blocks average `ball_y_at_top ≈ 253` and follow blocks ≈ 423, and drift
  blocks have less planning. A pooled analysis that treats ball height as a
  continuous threat proxy therefore conflates the block-level drift effect with
  the trial-level ball-height effect. Centering ball height within block barely
  changes the cloud coefficient (−0.073, p = 0.17), so the trial-level effect
  is weak.
- **Five-bin ball-height profile (mid-screen peak).** `analysis/threat_bins_analysis.py`
  (`threat_bins5_*.csv`, `fig_threat_bins5.png`) segments `ball_y_at_top` into
  quintiles. The pooled cloud profile is an inverted-U — planning peaks mid-screen
  (q0 0.279, q2 0.379, q4 0.319; quadratic GEE p = 0.041, peak `ball_y ≈ 378 px`;
  drift-only p = 0.0002). **It does not survive within-block centring**
  (p = 0.87; drift-only p = 0.55), so it is substantially the between-block
  drift/camera effect: block-mean ball height predicts block planning rate
  (p = 0.012). Dropping centration and looking at drift blocks only does not
  rescue a trial-level effect: the drift-only, non-centred quadratic is even
  stronger (b2 = −5.5e-6, p = 0.0002, peak ≈ 270 px) but realizes only 4 quantile
  bins (trials tie at 100 px) and the drift block-mean relation remains
  (slope = −7.8e-5, p = 0.0024, per-participant r ≈ +0.03), i.e. still
  between-block (`threat_bins5_drift_pooled.csv`, panel A of
  `fig_threat_bins5.png`). The EMU (all-drift) profile is noisy (p = 0.23). The
  neural 5-bin decode (`xd_neural_planning_bally.csv`) mirrors the pooled shape
  (rate 0.64–0.68 mid vs 0.56–0.57 at extremes) but the near/far difference is
  n.s. after side/RT/block matching.
- **No shared per-participant peak.** `analysis/participant_peak_location.py`
  (`participant_peak_location.csv`, `participant_peak_summary.csv`,
  `fig_participant_peak_location.png`) gives each participant their own
  quintiles and asks whether their best bin is interior. It is not: raw argmax
  bins are bimodal (bins 1 and 4: 4/11 each), interior peaks are at chance
  (6/11 raw, permutation p = 0.77; 11/18 within-block, p = 0.60), per-participant
  quadratic curvature is ~50/50 (7/11 raw inverted-U, 9/18 within-block), and
  peak ball_y locations scatter (SD ≈ 131 px raw, 61 px within-block). The
  pooled mid-screen peak is therefore not a shared individual rule; individuals
  differ and many peak at an extreme. Drift-only cannot be tested per
  participant (only 1 participant has all 5 bins populated).
- Neural: the planning-vs-greedy signal is present under both high and low
  threat halves with no reliable interaction (from `threat_modulation.csv`),
  consistent with the neural threat null.

**Figures:** `fig_xd_threat.png`, `fig_xd_threat_weight.png`,
`fig_xd_threat_neural.png`.

---

## Part 4b — Level decomposition (within-block / block / participant)

`analysis/level_decomposition.py` (`level_decomposition_*.csv`,
`fig_level_decomposition.png`) splits each predictor into a within-block trial
deviation (`_w`), a block mean within participant (`_b`), and a participant mean
(`_p`), fitted jointly (Mundlak). Cloud conflict trials, outcome
`chose_planning`, clustered by participant.

- **Geometry is trial-level.** Value framing (`plan_advantage` +,
  `greedy_advantage` −): within coefficients **+0.108** and **−0.273**
  (both p < 1e‑6); block `greedy_advantage` **−0.38** (p < 1e‑6). The
  between-level terms are large and opposite (±5.2) but unstable (se ≈ 2,
  `unstable` flag) — with only 18 participants the participant-means of the two
  advantages are collinear. `conflict_mag` standalone: within **−0.078**,
  block **−0.16**, between n.s.
- **`conflict_mag = plan_advantage + greedy_advantage` exactly** (identity
  residual 0, design rank 2 of 3). The three are one redundant triple: never fit
  all three together, and the `conflict_mag` coefficient in a joint model equals
  the planning-advantage weight, not a "magnitude" effect. `geometry_value`
  (value framing) is preferred for the choice model; `conflict_mag` is the
  standalone choice-independent target for the neural link.
- **Context is block/participant-level, not trial-level.** `ball_y_at_top`:
  within +0.0004 (p = 0.51, null), block +0.0007 (p = 0.17), between n.s. —
  so the pooled ball-height U was a block/camera effect with no trial-level
  action. `block_conflict_rate` and `block_plan_advantage` act between/block,
  not within.
- **Variance partition (LPM R²).** Value-framing geometry: **trial 72%**,
  block 9%, participant 19% of the (small, R² ≈ 0.07) explained variance.
  `conflict_mag` alone explains far less overall (R² ≈ 0.016) and is more
  between-loaded. Context is ~**83% participant-level** — the confound, made
  quantitative.
- **Planning weight by level** (canonical `chosen_left ~ diff_1step +
  diff_planning`, all trials): within **0.18**, block **0.21**, participant
  **0.16** — the geometry weighting is stable across levels (no Simpson flip).
- **EMU replication** (`level_decomposition_*.csv`, dataset `emu`): geometry
  within terms reproduce (block/participant cells mostly separation-unstable at
  n = 4 and are flagged).

---

## Part 5 — Environmental statistics

`xd_environment.csv`, `xd_block_neural.csv`, `xd_environment_weight.csv`.

- Cloud: within-subject slope of planning rate on the block's conflict rate is
  on average **+0.57** — but this is an unweighted mean of noisy per-participant
  slopes (split-half SB = 0.52). The robust estimate is the within-participant
  GEE **+0.30 (p = 0.03)**, which is stable under block-height and threat
  controls; the pooled cross-block GEE is n.s. (`planning_reanalysis_block.py`).
- EMU: the same slope is on average **−0.01** (range −0.17 to +0.16) — the EMU
  sessions do **not** show the cloud within-session environment sensitivity.
- **Coefficient space** (`xd_environment_weight.csv`): the
  `block_conflict_rate × planning` interaction averages −0.14 for cloud
  (5/18 positive) and −0.08 for EMU (1/4 positive) — the same null-to-negative
  picture as the rate-based slope.
- Per-block neural planning score vs block conflict rate is weakly positive
  (r = 0.09–0.27, n.s.).

**Figures:** `fig_xd_environment.png`, `fig_xd_environment_weight.png`,
`fig_xd_block_neural.png`.

---

## Part 6 — Individual differences

`xd_individual.csv`. The mixture parameters are unstable, so EMU participants
are placed in the cloud trait space using model-free quantities (conflict-trial
planning rate and the `ball_y` slope).

- EMU planning propensity percentiles vs cloud: 0.11–0.33.
- EMU model-based planning-weight percentiles vs cloud: 0.17–0.44.
- EMU threat-sensitivity percentiles vs cloud: 0.28–0.39.
- With n = 2 participants this is a **case study**: the EMU participants are
  unremarkable within the cloud distribution.
- **Reliability caveat** (`planning_reanalysis_reliability.py`): `P(plan)` and
  the planning weight are reliable participant summaries (split-half SB = 0.93),
  but the choice ICC(participant) = 0.02 — the trait explains little of the
  choice variance, which is dominated by within-person trial/block variation.
  The `ball_y` slope is only moderately reliable (SB = 0.77) and the environment
  slope is weak (SB = 0.52), so individual differences in those are noisy.

**Figure:** `fig_xd_individual.png` (rate-based), `fig_xd_individual_weight.png`
(model-based).

---

## Figure-by-figure detail

This section explains exactly what is plotted in each of the 18 figures in
`analysis/cross_dataset_outputs/figures/`. For each figure it gives the
scientific question, the conceptual meaning of the quantity being plotted, how
that quantity is derived, what every axis and mark encodes, how to read the
actual pattern, what the figure would look like if the effect were absent or
confounded, and the caveats. All figures are produced by
`analysis/cross_dataset_figures.py` from the CSVs written by
`analysis/cross_dataset_planning.py`.

### Shared metric definitions — what the numbers actually mean

**The harmonized trial table (Part 0).** Every row is one 1-2-1 sequence — an
entry hole, a 2-hole choice level, and a goal hole — reconstructed identically
for the cloud and EMU data so the two can be compared trial by trial:

- `entry_hole`, `goal_hole`, `chosen_hole`, `unchosen_hole` are segment indices
  (0–11).
- `greedy_cost(h) = |entry − h|` is the **1-step** cost of choosing hole `h`;
  `planning_cost(h) = |entry − h| + |h − goal|` is the **2-step** cost. These
  are the two decision rules the whole project is about.
- `conflict = 1` when the greedy-optimal and planning-optimal holes differ — the
  trials where the two rules make different predictions. `chose_planning` and
  `chose_greedy` mark which rule the participant's actual hole matched;
  `condition ∈ {planning, greedy, agree_optimal, lapse}` combines them
  (`agree_optimal`/`lapse` are the agreement trials where the chosen hole was
  the best/worse one).
- `diff_planning = L1 + L2 − R1 − R2` is the **signed 2-step advantage of the
  left hole** (L/R are the chosen/unchosen 1-step and 2-step distances). It is
  the single geometry feature that drives the choice models: on conflict trials
  it is the model's only input, so "does the neural score add information beyond
  geometry?" means "beyond `diff_planning`?".
- `ball_y_at_top = ball_y − camera_y` at the entry event (pixels). This is the
  ball's **screen-y measured from the top**: the game draws objects at
  `y − cameraY` (`app/static/objects.js:34`) and declares death when
  `ball.y − cameraY < 0` (`app/static/sketch.js:338`). So **small values = the
  ball is high on screen (near the top) / near death**; large values = the ball
  is lower on screen / safe. It is the continuous threat regressor. (An earlier
  draft of this note said "low on screen" — that was backwards.)
- `block_conflict_rate = mean(conflict)` over a block's experimental trials —
  the block's "environment" (how often planning and greed actually disagree).

**Behavioural quantities and what they mean about the participant.**

- `p_plan = P(plan | conflict) = mean(chose_planning)` over a participant's
  conflict trials. This is the participant's **propensity to plan when the two
  rules disagree** — the central individual-difference measure. 0.5 would mean
  they pick the planning hole half the time; below 0.5 means greed dominates.
- `p_optimal = P(optimal | agreement) = mean(chose_greedy)` over agreement
  trials; `p_lapse = 1 − p_optimal` is the **lapse rate** — how often they pick
  the worse hole when there is no conflict, a measure of inattention/error.
- `rt_decision = t(choice) − t(entry)` (ms) is the time from entering the
  sequence to committing at the choice hole; `rt_exec = t(goal) − t(choice)` is
  the time from commitment to the goal. They separate deliberation from
  execution.
- `slope_ball_y` = OLS slope of `chose_planning` on `ball_y_at_top` (per pixel)
  over conflict trials — a **threat-sensitivity index** (negative = plans less
  when closer to death). Per-pixel units make the raw magnitude tiny, which is
  why the tercile split is the more interpretable view.

**Neural quantities (out-of-fold, EMU only) and why out-of-fold matters.** A
decoder is trained on 9/10 of a session's trials and applied to the held-out
tenth; repeating over folds gives every trial a score from a model that never
saw it. This prevents the circularity of using a model's in-sample fit to claim
it "predicts" the behaviour.

- **planning score** = LDA decision score for `planning_vs_greedy` (rate
  representation, post-choice window `[0, +1000]` ms). The decision score is the
  signed projection of the trial onto the population axis that separates the two
  policies: large positive = "this trial looks like a planning trial to the
  population".
- **pre-decision score** = the same LDA score from the entry-anchored window
  `[0, +250]` ms after the entry pass (only trials with `rt_decision ≥ 250` ms,
  so the window never crosses the choice pass).
- **conflict_mag prediction** = ridge out-of-fold prediction of
  `conflict_mag = |planning_gap − greedy_gap|` from the PCA time-resolved
  representation. `conflict_mag` is the maze's magnitude of policy disagreement
  (a property of the geometry, not the choice).

**Choice-information test.** A logistic regression predicts the trial's choice
(planning vs greedy) from geometry alone (`diff_planning`) versus geometry plus
the neural score, evaluated by **cross-validated accuracy** (2 repeats × 10
folds). Because the neural score is itself out-of-fold and the logistic is
cross-validated, the comparison asks a clean question: *does population activity
carry choice information that the maze geometry alone does not?* Accuracy here
is plain accuracy on the conflict subset (which is roughly balanced, so chance is
0.5); a model that ignored the data would score 0.5.

**Cross-dataset transfer.** A logistic model is fit on the cloud all-drift
cohort's conflict trials and applied unchanged to the EMU conflict trials; the
comparison is an EMU model fit with whole blocks held out. This is a
**generalization test**: if the online decision rule transfers to intracranial
patients, the task is being solved the same way across populations. Metrics are
accuracy (does the predicted side match the choice?) and mean held-out
log-likelihood (how confident and correct the probabilities are; 0 is perfect,
−ln 2 ≈ −0.69 is chance).

**Mixture model.** `fit_mixture_model` fits a greedy/planning mixture with a
lapse rate (`p_lapse`), a baseline planning probability (`p_plan_base`), a threat
coefficient (`w1`), inverse temperatures (`s_greedy`, `s_plan`) and a direction
bias. `w1` is bounded to ±8 and `p_plan_base` to (0.02, 0.98): hitting a bound
means the parameter is **not identified** by the data, so those values are
reported with a warning rather than interpreted.

---

### `fig_xd_prevalence.png` — is EMU behaviour inside the cloud range?

- **The question.** Before linking neural activity to behaviour, is the EMU
  participant's decision policy the *same kind of thing* as the online sample's,
  or is the intracranial setting producing a different task?
- **What the quantity means.** `P(plan | conflict)` is the participant's
  propensity to choose the 2-step-optimal hole when greed and planning disagree —
  a stable individual signature of how much they plan.
- **X-axis.** Participant. Cloud participants are ordered by value (so the cloud
  block is a sorted distribution); the four EMU runs are appended at the right so
  they can be read against that distribution.
- **Y-axis.** `P(plan | conflict)`.
- **Marks.** Grey circles = the 18 cloud participants (all datasets); coloured
  diamonds = the four EMU runs; dashed grey line = cloud mean (0.34); dotted
  black line at 0.5 (the point at which the two rules are equally likely).
- **How to read it.** The EMU diamonds (0.25–0.29) sit inside the cloud range
  (0.13–0.46) and almost exactly on the all-drift cohort (0.24–0.28): the
  intracranial sample behaves like the online one. This licenses interpreting the
  EMU neural effects as being about the same behavioural phenomenon.
- **What it would look like if absent.** EMU diamonds outside the cloud range
  (e.g. near 0 or 1) would mean the neural data come from a different regime and
  the cross-dataset comparison is not meaningful.
- **Caveat.** Prevalence is a single number per participant; the figure is a
  comparability check, not a test of a difference.

---

### `fig_xd_rt_tradeoff.png` — is the planning/greedy time trade-off the same?

- **The question.** Does choosing to plan cost and save the same time in the EMU
  sessions as it does online?
- **What the quantity means.** `rt_decision` is deliberation time before
  committing; `rt_exec` is the time to reach the goal after committing. Planning
  is expected to be slower to commit (the farther hole) but faster to execute
  (the shorter remaining path) — a purely kinematic consequence of choosing the
  farther/shorter path on conflict trials.
- **X-axis.** Choice (`planning`, `greedy`), in two panels.
- **Y-axis.** RT in ms: left = decision RT; right = execution RT.
- **Marks.** Each dot = one participant's mean RT for that choice (grey = cloud,
  blue = EMU); the connected line = the group mean.
- **How to read it.** Both datasets show the same shape: planning takes longer to
  commit (left panel higher) and reaches the goal sooner (right panel lower).
  This confirms the EMU task reproduces the cloud task's decision kinematics.
- **What it would look like if absent.** Flat or reversed lines would mean the
  EMU task's physics differ (e.g. different geometry).
- **Caveat.** These are raw means and the trade-off is largely kinematic (the
  farther 1-step hole and shorter 2-step path are properties of conflict trials),
  so the figure is a comparability check, not evidence about decision time. The
  neural RT analysis shows the raw effect is ~90% geometry.

---

### `fig_xd_mixture.png` — model-based summaries

- **The question.** Do the model-based behavioural parameters agree between the
  cloud and EMU samples?
- **What the quantities mean.** `p_lapse` is the probability of choosing the
  worse hole when there is no conflict (inattention). `w1` is the mixture's
  threat coefficient: the effect of `ball_y` on the probability of planning
  (negative = plans more when closer to death).
- **Axes.** **X-axis:** cloud vs EMU. **Y-axis:** the parameter value. Grey dots =
  cloud participants; coloured diamonds = EMU runs.
- **How to read it.** Lapse rates are comparable across datasets (the mixture's
  `p_lapse` sits around 0.4–0.6). `w1` is **unstable** — it hits the ±8 bound for
  several participants in *both* datasets — so the panel carries an explicit
  warning and Part 6 uses model-free quantities instead.
- **What it would look like if absent.** If lapse rates were wildly different, the
  samples would not be comparable; if `w1` were stable and negative in both, that
  would be a model-based threat effect.
- **Caveat.** `p_plan_base` saturates at 0.98 for most participants and is not
  plotted. Bound-hitting means the parameter is not identified, not that the
  effect is large.

---

### `fig_xd_neural_choice.png` — does neural activity add choice information?

- **The question.** Beyond the maze geometry, does the population carry
  information about which hole the participant will choose?
- **What the quantity means.** Choice accuracy is the probability that a
  cross-validated logistic model assigns the correct policy (planning vs greedy)
  to a held-out conflict trial. Comparing geometry-only vs geometry+neural asks
  whether the neural score contains choice information *the geometry lacks*.
- **X-axis.** EMU run. **Y-axis.** Choice accuracy (chance 0.5, dashed).
- **Bars.** Grey = geometry-only logistic (`diff_planning`); blue = geometry plus
  the out-of-fold neural planning score. The text above each pair gives
  `geometry → +neural`.
- **How to read it.** The neural score improves accuracy in **all four runs**
  (0.71→0.79, 0.75→0.88, 0.75→0.78, 0.77→0.78, mean +0.06): the population holds
  choice-relevant information over and above the geometry. This ties the cloud
  behavioural variable (`diff_planning`) to the EMU neural code.
- **What it would look like if absent.** The two bars equal (the neural score
  adds nothing) or the neural bar at chance.
- **Caveat.** The neural score is a decoder *for the choice*, so this establishes
  a coupling between population activity and the decision, not that the neural
  signal causes the choice. The accuracy is not corrected for the number of
  features, but the CV and the out-of-fold score make the comparison fair.

---

### `fig_xd_neural_rt.png` — does the conflict representation track time?

- **The question.** Is the neural representation of policy conflict simply a
  proxy for how long the trial took (a decision-time/kinematic signal)?
- **What the quantity means.** The kinematic-adjusted RT residual is the part of
  `rt_decision` not explained by the chosen 1-step distance (the main kinematic
  driver of RT). Correlating the neural `conflict_mag` prediction with it asks
  whether the conflict code carries timing information beyond the geometry.
- **Left panel.** **X-axis:** run; **y-axis:** Pearson r between the out-of-fold
  `conflict_mag` prediction and the RT residual; bars coloured by run; dashed
  zero line.
- **Right panel.** The trial-level scatter for yfz_1: **x-axis** = neural
  `conflict_mag` prediction; **y-axis** = RT residual (ms); each dot = one trial.
- **How to read it.** Correlations are near zero (|r| ≤ 0.06): the neural
  representation of policy conflict is not a decision-time or kinematic signal.
  This matters because the raw `conflict_mag` decode could otherwise be dismissed
  as "just timing".
- **What it would look like if absent (i.e. if it were timing).** A clear
  positive correlation, especially a scatter with a visible upward trend.
- **Caveat.** The residual controls only the 1-step distance; other kinematic
  factors are handled in the neural confound audit.

---

### `fig_xd_predecision.png` — is there a pre-decision signal?

- **The question.** Is the choice already readable from the population *before*
  the ball reaches the choice hole? This is the "computation vs consequence"
  question posed as a behavioural-prediction test.
- **What the quantity means.** Same as `fig_xd_neural_choice.png`, but the neural
  feature is the **entry-anchored pre-decision** score — activity in `[0, +250]`
  ms after the entry pass, strictly before the choice.
- **X-axis.** Run; **y-axis:** choice accuracy (chance 0.5). Grey = geometry only;
  red = geometry plus the pre-decision neural score.
- **How to read it.** The pre-decision score adds **nothing** in all four runs
  (Δaccuracy ≤ 0.005, slightly negative in two): there is no pre-decision neural
  read-out of the plan. This mirrors the cloud finding that behaviour shows no
  pre-choice strategy state, and the neural finding that the post-choice
  `planning_vs_greedy` effect is a consequence of the choice.
- **What it would look like if absent.** A clear accuracy increase would mean the
  plan is computed and held before the choice.
- **Caveat.** The window is fixed at 250 ms post-entry; a very early signal could
  in principle be missed, but the wider entry-anchored analyses in the neural
  pipeline find the same null.

---

### `fig_xd_transfer.png` — does a cloud-fitted model predict EMU choices?

- **The question.** Does the decision rule learned from the online sample
  generalize to the intracranial participants?
- **What the quantity means.** Accuracy is whether the predicted policy matches
  the participant's actual choice on held-out EMU conflict trials; mean
  log-likelihood measures how good the predicted probabilities are (0 = perfect,
  −ln 2 ≈ −0.69 = chance).
- **X-axis.** Model: `cloud->EMU` (logistic fit on the cloud all-drift cohort,
  tested on EMU) vs `EMU(block-CV)->EMU` (logistic fit on EMU with whole blocks
  held out — the strongest within-EMU baseline).
- **Y-axis.** Left panel = accuracy (chance 0.5, dashed); right panel = mean
  held-out log-likelihood.
- **How to read it.** The cloud-fitted model performs **as well as** the
  EMU-fitted model (accuracy 0.738 both; log-likelihood −0.568 vs −0.565): the
  online decision rule transfers to the patients without refitting. This is the
  strongest statement of task-level comparability in the project.
- **What it would look like if absent.** The cloud model clearly worse than the
  EMU-fitted model (the task would be solved differently by the patients).
- **Caveat.** Only the geometry features (`diff_1step`, `diff_planning`) are
  transferred — neural data cannot be transferred across datasets because there
  is no neural recording in the cloud sample.

---

### `fig_xd_model_neural.png` — does the neural code match the model variable?

- **The question.** Is the quantity the neural population encodes the *same*
  quantity the behavioural model uses to explain choices?
- **What the quantity means.** `conflict_mag` is the maze's magnitude of
  disagreement between the greedy and planning rules — a property of the
  geometry. The correlation between the out-of-fold neural prediction and the
  analytic value asks whether the population tracks this model variable.
- **X-axis.** Run; **y-axis:** Pearson r; bars coloured by run; dashed zero line.
- **How to read it.** The neural representation matches the model quantity
  strongly in the two YFZ sessions (r = 0.26, 0.41) and weakly in YGA
  (r = 0.07, 0.08). Where the match is strong, the neural code and the
  behavioural model are speaking about the same thing.
- **What it would look like if absent.** All bars near zero (the neural code
  would encode something unrelated to the model's conflict variable).
- **Caveat.** The YGA weakness mirrors the smaller `conflict_mag` decoding in
  those sessions (fewer units, more bursty units), not necessarily a different
  code.

---

### `fig_xd_threat.png` — is the behavioural threat effect present in EMU?

- **The question.** Does the cloud threat effect (planning falls under threat)
  also appear in the EMU sessions?
- **Left panel (behavioural threat effect).** For each participant, the planning
  rate in the **safe** third of `ball_y` (high, left) vs the **near-death** third
  (low, right). **X-axis:** safe → near death; **y-axis:** `P(plan | conflict)`.
  Grey lines = cloud participants (thick grey = cloud mean); coloured lines =
  EMU runs. A downward line means less planning near death.
- **Right panel (threat sensitivity).** The per-participant `slope_ball_y`.
  **X-axis:** cloud vs EMU; **y-axis:** slope of `chose_planning` on `ball_y`
  (per pixel); each dot = one participant.
- **How to read it.** Cloud shows lower planning near death (thick grey line
  slopes down), and 3 of 4 EMU runs show the same direction (−0.015, −0.055,
  0.000, −0.028). The linear slopes are near zero because `ball_y` is in pixels,
  which is why the tercile split is the clearer view.
- **What it would look like if absent.** Flat lines in the left panel and slopes
  centred on zero in the right panel.
- **Caveat.** EMU is all-drift, so the cloud drift-vs-follow contrast has no EMU
  analogue; the threat test uses within-session `ball_y` variation only, which is
  a weaker manipulation than the block-level drift manipulation.

---

### `fig_xd_threat_neural.png` — does threat change the neural planning signal?

- **The question.** Does the population's planning code change under threat, as
  it might if threat shifted cognitive resources?
- **What the quantity means.** `planning_vs_greedy` decoding accuracy (balanced)
  computed separately in the low- and high-threat halves of each session (median
  split on `ball_y`). If threat modulated the planning representation, the two
  bars would differ.
- **X-axis.** Run; **y-axis:** balanced accuracy (chance 0.5, dashed). Grey = low
  threat; red = high threat.
- **How to read it.** The planning signal is present under both threat levels
  with no consistent difference — the population distinguishes the policies
  regardless of threat. This is consistent with the neural threat null: there is
  no dedicated threat representation and no threat modulation of the planning
  code.
- **What it would look like if absent (i.e. if threat mattered).** A systematic
  accuracy difference between the grey and red bars.
- **Caveat.** The median split gives ~215 trials per half; a subtler modulation
  could be missed.

---

### `fig_xd_environment.png` — does planning track the block environment?

- **The question.** Does a participant adapt how much they plan to the local
  environment (how often planning actually matters in that block)?
- **What the quantity means.** For each participant, the OLS slope of their
  per-block planning rate on that block's conflict rate. A positive slope means
  they plan more in blocks where the two rules disagree more often — the Exp 2
  sensitivity to environmental statistics.
- **X-axis.** Cloud vs EMU; **y-axis:** the slope. Each dot = one participant;
  the black diamond ± error bar = group mean ± SEM; dashed zero line.
- **How to read it.** The cloud mean slope is positive (≈ +0.57 unweighted;
  the robust within-participant GEE is **+0.30, p = 0.03**, robust to block
  height/threat controls) — the online sensitivity. The mean per-participant
  slope is noisy (split-half SB = 0.52), so treat it as a group-level effect, not
  a precise individual trait. The EMU mean is ≈ −0.01, i.e. the EMU sessions do
  **not** show within-session adaptation to the block's conflict rate. This is the
  one clear cross-dataset divergence.
- **What it would look like if absent.** Dots centred on zero (which is what EMU
  shows).
- **Caveat.** EMU has only 26 blocks per session and a single environment
  (all-drift), so this is a genuine but modest-powered null; it does not rule out
  a smaller effect.

---

### `fig_xd_block_neural.png` — does the neural code track the block environment?

- **The question.** Does the neural planning representation itself strengthen in
  blocks where planning matters more?
- **What the quantity means.** Per run, the correlation between the block-mean
  out-of-fold neural planning score and the block's conflict rate. A positive
  correlation would mean the population's policy code is stronger in
  high-conflict environments.
- **X-axis.** Run; **y-axis:** Pearson r; bars coloured by run; dashed zero line.
- **How to read it.** Correlations are weakly positive (r = 0.09–0.27) and not
  significant — consistent with the behavioural null in EMU, the neural
  representation does not track the block environment.
- **What it would look like if absent.** Bars at zero; what a real effect would
  look like: consistently positive, significant bars.
- **Caveat.** This uses the cheap block-rate correlation (trial-level scores
  averaged per block) rather than a true per-block decoder, so it is a
  conservative screen.

---

### `fig_xd_individual.png` — where do EMU participants sit?

- **The question.** Are the intracranial participants unusual within the online
  population, or ordinary points in its distribution?
- **What the quantities mean.** X = planning propensity (`P(plan | conflict)`);
  y = threat sensitivity (`slope_ball_y`). Together they are a two-dimensional
  behavioural trait space for the task.
- **Marks.** Each grey dot = one cloud participant; coloured diamonds = the EMU
  runs (annotated).
- **How to read it.** The EMU participants cluster near the cloud centre
  (planning-propensity percentiles 0.11–0.33; threat-slope percentiles
  0.28–0.39) — unremarkable within the online distribution. This supports
  treating them as representative rather than special cases.
- **What it would look like if absent.** EMU points far outside the cloud cloud,
  which would caution against generalizing the neural findings.
- **Caveat.** n = 2 participants → a case study, not an individual-differences
  result; no correlation between neural and behavioural traits can be estimated.
  Mixture parameters are not used here because they are unstable.

---

### `fig_xd_prevalence_weight.png` — model-based prevalence

- **The question.** Same comparability question as `fig_xd_prevalence.png`, but
  with the coefficient-based planning weight instead of the raw conflict rate.
- **What the quantity means.** `planning_weight = |b_plan| / (|b_plan| +
  |b_1step|)` from the per-participant choice logistic (Part 1b): the share of
  the geometry log-odds explained by the 2-step (planning) rule.
- **Axes/marks.** Same layout as `fig_xd_prevalence.png`: x = participant
  (cloud sorted, EMU appended), y = planning weight; grey circles = cloud,
  coloured diamonds = EMU, dashed line = cloud mean (0.27).
- **How to read it.** EMU diamonds (0.08–0.22) sit inside the cloud range
  (0.02–0.58) — the model-based measure reproduces the Part 1 conclusion.
- **What it would look like if absent.** EMU outside the cloud range, or the
  cloud distribution compressed to a single value.
- **Caveat.** The weight is a ratio of two fitted coefficients, so it is noisier
  than the raw rate (it also moves with `b_1step`); `corr(p_plan, weight) =
  +0.90`, not 1.

---

### `fig_xd_threat_weight.png` — threat in coefficient space

- **The question.** Does the *planning coefficient* change with ball height, and
  in which direction?
- **What the quantities mean.** Left: the normalized planning weight in the
  near-death vs safe thirds of `ball_y_at_top`. Right: the raw `b_plan` (more
  negative = more planning) in the same thirds.
- **X-axis.** Near death (ball high on screen, low `ball_y_at_top`) → safe (ball
  low on screen). **Y-axis:** planning weight (left) / `b_plan` (right). Thin
  lines = participants; thick line = group mean.
- **How to read it.** Cloud: the mean raw `b_plan` moves from −0.15 (near death)
  to −0.24 (safe), i.e. the planning coefficient is *larger when the ball is
  lower/safer* — opposite to the intuitive "threat increases planning". EMU: the
  mean raw `b_plan` moves the other way (3/4 runs), but the effect is tiny and
  the two estimation methods disagree in sign.
- **What it would look like if absent.** Flat thick lines.
- **Caveat.** The normalized weight and the raw coefficient can move in opposite
  directions because the greedy coefficient also varies with ball height; use
  the raw coefficient (right panel) to answer "is the planning weight larger?".

---

### `fig_xd_environment_weight.png` — environment in coefficient space

- **The question.** Does the planning coefficient adapt to the block's conflict
  rate?
- **What the quantity means.** Per-participant coefficient on
  `block_conflict_rate × planning` from the conflict-trial choice logistic. A
  positive value means the planning weight grows in high-conflict blocks.
- **X-axis.** Cloud vs EMU; **y-axis:** the interaction coefficient; dots =
  participants, diamond ± SEM = group mean; dashed zero line.
- **How to read it.** Cloud mean −0.14 (5/18 positive) and EMU mean −0.08
  (1/4 positive): no positive environment sensitivity in either sample,
  matching the rate-based null in EMU.
- **What it would look like if absent.** Dots centred on zero (which is roughly
  what both show).
- **Caveat.** `block_conflict_rate` is constant within a block, so this is a
  between-block moderator estimated from trial-level choices; EMU has only 26
  blocks per run.

---

### `fig_xd_individual_weight.png` — model-based trait space

- **The question.** Same as `fig_xd_individual.png`, using the model-based
  planning weight on the x-axis.
- **What the quantities mean.** X = planning weight; y = threat sensitivity
  (slope of planning on `ball_y_at_top`).
- **How to read it.** EMU runs have planning-weight percentiles 0.17–0.44 within
  the cloud distribution — unremarkable, as in the rate-based view.
- **What it would look like if absent.** EMU points outside the cloud cloud.
- **Caveat.** n = 2 participants; case study only.

---

### `fig_xd_summary.png` — scorecard

- **The question.** At a glance, which cross-dataset findings hold and which do
  not?
- **What it shows.** Each row is a finding; a green / orange / red dot marks
  pass / partial / fail, with the supporting number in the right column.
- **How to read it.** It is a navigation aid, not a statistical test: the
  per-figure numbers and the Parts above are the evidence. Read the coloured dots
  as "did the EMU data reproduce the cloud finding?".

---

## Scorecard

`fig_xd_summary.png` summarizes the findings:

| finding | status |
|---|---|
| prevalence inside cloud range | pass |
| model-based planning weight inside cloud range | pass |
| planning/greedy RT trade-off | pass |
| neural score adds choice info | pass (4/4) |
| pre-decision adds nothing | pass (4/4) |
| cloud model transfers to EMU | pass |
| neural matches model conflict | partial (strong YFZ, weak YGA) |
| behavioral threat effect | pass in cloud; **inconclusive in EMU** (rate vs coefficient disagree) |
| environment sensitivity | **fail in EMU** (cloud positive, EMU null) |

## Caveats

- n = 2 EMU participants → individual-difference and transfer claims are a case
  study, not inference.
- EMU is all-drift only → no neural drift-vs-follow contrast; the threat test is
  limited to within-session `ball_y` variation.
- Cloud has no neural data → the combination is behavioural comparability plus
  within-EMU neural linkage, not a joint neural model.
- The mixture model is unstable in both datasets; model-free summaries are used.
- Neural out-of-fold scores are recomputed per run and are not unit-matched
  across runs.
