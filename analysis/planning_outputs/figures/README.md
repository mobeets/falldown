# Planning experiments — figure captions, axes, and methods

This document explains every figure in `analysis/planning_outputs/figures/`,
including what each axis is, exactly how it is calculated, and what the figure
shows. It is generated alongside the outputs of
`analysis/planning_experiments.py` and `analysis/planning_figures.py`.

---

## 1. Data and inclusion

**Source.** `data/logs_sorted/` — the three online experiments:

| dataset | config | threat | environment |
|---|---|---|---|
| `short_trials_experiment_nodrift` | `short_trials_experiment_nodrift.json` | follow only (mode 0) | balanced (~0.53 agreement) |
| `short_trials_experiment` | `short_trials_experiment.json` | alternating drift/follow | high agreement (~0.79) |
| `short_trials_experiment-7-10` | `short_trials_experiment-7-10.json` | drift only (mode 1) | balanced (~0.53) |

**Inclusion (Definition A).** A participant is retained if

```
distinct experimental blocks played (>4 trials) / experimental blocks in the config  >= 0.6
```

where the denominator is 40 for the two 44-block configs and 32 for the
36-block `short_trials_experiment` config. **Exp 1 (`nodrift`) uses a relaxed
0.40 cutoff** because the strict 0.6 rule left only 3 participants. Retained:
`exp1_nodrift` 5, `exp3_altdrift` 9, `exp3_alldrift` 4 — **18 participants**.
Every participant's numerator, denominator, fraction and retained flag is in
`completion_table.csv`.

---

## 2. Sequence construction and derived quantities

The game presents levels in **3-level sequences**: entry → choice → goal. For
sequence `i` of a block the three trials are `trials[3i]`, `trials[3i+1]`,
`trials[3i+2]`. Only sequences where all three levels were passed are used.

**Geometry** (hole positions are segment indices 0–11):

| symbol | definition |
|---|---|
| `entry_hole` | hole used at level `3i` |
| `choice_holes` | the two holes at level `3i+1` |
| `goal_hole` | hole used at level `3i+2` |
| `chosen_hole` | hole actually used at level `3i+1` |
| `unchosen_hole` | the other hole at level `3i+1` |

**Costs** (distance is the absolute difference of segment indices):

- greedy (1-step): `greedy_cost(entry, h) = |entry − h|`
- planning (2-step): `planning_cost(entry, h, goal) = |entry − h| + |h − goal|`
- `greedy_optimal_hole = argmin_h greedy_cost(entry, h)`
- `planning_optimal_hole = argmin_h planning_cost(entry, h, goal)`
- **`conflict`** = greedy and planning prescribe different holes
- `chose_greedy = [chosen_hole == greedy_optimal_hole]`
- `chose_planning = [chosen_hole == planning_optimal_hole]`
- `plan_advantage = planning_cost(entry, greedy_optimal_hole, goal) − planning_cost(entry, planning_optimal_hole, goal)`
  (how much 2-step cost the participant would save by following planning)
- `greedy_advantage = greedy_cost(entry, planning_optimal_hole) − greedy_cost(entry, greedy_optimal_hole)`

**Distances travelled** (used for RT controls):

- `chosen_1step_dist = |chosen_hole − entry_hole|`
- `chosen_2step_dist = |chosen_hole − entry_hole| + |chosen_hole − goal_hole|`
- `unchosen_1step_dist`, `unchosen_2step_dist` are the same for the other hole.

**Direction coding** (for logistic models):

- `chosen_left = [chosen_hole < unchosen_hole]`
- `L1 = chosen_1step_dist` if the left hole was chosen else `unchosen_1step_dist`;
  `R1` is the mirror. Likewise `L2`, `R2` for the second step.
- `diff_1step = L1 − R1` (signed 1-step advantage of the left hole)
- `diff_planning = L1 + L2 − R1 − R2` (signed 2-step advantage of the left hole)

**Reaction times** (ms, from `performance.now()` timestamps):

- `observed_rt = t(goal) − t(entry)` (total sequence time)
- `rt_decision = t(choice) − t(entry)` (entry → choice: time to commit)
- `rt_exec = t(goal) − t(choice)` (choice → goal: execution of the chosen path)
- `rt_total = rt_decision + rt_exec`
- Sequences outside `[Q1 − 2.5·IQR, Q3 + 2.5·IQR]` of `observed_rt` are dropped.

**Threat / context:**

- `block_drift` = `block_config.params.startCameraMode` (1 = drift, ball can die; 0 = follow)
- `ball_y_at_top` = `ball_y − camera_y` at the game-state frame nearest `t(entry)`
  (the ball's screen y; it approaches 0 as the camera catches a dying ball, so
  **small values = close to death**)
- `incoming_direction = −sign(goal_{i−1} − entry_i)` for consecutive sequences in
  the same block (the lateral direction the ball is already moving)
- `scroll_speed` at the entry event

**Environment statistics** (per block, over experimental trials):

- `block_conflict_rate = mean(conflict)`
- `block_plan_advantage = mean(plan_advantage)`

**Standardization.** `z_ball_y` and `z_block_conflict` are z-scores
(`(x − mean) / sd`, population sd) computed on the pooled analysis table.

---

## 3. Figures

Figures 3.1 (`fig_planning_prevalence.png`) and 3.2
(`fig_planning_temporal.png`) cover **all 18 online participants** and colour
markers by dataset: nodrift (green), alternating-drift (orange), all-drift (red).
The remaining Exp 1 / Exp 2 / Exp 3 figures use the subsets described in their
captions.

### 3.1 `fig_planning_prevalence.png` — planning prevalence (all online participants)

**Scope.** All 18 retained participants (5 nodrift, 9 alternating-drift, 4
all-drift), coloured by dataset (nodrift green, alt-drift orange, all-drift red).

**Left panel.**
- **Y-axis:** participants (18), sorted by planning rate.
- **X-axis:** `P(plan | conflict) = mean(chose_planning)` over that
  participant's experimental **conflict** trials.
- Marker colour = dataset; the legend gives each dataset's mean; dashed line at
  0.5.
- Annotation: one-sample *t*-test of the 18 per-participant values against 0.5
  (`t = −8.08`, `p = 3.2e-07`, N = 18).

**Right panel — per-participant proportions grouped by agreement.**
- **X-axis:** four outcome proportions, grouped by whether greedy and planning
  prescribe the same hole:
  - *agreement trials:* `optimal = P(optimal | agreement) = mean(chose_greedy)`
    on agreement trials; `lapse = 1 − optimal`.
  - *conflict trials:* `planning = P(planning | conflict) = mean(chose_planning)`;
    `greedy = 1 − planning`.
- **Y-axis:** proportion of trials. Bars = mean across the 18 participants,
  error bars = SEM; black dots = individual participants (jittered within each
  bar). A vertical dashed line separates the agreement and conflict groups.
- Values: optimal 0.84, lapse 0.16, planning 0.34, greedy 0.66.

**Reading.** People use both strategies but plan in the minority of conflict
trials (~34%), and are near-optimal on agreement trials (~84% optimal / 16%
lapse). Planning rate varies across participants (0.13–0.46) and is somewhat
higher in the alt-drift dataset.

**Note.** The previous version of this figure pooled raw sequence counts over
all datasets on the right while showing only Exp 1 participants on the left.
Both panels are now computed **per participant over the same 18-participant
scope** (means are unweighted by trial count).

---

### 3.2 `fig_planning_temporal.png` — temporal structure (all online participants) [SUPERSEDED]

> **Superseded** by the sequential re-analysis
> (`fig_planning_reanalysis_sequential.png`). This stay/switch test was one-sided
> for *excess stay* and so could not detect **alternation**; the re-analysis
> finds a small negative within-block lag-1 dependence (LPM beta = -0.089,
> permutation p < 0.001). The file has been moved to
> `analysis/superseded_figures/`. The description below is kept for provenance.

**Scope.** All 18 retained participants, coloured by dataset.

For each participant, the sequence of **conflict trials** is reduced to a binary
strategy string (`1 = chose_planning`). Stay probability is computed over
adjacent conflict trials **within the same block**.

**Left panel.**
- **X-axis:** memoryless-null stay probability = mean over 1000 permutations of
  the strategy string, where within each block the `0/1` labels are shuffled
  (preserving each block's length and base rate). Error bars = 2.5th/97.5th
  percentiles of the permutation distribution.
- **Y-axis:** observed stay probability = fraction of adjacent same-block
  conflict pairs with identical strategy.
- Dashed diagonal `y = x`; marker colour = dataset.

**Right panel.**
- **Y-axis:** participants (18), sorted by excess persistence.
- **X-axis:** excess persistence = `observed_stay − permutation_mean`; error bars
  from the permutation percentiles; bar colour = dataset; `perm_p` = fraction of
  permutations with stay ≥ observed.
- Annotation reports how many participants survive permutation `p < 0.05`
  (here **0 of 18**).

**Reading.** All points fall on the diagonal and every CI crosses zero: once the
base planning rate is controlled, strategy choices look **memoryless** across the
whole online sample — no evidence of sequential switching structure.

---

### 3.3 `fig_hmm_selection.png` — choosing the number of HMM states

**Model.** A K-state GLM-HMM is fit per participant on the choice trials.
Features are `[z(diff_1step), z(diff_planning), incoming_direction, 1]`, where the
first two are z-scored on the training split. The first 80% of a participant's
choice trials are used for fitting, the last 20% are held out. `K=1` is a
plain logistic regression (MLE via `statsmodels.Logit`); `K≥2` uses
`ssm.HMM` with an input-driven categorical emission (`C=2`) fit by EM (100
iterations).

**Free parameters:** `p = K·M + K·(K−1) + (K−1)` with `M = 4` (K emission weight
vectors, `K(K−1)` transition probabilities, `K−1` initial-state probabilities).

- **BIC** `= −2·LL_train + p·ln(T_train)`
- **AIC** `= −2·LL_train + 2·p`
- **held-out LL** = log-likelihood of the test split

**Panel 1–2 (BIC vs K).**
- **X-axis:** number of states K ∈ {1,2,3,4}.
- **Y-axis:** BIC.
- Thin gray lines = individual participants; thick colored line = group sum;
  dashed line marks the group `K*` = argmin of the summed BIC.
- Result: **K* = 1** for both experiments (per-participant: 5/5 and 13/13 at K=1).

**Panel 3 (held-out LL vs K).**
- **X-axis:** K; **Y-axis:** sum of held-out log-likelihoods (higher = better).
- Result: Exp 3 picks K*=1; Exp 1 nominally picks K*=4 by ≈1 nat total — a flat,
  non-monotone curve consistent with noise.

**Panel 4 (criterion sensitivity).**
- **X-axis:** per-participant optimal K; **Y-axis:** number of participants
  (both experiments pooled).
- Solid bars = BIC-optimal K (all 18 at K=1); hatched bars = held-out-LL-optimal
  K (spread across 1–4).

**Reading.** BIC consistently selects a single state; held-out LL is flat/noisy;
in-sample train LL (not shown) is monotone and always picks the largest K. The
justified model is therefore K=1.

---

### 3.4 `fig_hmm_weights.png` — emission coefficients

**Panel 1 — justified single state (K*=1).**
- **X-axis:** the four emission features: `L1-R1` (greedy/1-step signed
  advantage), `L1+L2-R1-R2` (planning/2-step signed advantage),
  `incoming_dir`, `bias`.
- **Y-axis:** coefficient on the log-odds scale. For K=1 this is the logistic
  coefficient predicting `P(chose left)` per one SD of the feature (the first
  two features are z-scored; `incoming_dir` and the bias are as-is).
- Blue bar = group mean; black dots = individual participants.

**Panel 2 — forced K=2 (not BIC-supported).**
- Same x-axis; grouped bars for state 0 and state 1 with individual dots.
- For K≥2 these are the `ssm` emission weights, which parameterize the two
  outcome categories (the last category is the softmax baseline); interpret the
  *differences between states*, not the absolute sign.
- The states differ mainly on `L1-R1` (≈3.7 vs 1.1) — weak separation.

**Panel 3 — individual coefficients (forced K=2).**
- **Y-axis:** participants; **X-axis:** `(state, feature)` pairs.
- Colour = coefficient value (diverging around 0), showing individual variation.

**Reading.** With the justified single state the coefficients are stable across
participants; forcing two states produces only weak, unsupported differences.

---

### 3.5 `fig_hmm_k2_states.png` — forced two-state solution (interpretation only)

**Panel 1 — mean transition matrix.**
- **X-axis:** to-state; **Y-axis:** from-state.
- Values = `exp(log_Ps)` averaged over participants for the forced K=2 fit.

**Panel 2 — state occupancy.**
- **X-axis:** participants; **Y-axis:** mean filtered posterior per state,
  stacked to 1. Occupancy = `mean_t P(state | choices_{1:t})`.

**Panel 3 — state-2 posterior over time.**
- **X-axis:** choice-trial index; **Y-axis:** filtered posterior
  `P(state 2 | choices up to t)` for each participant.

**Reading.** The forced K=2 solution shows high "stickiness" (diagonal ≈
0.95/0.92) plus flickering posteriors — an artifact rather than real states.
BIC rejects this model; the panel exists only to show what the states look like.

---

### 3.6 `fig_exp2_within.png` — sensitivity to environment statistics

Per block we compute `block_conflict_rate`, `block_plan_advantage` (both over the
block's experimental trials) and `planning_rate = mean(chose_planning)` over the
block's conflict trials. A participant needs ≥6 blocks.

**Panels 1–2 (participant-centred scatter).**
- **X-axis:** the block statistic **minus that participant's mean**
  (participant-centred conflict rate / planning advantage).
- **Y-axis:** `planning_rate` **minus that participant's mean**.
- Thin gray lines connect a participant's blocks; the red line is the pooled
  within-participant OLS fit on the centred data.
- Annotation: mean of the per-participant slopes and a one-sample t-test of the
  slopes against 0 (conflict rate: +0.31, p = 0.044; planning advantage: +0.14,
  p = 0.012; N = 18).

**Panel 3 (slopes).**
- **X-axis:** predictor (conflict rate vs planning advantage).
- **Y-axis:** per-participant OLS slope of planning rate on the block statistic.
- Dots = participants; diamond ± error bar = group mean ± SEM; dashed line at 0.

**Reading.** Within a participant, planning rate rises with the block's conflict
rate and expected planning advantage — sensitivity to environment statistics,
the opposite of insensitivity. Centring matters: a naive pooled fit shows the
wrong sign (Simpson's paradox), which is why the figure uses within-participant
centred data.

---

### 3.7 `fig_exp2_between.png` — between-cohort environment comparison

- **X-axis:** cohort — `balanced` (`nodrift` + `-7-10`, ~0.53 agreement) vs
  `high_agree` (`short_trials_experiment`, ~0.79 agreement).
- **Y-axis:** `P(plan | conflict)` pooled over each participant's experimental
  conflict trials.
- Coloured dots = participants (jittered); black diamond ± error bar = cohort
  mean ± SEM; `n` printed above each group.
- Annotation: Welch t-test between cohorts (t = −3.46, p = 0.004).

**Reading.** The high-agreement environment does **not** reduce planning
(0.39 vs 0.28, if anything higher). Caveat: this is a between-cohort comparison
confounded with the drift design.

---

### 3.8 `fig_exp3_threat.png` — threat and planning

**Left panel (block-level threat).**
- **X-axis:** condition — `follow` (no threat) vs `drift` (threat).
- **Y-axis:** `P(plan | conflict)` pooled within each participant × condition
  (requires ≥15 conflict trials in both).
- Gray lines = individual participants; thick line = group mean; annotation =
  paired t-test (drift − follow = −0.05, t = −3.38, p = 0.010, N = 9).

**Right panel (continuous proximity to death).**
- **X-axis:** `safe` (highest third of that participant's `ball_y_at_top`) →
  `near death` (lowest third). The axis is ordered **safe → near death** so that,
  as in the left panel, moving right means more threat; a negative slope then
  means "less planning under threat" in both panels.
- **Y-axis:** `P(plan | conflict)` in each third.
- Paired lines + group mean; paired t-test (near − safe = −0.03, p = 0.296,
  N = 13).

**Reading.** Both panels run no-threat → threat, and both group means slope
**down**: threat (drift, and weakly proximity to death) is associated with
**less** planning.

---

### 3.9 `fig_exp3_forest.png` — threat and RT coefficients

Generalized estimating equations (GEE) with a clustered-by-participant
exchangeable working correlation. Planning models are Binomial; RT models are
Gaussian on `log(RT)`.

**Panel 1 — `chose_planning` (Binomial).**
`chose_planning ~ diff_planning + z_ball_y + block_drift + z_ball_y×block_drift + incoming_direction`
- **X-axis:** coefficient (log-odds); **Y-axis:** term, with stars from the
  p-value. Whiskers = 95% CI.
- `block_drift` = −0.33*** (less planning under threat); `z_ball_y`,
  `diff_planning` n.s.

**Panel 2 — `log(rt_decision)` (Gaussian), kinematic-controlled.**
`log(rt_decision) ~ chose_planning + chosen_1step_dist + block_drift + z_ball_y`
- `chose_planning` = +0.04 (n.s.); `chosen_1step_dist` = +0.23***.

**Panel 3 — `log(rt_exec)` (Gaussian), kinematic-controlled.**
`log(rt_exec) ~ chose_planning + chosen_2step_dist + chosen_1step_dist + block_drift + z_ball_y`
- `chose_planning` = −0.28***; `chosen_2step_dist` = +0.15***;
  `chosen_1step_dist` = −0.16***.

**Reading.** Threat reduces planning. Once the kinematic distances are
controlled, planning's decision-time cost disappears and only a small
execution-time benefit remains.

---

### 3.10 `fig_exp3_exec.png` — distance-adjusted reaction times (conflict trials)

On conflict trials the planning hole is necessarily the **farther** 1-step hole
and the **shorter** 2-step path, so raw RT gaps are geometric. This figure
removes that geometry.

**Adjusted RTs** are OLS residuals (with intercept):

- `rt_decision_adj` = residual of `rt_decision ~ 1 + chosen_1step_dist`
- `rt_exec_adj` = residual of `rt_exec ~ 1 + chosen_1step_dist + chosen_2step_dist`
- `rt_total_adj` = residual of `rt_total ~ 1 + chosen_1step_dist + chosen_2step_dist`

**Panels.**
- **X-axis:** choice (greedy vs planning).
- **Y-axis:** mean adjusted RT (ms); bars are group means, error bars SEM,
  zero line marked, and the planning − greedy difference printed.
- Result: **+33 / −4 / +4 ms** (decision / execution / total) versus raw
  **+443 / −879 / −435 ms**.
- Footnote gives the kinematic-controlled log-RT GEE: decision +0.04 (n.s.),
  execution −0.28 (p < .001), total +0.01 (n.s.).

**Reading.** ~90% of the raw RT effect was kinematic; after adjustment the
differences are near zero.

---

### 3.11 Re-analysis figures (level / reliability / sequential / block audit)

Generated by the `planning_reanalysis_*` scripts and `claim_audit.py`.

- `fig_planning_reanalysis_reliability.png` — split-half reliability of the
  participant-level summaries: `P(plan)` and planning weight are reliable
  (Spearman-Brown 0.93), `slope_ball_y` moderate (0.77), environment slope weak
  (0.52). Choice ICC(participant) = 0.02: most choice variance is within-person.
- `fig_planning_reanalysis_sequential.png` — lag Mundlak, pooled lag, and
  GLM-HMM BIC with/without a lag input. The within-block lag effect is negative
  (alternation-like), survives uncentered block fixed effects, and permutes
  p < 0.001; adding a lag input does not improve HMM BIC.
- `fig_planning_reanalysis_block.png` — environment slope pooled vs
  within-participant (with height/threat controls), per-participant slope spread
  (unreliable), and threat vs block-mean ball height jointly.
- `fig_claim_audit.png` — claim-by-claim status board (robust / relabeled /
  qualified / revised / confounded) with the re-estimated effects.
- `analysis/level_decomposition.py` → `fig_level_decomposition.png` (Part 4b).
- `analysis/participant_peak_location.py` →
  `fig_participant_peak_location.png`.
- `analysis/threat_bins_analysis.py` → `fig_threat_bins5.png`.

## 4. Outputs and how to rerun

Analysis tables and stats are written to `analysis/planning_outputs/`
(`sequence_table.csv`, `completion_table.csv`, `exp1_*`, `exp2_*`, `exp3_*`).
Note that `exp1_prevalence.csv` and `exp1_transitions.csv` now hold **all 18
participants** (per-participant rows carry `experiment`/`dataset` labels), while
the HMM selection and `exp1_glmhmm` outputs remain scoped to Exp 1.
Figures are written to `analysis/planning_outputs/figures/`.

```
# create the cross-platform analysis env once:
#   bash analysis/setup_env.sh
python analysis/planning_experiments.py            # full reload
python analysis/planning_experiments.py --from-cache
python analysis/planning_figures.py
python analysis/planning_reanalysis_reliability.py
python analysis/planning_reanalysis_sequential.py
python analysis/planning_reanalysis_block.py
python analysis/claim_audit.py
```

`planning_experiments.py` builds the sequence table, applies the completion
filter, runs the HMM model selection (BIC) and the Exp 1/2/3 analyses, and
writes the CSVs. `planning_figures.py` reads those outputs and renders the
figures above.
