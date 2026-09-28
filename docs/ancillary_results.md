# Ancillary results

Results produced across the Falldown project that are **not** part of the main
planning/threat/neural narrative in `full_results_synthesis.md`, gathered by a
systematic sweep of every output CSV and findings doc. Grouped by topic, with
source files and scripts. Where a doc was stale or an output failed, that is
flagged.

---

## Behaviour and modelling

### 1. Transition / persistence structure (`exp1_transitions.csv`)
For each participant, the stay probability of the planning choice between
adjacent same-block conflict trials is compared with a within-block permutation
null (labels shuffled, blocking/base rate preserved). Observed stay ≈ the
memoryless null (e.g. 0.539 vs 0.527, perm p = 0.77; 0.776 vs 0.773, p = 0.39;
mean run length ≈ 2.2–4.4). **0 of 18** participants have excess persistence at
perm p < 0.05. This backs the "no latent strategy state" claim. **Caveat
(2026 re-analysis):** this test is one-sided for excess *stay* and so cannot see
alternation; `planning_reanalysis_sequential.py` finds a small negative
within-block lag-1 effect (LPM β = −0.09, permutation p < 0.001), so choices are
not strictly memoryless.

### 2. GLM-HMM beyond the state count (`exp1_glmhmm*.csv`, `exp1_hmm_state_*.csv`)
- BIC selects K* = 1 for 18/18 cloud participants and 4/4 EMU runs.
- Held-out LL is flat/noisy: Exp-1 nominally prefers K* = 4 by ≈1 nat total
  (noise); Exp-3 prefers K* = 1.
- Forced K = 2 gives near-absorbing transitions (diagonal ≈ 0.95 / 0.92),
  state occupancies such as [0.775, 0.225], and weak emission separation
  (`L1−R1` ≈ 3.7 vs 1.1) — an artifact, not a supported state structure.
- Single-state emission weights: `L1−R1`, `L1+L2−R1−R2`, `incoming_dir`, bias.

### 3. Environment sensitivity (`exp2_*.csv`)
- Within-subject slope of planning rate on block conflict rate: mean **+0.312,
  t = 2.18, p = 0.044, N = 18** (heterogeneous; one participant −0.79, p = 0.024).
- Sensitivity to expected planning advantage: mean **+0.144, t = 2.80, p = 0.012**.
- Between-cohort (balanced vs high-agreement): **0.282 vs 0.391, Welch t = −3.46,
  p = 0.004** — but confounded with the drift design (Simpson's paradox: the
  pooled vs within-participant signs differ).
- **Deprecated output:** `exp2_shift_blocks.csv` / `exp2_shift_summary.csv` were
  being written empty because the `exp2_shift` dataset (YFX `default_experiment`)
  is no longer registered. The writer is now guarded and the empty files were
  removed (`planning_experiments.py`, `planning_figures.py`); the figure
  `fig_exp2_shift.png` is skipped when the data is absent.

### 4. Continuous threat and per-participant slopes (`exp3_*.csv`)
- Planning near-death vs safe terciles: near 0.338 vs safe 0.363, paired
  t = −1.09, **p = 0.296, N = 13** (weak; the block-level drift contrast is the
  robust threat result).
- Per-participant logistic slope of planning on `z(ball_y)`: mean **+0.204,
  t = 2.54, p = 0.026, N = 13**, but signs are mixed (one participant +0.79,
  p = 0.038; several negative). The direction is opposite to the intuitive
  "plan more near death" — planning is *lower* when the ball is high/near death
  in the block manipulation.
- Mixture `w1` (threat gate): mean −0.049, t = −0.05, p = 0.958, and it hits the
  ±8 bound for several participants → **unstable, not interpretable**.

### 5. EMU threat is opposite/inconclusive (`xd_threat_behavior.csv`)
On all trials (all-drift), planning is **higher** in the near-death third than
the safe third in 4/4 runs (e.g. YFZ_1 0.618 vs 0.530); on conflict trials the
difference is small and mixed. This contradicts the cloud block-level effect and
is reported as inconclusive (no follow contrast in EMU).

### 6. Incoming direction / choice stickiness (`variations_of_adding_incoming_direction_to_custom_model.md`)
The custom mixture-of-experts model added a `bias_dir` term (variant C,
implemented): positive = repeat previous direction (perseveration), negative =
alternate. Variants A (incoming direction modulates the mixture weight, 6 params)
and B (per-strategy direction bias, 7 params) were specified but not adopted.
The doc's sign convention (`incoming_direction = +1` if the previous direction
was left) **matches** `build_sequence_table` (`incoming_direction =
−sign(prev_goal − entry)`: moving left gives prev_goal < entry, so `+1`), so
there is no inconsistency. `bias_dir` reaches the ±8 bound in many
`exp3_mixture` fits, so its sign is not reliable.

### 7. Model comparison and transfer scaling (`scaling_*.csv`, `model_comparison_*.png`)
- `model_comparison.py` compares 10 models (logistic, GRU `TinyDecisionRNN`,
  feedforward distance/raw-position NNs, 2-state GLM-HMM, CognitiveDeepONet,
  gated/multi-task/time-binned StrategyDeepONets, custom mixture).
- **Transfer scaling** (`scaling_analysis.py`, output CSVs in `analysis/`):
  frozen-basis held-out accuracy rises with pool size N, cognitive
  0.754 (N1) → 0.773 (N12) at full data; fewer-shot degrades less with larger N
  (N1 0.754→0.717 from fit_frac 1.0→0.1; N12 0.773→0.757) — transfer substitutes
  for within-participant data. Matched within-participant logistic ≈ **0.780**.
- **Strategy scaling is complete** (`scaling_strategy_summary.csv`), contrary to
  an earlier doc; strategy N12 ≈ 0.777 at full data.
- `analysis/scaling_figures.py` regenerates the two scaling figures from the
  CSVs (retraining needs `torch` + `ssm`, not co-installed here).
- **Multi-task DeepONet bug** (`7_24_updates.md`): raw RT scale (~10³) made the RT
  MSE ~10⁶× the choice BCE, collapsing choice accuracy to ~50%; fixed by
  z-scoring RT.

### 8. Mixture model instability (`exp1_mixture.csv`, `exp3_mixture.csv`, `xd_mixture.csv`)
`p_plan_base` saturates at 0.98, inverse temperatures hit the 20 bound, and
`bias_dir` hits ±8 for many participants in both datasets. Parameters are
reported but treated as unidentified; downstream uses model-free summaries.

---

## Neural

### 9. Temporal decoding (`temporal_decoding_*.csv`)
Sliding-window decoding is at chance for `side` (0.478 pre) in yfz_1; the only
notable time-resolved effect is `agree` pre-1000 ms at 0.543 (p = 0.017) in
yfz_1 alone (not replicated). `planning_vs_greedy` pre-1000 ms = 0.454 (p = 0.96).

### 10. Demixed PCA (`dpca_*.csv`)
Variance explained per demixed term is small; axis cosines near 0. The one above
-chance decoding is `side` via the stimulus (`s`) term (0.596) but it is flagged
`circular=True` (a within-condition alignment artifact), so it is not evidence of
a clean side code.

### 11. Per-unit selectivity (`selectivity_results.csv`, `left_right_selectivity_results.csv`)
Permutation + BH-FDR per unit. Planning-related significant units are **few**:
yfz_1 4 post / 1 pre `planning_vs_greedy`; yfz_2 2 post / 2 whole-trial; none in
YGA. By contrast **left/right (motor/direction) selectivity is far more
prevalent**: 54 (yfz_1), 58 (yfz_2), 4 (yga_1), 10 (yga_2) significant units.
YGA's significant units are dominated by `death_vs_normal` (10/15), i.e. the
exploratory death effect. This is a key control: whatever the population
decoding shows, single-unit choice/planning selectivity is weak relative to
direction selectivity.

### 12. Spatial tuning (`spatial_tuning_results.csv`)
Skaggs spatial information per unit (x = 12 `ball_x` bins, y = 8 `ball_y` bins),
circular time-shift null calibrated at ~5% false positives. Only **one** unit
(yga_2, unit 1) is significant (x = 0.698 bits, y = 0.893 bits, q = 0) — no
population-level place-field-like structure in this MTL sample.

### 13. RSA, value decoding, planning history
- `rsa_bally.csv`: neural RDM vs model RDMs (`greedy_gap`, `planning_gap`,
  `conflict_mag`) Spearman r ≈ 0.000–0.006, all n.s. — the population geometry
  does not align with the model variable RDM at this resolution (exploratory
  negative).
- `value_bally.csv`: ridge decoding of `greedy_gap`/`planning_gap` by ball-y bin
  gives negative R² (no reliable value code); `neural_choice_bally.csv` reports
  the OOF neural score × ball_y choice-logistic coefficients (neural × ball_y
  interaction n.s.).
- `planning_history.csv`: choice model with previous-trial planning
  (`prev_plan` ≈ +0.12) and `threat × prev_plan` n.s. — no history effect.

### 14. Unit-level threat × planning (`unit_threat_planning.csv`)
Per-unit near-death vs far planning modulation: **0 of 122 (yfz_1), 0 of 124
(yfz_2), 0 of 81 (yga_1), 1 of 77 (yga_2)** significant — no per-unit threat
modulation of the planning code.

### 15. Death (`death_decoding.csv`, `death_locked_rates.csv`)
A death-locked population signal is present in yga_2 (0.552, p = 0.007) but not
yga_1 (0.497); block-matching barely changes it (0.543). YFZ has ≤1 death.
Exploratory (1 of 2 sessions).

---

## Methods, QC, and data

### 16. Recording and alignment (`analysis/neural_outputs/DATA_STRUCTURE.md`)
Bilateral mesial-temporal depth electrodes, 64 channels = 8 leads × 8 contacts;
**yfz_1: 122 units after QC, 910 trials**, 25 ms bins, ±2 s. Photodiode↔JSON DTW
offset ≈ 126.9 s with median residual ≈ 4.5 ms. Entry-anchored dataset parity
gate passed. Median entry→choice RT: planning ≈ 950 ms, greedy ≈ 417 ms.

### 17. Death logging (`docs/logging_deaths_in_json_files`)
`E.block_deaths` is now written at the death site in `app/static/sketch.js`, so
every JSON snapshot carries per-block death counts (backward-compatible
heuristic fallback). yfz_1 contains **one** genuine death (~758.9 s, block 8);
the 10 "empty trials" are levels scrolled past during that fall, not 10 deaths.

### 18. Known discrepancies and deprecations
- `DATA_STRUCTURE.md` unit counts and QC floor have been reconciled to **122
  units** and **< 0.5 Hz** (verified against `unit_metadata.csv`,
  `segmented_spikes_binned.npz` shape, and `segment_trials.MIN_FIRING_RATE_HZ`).
  The two spike totals are both correct and now documented as such: 1,521,368
  raw (pre-window) vs 1,514,339 inside the ±2 s window.
- `scaling_analysis_explanations.md` was stale (fit_frac 0.5 row not in any CSV;
  claimed strategy scaling not run). Corrected.
- The per-run `planning_cross_bally.csv` is combined into
  `xd_neural_cross_bally.csv` (present); there is no
  `xd_neural_planning_cross_bally.csv`.
- `exp2_shift_*` deprecated (dataset removed); outputs removed, writers guarded.
- `analysis/figures/scaling_*` regenerated by `analysis/scaling_figures.py`.

## Reproduce

```
.venv-analysis\Scripts\python.exe analysis\threat_bins_analysis.py
.venv-analysis\Scripts\python.exe analysis\scaling_figures.py
.venv-analysis\Scripts\python.exe analysis\planning_experiments.py
.venv-analysis\Scripts\python.exe analysis\planning_figures.py
```
