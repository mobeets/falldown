# Neural LDA decoding: confound analysis and validation

Status: results of four follow-up analyses on the 122-unit mesial-temporal
population recorded during the YFZ EMU Falldown session (910 trials).

Companion scripts (all in `analysis/`, outputs in `analysis/neural_outputs/`):

| Script | Outputs |
|---|---|
| `neural_lda_decoding.py` | `neural_lda_decoding_results.csv`, `neural_lda_decoding_cross.csv`, `neural_lda_decoding_loadings.csv`, `neural_lda_decoding_similarity.csv` |
| `neural_decoding_confounds.py` | `decoding_nuisance_table.csv`, `decoding_label_association.csv`, `decoding_matched_redecode.csv` |
| `neural_temporal_decoding.py` | `temporal_decoding_sliding.csv`, `temporal_decoding_summary.csv`, `temporal_decoding_generalization.csv` |
| `neural_dpca.py` | `dpca_variance_decomposition.csv`, `dpca_decoding.csv`, `dpca_axis_similarity.csv`, `dpca_components.csv` |
| `neural_continuous_decoding.py` | `continuous_decoding_results.csv` |

---

## 1. Motivation

`neural_lda_decoding.py` asks which trial-level properties the mesial-temporal
population can linearly discriminate with an LDA decoder. The headline results
(post-choice window, rate rep):

| hypothesis | acc_mean | chance | perm_p |
|---|---|---|---|
| side | 0.509 | 0.50 | 0.330 |
| agree | 0.511 | 0.50 | 0.284 |
| planning_optimal | 0.575 | 0.50 | 0.000 |
| move_dir | 0.507 | 0.50 | 0.378 |
| move_dir_vx | 0.516 | 0.50 | 0.206 |
| condition | 0.339 | 0.25 | 0.000 |
| planning_vs_greedy | 0.697 | 0.50 | 0.000 |
| agree_optimal_vs_lapse | 0.528 | 0.50 | 0.078 |

With the time-resolved (PCA) representation, `planning_vs_greedy` reaches
0.964 and `condition` 0.475 in the post window.

The central concern with any decoding study is **confounding**: the trial
labels are defined from the hole geometry and the participant's choice
(`classify_trials.py`), so they are mechanically entangled with the
consequences of that choice — the ball trajectory, timing, and outcome in the
post-choice window. A decoder that separates "planning vs greedy" trials could
be reading the *aftermath* of different choices (kinematics, timing) rather
than a planning computation. These four analyses make that entanglement
explicit and test how much of the decoding survives control.

---

## 2. Common methodology

All decoding uses the same conventions as `neural_lda_decoding.py`:

- **Features** (`rate` / `pca`): per-unit mean firing rate (Hz,
  sqrt-transformed) in a window, or time-resolved 25 ms bin counts
  PCA-reduced to 30 components. The PCA is fit once on all trials
  (unsupervised); only the CV step re-standardizes per fold.
- **Windows**: `pre` `[-1000, 0]`, `post` `[0, +1000]` ms relative to the
  choice (t = 0), and `whole` trial.
- **Classifier**: LDA (Ledoit–Wolf shrunk within-class covariance,
  `solver='eigen'`), repeated stratified k-fold CV, balanced accuracy
  (imbalance-proof), `chance = 1/n_classes`, and an empirical permutation
  p (label shuffle through the same CV). p is two-sided unless noted.

---

## 3. Analysis 1 — confound diagnostics and matched re-decoding

### Methodology

**Part 1 (nuisance table + associations).** For every trial we build nuisance
covariates: `rt_ms` (choice − entry), `ball_time_ms` (exit − choice), trial
duration, block index, position in block, the signed left/right greedy and
planning cost gaps, and categorical `side`, `move_dir`, and previous-trial
condition. Each hypothesis × nuisance association is tested against a
label-shuffle null (300 shuffles): point-biserial r (binary labels) or η²
(4-class) for continuous nuisances, Cramér's V for categorical ones. This is
the "confound table".

**Part 2 (matched re-decoding).** For the key binary hypotheses we re-run the
post-window decoder on subsets where the two classes are matched on nuisance
variables — keeping equal per-stratum counts within strata defined by
(a) `side`, (b) `side` × RT-quartile × block-half, (c) adding ball-time
quartiles. If accuracy collapses under matching, the original number rode on
the confound. Subsets smaller than 40 trials are not decoded (marked
"insufficient overlap").

### Results

**Nuisance–label associations (selected):**

| hypothesis | nuisance | stat | p |
|---|---|---|---|
| planning_vs_greedy | ball_time_ms | r = 0.872 | <0.001 |
| planning_vs_greedy | trial_duration_ms | r = 0.344 | <0.001 |
| planning_vs_greedy | block_index | r = 0.075 | 0.033 |
| condition | ball_time_ms | η² = 0.479 | <0.001 |
| condition | rt_ms | η² = 0.401 | <0.001 |
| agree_optimal_vs_lapse | rt_ms | r = 0.571 | <0.001 |
| side | move_dir | Cramér's V = 1.000 | <0.001 |

The association between `planning_vs_greedy` and ball time is extreme: mean
post-choice traversal time is **438 ms for planning trials vs 1455 ms for
greedy trials** — nearly disjoint distributions. Any post-window decoder has
ample kinematic structure to exploit.

**Matched re-decoding (post window):**

| hypothesis | rep | unmatched | matched_side | matched_side_rt_block | matched_side_rt_ball |
|---|---|---|---|---|---|
| planning_vs_greedy | rate | 0.695 | 0.721 | 0.679 | n=16 (no overlap) |
| planning_vs_greedy | pca | 0.964 | 0.968 | 0.947 | n=16 (no overlap) |
| planning_optimal | rate | 0.569 | 0.560 | 0.604 | 0.521 (p=0.27) |
| planning_optimal | pca | 0.699 | 0.734 | 0.798 | **0.508 (p=0.44)** |
| agree | pca | 0.605 | 0.608 | 0.620 | **0.500 (p=0.52)** |
| agree_optimal_vs_lapse | rate | 0.506 | 0.566 | 0.559 | 0.473 (p=0.61) |

### Interpretation

- Matching on `side` or `side`×RT×block does **not** remove the
  `planning_vs_greedy` signal — it survives at 0.68–0.72 (rate) and
  >0.95 (PCA). It is not a trivial left/right or reaction-time artifact.
- Matching on `ball_time_ms` is *impossible* for `planning_vs_greedy` (only
  16/431 trials have both classes in any shared stratum) — the two classes
  differ so strongly in traversal time that the confound cannot be
  matched away. That infeasibility is itself the strongest possible
  statement of the kinematic confound.
- For `planning_optimal` and `agree`, matching on ball time **collapses the
  decoding to chance** (PCA 0.699→0.51, 0.605→0.50). For these two
  geometry labels the post-window signal was carried by the traversal-timing
  difference, not by a task-value representation.

---

## 4. Analysis 2 — temporal generalization and sliding-window decoding

### Methodology

The fixed pre/post/whole windows hide *when* the signal exists. We decode the
`rate` representation on a sliding 150 ms window stepping 50 ms across
`[-2000, +2000]` ms (repeated 5-fold CV), and run full permutation tests
(300 shuffles) at four a-priori windows (pre `[-1000,0]`, pre `[-500,0]`,
post `[0,+500]`, post `[0,+1000]`). A King–Dehaene-style train×test
generalization matrix over a 6-window grid (rate rep) asks whether the
representation is stable across time or a post-decision transient.

### Results (a-priori windows, balanced accuracy / p)

| hypothesis | pre [-1000,0] | pre [-500,0] | post [0,+500] | post [0,+1000] |
|---|---|---|---|---|
| planning_vs_greedy | 0.454 / p=0.96 | 0.499 / p=0.50 | 0.507 / p=0.36 | **0.699 / p<0.001** |
| planning_optimal | 0.489 / p=0.73 | 0.494 / p=0.56 | 0.497 / p=0.57 | **0.561 / p<0.001** |
| condition | 0.235 / p=0.86 | 0.268 / p=0.10 | 0.251 / p=0.46 | **0.333 / p<0.001** |
| agree | 0.543 / p=0.017 | 0.525 / p=0.08 | 0.506 / p=0.38 | 0.503 / p=0.44 |

The sliding trace shows the `planning_vs_greedy` signal rising sharply only
after choice (~0.96 at +1000 ms, chance before).

### Interpretation

- `planning_vs_greedy`, `planning_optimal`, and `condition` are **at chance
  before and during the choice** and become decodable **only in the late
  post-choice window** (`[0, +1000]`, not even `[0, +500]`).
- If the population were *computing* planning versus greedy, the signal
  should exist before/during the decision. Its confinement to the late
  post-choice period is the signature of decoding the *consequences* of the
  choice (ball traversal into the hole, outcome), not the decision process.
- The one weak pre-choice effect is `agree` (`[-1000,0]`: 0.543, p=0.017) —
  an a-priori window, so interpret with caution given multiple tests.

---

## 5. Analysis 3 — demixed PCA (dPCA)

### Methodology

LDA decoding says *that* a label is decodable, not *which population axis*
carries it. dPCA (Kobak et al. 2016, `machenslab/dPCA`) demixes the
population PSTHs (8 cells = condition × side, time-resolved, coverage-trimmed,
imputed) into marginalizations over `t` (time, condition-independent),
`c` (condition), `s` (side), and the interactions `cs`, `ct`, `st`, `cst`
(regularizer off; 20/6/4/3/6/4/4 components per term). We then (a) report the
explained-variance decomposition, and (b) project single trials onto each
term's decoder axes and run the usual LDA balanced-accuracy decoding. Rows
whose term contains the hypothesis's defining parameter (`c` for
condition-based labels, `s` for side) are flagged `circular=True` — dPCA was
given those labels — so the meaningful confound controls are the
non-circular decodes (the `t`, `s`, `st` axes for condition-based labels).

### Results

**Variance decomposition (post window, fraction of total):**

| term | ct | cst | t | st | c | cs | s |
|---|---|---|---|---|---|---|---|
| explained variance | 0.123 | 0.084 | 0.053 | 0.023 | 0.017 | 0.014 | 0.002 |

Condition×time structure (ct + cst ≈ 21%) outweighs condition-independent
time (`t` ≈ 5%), opposite to the usual dPCA finding in PFC-like areas where
the time term dominates.

**Demixed-axis decoding (post window, balanced accuracy / p):**

| hypothesis | t (time) | c (condition) | s (side) | ct | st |
|---|---|---|---|---|---|
| planning_vs_greedy | 0.500 / 0.89 | **0.649 / <0.001** (circ) | 0.503 / 0.06 | 0.505 / 0.01 (circ) | 0.500 / 0.81 |
| condition | 0.247 / 0.85 | **0.337 / <0.001** (circ) | 0.249 / 0.61 | 0.248 / 0.75 (circ) | 0.249 / 0.68 |
| planning_optimal | 0.499 / 0.73 | 0.534 / <0.001 (circ) | 0.498 / 0.78 | 0.497 / 0.85 (circ) | 0.498 / 0.81 |
| agree | 0.515 / 0.15 | 0.587 / <0.001 (circ) | 0.496 / 0.57 | 0.502 / 0.37 (circ) | 0.486 / 0.89 |
| side | 0.488 / 0.76 | 0.492 / 0.64 | **0.605 / <0.001** | 0.509 / 0.24 | 0.479 / 0.85 |

### Interpretation

- `planning_vs_greedy`, `condition`, and `agree` decode **only from the
  condition-demixed `c` axis** and are at chance from every non-circular axis
  (`t`, `s`, `st`). The signal is **not** carried by the condition-independent
  temporal axis (which would be a pure timing/kinematics artifact) and **not**
  by the side axis.
- `side` decodes only from its own `s` axis.
- So the post-choice activity genuinely separates the conditions on their own
  population axis — it is condition-specific, not generic trial-locked
  dynamics. Combined with Analyses 1–2, the honest statement is: the
  population distinguishes the *different trajectories/outcomes the two
  policies produce*, on a dedicated axis, and only after the choice.

---

## 6. Analysis 4 — model-based continuous decoding

### Methodology

Discrete labels waste information and confound trivially. We instead regress
continuous task-value quantities from `trial_labels.csv` onto neural features
with ridge regression (10-fold CV, repeated 3×, correlation + R², 150-shuffle
permutation p):

- `greedy_gap` / `planning_gap` — signed 1-step / 2-step value difference
  between the two choice holes;
- `model_conflict` — `planning_gap − greedy_gap` (signed disagreement);
- `conflict_mag` — its absolute value (disagreement magnitude);
- `chosen_greedy_cost` / `chosen_planning_cost` — the model cost of the hole
  actually chosen (decision quality).

Three controls: `none` (raw), `target` (regressor residualized on RT, ball
time, duration, block, side, move_dir), and `both` (features additionally
residualized per unit). Collapsing under residualization means the signal was
the confound.

### Results (CV correlation, rate / pca)

| regressor | window | none | target | both |
|---|---|---|---|---|
| chosen_planning_cost | post/rate | **0.165 / <0.001** | −0.053 / 0.84 | −0.052 / 0.85 |
| chosen_planning_cost | post/pca | **0.501 / <0.001** | 0.105 / 0.007 | 0.129 / <0.001 |
| chosen_greedy_cost | post/pca | **0.339 / <0.001** | 0.123 / <0.001 | 0.128 / <0.001 |
| conflict_mag | post/pca | **0.269 / <0.001** | 0.241 / <0.001 | **0.301 / <0.001** |
| conflict_mag | whole/pca | **0.431 / <0.001** | 0.322 / <0.001 | 0.390 / <0.001 |
| greedy_gap / planning_gap | any | ≈0 | ≈0 | ≈0 |

### Interpretation

- The raw value gaps (`greedy_gap`, `planning_gap`) barely decode anywhere:
  the population is **not continuously tracking the cost difference between
  the two holes**.
- `chosen_planning_cost` is strongly decodable post-choice (r ≈ 0.5 in PCA)
  but ~80% of it is confound: it collapses toward 0 after nuisance
  residualization, though a small significant residue (r ≈ 0.11–0.13)
  survives in PCA.
- **`conflict_mag` — the magnitude of disagreement between the greedy and
  2-step models — survives all three controls** (post/pca ≈ 0.24–0.30,
  whole/pca ≈ 0.32–0.43, p<0.001 in every variant). This is the most
  confound-resistant finding: a planning-relevant quantity (how much the
  models disagree) is genuinely represented even after removing everything
  correlated with RT, kinematics, position, and block drift.

---

## 7. Synthesis

The four analyses triangulate on a coherent picture:

1. **The post-choice decoding is largely a consequence, not a computation.**
   The planning-vs-greedy signal appears only in the late post-choice window
   (`[0,+1000]` ms), tracks ball-traversal time (planning 438 ms vs greedy
   1455 ms; r = 0.87), and for the geometry labels `planning_optimal` and
   `agree` it is completely removed by ball-time matching. There is no
   evidence of a pre-decision planning readout.

2. **It is still condition-specific.** The signal lives on the
   condition-demixed dPCA axis, not the condition-independent time axis or
   the side axis, and it survives matching on side/RT/block. It is a genuine
   population distinction between what the two policies do — but it encodes
   the differing post-choice trajectories/outcomes on a dedicated axis, not a
   generic timing artifact.

3. **One robust planning-relevant quantity survives.** `conflict_mag` — the
   degree to which the greedy and 2-step models disagree — is decodable in
   the post and whole windows and survives full nuisance residualization.
   This is the strongest candidate for a real planning-related signal in this
   population.

Caveats: single session (n = 1 participant, 910 trials); the small `lapse`
(n=67) and `planning` (n=125) classes limit power; the a-priori window tests
involve multiple comparisons; dPCA same-parameter decodes are circular by
construction (flagged); and residualization removes confound-aligned variance
from the target, which can also attenuate genuine signal that happens to
correlate with kinematics.

---

## 8. Entry-anchored follow-up: is the negative pre-choice result an anchoring artifact?

### Motivation

The whole synthesis above leans on one claim: `planning_vs_greedy`,
`planning_optimal`, and `condition` are **at/below chance before and during the
choice** and only decode in the late post-choice window `[0, +1000]`. That
claim rests on choice-anchored windows, and the choice-anchored
`pre [-1000, 0]` window is *heterogeneous*: for slow-approach trials
(planning: entry→choice rt ≈ 950 ms median) it covers the whole approach
interval, but for fast trials (agree_optimal: rt ≈ 267 ms) it extends
**before the entry pass** — into a period where the decision hasn't started.
If the planning computation is carried in the early approach (right after the
participant passes the entry hole and commits to a direction), a
choice-anchored pre window could smear or miss it.

**Fix tested here:** re-anchor t = 0 to the **entry pass** (the 1-hole level
that opens each 1-2-1 sequence) and measure the actual decision/approach
interval `[entry, choice)` in windows that never run past the choice pass.
Dataset: `segmented_spikes_entrylocked_binned.npz`
(`analysis/neural_entry_locked.py`, parity-checked against the choice-anchored
npz — see `analysis/neural_outputs/DATA_STRUCTURE.md`).

### LDA results (rate rep, entry-anchored; `neural_entry_decoding.py`)

| window | hypothesis | acc | chance | p | n |
|---|---|---|---|---|---|
| entry+250 | planning_vs_greedy | 0.503 | 0.50 | 0.43 | 431 |
| entry+250 | planning_optimal | 0.505 | 0.50 | 0.43 | 707 |
| entry+250 | condition | 0.257 | 0.25 | 0.29 | 707 |
| entry+250 | agree | 0.497 | 0.50 | 0.50 | 707 |
| entry+500 | planning_vs_greedy | 0.505 | 0.50 | 0.50 | 237 |
| entry+500 | planning_optimal | 0.518 | 0.50 | 0.29 | 390 |
| entry+500 | agree | 0.530 | 0.50 | 0.18 | 390 |
| entry+750 | planning_vs_greedy | 0.431 | 0.50 | 0.89 | 133 |
| entry+750 | planning_optimal | 0.491 | 0.50 | 0.58 | 210 |
| entry+750 | agree | 0.567 | 0.50 | 0.077 | 210 |
| approach [entry, choice) | planning_vs_greedy | 0.645 | 0.50 | <0.001 | 431 |
| approach | planning_optimal | 0.536 | 0.50 | 0.010 | 910 |
| approach | condition | 0.333 | 0.25 | <0.001 | 910 |
| approach | agree | 0.598 | 0.50 | <0.001 | 910 |
| pre-entry [-500,0] | (all) | ≈chance | — | n.s. | — |

**Fixed windows that contain the same absolute post-entry milliseconds for
both classes (`[0, 250]`, `[0, 500]`, `[0, 750]`) show NO planning decoding —
all at chance.** The only windows that decode are the **whole-approach**
interval `[entry, choice)` (variable length = rt per trial) and, marginally,
`agree` at entry+750 (0.567, p=0.077, one of many tests). `move_dir` and
`move_dir_vx` (steering direction) are also at chance in every entry window,
so the result is not a direction/steering confound.

**Why the approach window decodes but fixed windows don't:** the approach
window's length IS rt, and rt differs sharply by condition (planning ≈ 950 ms
vs greedy ≈ 417 ms). Matched re-decoding of the approach window
(`entry_locked_matched_redecode.csv`) collapses the effect:

| hypothesis | unmatched | +side | +side·rt·block | +side·rt·ball(+block) |
|---|---|---|---|---|
| planning_vs_greedy | 0.645 | 0.773 | 0.576 (p=0.053) | n/a (no overlap) |
| planning_optimal | 0.536 | 0.552 | 0.550 | 0.511 (n.s.) |
| agree | 0.598 | 0.590 | 0.568 | 0.515 (n.s.) |

The approach-window "signal" is largely a **duration/timing structure**: the
two classes simply occupy the pre-choice interval for very different lengths,
and equalizing rt removes most of the decode (the remainder, e.g.
planning_vs_greedy 0.576 p=0.053 at n=190, is not significant).

### Continuous (ridge) results (rate rep; `neural_entry_continuous.py`)

`conflict_mag` — the headline confound-resistant quantity from the post-choice
analysis — is **not** decodable in any fixed pre-decision window
(entry+250: r≈0.05 p>0.1; entry+500: r≈0 n.s.). In the whole-approach window
it decodes raw (r = 0.158, p<0.001) but **collapses to r≈0.02–0.05 (n.s.)
under residualization on rt/block/side/move_dir alone** (no post-choice
nuisances needed) — i.e. the raw approach-window effect is downstream of the
slow-approach structure, not a planning representation. The value gaps
(`greedy_gap`, `planning_gap`) are ~0 everywhere, as in the choice-anchored
analysis. `chosen_planning_cost` shows a small residual (r≈0.11–0.12, p<0.001)
late in the approach under the no-post-choice nuisance set — comparable in
size to the post-choice residue — but it is sensitive to nuisance choice
(collapses to n.s. once ball_time is added as a nuisance).

### Interpretation

Anchoring at the entry hole and measuring the decision interval in fixed
windows that never cross the choice pass **does not reveal a pre-decision
planning signal**. The choice-anchored "at chance before/during choice"
conclusion is not an anchoring artifact. The one pre-choice decode that does
appear — from the whole variable-length approach interval — is explained by
the RT/duration difference between the classes and collapses under rt-matched
control, consistent with (not contradicting) the post-choice-consequence
story: the population distinguishes the *trajectories/outcomes* the two
policies produce, on a dedicated axis, and only after the choice.

Caveats: rate rep only (no PCA on the variable-length approach window);
fixed-window eligibility is condition-dependent so n drops across windows
(e.g. planning_vs_greedy at entry+750: n=133); single session; and the
variable-length approach window necessarily confounds interval length with
condition, which is precisely why the fixed windows are the cleaner test.

---

## 9. Reproducibility

All scripts run with
`C:\Users\manik\AppData\Local\Programs\Python\Python311\python.exe` from the
repo root, e.g.:

```
python analysis\neural_decoding_confounds.py
python analysis\neural_temporal_decoding.py
python analysis\neural_dpca.py
python analysis\neural_continuous_decoding.py
```

Entry-anchored follow-up (section 8):

```
python analysis\neural_entry_locked.py           # builds entry-anchored npz
python analysis\neural_entry_locked.py --check-parity   # parity gate
python analysis\neural_entry_decoding.py
python analysis\neural_entry_continuous.py
```

`neural_dpca.py` requires `pip install dPCA numexpr numba`. Each script
writes CSVs to `analysis/neural_outputs/` and ships `plot_*` functions
(`plot_sliding_trace`, `plot_tg_matrix`, `plot_demixed_axis_components`,
`plot_continuous_bars`, `plot_matched_redecode`, `plot_association_heatmap`)
for use in Jupyter.
