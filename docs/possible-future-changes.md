# Possible Future Changes

A roadmap of analysis expansions discussed but not yet implemented.
Each entry links to the motivating paper, sketches the approach, and
estimates effort. Priorities are ranked by insight-to-effort ratio.

Items that have been implemented are moved to the **Completed** section at
the bottom. The newest direction is **Neural Data Analysis** — the EMU patient
has intracranial single-unit recordings synchronized to the task, so the
roadmap now spans behavior → model → neural. All spike-sorted units come from
bilateral **mesial temporal lobe depth electrodes** (hippocampus: CA fields,
anterior hippocampus, hippocampal body; plus amygdala) — see the electrode
labels in `times_*.mat` — so the neural hypotheses below should be read
through that hippocampal/amygdalar lens.

---

## Neural Data Analysis (current focus)

The spike–behavior pipeline is done: `spike_data_alignment.py` (photodiode DTW
offset), `spike_unit_conversion.py` (unit-level spike times), and
`segment_trials.py` (per-trial windows aligned to choice time, ±2 s, 137 units
after QC, 910 trials). See `analysis/spike_data_alignment_output/DATA_STRUCTURE.md`.
These are concrete analyses to run on that segmented data; several map onto the
[`consensus-ai-prompts.md`](consensus-ai-prompts.md) literature questions.

### Choice-Selectivity PSTHs

**What:** For each unit, compare firing around the choice moment (t=0) across
trials split by which hole was chosen (left vs right, or greedy-preferred vs
planning-preferred hole). Compute a selectivity index per unit and a
population histogram of significant units.

**Approach:** Reuse `segment_trials.py` output; bin/smooth (σ ≈ 25 ms),
z-score against a pre-choice baseline, then per-unit signed difference between
conditions. Permutation test across trials for significance.

**Impact:** Establishes whether any hippocampal/amygdalar units encode the
choice itself (first question for a new dataset).

**Effort:** ~150 lines in `analysis/neural_choice_selectivity.py`.

### Conflict vs Agreement Neural Correlates

**Paper:** the task's `greedy_planning_agreement` per-trial label (see
`level_generation`); Ashwood et al 2022.

**What:** Split trials by whether greedy and planning strategies point to the
same hole (agree) or different holes (conflict). Compare choice-locked PSTHs
and RT-locked activity. Tests whether conflict (anterior-cingulate-style
signature) is visible in this task, or whether hippocampal/amygdalar units
show their own conflict/agreement modulation.

**Impact:** Directly tests the cognitive-map/planning hypotheses — especially
hippocampal successor-representation and memory-based accounts — against the
only neural data we have.

**Effort:** ~100 lines (extends the selectivity script).

### Strategy-State Decoding from Population

**Paper:** Ji-An et al 2025; Ashwood et al 2022.

**What:** Fit the GLM-HMM to behavior (already done), get per-trial state
posteriors, then ask whether a linear decoder trained on the 137-unit
population (trial-mean firing, choice window) can recover the strategy state
on held-out trials, and whether it precedes the behavioral switch.

**Effort:** ~150 lines in `analysis/neural_strategy_decode.py`.

### Single-Trial Choice Decoding

**What:** Decode left/right choice from population spike counts (binned 25 ms)
using logistic regression / LDA with proper temporal cross-validation; report
accuracy vs chance per time bin around choice to show when choice information
appears.

**Effort:** ~120 lines in `analysis/neural_choice_decode.py`.

### Reaction-Time Modulation

**What:** Regress trial RT on per-trial mean firing (or time-to-peak PSTH)
across units; identify units whose activity tracks decision speed, and whether
any correlate with the planning-depth signature.

**Effort:** ~80 lines.

---

---

## Instrumentation (participant-facing changes)

### Confidence Ratings

**Paper:** Resulaj et al 2009 (change-of-mind bounded diffusion); Kirsch 2019
(unified model with confidence dimension).

**What:** Add a 1–7 slider after each trial asking "How confident were you that
you chose the best hole?" Logged alongside trial events.

**Impact:** Enables fitting bounded diffusion models with confidence boundaries,
detecting changes of mind (low confidence trials → trajectory corrections),
and measuring metacognitive sensitivity.

**Effort:** ~20 lines in `sketch.js` (UI + event logging) + new `analysis/confidence_diffusion.py`.

---

### Post-Experiment Trait Questionnaires

**Paper:** Anxiety-Depression-and-Decision-Making (computational psychiatry);
Jach et al 2024 (curiosity, need for cognition, BIS/BAS).

**What:** Brief validated scales administered after the task:
- BIS/BAS (Carver & White, 1994) — behavioral inhibition/activation
- Need for Cognition (Cacioppo & Petty, 1982)
- State-Trait Anxiety Inventory (short form)
- Curiosity and Exploration Inventory (Kashdan et al, 2004)

**Impact:** Enables computational psychiatry analysis: do trait anxiety or BIS
scores predict strategy type, planning depth, or RT variability?

**Effort:** ~100 lines of HTML/JS for the questionnaire UI + CSV export.

---

### Information Demand Probes

**Paper:** Jach et al 2024 (individual differences in information demand).

**What:** Intersperse "preview" trials where participants choose between seeing
the next 2 levels vs. the next 5 levels (Horizon Task adaptation). Measure
how often they pay a cost (time or points) for more information.

**Impact:** Directly measures the latent dimension Jach's paper claims organizes
individual differences. Correlate with strategy clusters found by the DeepONet.

**Effort:** ~80 lines in `sketch.js` + `experiment.js` + new config.

---

### Decision-Locked Epoch Markers

**Paper:** Resulaj et al 2009; Keung et al 2020 (evidence accumulation timing).

**What:** Log the exact timestamp when (a) the first level of a 3-level sequence
becomes visible, (b) the participant commits to a lateral position (detected
by sustained directional input), and (c) the ball passes through the hole.
This decomposes RT into encoding + deliberation + execution phases.

**Impact:** Enables drift-diffusion model fitting with separate non-decision time
parameters per phase. Tests Keung's divisive model prediction that evidence is
weighted unevenly over time.

**Effort:** ~40 lines in `sketch.js` (timestamp logging) + new analysis script.

---

### Gaze Proxy via Mouse Tracking

**Paper:** Peer et al 2021 (cognitive maps, spatial attention); Kirsch 2019
(adaptive toolbox — attention is the strategy selector).

**What:** Log mouse position during inter-trial intervals (participants often
move the mouse to where they're looking). Coarse proxy for gaze allocation
across holes without eye-tracking hardware.

**Impact:** Reveals whether participants attend to the immediate hole, the
next hole, or the full trajectory. Tests Peer's claim that spatial attention
patterns reflect cognitive map structure.

**Effort:** ~15 lines in `sketch.js` (additional `log_states` field) + new
visualization in `analysis/gaze_proxy.py`.

---

## Analysis & Modeling (code-only changes)

### Bounded Diffusion / Drift-Diffusion Model Fitting

**Paper:** Resulaj et al 2009; Keung et al 2020.

**What:** Fit a drift-diffusion model (DDM) to the continuous ball-trajectory
data already logged by `log_states()`. The ball's x-position over time is
treated as an evidence accumulation process; the "decision" is the final
lateral position when crossing the hole threshold.

**Approach:**
- Use `pyddm` or `HDDM` package
- Evidence = ball x-coordinate relative to hole center
- Drift rate = participant's steering input minus noise
- Decision boundary = level width / number of segments
- Fit per-participant and per-condition drift rates

**Impact:** Maps the continuous control task onto the standard evidence
accumulation framework. Enables testing whether drift rate varies with
planning horizon (Kirsch), whether boundary separation changes with
difficulty (Resulaj), and whether evidence is weighted unevenly (Keung).

**Effort:** ~150 lines in `analysis/diffusion_model.py`.

---

### Planning Depth Estimation (partial — see Completed)

**Paper:** Mattar et al 2025 (few rollouts); Keramati et al 2016 (depth-limited
planning along habit–goal spectrum).

**Status:** Core cost functions live in `level_generation/agentic_decision_making.py`
(`calculate_greedy_cost`, `calculate_planning_cost`, `calculate_agreement`).
Not yet wired into a per-participant "which depth explains their choice" score
or a behavioral-agreement figure of merit.

**What remains:** Compare participant choices against optimal play computed by
tree search at different depths (1-step greedy, 2-step, 3-step), find the
minimum planning depth that reproduces each choice, and summarize per
participant/strategy.

**Effort:** ~100 lines reusing the existing cost functions.

---

### Cognitive Map Analysis

**Paper:** Peer et al 2021 (cognitive maps vs cognitive graphs; grid cells;
hippocampal place fields).

**What:** Model the level sequence as a graph where nodes are hole positions
and edges are transitions. Apply successor representation (SR) / graph
analysis to test whether participants learn the transition structure and
whether their choices reflect SR-based planning.

**Impact:** Bridge between the neuro-cognitive map literature and the
behavioral planning literature already in the corpus.

**Effort:** ~200 lines in `analysis/cognitive_map.py`. Requires `numpy` only.

**Priority:** Medium. Conceptually rich but needs clearer mapping to the
Falldown task structure.

---

## Cross-Cutting

### Basis Interpretability Labels (partial — see Completed)

**Status:** Post-hoc basis sweeps with named feature labels exist
(`cognitivedeepOnet.py` `plot_3d_basis_sweeps`). The proposed supervised
auxiliary loss that maps each basis to a named cognitive construct is not
implemented.

**What remains:** Add a supervised auxiliary loss to the DeepONet that maps each
basis function to a named cognitive construct (e.g., "immediate optimizer",
"two-step planner", "ball-y follower", "direction bias"). Could use weak
supervision: label a small set of trials with ground-truth strategy labels
and propagate through the basis network.

**Effort:** ~80 lines in `strategy_deeponet.py` (modification). Requires manual
labeling of ~100 example trials.

---

## Summary by Priority

| Priority | Change | Paper | Lines | Category |
|---|---|---|---|---|
| 1 | Choice-Selectivity PSTHs | — | ~150 | Neural |
| 2 | Conflict vs Agreement Neural Correlates | Ashwood 2022 | ~100 | Neural |
| 3 | Strategy-State Decoding from Population | Ji-An 2025 / Ashwood 2022 | ~150 | Neural |
| 4 | Confidence Ratings | Resulaj 2009 | ~20 + ~150 | Instrumentation + Analysis |
| 5 | Trait Questionnaires | Anxiety-Depression paper | ~100 HTML/JS | Instrumentation |
| 6 | Bounded Diffusion Fitting | Resulaj 2009 / Keung 2020 | ~150 | Analysis |
| 7 | Single-Trial Choice Decoding | — | ~120 | Neural |
| 8 | Decision-Locked Epochs | Resulaj 2009 | ~40 | Instrumentation |
| 9 | Information Demand Probes | Jach 2024 | ~80 | Instrumentation |
| 10 | Reaction-Time Neural Modulation | — | ~80 | Neural |
| 11 | Gaze Proxy | Peer 2021 | ~15 + ~50 | Instrumentation + Analysis |
| 12 | Cognitive Map Analysis | Peer 2021 | ~200 | Analysis |
| 13 | Planning Depth (finish) | Mattar 2025 / Keramati 2016 | ~100 | Analysis |
| 14 | Basis Interpretability (finish) | — | ~80 | Model Modification |

---

## Completed

Implemented items from earlier roadmap iterations. These are kept at the back
for provenance; the main sections above are the current backlog.

### GLM-HMM Strategy Switching Model

**Paper:** Ashwood et al 2022.

**What was done:** Per-participant K-state GLM-HMM fit on trial features with
EM, posterior state inference, transition matrices, and held-out
log-likelihood, integrated into model comparison and scaling analyses.

**Where:** `analysis/model_comparison.py` (`run_glmhmm_for_participant`,
`print_glmhmm_summary`), `analysis/scaling_analysis.py`
(`fit_heldout_glmhmm`), `analysis/behavior_analysis.py`,
`analysis/synthetic_data_experiments.py`.

---

### Tiny RNN Strategy Discovery (Ji-An-style)

**Paper:** Ji-An et al 2025.

**What was done:** `TinyDecisionRNN` (GRU) and `FeedforwardDecisionNN` variants
with per-participant training, RT/accuracy evaluation, and L1 regularization on
the recurrent weights.

**Where:** `analysis/RNN.py`.

---

### Unified Model Evaluation Framework

**What was done:** `model_comparison.py` runs every model (CognitiveDeepONet,
StrategyDeepONet family, GLM-HMM, RNN, feedforward NN, logistic baselines) on
every participant and produces comparison DataFrames, log-likelihood box plots,
and participant×model heatmaps.

**Where:** `analysis/model_comparison.py` (`compare_all_models`, `_plot_comparison*`,
`_plot_participant_heatmap`).

---

### Planning Depth Estimation (partial)

**Paper:** Mattar et al 2025; Keramati et al 2016.

**What was done:** Greedy (1-step) and planning (2-step) cost functions plus
strategy-agreement computation for generated level configurations.

**Where:** `level_generation/agentic_decision_making.py`
(`calculate_greedy_cost`, `calculate_planning_cost`, `get_trial_features`,
`calculate_agreement`). The per-participant planning-depth score is still open
(see main section).

---

### Basis Interpretability Labels (partial)

**What was done:** Post-hoc basis sweeps plotted with named feature labels to
interpret what each DeepONet basis encodes.

**Where:** `analysis/cognitivedeepOnet.py` (`plot_3d_basis_sweeps`). The
supervised auxiliary-loss labeling is still open (see main section).

---

### Spike–Behavior Alignment & Per-Trial Segmentation

**What was done:** Full EMU spike-alignment pipeline: photodiode flash ↔ JSON
event DTW offset model (`spike_data_alignment.py`), unit-level spike time
conversion (`spike_unit_conversion.py`), and per-trial choice-locked
segmentation with unit QC (`segment_trials.py`). Data structure documented in
`analysis/spike_data_alignment_output/DATA_STRUCTURE.md`.

**Where:** `analysis/spike_data_alignment.py`, `analysis/spike_unit_conversion.py`,
`analysis/segment_trials.py`.
