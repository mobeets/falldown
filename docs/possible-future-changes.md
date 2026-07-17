# Possible Future Changes

A roadmap of analysis expansions discussed but not yet implemented.
Each entry links to the motivating paper, sketches the approach, and
estimates effort. Priorities are ranked by insight-to-effort ratio.

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

### GLM-HMM Strategy Switching Model

**Paper:** Ashwood et al 2022 (GLM-HMM for discrete strategy detection).

**What:** Fit a Generalized Linear Model — Hidden Markov Model to participant
choice data. Each HMM state is a logistic regression on trial features; the
HMM transition matrix captures strategy switching rates. Use Expectation
Maximization (as Ashwood does) or stochastic variational inference.

**Approach:**
```python
# State k: logit = β_k0 + β_k1 × (L1−R1) + β_k2 × (Total_L−Total_R) + ...
# Transition matrix: T[i,j] = P(state j at t+1 | state i at t)
# EM: estimate β_k and T from choices
```

**Impact:** Validates and provides a frequentist counterpart to the DeepONet's
learned gate weights. If the HMM finds 3 states and the DeepONet finds the same
3 strategies, confidence in the result is high. If they disagree, the DeepONet
is capturing something the GLM-HMM misses (or vice versa).

**Effort:** ~300 lines in `analysis/glm_hmm.py`. Depends on `ssm` or `hmmlearn`
package.

**Priority:** High. This is the canonical approach in the paper corpus for
strategy detection and would provide convergent validation.

---

### Tiny RNN Strategy Discovery (Ji-An-style)

**Paper:** Ji-An et al 2025 (discovering cognitive strategies with tiny RNNs).

**What:** Train a small RNN (input: trial features + previous choice; output:
next choice) per participant. Extract the RNN's hidden state dynamics via PCA;
cluster participants by RNN latent trajectories. This discovers strategies
from raw behavior without imposing a parametric form.

**Approach:**
```python
class StrategyRNN(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=8):
        self.rnn = nn.GRUCell(input_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)
    def forward(self, x_seq, h0):
        # Process trial sequence; return choices and hidden states
```

**Impact:** The Ji-An approach is complementary to the DeepONet: the DeepONet
uses pre-engineered features to parameterize strategies; the RNN discovers
strategies from raw sequences. Comparing their outputs tests whether the
engineered features capture everything relevant.

**Effort:** ~200 lines in `analysis/strategy_rnn.py`. Uses existing PyTorch
stack.

---

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

### Planning Depth Estimation

**Paper:** Mattar et al 2025 (few rollouts); Keramati et al 2016 (depth-limited
planning along habit–goal spectrum).

**What:** Compare participant choices against optimal play computed by tree
search at different depths (1-step greedy, 2-step, 3-step). For each trial,
find the minimum planning depth needed to produce the observed choice. A
participant who consistently matches depth-1 optimal is a heuristic planner;
one who matches depth-3 is a deliberative planner.

**Approach:** Use existing `agentic_decision_making.py`'s `calculate_greedy_cost`
and `calculate_planning_cost` to compute optimal paths at each depth, then
compute agreement with participant choices.

**Impact:** Directly quantifies the Keramati habit–goal continuum. Connects
to the DeepONet by testing whether specific strategies (e.g., Strategy 2)
correspond to specific planning depths.

**Effort:** ~100 lines in `analysis/planning_depth.py` (reuses existing code).

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

### Unified Model Evaluation Framework

**What:** A script that loads all trained models (original DeepONet, gated
DeepONet, GLM-HMM, RNN, DDM) and evaluates them on common metrics: choice
prediction accuracy, RT prediction R², strategy cluster consistency, and
cross-prediction (does the HMM's state probability correlate with the
DeepONet's gate weight?).

**Impact:** Prevents fragmentation. Every new model gets benchmarked against
existing ones on the same train/test split.

**Effort:** ~150 lines in `analysis/model_comparison.py`.

---

### Basis Interpretability Labels

**What:** Add a supervised auxiliary loss to the DeepONet that maps each basis
function to a named cognitive construct (e.g., "immediate optimizer",
"two-step planner", "ball-y follower", "direction bias"). Could use weak
supervision: label a small set of trials with ground-truth strategy labels
and propagate through the basis network.

**Impact:** Makes DeepONet bases directly interpretable rather than requiring
post-hoc inspection of basis sweeps.

**Effort:** ~80 lines in strategy_deeponet.py (modification). Requires manual
labeling of ~100 example trials.

---

## Summary by Priority

| Priority | Change | Paper | Lines | Category |
|---|---|---|---|---|
| 1 | GLM-HMM Strategy Switching | Ashwood 2022 | ~300 | Analysis |
| 2 | Confidence Ratings | Resulaj 2009 | ~20 + ~150 | Instrumentation + Analysis |
| 3 | Trait Questionnaires | Anxiety-Depression paper | ~100 HTML/JS | Instrumentation |
| 4 | Planning Depth Estimation | Mattar 2025 / Keramati 2016 | ~100 | Analysis |
| 5 | Tiny RNN Strategy Discovery | Ji-An 2025 | ~200 | Analysis |
| 6 | Decision-Locked Epochs | Resulaj 2009 | ~40 | Instrumentation |
| 7 | Bounded Diffusion Fitting | Resulaj 2009 / Keung 2020 | ~150 | Analysis |
| 8 | Information Demand Probes | Jach 2024 | ~80 | Instrumentation |
| 9 | Gaze Proxy | Peer 2021 | ~15 + ~50 | Instrumentation + Analysis |
| 10 | Unified Evaluation Framework | — | ~150 | Cross-cutting |
| 11 | Cognitive Map Analysis | Peer 2021 | ~200 | Analysis |
| 12 | Basis Interpretability Labels | — | ~80 | Model Modification |
