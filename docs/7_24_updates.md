# July 24 Updates

## 1. Feedforward Neural Network Baseline — `analysis/RNN.py`

Added a feedforward decision neural network for direct comparison against the existing TinyDecisionRNN (GRU).

**New code:**
- `prepare_ff_tensors()` — flattens per-block padded sequences into individual trial features (5-dim: L1-R1, L2-R2, block_drift, ball_y, cost). Uses the same chronological train/test split logic as the RNN.
- `FeedforwardDecisionNN` — 2 hidden layers (16 → 8) with ReLU, output softmax over 2 actions. No temporal/sequential processing.
- `train_feedforward()` — standard mini-batch training (Adam, CrossEntropyLoss).
- `evaluate_feedforward()` — computes log-likelihood, accuracy, and confusion matrix (same metrics as RNN).
- `run_FF_for_eval()` — end-to-end runner mirroring `run_RNN_for_eval()`.

**How to compare:**
```python
from RNN import run_FF_for_eval, run_RNN_for_eval
ff_metrics = run_FF_for_eval(participant_data)
rnn_metrics = run_RNN_for_eval(participant_data)
```

---

## 2. Multi-task DeepONet Bug Fix — `analysis/strategy_deeponet.py`

**Root cause:** Raw RT values (hundreds to thousands of ms) were fed directly into `MSELoss` for the RT prediction head. With values in the range of 10³, MSE was ~10⁶× larger than the `BCEWithLogitsLoss` on the choice head (~0.7). The shared basis network was trained almost exclusively to predict RT, collapsing choice accuracy to ~50% (random chance).

**Fix:** RT values are now z-score normalized (`(rt - mean) / std`) before training. Both loss terms now operate on comparable scales (~1), so the model learns meaningful choice-predictive features.

---

## 3. Direction Switching Histogram — `analysis/checking_data_validity.py`

Modified the histogram to split by block type (drift vs. follow).

- `get_switches_per_level()` now accepts `return_drift=False` (backward compatible). When `True`, returns `(switches, drift_flags)` where `drift_flags[i]` indicates whether that level was in a drift block.
- `plot_switch_distribution()` shows two side-by-side histograms: Follow blocks (skyblue, left) vs. Drift blocks (lightcoral, right).
- Handles edge case where a participant has only one block type — the empty subplot displays a "No X blocks" message instead of crashing.

---

## 4. Deaths per Block Fix — `analysis/checking_data_validity.py`

`plot_deaths_per_block()` was counting all trials without events as "deaths", which inflated counts by including calibration/instruction screens.

**Fix:** The function now:
- Skips block 0 (instruction block)
- Skips pure calibration blocks (exactly 4 trials)
- Skips the first 4 calibration trials within each remaining block before counting
- Only counts active gameplay trials without events as deaths

---

## 5. Position-based Feedforward NN — `analysis/RNN.py`

Added a variant of the feedforward network that takes raw hole positions instead of pre-computed distance differences.

**New code:**
- `prepare_ff_position_tensors()` — extracts 4 features per trial: `[entry_hole, left_hole, right_hole, exit_hole]`. No hand-crafted distance features — the model learns spatial relationships directly.
- `run_FF_position_for_eval()` — uses the same `FeedforwardDecisionNN` architecture but with `input_size=4`.

**How to use:**
```python
from RNN import run_FF_position_for_eval
metrics = run_FF_position_for_eval(participant_data)
```

Compare all three: `run_RNN_for_eval` (GRU, distances), `run_FF_for_eval` (feedforward, distances), `run_FF_position_for_eval` (feedforward, raw positions).

---

## 6. Strategy Preference Over Time — `analysis/exploratory_data_analysis.py`

Added `plot_strategy_over_time()` — a rolling-window analysis showing how each participant's strategy preference evolves over the experiment.

**What it shows:**
- **Gray dotted line:** how often the participant follows the greedy strategy across all trials.
- **Coral solid line:** how often they follow greedy specifically on **disagreement trials** — when greedy (1-step) and planning (2-step) prescribe different actions. This is the purer measure of strategy preference.
- A reference box in the corner shows the count of agree vs disagree trials.

**How to use:**
```python
from exploratory_data_analysis import plot_strategy_over_time, load
data = load("path/to/participant.json")
plot_strategy_over_time(data, participant_label="P1", window_size=30)
```

---

## 7. Model Comparison Runner — `analysis/model_comparison.py`

Added a unified comparison script that runs every model on every participant and produces a comparison table + accuracy box plot.

**Models compared:**
1. Logistic Regression — `evaluate_logistic_baseline()`
2. RNN (TinyDecisionRNN, GRU) — `run_RNN_for_eval()`
3. Feedforward NN (distance features) — `run_FF_for_eval()`
4. Feedforward NN (raw positions) — `run_FF_position_for_eval()`
5. GLM-HMM (2-state input-driven) — wrapped `ssm.HMM`
6. CognitiveDeepONet — shared basis network
7. StrategyDeepONet (gated) — mixture-of-strategies
8. StrategyDeepONet (multi-task) — choice + RT heads
9. StrategyDeepONet (time-binned) — temporal coefficient trajectories
10. Custom cognitive model — `run_participant_fits()` (planning + greedy mixture)

**Output:**
- Per-participant accuracy matrix (participants × models)
- Mean accuracy ranking across models
- `model_comparison.png` — box plot with individual participant points

**How to use:**
```python
from model_comparison import compare_all_models
df = compare_all_models()
```
