# Variations for Adding Incoming Direction to the Custom Cognitive Model

The current custom model (`fit_dynamic_model`) is a mixture-of-experts with a greedy strategy and a planning strategy. The mixing weight (probability of using planning) is modulated by `ball_y_at_top` (time pressure). The logistic regression baseline additionally uses `Incoming Direction` as a feature, but the custom model ignores it entirely.

Below are three principled ways to add `incoming_direction` to the custom model, ordered by complexity and theoretical scope.

---

## Current Model Architecture (for reference)

```
Parameters: [p_lapse, p_plan_base, w1, s_greedy, s_plan]

P_greedy(R) = sigmoid(d_greedy / s_greedy)
P_plan(R)   = sigmoid(d_plan / s_plan)

p_plan = logistic(logit(p_plan_base) + w1 * ball_y)    ← mixture weight

P_model(R) = (1 - p_plan) * P_greedy(R) + p_plan * P_plan(R)
P_final(R) = p_lapse * 0.5 + (1 - p_lapse) * P_model(R)
```

where `d_greedy = L1 - R1` (1-step distance advantage) and `d_plan = Total_L - Total_R` (2-step distance advantage).

### What each parameter captures

| Param | Role | Bounds |
|---|---|---|
| `p_lapse` | Probability of random guess | [0, 0.99] |
| `p_plan_base` | Baseline planning probability | [0, 1] |
| `w1` | Effect of ball height on planning probability | unbounded |
| `s_greedy` | Softmax temperature for greedy strategy | [0.001, ∞) |
| `s_plan` | Softmax temperature for planning strategy | [0.001, ∞) |

---

## Variation A — Incoming direction modulates the greedy/planning mixture weight

**New parameter**: `w2` (unbounded)

```
p_plan = logistic(logit(p_plan_base) + w1 * ball_y + w2 * incoming_dir)
```

**Total parameters**: 6

**What it captures**: Whether directional inertia shifts reliance from greedy to planning (or vice versa). A positive `w2` means that when the previous direction was right (`incoming_dir = +1`), the participant is more likely to use planning. Directly comparable to the `Incoming Direction` coefficient in the logistic regression.

**Pros**: 
- Closest analogue to the logistic regression feature set
- Single additional parameter
- Fits naturally into the existing mixture-of-experts structure

**Cons**:
- Cannot distinguish whether the effect lives in strategy selection vs. execution
- Assumes the same directional modulation applies equally to both strategies

**When to use**: If you want the most parsimonious comparison with logistic regression.

---

## Variation B — Incoming direction as a bias within each strategy

**New parameters**: `b_greedy_dir`, `b_plan_dir` (unbounded)

```
P_greedy(R) = sigmoid((d_greedy + b_greedy_dir * incoming_dir) / s_greedy)
P_plan(R)   = sigmoid((d_plan   + b_plan_dir   * incoming_dir) / s_plan)
```

**Total parameters**: 7

**What it captures**: Whether directional inertia affects how the participant executes each strategy separately. `b_greedy_dir` captures bias in the greedy system (e.g., tendency to repeat direction when not planning), while `b_plan_dir` captures bias in the planning system.

**Pros**:
- Separates strategy-selection effects from strategy-execution effects
- Can reveal whether directional bias lives in the automatic (greedy) vs. deliberative (planning) system
- Each bias operates in log-odds space, matching the softmax formulation

**Cons**:
- Two additional parameters (higher risk of overfitting)
- More complex to interpret the interaction

**When to use**: If you have a theoretical reason to think execution vs. selection differ, or you want to compare the magnitude of stickiness in greedy vs. planning behavior.

---

## Variation C — Choice stickiness bias (simple perseveration)

**New parameter**: `bias_dir` (unbounded)

```
# Add stickiness bias in log-odds space
logit_model = safe_logit(P_model(R))
logit_biased = logit_model + bias_dir * incoming_dir
P_biased(R) = expit(logit_biased)

P_final(R) = p_lapse * 0.5 + (1 - p_lapse) * P_biased(R)
```

**Total parameters**: 6

**What it captures**: Low-level perseveration — a tendency to repeat the previous choice regardless of the strategy used. A positive `bias_dir` means the participant is more likely to choose right when they chose right on the previous trial (and more likely to choose left when they chose left), independent of the task structure.

The bias operates in log-odds space (same as a logistic regression coefficient), which means:
- `bias_dir = 0` → no stickiness
- `bias_dir > 0` → repeat previous direction
- `bias_dir < 0` → alternate (switch away from previous direction)

**Pros**:
- Simplest possible addition (one parameter)
- Log-odds formulation guarantees P stays in [0, 1] without clipping
- Directly comparable to the logistic regression's `Incoming Direction` coefficient
- Separates "automatic perseveration" from "strategy-driven" choice

**Cons**:
- Cannot tell you whether the stickiness lives in the greedy system, planning system, or mixture
- Purely descriptive — doesn't explain why perseveration occurs

**When to use**: As a baseline comparison to the logistic regression. If the logistic regression's `Incoming Direction` coefficient is significant but the custom model with Variation C shows no improvement, the stickiness is already fully captured by the strategy mixture.

---

## Summary Table

| Variation           | Params | What it adds                 | Comparable to LR               |
| ------------------- | ------ | ---------------------------- | ------------------------------ |
| A                   | 6      | `w2` on mixture weight       | Yes — `Incoming Direction`     |
| B                   | 7      | `b_greedy_dir`, `b_plan_dir` | Partial — splits the effect    |
| **C (implemented)** | **6**  | **`bias_dir` on log-odds**   | **Yes — `Incoming Direction`** |

---

## Implementation notes for Variation C

The log-odds bias requires a safe logit function to handle edge cases:

```python
def safe_logit(p, eps=1e-10):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))
```

Then in `calculate_p_right_dynamic`:

```python
p_lapse, p_plan_base, w1, s_greedy, s_plan, bias_dir = params

# ... existing greedy/plan probability calculation ...

model_prob = (1 - dynamic_p_plan) * prob_greedy + dynamic_p_plan * prob_plan

# Apply stickiness bias in log-odds space
logit_prob = safe_logit(model_prob)
logit_biased = logit_prob + bias_dir * incoming_dir
biased_prob = expit(logit_biased)

final_prob = p_lapse * 0.5 + (1 - p_lapse) * biased_prob
```

The `incoming_direction` column from the processed data is already in the correct format: `+1` if the previous direction was left, `-1` if the previous direction was right. The bias direction produces stickiness when positive, alternation when negative.
