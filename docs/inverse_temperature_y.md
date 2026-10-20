# Inverse temperature vs. ball y-position (Lei 2026 analogy)

Read-only analysis of whether the **inverse temperature** principle from Lei et
al. 2026 ("Environmental stochasticity reduces human planning effort") applies
to participants' behaviour at different ball y-positions in Falldown.

> **Provenance.** This doc reconstructs a discussion that was never written to
> disk. It was recovered (2026-10-04) from the opencode session store
> (`~/.local/share/opencode/opencode.db`, session *"splinemodel.py planning
> discrepancy"*, `ses_f111addc8ffeEPOgZ2Nc2lwhCF`). The original runs were
> **read-only** — no scripts were committed, so the numbers below are transcribed
> from the session transcript, not regenerated. The planned script was never
> implemented.

---

## Background

Lei 2026 reports **policy compression**: the choice inverse temperature `β`
(reciprocal temperature in `softmax(βV)`, i.e. the sensitivity of choice to
value) is *modulated by environmental stochasticity* — lower `β` under higher
stochasticity. The question here was whether `β` in Falldown is modulated by
**ball height** `y` (`ball_y_at_top`), a threat / time-pressure proxy.

Key framing established during the discussion:

- A **constant** `β` is still a legitimate inverse temperature. What Lei reports
  is not "β exists" but "**β is modulated** by stochasticity."
- Height is a **time-pressure/threat proxy, not a stochasticity manipulation**,
  so a null result for `β(y)` is *not* a failure of Lei's principle — it is a
  test of a different hypothesis (does threat/time-pressure compress the policy?).

---

## Data and method

- Data: `cross_dataset_outputs/cross_dataset_trials.csv`, cloud experimental
  participants (`drift` blocks for the focused analysis: 8,717 trials, 13
  participants; all cloud experimental: 17,721 trials, 18 participants).
- Value advantage: `x1 = −diff_1step` (greedy) and `x2 = −diff_planning`
  (planning); positive → left option better.
- Per participant: `logit P(chosen_left) ~ x1 + x2 + y + x1·y + x2·y`, values and
  height (`ball_y_at_top`) z-scored within participant. `β` = slope on the value
  advantage; `β·y` = change in `β` per +1 SD height toward the bottom.
- Also computed per-participant 5-quantile `β(y)` profiles.
- No writes; no lapse parameter; single `y` per sequence.

---

## Results

### Preliminary (all cloud experimental, 18 participants)

| subset | greedy β | planning β | greedy β×height | planning β×height |
|---|---|---|---|---|
| all | +1.62 | +0.47 | **−0.072 (p=0.023, 5/18+)** | +0.018 (p=0.64) |
| drift only | +1.62 | +0.38 | −0.081 (p=0.15) | +0.047 (p=0.36) |
| follow only | +1.45 | +0.57 | +0.047 (p=0.46) | −0.067 (p=0.22) |
| conflict only | +1.13 | +0.19 | +0.085 (p=0.16) | **+0.114 (p=0.048, 11/18+)** |

β by height quintile (top→bottom, per-SD), participants with all 5 bins:

```
greedy:   1.95  2.29  1.92  1.65  1.67
planning: 1.24  1.61  1.02  0.98  0.96
```

### Focused (drift blocks only, entry height, 13 participants)

Baseline inverse temperature (slope of `logit P(left)` on value advantage, per SD):

- Greedy value: `β ≈ +1.81` (SD 0.45)
- Planning value: `β ≈ +1.11` (SD 0.37)

| test | greedy β | planning β |
|---|---|---|
| interaction β×height (all 13 subj, z-scored) | −0.044 (p=0.28, 7/13 neg) | −0.025 (p=0.45, 9/13 neg) |
| OLS slope of β on height (6 subj with all 5 quintiles) | −0.257 /100px (p=0.028) | −0.148 /100px (p=0.003) |
| bottom − top β difference (same 6 subj) | −0.874 (p=0.016) | −0.502 (p=0.008) |

Pooled β by height quintile (top→bottom, drift only):

```
greedy:   bin0 2.22  bin1 2.01  bin2 2.06  bin3 1.97  bin4 1.35
planning: bin0 1.31  bin1 1.47  bin2 1.16  bin3 1.28  bin4 0.81
```

### Cross-validation (drift blocks, entry height, 13 participants)

Per-participant `β` and 5-fold CV gain `dLL` of the height-dependent `β` over
constant `β` (positive = height helps; `*_y` = change in β per +1 SD height):

```
 participant     n | b_greedy   g_y | b_plan   p_y |  dLL_g  dLL_p
 32FC87F1C127  1120 |    2.46  0.037 |   0.91 -0.196 |  -0.9    2.5
 46331EBA4F49   508 |    2.09 -0.016 |   1.57  0.053 |  -2.3   -4.3
 6462D588260B   532 |    1.66 -0.391 |   1.14 -0.184 |   1.2   -0.4
 65D6694BE069   533 |    2.05 -0.194 |   1.51 -0.030 |  -0.3   -1.6
 88AD64F00C6B   531 |    1.62 -0.143 |   1.07 -0.127 |  -0.3   -0.5
 96CA2FB77099  1148 |    1.87  0.153 |   0.73 -0.010 |  -0.6   -1.3
 B0525260D0F8   810 |    1.70 -0.078 |   0.65  0.058 |  -0.5   -2.5
 BB4D2ACD4DAB   521 |    1.69 -0.028 |   1.15  0.206 |  -1.5    0.9
 C47CEEC22AD9   440 |    0.87  0.028 |   0.63 -0.035 |  -0.0   -0.4
 C8C4C97C01AA  1019 |    1.90 -0.072 |   0.81 -0.009 |  -2.8   -1.2
 CEFD2FE92E68   490 |    2.07  0.037 |   1.83  0.113 |  -3.6   -3.9
 EC07396CE232   533 |    1.03  0.088 |   0.94 -0.036 |   0.3    0.0
 FD2A6686546A   532 |    2.43  0.012 |   1.42 -0.129 |  -0.8   -2.7
```

Aggregate (one-sample t over 13 participants):

- **Baseline β is real**: greedy +1.81 (13/13 positive, p<0.001); planning +1.11
  (13/13, p<0.001).
- Height slope: greedy −0.044 (p=0.28); planning −0.025 (p=0.45) — **ns**.
- Permutation test (shuffle height within participant, 500×): greedy p=0.31,
  planning p=0.45 — within the null band.
- **CV: the height-dependent model is worse out-of-sample than constant β** —
  mean test-log-likelihood change −0.94 (greedy, p=0.025) and −1.18 (planning,
  p=0.041); only 2–3/13 participants improve.

---

## Conclusions

1. **β is a real, robust quantity** in Falldown — a stable per-participant
   inverse temperature, larger for greedy than planning value, positive in every
   participant.
2. **β(y) is not real** — the apparent height dependence does not survive 5-fold
   CV or permutation. Earlier "significant" per-bin slopes were driven by the 6
   participants who covered all height bins, plus overfitting. To a good
   approximation **β is constant across y**.
3. A constant β is still an inverse temperature; what is *not* reproduced is
   Lei's **modulation** of β. But **height is the wrong modulator** — it is a
   threat/time-pressure proxy, not environmental stochasticity.

### Environment-modulator follow-up (cloud data, read-only)

| modulator | greedy β effect | planning β effect |
|---|---|---|
| height `ball_y_at_top` | ns (CV worse) | ns (CV worse) |
| block conflict rate (stochasticity proxy) | ns (CV worse, p=0.008) | marginal −0.07 (p=0.085), CV neutral |
| camera mode (drift vs follow) | **drift 1.72 vs follow 1.54, p=0.014** | 1.25 vs 1.25, p=0.97 |

- The only reliable condition difference is **drift vs follow for the greedy
  value, within the 9 alternating participants** (8/9 higher in drift). Planning
  β is condition-invariant in these comparisons.
- Only **9 of 18** participants completed alternating drift/follow blocks; the
  other 9 are drift-only (exp3) or follow-only (exp1), so any drift/follow
  comparison over all 18 would be between-subjects and confounded with the
  experiment.
- Drift and follow differ in conflict rate and value distribution, so the greedy
  β difference could partly reflect context rather than a pure temperature change.

Per-participant β for the 9 alternating participants (per-SD value):

```
   pid      greedy: drift follow | planning: drift follow
46331EBA4F     2.08   2.07     |     1.57   1.83
6462D58826     1.64   1.47     |     1.15   1.14
65D6694BE0     2.04   1.78     |     1.51   1.37
88AD64F00C     1.63   1.68     |     1.06   1.26
BB4D2ACD4D     1.68   1.59     |     1.13   1.11
C47CEEC22A     0.88   0.57     |     0.63   0.57
CEFD2FE92E     2.07   1.57     |     1.84   1.57
EC07396CE2     1.01   0.82     |     0.93   0.79
FD2A668654     2.44   2.34     |     1.41   1.58
```

---

## Caveats

- Only 6 of 13 drift participants have all 5 height quintiles; the significant
  per-bin slope tests rest on them. The all-participant interaction is
  underpowered.
- Entry-time height (`ball_y_at_top`) only. The choice-event height
  (`cloudio.heightOnScreen`) is post-choice and behaves differently (note: the
  column is named `heightOnScreen` in the session).
- Single `y` per sequence; no lapse parameter; values in distance units
  (z-scored per participant per SD).
- Risk of value-representation mismatch: Lei's β sits on a single subjective
  value `softmax(βV)`, whereas this analysis used separate greedy and planning
  slopes — a different parameterization that may be less sensitive to
  modulation.

---

## Unimplemented follow-up plan

A proposed (never-run) read-only script `analysis/inverse_temperature_y.py`:

1. Load `cross_dataset_trials.csv`; filter cloud + experimental + drift blocks.
2. Per participant fit `logit P(left) ~ x + x·y` for `x = −diff_1step` and
   `x = −diff_planning`; plot `β(y)` over the participant's height range with
   bootstrap 95% CI, plus the constant-β estimate as a horizontal line.
3. Add a CV panel: 5-fold within-participant ΔLL (height vs constant) per
   participant with the paired test, and the permutation null distribution of the
   mean slope.
4. Write a figure to `analysis/figures/` and a CSV of per-participant β, slope,
   CV ΔLL, and permutation p.
5. Fixed seed; no data modified.

To test the Lei principle more directly, modulate β by a
stochasticity-style variable (block conflict rate, drift vs follow,
incoming-direction uncertainty, or a composite environmental-uncertainty index)
using a single subjective-value β rather than the greedy/planning split.
