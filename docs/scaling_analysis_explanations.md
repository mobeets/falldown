# Scaling Analysis — How to Interpret the Output

Companion to `analysis/scaling_analysis.py` (the DeepONet transfer-scaling study).
This document explains what the table and plots mean so the results can be read
correctly.

---

## The mean held-out accuracy table

Example output:

```
--- Mean held-out accuracy by N and fit_frac ---
  fit_frac=1.0: N1=0.753, N2=0.747, N4=0.758, N6=0.772, N8=0.769, N12=0.773
  fit_frac=0.5: N1=0.751, N2=0.745, N4=0.753, N6=0.769, N8=0.764, N12=0.767
  fit_frac=0.25: N1=0.734, N2=0.736, N4=0.730, N6=0.766, N8=0.757, N12=0.762
```

### What each number means

Each value is the **mean held-out test accuracy** — accuracy on a participant whose
data was **never** used to train the basis — averaged over the held-out participants
and all subset × seed repeats.

- **Columns (N):** the number of participants the shared basis was trained on before
  being frozen. N=1 is a basis fit on one random participant; N=12 is the full pool.
- **Rows (fit_frac):** the fraction of the *held-out participant's own* train trials
  used to fit their embedding with the frozen basis. 1.0 = all their data; 0.25 = a
  quarter (few-shot).

### Reading across N (the transfer-scaling question)

For `fit_frac=1.0`, accuracy goes **0.753 → 0.773** as the pool grows 1 → 12. There is
a positive but **modest** transfer effect: a basis trained on more participants helps a
brand-new participant by roughly +2 accuracy points, with the gain concentrated around
N=4 → 6.

### Reading across fit_frac (the few-shot question)

- At N=1, cutting the new participant's data 4× costs ~2 points (0.753 → 0.734).
- At N=12, the same cut costs only ~1 point (0.773 → 0.762).

So a basis trained on more participants **partly compensates for having less of the new
participant's own data** — transfer substitutes for within-participant data. This is the
cleanest finding in the table.

### Benchmarking

- Chance = 0.5 (the study runs ~0.25 above it).
- Within-participant references (printed in the run header as `Baseline ...`):
  - `logistic` ≈ 0.75–0.82
  - `scratch_deeponet` ≈ 0.65–0.86
- The frozen-basis fit (~0.77 with full data) lands **right in line with a full
  within-participant logistic** — strong evidence the basis carries the features that
  matter, since a stranger's basis + a quick embedding fit nearly matches a model fit
  entirely on that participant.

### Caveats before trusting the numbers

1. **Only 3 held-out participants** — error bars are likely ±0.02–0.03, so differences
   between *adjacent* N (e.g., N8 vs N12) are noise; only the N1 → N12 trend is
   meaningful.
2. Check the per-participant lines in the plot — if one held-out participant dominates,
   the aggregate is misleading.
3. To claim significance, run a paired test per held-out participant (N=1 vs N=12) or
   bootstrap the difference.

---

## The plot legend

The accuracy-vs-N figure (one panel per `fit_frac`) shows:

| Line | Meaning |
|---|---|
| **Black mean ± SE** | Aggregate frozen-basis transfer accuracy across held-out participants |
| **Thin colored lines** (blue / orange / green — one per held-out participant) | Individual held-out participants' transfer curves; the "yellow" line is matplotlib's orange |
| **Gray dashed `chance`** | 0.5 |
| **Red dotted `logistic (no transfer)`** | Within-participant logistic regression on raw features |
| **Blue dotted `scratch deeponet`** | Within-participant DeepONet, basis fit on that participant alone |

### What "logistic (no transfer)" means

Computed by `logistic_baseline_accuracy` in `scaling_analysis.py`: it calls
`evaluate_logistic_baseline` (exploratory_data_analysis.py), which fits a plain
`sklearn` logistic regression **entirely on that participant's own data** — features
`L1-R1`, `L1+L2-R1-R2`, incoming direction (+ drift interaction), z-scored, evaluated on
a chronological held-out block split. No shared basis, no other participants' data.

It is the **per-participant no-transfer reference**: the accuracy a simple linear model
can reach using that participant's full data alone. The line drawn is the mean across
the held-out participants. Comparing the frozen-basis curves (the rising points) to it
answers *"is borrowing the basis worth it?"* — if the transfer curve approaches or
crosses the line, the transferred basis is nearly as good as a model fit on the
participant's own full data.

Caveat: it is a strong reference, not a ceiling — the logistic baseline uses ~80% of
the participant's data and its own raw features, while the transfer fit sees only
`fit_frac` of their trials and is constrained to the frozen basis. It also uses a
block-based split (from the logistic pipeline), slightly different from the temporal
80/20 split used for the transfer curves.

### What "scratch deeponet" means

Computed by `scratch_deeponet_accuracy` in `scaling_analysis.py`: for each held-out
participant, train a full `CognitiveDeepONet` **from scratch on that participant's own
train trials alone** (`num_participants=1`, self-scaled, orthogonality penalty), then
evaluate on their test trials. The plotted line is the mean across the held-out
participants.

It is the **no-transfer DeepONet baseline**: same architecture as the study, but the
basis is fit to this one participant instead of borrowed from a pool. The transfer
question becomes *"does a basis trained on N other participants (black curve) beat a
basis trained on this participant (blue line)?"* If the black curve approaches the blue
line, transfer is working well.

Note: only computed for `model_type='cognitive'` — it is `NaN` for the strategy run, so
the line does not appear there.

### What the thin "green and yellow" lines mean

The thin low-alpha lines are the **per-held-out-participant transfer curves** — one per
held-out participant, using matplotlib's default color cycle (blue, orange, green for
the 3 held-out participants; the "yellow" is the orange one). Each shows that
individual participant's frozen-basis accuracy vs N (embedding fit on `fit_frac` of
their data, evaluated on their test trials).

They matter because the black mean ± SE line is just their average: if the per-participant
lines are far apart, one participant is driving the aggregate and the trend may not
generalize.

---

## Addendum — Strategy DeepONet still to run

The strategy (HMM-gated `StrategyDeepONet`) scaling study has **not yet been run with
lower `fit_frac` values**. In `analysis/scaling_analysis.py`, the strategy call in the
`__main__` block is currently commented out and configured with `fit_fracs=(1.0,)`
only.

To complete it, uncomment the strategy block and run with the full few-shot grid, e.g.:

```python
results_s, summary_s, _, _, _ = run_scaling_study(
    model_type="strategy",
    data_dir=os.path.join(_SCRIPT_DIR, "cloud study data"),
    num_states=3,
    subsets_per_size=2,
    seeds=(0,),
    num_epochs=100,
    fit_fracs=(1.0, 0.5, 0.25),   # add the lower fit_fracs
)
summary_s.to_csv("scaling_strategy_summary.csv", index=False)
```

Expected runtime is substantially slower than the cognitive study (~30–60 min for the
full grid) because each strategy pool model is trained on sequences via the Markov-chain
NLL objective. The held-out fit for the strategy model is a GLM-HMM on the frozen basis
features (`num_states=3`, fit via `ssm`).
