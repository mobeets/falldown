# Superseded figures

Figures moved here are no longer current for the behavioural planning narrative.
They are kept for provenance. Do not cite them; cite the replacement.

| Figure | Was | Why superseded | Replacement |
|---|---|---|---|
| `fig_planning_temporal.png` | Stay/switch temporal structure, read as "memoryless" (0/18 excess stay) | The test was one-sided for **excess stay** and therefore blind to **alternation**. The level/sequential re-analysis finds a small but significant negative within-block lag-1 dependence (LPM beta = -0.089, within-participant permutation p < 0.001), which survives uncentered block fixed effects. The "no latent strategy state (K*=1)" part still holds, but "memoryless" does not. | `analysis/planning_outputs/figures/fig_planning_reanalysis_sequential.png`; claim audit `fig_claim_audit.png` |

## Kept, but with revised interpretation (not moved)
- `fig_exp3_threat.png` — threat (drift < follow) attenuates to -0.036 (p = 0.09) once block-mean ball height is controlled; threat and height are coupled.
- `fig_exp2_within.png` — within-participant environment slope is real but modest (+0.30, p = 0.03) and robust to controls; the +0.57 headline is an unreliable unweighted mean (split-half SB = 0.52).
- Prevalence / individual-difference figures — `P(plan)` and planning weight are reliable participant summaries (SB = 0.93) but explain little choice variance (ICC = 0.02).

Re-running `analysis/planning_figures.py` will recreate the superseded
`fig_planning_temporal.png` under `analysis/planning_outputs/figures/`; treat it
as historical only.
