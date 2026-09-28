# %% [markdown]
# # Behavioural claim audit
#
# Aggregates the level/reliability/sequential/block re-analyses into a
# claim-by-claim status: robust / relabeled / qualified / revised / confounded.
# Reads the re-analysis outputs and writes:
#   analysis/planning_outputs/claim_audit.csv
#   analysis/planning_outputs/figures/fig_claim_audit.png
#
# Run with:
#   python analysis/claim_audit.py

# %%
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

_REPO = Path(__file__).resolve().parent.parent
PO = _REPO / "analysis" / "planning_outputs"

STATUS_COLORS = {
    "robust": "#2a9d8f",
    "relabeled": "#457b9d",
    "qualified": "#e9c46a",
    "revised": "#e76f51",
    "confounded": "#9d0208",
}


def load():
    rel = pd.read_csv(PO / "planning_reanalysis_reliability.csv")
    seq = pd.read_csv(PO / "planning_reanalysis_sequential_coefs.csv")
    blk = pd.read_csv(PO / "planning_reanalysis_block.csv")
    return rel, seq, blk


def build_claims(rel, seq, blk):
    sb = rel.set_index("measure")["spearman_brown"]

    def blk_val(spec, term):
        r = blk[(blk.spec == spec) & (blk.term == term)]
        return float(r["coef"].iloc[0]) if len(r) else np.nan

    def blk_p(spec, term):
        r = blk[(blk.spec == spec) & (blk.term == term)]
        return float(r["p"].iloc[0]) if len(r) else np.nan

    perm_path = PO / "planning_reanalysis_sequential_perm.csv"
    lag_obs = (float(pd.read_csv(perm_path)["observed"].iloc[0])
               if perm_path.exists() else np.nan)

    rows = [
        dict(claim="Planning is graded / minority (P(plan|conflict)~0.34)",
             level="participant", status="robust",
             evidence=f"P(plan) split-half SB={sb.get('p_plan', float('nan')):.2f}"),
        dict(claim="Planning is an individual trait",
             level="participant", status="relabeled",
             evidence="ICC(participant)=0.02; most choice variance is within-person/trial"),
        dict(claim="No latent strategy state (GLM-HMM K*=1)",
             level="participant", status="robust",
             evidence="K*=1; adding a lag input worsens BIC"),
        dict(claim="Strategy choices are memoryless",
             level="trial", status="qualified",
             evidence=f"within-block lag-1 beta={lag_obs:.3f} (perm p<0.001; "
                      f"alternation); one-sided stay test missed it"),
        dict(claim="Planning rises with block conflict rate (+0.57)",
             level="block", status="revised",
             evidence=f"within-participant GEE {blk_val('gee_within','block_conflict_rate_c'):.2f} "
                      f"(p={blk_p('gee_within','block_conflict_rate_c'):.3f}); "
                      f"+0.57 is a noisy unweighted mean (SB={sb.get('env_slope', float('nan')):.2f})"),
        dict(claim="Drift blocks plan less (Threat, delta=-0.05)",
             level="block", status="qualified",
             evidence=f"block_drift {blk_val('gee_plus_height','block_drift'):.3f} "
                      f"(p={blk_p('gee_plus_height','block_drift'):.3f}) with block height; coupled"),
        dict(claim="Ball-height mid-screen peak",
             level="block/participant", status="confounded",
             evidence="within-block p=0.51 (level decomposition); block/camera effect"),
        dict(claim="Model planning weight agrees with rate (r=+0.90)",
             level="participant", status="robust",
             evidence=f"weight split-half SB={sb.get('planning_weight', float('nan')):.2f}; "
                      f"level-stable 0.18/0.21/0.16"),
        dict(claim="Choice-independent geometry conflict_mag",
             level="trial", status="relabeled",
             evidence="conflict_mag = plan_advantage + greedy_advantage exactly; "
                      "value_pair AIC 7091 vs 7394"),
        dict(claim="RT trade-off is transport, not deliberation",
             level="trial/block", status="robust",
             evidence="constant fall ~210 ms, no dwell (timing decomposition)"),
    ]
    return pd.DataFrame(rows)


def make_figure(claims, rel, seq, blk):
    fig, axes = plt.subplots(1, 2, figsize=(17, 8),
                             gridspec_kw={"width_ratios": [2.4, 1]})

    ax = axes[0]
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, len(claims))
    for i, r in claims.iloc[::-1].reset_index(drop=True).iterrows():
        y = i + 0.5
        ax.add_patch(plt.Rectangle((0.0, y - 0.38), 0.012, 0.76,
                                   color=STATUS_COLORS[r.status]))
        ax.text(0.03, y + 0.12, r.claim, fontsize=9.5, va="center").set_weight("bold")
        ax.text(0.03, y - 0.22, f"[{r.level}] {r.evidence}", fontsize=7.5,
                va="center", color="0.3")
    ax.set_title("Claim audit: status after the level / reliability / sequential "
                 "re-analysis", fontsize=13)
    handles = [plt.Line2D([0], [0], marker="s", ls="", ms=10,
                          color=c, label=s) for s, c in STATUS_COLORS.items()]
    ax.legend(handles=handles, loc="lower right", fontsize=8, ncol=5,
              frameon=False)

    ax = axes[1]
    e = blk[(blk.family == "environment")
            & (blk.term.isin(["block_conflict_rate", "block_conflict_rate_c"]))]
    t = blk[blk.family == "threat"]
    lag = seq[seq.term.isin(["prev_plan_w", "lpm_block_FE"])]
    items = []
    for _, r in e.iterrows():
        items.append((f"env {r['spec']}", r["coef"], 0.05))
    for _, r in t.iterrows():
        items.append((f"{r['spec']} {r['term']}", r["coef"], 0.05))
    for _, r in lag.iterrows():
        tag = r["spec"] if isinstance(r["spec"], str) else r["model"]
        items.append((f"lag {tag}", r["coef"], 0.02))
    labels = [x[0] for x in items]
    vals = np.array([x[1] for x in items], float)
    y = np.arange(len(items))
    ax.barh(y, vals, color="#457b9d")
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("effect size (logit / probability)")
    ax.set_title("Key re-estimated effects")
    ax.invert_yaxis()

    fig.tight_layout()
    out = PO / "figures" / "fig_claim_audit.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    PO.mkdir(parents=True, exist_ok=True)
    rel, seq, blk = load()
    claims = build_claims(rel, seq, blk)
    claims.to_csv(PO / "claim_audit.csv", index=False)
    fig = make_figure(claims, rel, seq, blk)
    pd.set_option("display.width", 200)
    print(claims.to_string(index=False))
    print(f"\nwrote {PO / 'claim_audit.csv'} and {fig}")


if __name__ == "__main__":
    main()
