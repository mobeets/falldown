# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format: .py
#     jupytext_version: 1.19.4
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Block conflict rate vs planning measures
#
# Pools every cloud participant (exp1 + exp3) at the **block** level and asks
# whether planning depends on how often the block's greedy and planning rules
# disagree.
#
# Unit = block. `x` = **conflict rate** = `mean(conflict)` over the block's
# 2-hole choice trials (conflict is the binary greedy!=planning indicator).
# Three planning measures:
#   1. `p_plan`      = P(plan | conflict) over the block's conflict trials;
#   2. `raw_beta`    = per-block logistic coefficient on the planning gap,
#                      plotted as `-beta_plan` so "up = more planning";
#   3. `planning_weight` = |b_plan| / (|b_plan| + |b_1step|).
#
# Per-block coefficients come from a per-block L2-penalized logistic
# `P(choose left) ~ diff_1step + diff_planning`, with both gaps z-scored using
# the participant's global mean/SD (so blocks are on a comparable scale).
# Drift blocks are marked separately; both pooled and within-participant
# (participant-centred) regressions are drawn.
#
# Outputs:
#   analysis/planning_outputs/planning_block_measures.csv
#   analysis/planning_outputs/planning_block_measures_summary.csv
#   analysis/planning_outputs/figures/fig_planning_block_conflict_scatter.png
#
# Run with:
#   python analysis/planning_block_conflict_scatter.py

# %%
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

_REPO = Path(__file__).resolve().parent.parent
XD = _REPO / "analysis" / "cross_dataset_outputs"
PO = _REPO / "analysis" / "planning_outputs"

MIN_CHOICE = 20      # min 2-hole choice trials per block
MIN_CONFLICT = 8     # min conflict trials per block
C_DRIFT = "#d1495b"
C_NON = "#00798c"
C_POOL = "0.35"


# %%
def load_cloud():
    d = pd.read_csv(XD / "cross_dataset_trials.csv")
    d = d[(d.is_experimental) & (d.source == "cloud")].copy()
    d = d.dropna(subset=["diff_1step", "diff_planning", "chosen_left"])
    return d


def block_measures(d):
    rows = []
    for (pid, bn), g in d.groupby(["participant", "block_number"]):
        n_choice = len(g)
        n_conf = int((g["conflict"] == 1).sum())
        if n_choice < MIN_CHOICE or n_conf < MIN_CONFLICT:
            continue
        if g["chosen_left"].nunique() < 2:
            continue
        pid_mean1 = d[d.participant == pid]["diff_1step"].mean()
        pid_sd1 = d[d.participant == pid]["diff_1step"].std()
        pid_meanp = d[d.participant == pid]["diff_planning"].mean()
        pid_sdp = d[d.participant == pid]["diff_planning"].std()
        z1 = (g["diff_1step"].to_numpy() - pid_mean1) / (pid_sd1 or 1.0)
        zp = (g["diff_planning"].to_numpy() - pid_meanp) / (pid_sdp or 1.0)
        X = np.column_stack([z1, zp])
        y = g["chosen_left"].astype(int).to_numpy()
        clf = LogisticRegression(penalty="l2", C=1.0, max_iter=500)
        clf.fit(X, y)
        b1, bp = float(clf.coef_[0][0]), float(clf.coef_[0][1])
        den = abs(b1) + abs(bp)
        conf = g[g["conflict"] == 1]
        rows.append({
            "participant": pid,
            "block_number": int(bn),
            "experiment": g["experiment"].iloc[0],
            "environment": g["environment"].iloc[0],
            "dataset": g["dataset"].iloc[0],
            "threat": g["threat"].iloc[0],
            "drift": int(g["block_drift"].iloc[0]),
            "n_choice": n_choice,
            "n_conflict": n_conf,
            "conflict_rate": float(g["conflict"].mean()),
            "p_plan": float(conf["chose_planning"].mean()),
            "beta_1step": b1,
            "beta_plan": bp,
            "neg_beta_plan": -bp,
            "planning_weight": abs(bp) / den if den > 1e-9 else np.nan,
        })
    return pd.DataFrame(rows)


# %% [markdown]
# ## Regressions (pooled and within-participant)

# %%
def fit_slope(df, ycol, within):
    d = df.dropna(subset=[ycol, "conflict_rate"]).copy()
    if len(d) < 10:
        return {"measure": ycol, "fit": "within" if within else "pooled",
                "n_blocks": len(d), "slope": np.nan, "p": np.nan}
    if within:
        d["x"] = d["conflict_rate"] - d.groupby("participant")["conflict_rate"].transform("mean")
        d["y"] = d[ycol] - d.groupby("participant")[ycol].transform("mean")
    else:
        d["x"] = d["conflict_rate"]
        d["y"] = d[ycol]
    m = sm.OLS(d["y"], sm.add_constant(d["x"])).fit(
        cov_type="cluster", cov_kwds={"groups": d["participant"]})
    return {"measure": ycol, "fit": "within" if within else "pooled",
            "n_blocks": len(d), "slope": float(m.params["x"]),
            "se": float(m.bse["x"]), "p": float(m.pvalues["x"])}


def make_figure(df, slopes):
    measures = [("p_plan", "P(plan | conflict)"),
                ("neg_beta_plan", "raw logistic: -beta_plan"),
                ("planning_weight", "normalized: planning weight")]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    for ax, (col, label) in zip(axes, measures):
        for drift, colr, mk in [(0, C_NON, "o"), (1, C_DRIFT, "D")]:
            s = df[df.drift == drift]
            ax.scatter(s["conflict_rate"], s[col], s=18, alpha=0.6, color=colr,
                       marker=mk, label="drift" if drift else "non-drift")
        # regression lines: pooled and within-participant
        d = df.dropna(subset=[col, "conflict_rate"])
        m = sm.OLS(d[col], sm.add_constant(d["conflict_rate"])).fit()
        xs = np.linspace(d["conflict_rate"].min(), d["conflict_rate"].max(), 50)
        ax.plot(xs, m.params["const"] + m.params["conflict_rate"] * xs,
                color=C_POOL, ls="--", lw=1.8, label="pooled fit")
        w = d.copy()
        w["xc"] = w["conflict_rate"] - w.groupby("participant")["conflict_rate"].transform("mean")
        w["yc"] = w[col] - w.groupby("participant")[col].transform("mean")
        mw = sm.OLS(w["yc"], w["xc"]).fit()
        ax.plot(xs, mw.params["xc"] * (xs - d["conflict_rate"].mean()), color="k",
                lw=2.2, label="within-participant fit")
        ps = slopes[(slopes.measure == col) & (slopes.fit == "pooled")].iloc[0]
        ws = slopes[(slopes.measure == col) & (slopes.fit == "within")].iloc[0]
        ax.text(0.02, 0.98,
                f"pooled slope={ps.slope:+.3f} (p={ps.p:.2g})\n"
                f"within slope={ws.slope:+.3f} (p={ws.p:.2g})",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(fc="white", ec="0.7", alpha=0.85))
        ax.set_xlabel("block conflict rate  mean(conflict)")
        ax.set_ylabel(label)
        ax.axhline(0, color="0.8", lw=0.8) if col != "planning_weight" else None
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Block conflict rate vs planning (cloud; blocks, drift marked; "
                 "4+9+5 participants)", fontsize=13)
    fig.tight_layout()
    out = PO / "figures" / "fig_planning_block_conflict_scatter.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    PO.mkdir(parents=True, exist_ok=True)
    d = load_cloud()
    df = block_measures(d)
    print(f"{len(df)} blocks, {df.participant.nunique()} participants")
    print(df.groupby(["experiment", "threat"]).agg(
        blocks=("conflict_rate", "size"),
        conflict_rate=("conflict_rate", "mean"),
        p_plan=("p_plan", "mean"),
        weight=("planning_weight", "mean")).round(3).to_string())

    slopes = [fit_slope(df, c, w) for c in
              ["p_plan", "neg_beta_plan", "planning_weight"] for w in [False, True]]
    slopes = pd.DataFrame(slopes)
    df.to_csv(PO / "planning_block_measures.csv", index=False)
    slopes.to_csv(PO / "planning_block_measures_summary.csv", index=False)
    fig = make_figure(df, slopes)
    print("\nSlopes:")
    print(slopes.round(4).to_string(index=False))
    print(f"\nwrote planning_block_measures*.csv and {fig}")


if __name__ == "__main__":
    main()
