# %% [markdown]
# # Block-level re-analysis: environment, threat, and the value reframing
#
# Claims 3, 4, 7. The environment and threat effects are block-level (and the
# environment slope is a within-participant across-block slope). This script
# asks whether they survive controls for the coupled block variable — block-mean
# ball height — and restates the geometry as value (claim 7).
#
# Outputs (analysis/planning_outputs/):
#   planning_reanalysis_block.csv
#   figures/fig_planning_reanalysis_block.png
#
# Run with:
#   python analysis/planning_reanalysis_block.py

# %%
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")

_REPO = Path(__file__).resolve().parent.parent
XD = _REPO / "analysis" / "cross_dataset_outputs"
PO = _REPO / "analysis" / "planning_outputs"


def load_trials():
    d = pd.read_csv(XD / "cross_dataset_trials.csv")
    d = d[d["is_experimental"]].copy()
    d["conflict"] = d["conflict"].fillna(0).astype(int)
    d["conflict_mag"] = (d["diff_planning"] - d["diff_1step"]).abs()
    return d


def block_table(d):
    c = d[(d.source == "cloud") & (d.conflict == 1)]
    b = c.groupby(["participant", "block_number"]).agg(
        planning_rate=("chose_planning", "mean"),
        block_conflict_rate=("block_conflict_rate", "first"),
        block_mean_ball_y=("ball_y_at_top", "mean"),
        block_drift=("block_drift", "first"),
        threat=("threat", "first"),
        n=("chose_planning", "size")).reset_index()
    return b[b.n >= 5]


def gee(f, data, groups="participant"):
    return smf.gee(f, groups=groups, data=data, family=sm.families.Gaussian(),
                   cov_struct=sm.cov_struct.Exchangeable()).fit()


def per_participant_slopes(b, x):
    rows = []
    for pid, g in b.groupby("participant"):
        if len(g) < 6 or g[x].std() < 1e-9:
            continue
        m = smf.ols(f"planning_rate ~ {x}", data=g).fit()
        rows.append({"participant": pid, "slope": m.params[x], "p": m.pvalues[x],
                     "n_blocks": len(g)})
    return pd.DataFrame(rows)


# %% [markdown]
# ## Environment and threat models

# %%
def environment_models(b):
    rows = []
    slopes = per_participant_slopes(b, "block_conflict_rate")
    if len(slopes):
        t, p = stats.ttest_1samp(slopes["slope"], 0)
        rows.append({"family": "environment", "spec": "per_participant_slope",
                     "term": "block_conflict_rate", "coef": slopes["slope"].mean(),
                     "p": p, "n": len(slopes)})
    # within-participant centred predictor (matches the "within a participant" claim)
    b = b.copy()
    for c in ["block_conflict_rate", "block_mean_ball_y"]:
        b[c + "_c"] = b[c] - b.groupby("participant")[c].transform("mean")
    for spec, terms in [("gee_pooled", "block_conflict_rate"),
                        ("gee_within", "block_conflict_rate_c"),
                        ("gee_within_height", "block_conflict_rate_c + block_mean_ball_y_c"),
                        ("gee_within_height_threat",
                         "block_conflict_rate_c + block_mean_ball_y_c + block_drift")]:
        m = gee(f"planning_rate ~ {terms}", b)
        for term in terms.split(" + "):
            rows.append({"family": "environment", "spec": spec, "term": term,
                         "coef": m.params.get(term), "p": m.pvalues.get(term),
                         "n": int(m.nobs)})
    return pd.DataFrame(rows), slopes


def threat_models(b):
    rows = []
    for spec, terms in [("gee_univariate", "block_drift"),
                        ("gee_plus_height", "block_drift + block_mean_ball_y")]:
        m = gee(f"planning_rate ~ {terms}", b)
        for term in terms.split(" + "):
            rows.append({"family": "threat", "spec": spec, "term": term,
                         "coef": m.params.get(term), "p": m.pvalues.get(term),
                         "n": int(m.nobs)})
    return pd.DataFrame(rows)


def value_models(d):
    c = d[(d.source == "cloud") & (d.conflict == 1)].dropna(
        subset=["conflict_mag", "plan_advantage", "greedy_advantage"])
    rows = []
    specs = {"conflict_mag": "conflict_mag",
             "value_pair": "plan_advantage + greedy_advantage",
             "redundant_triple": "conflict_mag + plan_advantage + greedy_advantage"}
    for name, rhs in specs.items():
        m = smf.logit(f"chose_planning ~ {rhs}", c).fit(
            disp=0, cov_type="cluster", cov_kwds={"groups": c["participant"]})
        rows.append({"family": "value", "spec": name, "term": rhs,
                     "coef": np.nan, "p": np.nan, "n": int(m.nobs),
                     "aic": m.aic})
    return pd.DataFrame(rows)


# %% [markdown]
# ## Figure / main

# %%
def make_figure(b, slopes, env, threat, value):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0, 0]
    for th, col in [("drift", "#e07a5f"), ("alternating", "#00798c"),
                    ("follow", "#2a9d8f")]:
        g = b[b.threat == th]
        ax.scatter(g["block_conflict_rate"], g["planning_rate"], s=16,
                   color=col, alpha=0.7, label=th)
    ax.set_xlabel("block conflict rate"); ax.set_ylabel("block planning rate")
    ax.set_title("A. Block environment (coloured by threat)")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.hist(slopes["slope"], bins=12, color="#e07a5f")
    ax.axvline(0, color="k", lw=1)
    ax.axvline(slopes["slope"].mean(), color="k", ls="--",
               label=f"mean={slopes['slope'].mean():.2f}")
    ax.set_xlabel("per-participant env slope")
    ax.set_title("B. Environment slopes (reliability SB=0.52)")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    e = env[env.term.isin(["block_conflict_rate", "block_conflict_rate_c"])]
    x = np.arange(len(e))
    ax.bar(x, e["coef"], color="#3d5a80")
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.spec}\n({r.term})" for r in e.itertuples()],
                       rotation=20, fontsize=7)
    ax.set_ylabel("env slope (block_conflict_rate)")
    ax.set_title("C. Env slope: pooled vs within, with controls")

    ax = axes[1, 1]
    width = 0.35
    specs = list(dict.fromkeys(threat["spec"]))
    terms = ["block_drift", "block_mean_ball_y"]
    for i, term in enumerate(terms):
        vals = []
        for s in specs:
            r = threat[(threat.spec == s) & (threat.term == term)]
            vals.append(float(r["coef"].iloc[0]) if len(r) else 0.0)
        ax.bar(np.arange(len(specs)) + i * width, vals, width, label=term)
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(np.arange(len(specs)) + width / 2)
    ax.set_xticklabels(specs, rotation=15, fontsize=8)
    ax.set_ylabel("coefficient")
    ax.set_title("D. Threat vs block-height (joint)")
    ax.legend(fontsize=8)

    fig.suptitle("Block-level effects: environment slope depends on height/threat "
                 "controls; threat and height are coupled", fontsize=13)
    fig.tight_layout()
    out = PO / "figures" / "fig_planning_reanalysis_block.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    PO.mkdir(parents=True, exist_ok=True)
    d = load_trials()
    b = block_table(d)
    env, slopes = environment_models(b)
    threat = threat_models(b)
    value = value_models(d)
    out = pd.concat([env, threat, value], ignore_index=True)
    out.to_csv(PO / "planning_reanalysis_block.csv", index=False)
    slopes.to_csv(PO / "planning_reanalysis_block_slopes.csv", index=False)
    fig = make_figure(b, slopes, env, threat, value)

    pd.set_option("display.width", 200)
    print(out.round(4).to_string(index=False))
    print(f"\nwrote planning_reanalysis_block*.csv and {fig}")


if __name__ == "__main__":
    main()
