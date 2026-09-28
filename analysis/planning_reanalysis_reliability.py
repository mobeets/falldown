# %% [markdown]
# # Reliability / level audit of behavioural planning (claims 1 and 6)
#
# "Planning is graded and individual, summarised by P(plan|conflict) and the
# model-based planning weight" mixes two things: how much of the choice is a
# stable participant trait versus within-person (block/trial) variation. This
# script quantifies both.
#
#   * ICC(1) and an LPM variance partition (participant vs block vs residual)
#     for `chose_planning`.
#   * Split-half reliability (odd/even trials, Spearman-Brown) of the
#     participant-level summaries: P(plan|conflict), planning weight,
#     slope_ball_y, and the environment slope.
#
# Outputs (analysis/planning_outputs/):
#   planning_reanalysis_reliability.csv
#   figures/fig_planning_reanalysis_reliability.png
#
# Run with:
#   python analysis/planning_reanalysis_reliability.py

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
C_RAW = "#e07a5f"


def load_trials():
    d = pd.read_csv(XD / "cross_dataset_trials.csv")
    d = d[d["is_experimental"]].copy()
    d["conflict"] = d["conflict"].fillna(0).astype(int)
    return d


def odd_even(df, by=("block_number", "sequence_index")):
    """Return boolean parity label for rows ordered within participant."""
    d = df.sort_values(list(by)).copy()
    d["_parity"] = d.groupby("participant").cumcount() % 2
    return d


# %% [markdown]
# ## ICC and variance partition

# %%
def icc_participant(df, outcome="chose_planning"):
    g = df.groupby("participant")[outcome]
    sizes = g.size()
    means = g.mean()
    grand = df[outcome].mean()
    k = sizes.mean()
    ssb = float((sizes * (means - grand) ** 2).sum())
    ssw = float(g.apply(lambda s: ((s - s.mean()) ** 2).sum()).sum())
    dfb = len(means) - 1
    dfw = len(df) - len(means)
    msb, msw = ssb / dfb, ssw / dfw
    icc = (msb - msw) / (msb + (k - 1) * msw)
    return {"icc_participant": icc, "n_participants": len(means),
            "n_trials": len(df), "ms_between": msb, "ms_within": msw}


def lpm_partition(df, outcome="chose_planning"):
    d = df.dropna(subset=[outcome]).copy()
    y = d[outcome].astype(float).to_numpy()

    def r2(f):
        m = smf.ols(f, data=d).fit()
        return m.rsquared

    r0 = 0.0
    r1 = r2(f"{outcome} ~ C(participant)")
    r2b = r2(f"{outcome} ~ C(participant) + C(participant) : C(block_number)")
    total = r2b - r0
    return {"participant": r1, "block": r2b - r1,
            "residual": 1 - r2b,
            "share_participant": (r1) / total if total > 0 else np.nan,
            "share_block": (r2b - r1) / total if total > 0 else np.nan,
            "total_explained": total}


# %% [markdown]
# ## Split-half reliability

# %%
def plan_rate(d):
    c = d[d.conflict == 1]
    return float(c["chose_planning"].mean()) if len(c) else np.nan


def planning_weight(d):
    d = d.dropna(subset=["diff_1step", "diff_planning", "chosen_left"]).copy()
    if len(d) < 20 or d["chosen_left"].nunique() < 2:
        return np.nan
    d["_z1"] = (d.diff_1step - d.diff_1step.mean()) / d.diff_1step.std()
    d["_zp"] = (d.diff_planning - d.diff_planning.mean()) / d.diff_planning.std()
    try:
        m = smf.logit("chosen_left ~ _z1 + _zp", data=d).fit(disp=0)
    except Exception:
        return np.nan
    den = abs(m.params["_z1"]) + abs(m.params["_zp"])
    return abs(m.params["_zp"]) / den if den > 1e-9 else np.nan


def ball_y_slope(d):
    c = d[d.conflict == 1].dropna(subset=["ball_y_at_top", "chose_planning"])
    if len(c) < 20 or c["chose_planning"].nunique() < 2:
        return np.nan
    try:
        m = smf.logit("chose_planning ~ ball_y_at_top", data=c).fit(disp=0)
        return float(m.params["ball_y_at_top"])
    except Exception:
        return np.nan


def env_slope(d):
    d = d[d.conflict == 1].copy()
    blk = d.groupby(["participant", "block_number"]).agg(
        pr=("chose_planning", "mean"),
        cr=("block_conflict_rate", "first")).reset_index()
    if len(blk) < 3:
        return np.nan
    return float(np.polyfit(blk["cr"], blk["pr"], 1)[0])


def split_half(df, measure, kind):
    """Per-participant split-half estimates for a participant-level measure."""
    if kind == "blocks":
        out = []
        for pid, g in df.groupby("participant"):
            blocks = (g[["block_number", "block_conflict_rate"]]
                      .drop_duplicates("block_number")
                      .sort_values("block_number").reset_index(drop=True))
            blocks["_p"] = blocks.index % 2
            for p in (0, 1):
                b = set(blocks.loc[blocks._p == p, "block_number"])
                out.append({"participant": pid, "half": p,
                            "value": measure(g[g.block_number.isin(b)])})
        return pd.DataFrame(out)
    d = odd_even(df)
    rows = []
    for pid, g in d.groupby("participant"):
        for p in (0, 1):
            rows.append({"participant": pid, "half": p,
                         "value": measure(g[g._parity == p])})
    return pd.DataFrame(rows)


def reliability(sh, name):
    wide = sh.pivot(index="participant", columns="half",
                    values="value").dropna()
    if len(wide) < 5:
        return {"measure": name, "n_participants": len(wide), "r": np.nan,
                "spearman_brown": np.nan, "pearson_r": np.nan}
    r, p = stats.pearsonr(wide[0], wide[1])
    rho, _ = stats.spearmanr(wide[0], wide[1])
    sb = 2 * r / (1 + r) if r > -1 else np.nan
    return {"measure": name, "n_participants": len(wide),
            "r": r, "r_p": p, "spearman_brown": sb, "pearson_r": rho}


# %% [markdown]
# ## Figure / main

# %%
def make_figure(wide_measures, rel, icc, part):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, (name, wide) in zip(axes.ravel(), wide_measures.items()):
        ax.scatter(wide[0], wide[1], color=C_RAW, s=30)
        lo = float(wide.min().min()); hi = float(wide.max().max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        row = rel[rel.measure == name].iloc[0]
        ax.set_xlabel(f"{name} (odd trials)")
        ax.set_ylabel(f"{name} (even trials)")
        ax.set_title(f"{name}: r={row['r']:.2f}, "
                     f"Spearman-Brown={row['spearman_brown']:.2f}")
    fig.suptitle(
        f"Split-half reliability of participant-level planning summaries\n"
        f"choice ICC(participant)={icc['icc_participant']:.2f}; "
        f"LPM share participant={part['share_participant']:.2f} / "
        f"block={part['share_block']:.2f}", fontsize=13)
    fig.tight_layout()
    out = PO / "figures" / "fig_planning_reanalysis_reliability.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    PO.mkdir(parents=True, exist_ok=True)
    d = load_trials()
    cloud = d[d.source == "cloud"].copy()

    icc = icc_participant(cloud)
    part = lpm_partition(cloud)
    print("ICC(participant) for chose_planning:", round(icc["icc_participant"], 3))
    print("LPM partition:", {k: round(v, 3) for k, v in part.items()
                             if k.startswith("share") or k == "total_explained"})

    specs = {"p_plan": (plan_rate, "trials", cloud),
             "planning_weight": (planning_weight, "trials", cloud),
             "slope_ball_y": (ball_y_slope, "trials", cloud),
             "env_slope": (env_slope, "blocks", cloud)}
    rel_rows, wide_measures = [], {}
    for name, (fn, kind, data) in specs.items():
        sh = split_half(data, fn, kind)
        wide = sh.pivot(index="participant", columns="half", values="value").dropna()
        wide_measures[name] = wide
        rel_rows.append(reliability(sh, name))
    rel = pd.DataFrame(rel_rows)

    summary = pd.DataFrame([{**icc, **{k: v for k, v in part.items()}}])
    summary.to_csv(PO / "planning_reanalysis_reliability_summary.csv", index=False)
    rel.to_csv(PO / "planning_reanalysis_reliability.csv", index=False)
    fig = make_figure(wide_measures, rel, icc, part)

    pd.set_option("display.width", 200)
    print("\nSplit-half reliability:")
    print(rel.round(3).to_string(index=False))
    print(f"\nwrote planning_reanalysis_reliability*.csv and {fig}")


if __name__ == "__main__":
    main()
