# %% [markdown]
# # Level decomposition of planning behaviour (within / block / participant)
#
# Where does each effect on planning live? This script splits every candidate
# variable into three nested components and estimates them jointly in the
# Mundlak / between-within sense:
#
#   * `_w`  within-participant, within-block trial deviation  (trial level)
#   * `_b`  block mean within participant                      (block level)
#   * `_p`  participant mean                                   (between level)
#
# Fitting `chose_planning ~ term_w + term_b + term_p` separates a variable's
# trial-level action from block/participant-level structure (Simpson's paradox),
# and a nested pseudo-R2 partition attributes variance to each level.
#
# Variable sets:
#   * **Geometry (value framing):** `plan_advantage` (+), `greedy_advantage` (-).
#   * **Geometry (choice-independent):** `conflict_mag` standalone. NOTE
#     `conflict_mag = plan_advantage + greedy_advantage` exactly, so the three
#     are one redundant triple -- never fit all three together (rank 2 of 3).
#   * **Context:** `ball_y_at_top`, `block_conflict_rate`, `block_plan_advantage`,
#     `block_drift` (block/participant only where constant within a block).
#
# A secondary decomposition of the canonical choice model
# (`chosen_left ~ diff_1step + diff_planning`, all trials) reports the normalized
# planning weight at each level.
#
# Outputs (analysis/planning_outputs/):
#   level_decomposition_coefs.csv           variable x level x model x dataset
#   level_decomposition_variance.csv        nested pseudo-R2 increments
#   level_decomposition_planning_weight.csv planning weight by level
#   figures/fig_level_decomposition.png
#
# Run with:
#   python analysis/level_decomposition.py
# (env: bash analysis/setup_env.sh)

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

warnings.filterwarnings("ignore")

_REPO = Path(__file__).resolve().parent.parent
XD = _REPO / "analysis" / "cross_dataset_outputs"
PO = _REPO / "analysis" / "planning_outputs"

LEVELS = ("w", "b", "p")
C_W = "#2a9d8f"
C_B = "#e07a5f"
C_P = "#3d5a80"


# %% [markdown]
# ## Data and level columns

# %%
def load_trials():
    d = pd.read_csv(XD / "cross_dataset_trials.csv")
    d = d[d["is_experimental"]].copy()
    d["conflict"] = d["conflict"].fillna(0).astype(int)
    d["conflict_mag"] = (d["diff_planning"] - d["diff_1step"]).abs()
    return d


def add_levels(df, cols):
    df = df.copy()
    for c in cols:
        by_block = df.groupby(["participant", "block_number"])[c]
        df[c + "_w"] = df[c] - by_block.transform("mean")
        df[c + "_b"] = by_block.transform("mean")
        df[c + "_p"] = df.groupby("participant")[c].transform("mean")
    return df


def geometry_columns():
    return ["conflict_mag", "plan_advantage", "greedy_advantage"]


def context_columns():
    return ["ball_y_at_top", "block_conflict_rate", "block_plan_advantage",
            "block_drift"]


def check_redundancy(df):
    """Verify conflict_mag == plan_advantage + greedy_advantage and rank 2/3."""
    x = df.dropna(subset=geometry_columns())
    resid = (x["conflict_mag"]
             - x["plan_advantage"] - x["greedy_advantage"]).abs().max()
    mat = x[geometry_columns()].to_numpy(float)
    rank = int(np.linalg.matrix_rank(mat - mat.mean(0)))
    return {"identity_max_abs_resid": float(resid), "rank": rank,
            "collinear": bool(resid < 1e-6 and rank < 3)}


# %% [markdown]
# ## Mundlak fit and variance partition

# %%
def _terms_for(name, levels, df):
    terms = []
    for lv in levels:
        col = f"{name}_{lv}"
        if col not in df.columns:
            continue
        if lv == "w" and df[col].std() < 1e-9:  # constant within block
            continue
        terms.append(col)
    return terms


def fit_mundlak(df, predictors, outcome, groups="participant"):
    """GEE-logit (fallback cluster-robust logit) on level-decomposed terms.

    `predictors` is a list of (name, levels) where levels is a subset of
    ("w", "b", "p").
    """
    terms = [t for name, lv in predictors for t in _terms_for(name, lv, df)]
    data = df.dropna(subset=terms + [outcome]).copy()
    formula = f"{outcome} ~ " + (" + ".join(terms) if terms else "1")
    try:
        m = smf.gee(formula, groups=groups, data=data,
                    family=sm.families.Binomial(),
                    cov_struct=sm.cov_struct.Exchangeable()).fit()
        kind = "gee"
    except Exception:
        try:
            m = smf.logit(formula, data=data).fit(
                disp=0, cov_type="cluster", cov_kwds={"groups": data[groups]})
            kind = "logit-cluster"
        except Exception:
            return None, None, None
    return m, kind, terms


def coef_rows(df, predictors, outcome, model, dataset, pooled=False):
    rows = []
    if pooled:
        for name, _ in predictors:
            data = df.dropna(subset=[name, outcome])
            try:
                mp = smf.logit(f"{outcome} ~ {name}", data=data).fit(
                    disp=0, cov_type="cluster",
                    cov_kwds={"groups": data["participant"]})
                kind = "logit-cluster"
            except Exception:
                mp, kind = None, "fail"
            rows.append({
                "dataset": dataset, "model": model, "variable": name,
                "level": "pooled", "n": int(len(data)),
                "coef": mp.params.get(name, np.nan) if mp is not None else np.nan,
                "se": mp.bse.get(name, np.nan) if mp is not None else np.nan,
                "p": mp.pvalues.get(name, np.nan) if mp is not None else np.nan,
                "fit": kind})
        return rows
    m, kind, terms = fit_mundlak(df, predictors, outcome)
    if m is None:
        return rows
    nobs = int(getattr(m, "nobs", len(df)))
    for name, levels in predictors:
        for lv in levels:
            col = f"{name}_{lv}"
            if col not in terms:
                continue
            rows.append({
                "dataset": dataset, "model": model, "variable": name,
                "level": lv, "n": nobs,
                "coef": m.params.get(col, np.nan),
                "se": m.bse.get(col, np.nan),
                "p": m.pvalues.get(col, np.nan), "fit": kind})
    return rows


def variance_partition(df, predictors, outcome, dataset, model):
    """Nested linear-probability R2 increments: participant -> block -> trial.

    OLS (linear-probability) is used rather than a likelihood R2 because the
    block/participant mean columns are collinear and logit R2 increments can be
    non-monotone/unstable; the LPM partition is monotone and is an exact
    variance decomposition of the binary outcome.
    """
    names = [n for n, _ in predictors]
    need = [f"{n}_{lv}" for n in names for lv in LEVELS
            if f"{n}_{lv}" in df.columns] + [outcome]
    data = df.dropna(subset=need).copy()
    if len(data) < 50 or data[outcome].nunique() < 2:
        return []
    y = data[outcome].astype(float).to_numpy()

    def r2(terms):
        if not terms:
            return 0.0
        X = sm.add_constant(data[terms].astype(float).to_numpy())
        return float(sm.OLS(y, X).fit().rsquared)

    P = [f"{n}_p" for n in names if f"{n}_p" in data.columns]
    B = [f"{n}_b" for n in names if f"{n}_b" in data.columns
         and data[f"{n}_b"].std() > 1e-9]
    W = [f"{n}_w" for n in names if f"{n}_w" in data.columns
         and data[f"{n}_w"].std() > 1e-9]
    r1, r2b, r3 = r2(P), r2(P + B), r2(P + B + W)
    inc = {"participant": r1, "block": r2b - r1, "trial": r3 - r2b}
    total = r3
    return [{"dataset": dataset, "model": model, "level_init": lvl,
             "delta_r2": inc[lvl],
             "share_of_explained": inc[lvl] / total if total > 1e-9 else np.nan,
             "cumulative_r2": sum(inc[k] for k in list(inc)[:i + 1]),
             "total_r2": total, "r2_type": "lpm"}
            for i, lvl in enumerate(inc)]


# %% [markdown]
# ## Planning-weight decomposition (secondary)

# %%
def planning_weight_by_level(d, dataset):
    """Canonical choice model `chosen_left ~ diff_1step + diff_planning` split
    by level; normalized planning weight per level."""
    data = d.dropna(subset=["diff_1step", "diff_planning", "chosen_left"]).copy()
    data = add_levels(data, ["diff_1step", "diff_planning"])
    data["chosen_left"] = data["chosen_left"].astype(int)
    predictors = [("diff_1step", list(LEVELS)), ("diff_planning", list(LEVELS))]
    m, kind, terms = fit_mundlak(data, predictors, "chosen_left")
    rows = []
    if m is None:
        return rows
    for lv in LEVELS:
        c1 = m.params.get(f"diff_1step_{lv}", np.nan)
        cp = m.params.get(f"diff_planning_{lv}", np.nan)
        den = abs(c1) + abs(cp)
        rows.append({"dataset": dataset, "level": lv,
                     "beta_1step": c1, "beta_planning": cp,
                     "planning_weight": abs(cp) / den if den > 1e-9 else np.nan,
                     "n": int(m.nobs), "fit": kind})
    return rows


# %% [markdown]
# ## Figure

# %%
def make_figure(coefs, var, weight):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    def dot(ax, sub, ylabel, title):
        y = np.arange(len(sub))
        colors = [C_W if r.level == "w" else C_B if r.level == "b"
                  else C_P for r in sub.itertuples()]
        ax.errorbar(sub["coef"], y, xerr=1.96 * sub["se"], fmt="o",
                    color="k", ecolor="0.6", capsize=3, ms=5)
        ax.scatter(sub["coef"], y, c=colors, zorder=3, s=45)
        ax.axvline(0, color="k", lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{r.variable} [{r.level}]" for r in sub.itertuples()],
                           fontsize=8)
        ax.set_xlabel("coefficient (logit)")
        ax.set_title(title, fontsize=11)

    geo = coefs[(coefs.dataset == "cloud") &
                (coefs.model.isin(["geometry_value", "geometry_conflict"])) &
                (coefs.level != "pooled")]
    dot(axes[0, 0], geo.reset_index(drop=True),
        "geometry", "A. Geometry, all levels\n(green=within, orange=block, blue=between)")

    ctx = coefs[(coefs.dataset == "cloud") & (coefs.model == "context") &
                (coefs.level != "pooled")]
    dot(axes[0, 1], ctx.reset_index(drop=True),
        "context", "B. Context, all levels")

    ax = axes[1, 0]
    v = var[var.dataset == "cloud"]
    models = v["model"].unique()
    bottom = np.zeros(len(models))
    for lvl, col in [("participant", C_P), ("block", C_B), ("trial", C_W)]:
        vals = [float(v[(v.model == m) & (v.level_init == lvl)]["delta_r2"].iloc[0])
                if len(v[(v.model == m) & (v.level_init == lvl)]) else 0
                for m in models]
        ax.bar(range(len(models)), vals, bottom=bottom, color=col, label=lvl)
        bottom += np.array(vals)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=15, fontsize=8)
    ax.set_ylabel("McFadden pseudo-R2")
    ax.set_title("C. Variance partition by level (cloud)")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    w = weight[weight.dataset == "cloud"]
    x = np.arange(len(w))
    ax.bar(x, w["planning_weight"], color=[C_W if l == "w" else C_B if l == "b"
                                           else C_P for l in w["level"]])
    ax.set_xticks(x); ax.set_xticklabels(w["level"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("planning weight  |b_plan|/(|b_plan|+|b_1step|)")
    ax.set_title("D. Planning weight by level (cloud, all choice trials)")
    for xi, r in zip(x, w.itertuples()):
        ax.text(xi, r.planning_weight + 0.02, f"{r.planning_weight:.2f}",
                ha="center", fontsize=8)

    fig.suptitle("Level decomposition: geometry acts trial-level; "
                 "context (ball height) is block/participant-level", fontsize=14)
    fig.tight_layout()
    out = PO / "figures" / "fig_level_decomposition.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


# %% [markdown]
# ## Main

# %%
def predictors_value():
    return [("plan_advantage", list(LEVELS)), ("greedy_advantage", list(LEVELS))]


def predictors_conflict():
    return [("conflict_mag", list(LEVELS))]


def predictors_context(dataset, df):
    preds = [("ball_y_at_top", list(LEVELS)),
             ("block_conflict_rate", ["b", "p"]),
             ("block_plan_advantage", ["b", "p"])]
    if df["block_drift"].nunique() > 1:
        preds.append(("block_drift", ["b", "p"]))
    return preds


def analyse(df, dataset):
    """Fit the three geometry/context models and variance partitions."""
    d = add_levels(df, geometry_columns() + context_columns())
    models = {
        "geometry_value": predictors_value(),
        "geometry_conflict": predictors_conflict(),
        "context": predictors_context(dataset, d),
    }
    coefs, vparts = [], []
    for model, preds in models.items():
        coefs += coef_rows(d, preds, "chose_planning", model, dataset)
        vparts += variance_partition(d, preds, "chose_planning", dataset, model)
    # pooled single-variable models for the classic geometry terms
    for name in ["conflict_mag", "plan_advantage", "greedy_advantage",
                 "ball_y_at_top"]:
        coefs += coef_rows(d, [(name, [])], "chose_planning", "pooled_single",
                           dataset, pooled=True)
    return coefs, vparts


def main():
    PO.mkdir(parents=True, exist_ok=True)
    d = load_trials()
    conflict = d[d["conflict"] == 1].copy()

    red_cloud = check_redundancy(conflict[conflict.source == "cloud"])
    red_emu = check_redundancy(conflict[conflict.source == "emu"])
    print("redundancy check (conflict_mag == plan_advantage + greedy_advantage):")
    print("  cloud:", red_cloud)
    print("  EMU  :", red_emu)
    assert red_cloud["collinear"], "cloud geometry triple is not collinear!"

    coefs, vparts = [], []
    for dataset in ["cloud", "emu"]:
        c, v = analyse(conflict[conflict.source == dataset], dataset)
        coefs += c
        vparts += v

    weight = []
    for dataset in ["cloud", "emu"]:
        weight += planning_weight_by_level(d[d.source == dataset], dataset)

    coefs = pd.DataFrame(coefs)
    coefs["unstable"] = ((coefs["coef"].abs() > 3) | (coefs["se"] > 3)
                         | coefs["p"].isna())
    vparts = pd.DataFrame(vparts)
    weight = pd.DataFrame(weight)

    coefs.to_csv(PO / "level_decomposition_coefs.csv", index=False)
    vparts.to_csv(PO / "level_decomposition_variance.csv", index=False)
    weight.to_csv(PO / "level_decomposition_planning_weight.csv", index=False)
    fig = make_figure(coefs, vparts, weight)

    pd.set_option("display.width", 220)
    print("\n=== cloud coefficients (within/block/between) ===")
    print(coefs[(coefs.dataset == "cloud") & (coefs.level != "pooled")]
          .round(4).to_string(index=False))
    print("\n=== variance partition (cloud) ===")
    print(vparts[vparts.dataset == "cloud"].round(4).to_string(index=False))
    print("\n=== planning weight by level (cloud) ===")
    print(weight[weight.dataset == "cloud"].round(3).to_string(index=False))
    print(f"\nwrote level_decomposition_*.csv and {fig}")


if __name__ == "__main__":
    main()
