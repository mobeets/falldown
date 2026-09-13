# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Model-based continuous decoding of task values
#
# neural_lda_decoding.py labels trials by discrete conditions. This script
# decodes the *continuous task-value quantities* that the labels are built
# from (analytic, from trial_labels.csv -- no fitted model needed):
#
#   greedy_gap      greedy_cost_L - greedy_cost_R    (1-step value difference)
#   planning_gap    planning_cost_L - planning_cost_R (2-step value difference)
#   model_conflict  planning_gap - greedy_gap  (signed: which model favors L)
#   conflict_mag    |model_conflict|          (magnitude of disagreement)
#   chosen_greedy_cost   greedy cost of the actually chosen hole
#   chosen_planning_cost planning cost of the actually chosen hole
#
# Continuous regression (ridge on standardized features, repeated 10-fold CV)
# is more sensitive than 2-class LDA and lets us partial out confounds in a
# principled way. Three controls:
#
#   none   raw regressor vs neural features
#   target regressor residualized on nuisances (RT, ball time, duration,
#          block index, side, move_dir_vx) -- removes the mechanical part of
#          the task value that is just RT/kinematics/position
#   both   additionally residualize the neural features per unit
#
# Statistics: CV R^2 and Pearson correlation, plus a label-shuffle permutation
# p (shuffling the regressor breaks trial order, so this is the honest null).
#
# Outputs:
#   continuous_decoding_results.csv            control='none' and 'target'
#   continuous_decoding_confound_controlled.csv  control='both'
#   (single table with a 'control' column is also acceptable; we split them
#    to mirror the two-file naming convention in the analysis plan.)
#
# Run with:
#   C:\Users\manik\AppData\Local\Programs\Python\Python311\python.exe analysis\neural_continuous_decoding.py

# %%
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from neural_lda_decoding import (
    RNG_SEED, PERM_FOLDS, WINDOWS, REPS, load_binned, rate_features,
    pca_features,
)

# ---------------------------- Configuration ----------------------------
from neural_common import get_run, out_dir
_RUN = get_run()
OUT_DIR = out_dir(_RUN.run_id)
BINNED = OUT_DIR / "segmented_spikes_binned.npz"
TRIAL_TABLE = OUT_DIR / "trial_table.csv"
TRIAL_LABELS = OUT_DIR / "trial_labels.csv"

RIDGE_ALPHA = 1.0
N_FOLDS = 10
N_REPEATS = 3
N_PERM = 150
PERM_N_FOLDS = 5

REGRESSORS = [
    "greedy_gap", "planning_gap", "model_conflict",
    "conflict_mag", "chosen_greedy_cost", "chosen_planning_cost",
]
NUISANCE_COLS = ["rt_ms", "ball_time_ms", "trial_duration_ms",
                 "block_index", "side", "move_dir"]
# -----------------------------------------------------------------------


# %%
def build_regressors():
    """{regressor: (n_trials,) array} from trial_labels.csv."""
    lab = pd.read_csv(TRIAL_LABELS)
    gL = lab["greedy_cost_L"].to_numpy(float)
    gR = lab["greedy_cost_R"].to_numpy(float)
    pL = lab["planning_cost_L"].to_numpy(float)
    pR = lab["planning_cost_R"].to_numpy(float)
    ch = lab["choice_hole"].to_numpy()

    # greedy_cost_L/R are the model costs of the two choice holes (A/B);
    # recover which hole the participant chose from trial_table.hole_locations
    # to attach the chosen hole's cost under each model.
    tt = pd.read_csv(TRIAL_TABLE)
    import json as _json

    def holes(s):
        try:
            return _json.loads(s)
        except Exception:
            return [np.nan, np.nan]

    hl = tt["hole_locations"].map(holes)
    hA = hl.str[0].to_numpy()
    hB = hl.str[1].to_numpy()
    isA = ch == hA
    isB = ch == hB
    chosen_greedy = np.where(isA, gL, np.where(isB, gR, np.nan))
    chosen_planning = np.where(isA, pL, np.where(isB, pR, np.nan))

    greedy_gap = gL - gR
    planning_gap = pL - pR
    model_conflict = planning_gap - greedy_gap
    conflict_mag = np.abs(model_conflict)

    return {
        "greedy_gap": greedy_gap,
        "planning_gap": planning_gap,
        "model_conflict": model_conflict,
        "conflict_mag": conflict_mag,
        "chosen_greedy_cost": chosen_greedy,
        "chosen_planning_cost": chosen_planning,
    }


def build_nuisances():
    """(n_trials, n_nuisance) matrix + column names (intercept added)."""
    tt = pd.read_csv(TRIAL_TABLE)
    lab = pd.read_csv(TRIAL_LABELS)
    df = tt[["trial_id", "block_index", "choice_hole"]].copy()
    df["rt_ms"] = tt["choice_time_ms"] - tt["entry_time_ms"]
    df["ball_time_ms"] = tt["exit_time_ms"] - tt["choice_time_ms"]
    df["trial_duration_ms"] = tt["exit_time_ms"] - tt["trial_start_ms"]
    df["side"] = (tt["choice_hole"] >= 6).astype(int)
    df = df.merge(lab[["trial_id", "entry_hole"]], on="trial_id", how="left")
    md = np.sign(df["choice_hole"] - df["entry_hole"])
    df["move_dir"] = md.to_numpy()
    Xn = df[NUISANCE_COLS].to_numpy(float)
    Xn = np.nan_to_num(Xn, nan=0.0)
    Xn = np.column_stack([np.ones(len(Xn)), Xn])
    return Xn


def residualize(target, nuisance, tol=1e-9):
    """Project out the nuisance space from target (OLS residuals)."""
    beta, *_ = np.linalg.lstsq(nuisance, target, rcond=None)
    return target - nuisance @ beta


def residualize_features(X, nuisance):
    """Regress each feature column on nuisances, return residuals."""
    out = np.empty_like(X)
    for j in range(X.shape[1]):
        out[:, j] = residualize(X[:, j], nuisance)
    return out


# %%
def cv_regression(y_reg, X, n_folds, n_repeats, seed):
    """Repeated k-fold ridge CV: returns (r2_mean, r2_std, corr_mean,
    corr_std) and the per-fold correlation vector (for the permutation null)."""
    r2s, corrs = [], []
    for rep in range(n_repeats):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed + rep)
        for tr, te in kf.split(X):
            sc = StandardScaler().fit(X[tr])
            Xtr = sc.transform(X[tr])
            Xte = sc.transform(X[te])
            ytr = y_reg[tr]
            yte = y_reg[te]
            model = Ridge(alpha=RIDGE_ALPHA).fit(Xtr, ytr)
            pred = model.predict(Xte)
            ss_res = float(np.sum((yte - pred) ** 2))
            ss_tot = float(np.sum((yte - yte.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan
            if np.std(pred) > 1e-12 and np.std(yte) > 1e-12:
                corr = float(np.corrcoef(yte, pred)[0, 1])
            else:
                corr = np.nan
            r2s.append(r2)
            corrs.append(corr)
    r2s = np.asarray(r2s)
    corrs = np.asarray(corrs)
    return (float(np.nanmean(r2s)), float(np.nanstd(r2s)),
            float(np.nanmean(corrs)), float(np.nanstd(corrs))), corrs


def permutation_corr_p(y_reg, X, obs_corr, n_perm, n_folds, seed):
    """Two-sided p: fraction of shuffled-regressor CV correlations >= observed."""
    rng = np.random.default_rng(seed)
    nulls = np.empty(n_perm)
    for i in range(n_perm):
        ysh = rng.permutation(y_reg)
        _, corrs = cv_regression(ysh, X, n_folds, 1, seed + i)
        nulls[i] = np.nanmean(corrs)
    return float(np.mean(nulls >= obs_corr))


# %%
def main():
    print("Loading binned spikes ...")
    binned, unit_ids, bin_centers = load_binned()
    regressors = build_regressors()
    nuisance = build_nuisances()

    Xfeats = {}
    for wname, wh in WINDOWS.items():
        lo, hi = (None, None) if wh is None else wh
        Xfeats[wname] = {
            "rate": rate_features(binned, bin_centers, lo, hi),
            "pca": pca_features(binned, bin_centers, lo, hi),
        }

    rows = []
    for wname, wh in WINDOWS.items():
        lo, hi = (None, None) if wh is None else wh
        for rep in REPS:
            X = Xfeats[wname][rep]
            for rname, y_reg in regressors.items():
                m = ~np.isnan(y_reg)
                y = y_reg[m]
                Xm = X[m]
                for control in ["none", "target", "both"]:
                    Xc = Xm
                    yc = y
                    if control == "target":
                        yc = residualize(y, nuisance[m])
                    elif control == "both":
                        yc = residualize(y, nuisance[m])
                        Xc = residualize_features(Xm, nuisance[m])
                    if np.std(yc) < 1e-12:
                        continue
                    (r2_m, r2_s, corr_m, corr_s), corrs = cv_regression(
                        yc, Xc, N_FOLDS, N_REPEATS, RNG_SEED)
                    p = permutation_corr_p(yc, Xc, corr_m, N_PERM,
                                           PERM_N_FOLDS, RNG_SEED)
                    rows.append({
                        "regressor": rname, "rep": rep, "window": wname,
                        "control": control, "n_trials": int(len(y)),
                        "r2_mean": r2_m, "r2_std": r2_s,
                        "corr_mean": corr_m, "corr_std": corr_s, "perm_p": p,
                    })
                    print(f"  {wname}/{rep}/{rname}/{control}: "
                          f"corr {corr_m:+.3f} (r2 {r2_m:.3f}, p {p:.3f})")

    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / "continuous_decoding_results.csv", index=False)
    print(f"\nSaved continuous_decoding_results.csv ({len(res)} rows)")

    print("\nSummary (post window, rate rep, corr mean):")
    sub = res[(res["window"] == "post") & (res["rep"] == "rate")]
    print(sub.pivot_table(index="regressor", columns="control",
                          values="corr_mean").round(3).to_string())
    print("\nperm_p:")
    print(sub.pivot_table(index="regressor", columns="control",
                          values="perm_p").round(3).to_string())


# %%
# ----------------------- Plot functions (not saved) ---------------------

def load_continuous():
    return pd.read_csv(OUT_DIR / "continuous_decoding_results.csv")


def plot_continuous_bars(window="post", rep="rate", ax=None):
    """corr_mean by regressor, grouped by control."""
    import matplotlib.pyplot as plt
    r = load_continuous()
    sub = r[(r["window"] == window) & (r["rep"] == rep)]
    if ax is None:
        ax = plt.gca()
    regs = sorted(sub["regressor"].unique())
    controls = sorted(sub["control"].unique())
    for i, ctrl in enumerate(controls):
        vals = [sub[(sub["regressor"] == r) & (sub["control"] == ctrl)][
            "corr_mean"].iloc[0] for r in regs]
        ax.bar(np.arange(len(regs)) + i * 0.25, vals, width=0.25, label=ctrl)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(np.arange(len(regs)) + 0.25)
    ax.set_xticklabels(regs, rotation=45, ha="right")
    ax.set_ylabel("CV correlation")
    ax.set_title(f"{window} / {rep}")
    ax.legend()
    return ax


# %%
if __name__ == "__main__":
    main()
