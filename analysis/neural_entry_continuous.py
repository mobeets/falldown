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
# # Entry-anchored continuous (ridge) decoding of task values
#
# Choice-anchored ridge decoding (`neural_continuous_decoding.py`) shows
# `conflict_mag` survives full nuisance residualization only in the POST-choice
# window and is ~0 pre-choice. This script repeats that regression on
# **entry-anchored** windows (t = 0 = the entry-hole pass of the 1-2-1
# sequence, never running past the choice pass), to test whether the
# confound-resistant `conflict_mag` signal exists DURING the decision/approach
# interval [entry, choice) — which the choice-anchored `pre [-1000,0]` window
# measured only coarsely (and heterogeneously across conditions).
#
# Windows:
#   entry+250   [0, 250] ms after entry (trials with rt_ms >= 250)
#   entry+500   [0, 500] ms (trials with rt_ms >= 500)
#   approach    [entry, choice) per trial (variable length = rt_ms)
#   (pre-entry  [-500,0] control, rate rep only)
#
# Statistics mirror neural_continuous_decoding.py: ridge (alpha=1.0) on
# standardized features, repeated 10-fold CV (x3), CV Pearson correlation,
# label-shuffle permutation p (150 shuffles). Controls none/target/both.
#
# NOTE on nuisances: for pre-decision windows `ball_time_ms` (choice->exit) is
# a *future* variable the subject could not know during the approach, so
# residualizing the target on it is over-controlling. We report the full
# nuisance set (for direct comparability with the choice-anchored tables) AND
# a reduced set excluding ball_time_ms / trial_duration_ms (which is post-choice
# too). Column `nuisance_set`: "full" | "no_postchoice".
#
# Output: analysis/neural_outputs/<run>/entry_locked_continuous_results.csv
#
# Run with:
#   C:\Users\manik\AppData\Local\Programs\Python\Python311\python.exe analysis\neural_entry_continuous.py

# %%
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from neural_entry_locked import load_entry_binned, rate_features_entry, \
    approach_rate_features
from neural_common import get_run, out_dir

# ---------------------------- Configuration ----------------------------
_RUN = get_run()
OUT_DIR = out_dir(_RUN.run_id)
TRIAL_TABLE = OUT_DIR / "trial_table.csv"
TRIAL_LABELS = OUT_DIR / "trial_labels.csv"

RIDGE_ALPHA = 1.0
N_FOLDS = 10
N_REPEATS = 3
N_PERM = 150
PERM_N_FOLDS = 5
RNG_SEED = 42

FIXED_WINDOWS_MS = [250.0, 500.0]

REGRESSORS = [
    "greedy_gap", "planning_gap", "model_conflict",
    "conflict_mag", "chosen_greedy_cost", "chosen_planning_cost",
]
NUISANCE_FULL = ["rt_ms", "ball_time_ms", "trial_duration_ms",
                 "block_index", "side", "move_dir"]
# rt_ms is partly "time available so far" (legitimate, known during approach);
# ball_time_ms / trial_duration_ms extend past the choice -> excluded here.
NUISANCE_NO_POST = ["rt_ms", "block_index", "side", "move_dir"]
# -----------------------------------------------------------------------


# %%
def load_trial_data():
    z = load_entry_binned()
    tt = pd.read_csv(TRIAL_TABLE)
    labels = pd.read_csv(TRIAL_LABELS)
    timing = tt[["trial_id", "trial_start_ms", "entry_time_ms",
                 "exit_time_ms"]].copy()
    timing["rt_ms"] = tt["choice_time_ms"] - tt["entry_time_ms"]
    timing["ball_time_ms"] = tt["exit_time_ms"] - tt["choice_time_ms"]
    timing["trial_duration_ms"] = tt["exit_time_ms"] - tt["trial_start_ms"]
    lab = labels.merge(timing, on="trial_id", how="left")
    # side / move_dir are derived (not in trial_labels.csv); build them here.
    # labels already carries entry_hole / choice_hole / greedy costs etc.
    lab["side"] = (lab["choice_hole"] >= 6).astype(int)
    lab["move_dir"] = np.sign(lab["choice_hole"] - lab["entry_hole"]).to_numpy()
    return z, lab


# %%
def build_regressors(lab):
    """{regressor: (n_trials,) array} from trial_labels columns."""
    gL = lab["greedy_cost_L"].to_numpy(float)
    gR = lab["greedy_cost_R"].to_numpy(float)
    pL = lab["planning_cost_L"].to_numpy(float)
    pR = lab["planning_cost_R"].to_numpy(float)
    ch = lab["choice_hole"].to_numpy()
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
    return {
        "greedy_gap": greedy_gap,
        "planning_gap": planning_gap,
        "model_conflict": model_conflict,
        "conflict_mag": np.abs(model_conflict),
        "chosen_greedy_cost": chosen_greedy,
        "chosen_planning_cost": chosen_planning,
    }


# %%
def build_nuisance_matrix(lab, cols):
    """(n_trials, n_cols+1) nuisance matrix with intercept column."""
    Xn = lab[cols].to_numpy(float)
    Xn = np.nan_to_num(Xn, nan=0.0)
    return np.column_stack([np.ones(len(Xn)), Xn])


def residualize(target, nuisance, tol=1e-9):
    beta, *_ = np.linalg.lstsq(nuisance, target, rcond=None)
    return target - nuisance @ beta


def residualize_features(X, nuisance):
    out = np.empty_like(X)
    for j in range(X.shape[1]):
        out[:, j] = residualize(X[:, j], nuisance)
    return out


# %%
def cv_regression(y_reg, X, n_folds, n_repeats, seed):
    """Repeated k-fold ridge CV -> (corr_mean, corr_std) and per-fold corrs."""
    corrs = []
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
            if np.std(pred) > 1e-12 and np.std(yte) > 1e-12:
                corrs.append(float(np.corrcoef(yte, pred)[0, 1]))
            else:
                corrs.append(np.nan)
    corrs = np.asarray(corrs)
    return float(np.nanmean(corrs)), float(np.nanstd(corrs)), corrs


def permutation_corr_p(y_reg, X, obs_corr, n_perm, n_folds, seed):
    rng = np.random.default_rng(seed)
    nulls = np.empty(n_perm)
    for i in range(n_perm):
        ysh = rng.permutation(y_reg)
        _, _, corrs = cv_regression(ysh, X, n_folds, 1, seed + i)
        nulls[i] = np.nanmean(corrs)
    return float(np.mean(nulls >= obs_corr))


# %%
def approach_rate_matrix(binned, bin_centers, rt_ms):
    return approach_rate_features(binned, bin_centers, np.asarray(rt_ms))


def fixed_rate_matrix(binned, bin_centers, lo, hi):
    return rate_features_entry(binned, bin_centers, lo, hi)


# %%
def main():
    z, lab = load_trial_data()
    binned = z["binned"]
    bin_centers = z["bin_centers"]
    rt = lab["rt_ms"].to_numpy(float)

    regressors = build_regressors(lab)
    nuis_full = build_nuisance_matrix(lab, NUISANCE_FULL)
    nuis_nopost = build_nuisance_matrix(lab, NUISANCE_NO_POST)

    # precompute fixed-window rate matrices (rate rep only, like the
    # choice-anchored continuous script's headline tables)
    Xwin = {"entry+250": fixed_rate_matrix(binned, bin_centers, 0.0, 250.0),
            "entry+500": fixed_rate_matrix(binned, bin_centers, 0.0, 500.0),
            "approach": approach_rate_matrix(binned, bin_centers, rt)}

    rows = []
    for wname, X in Xwin.items():
        # eligibility: fixed windows require rt >= window length
        if wname.startswith("entry+"):
            W = float(wname.split("+")[1])
            elig = rt >= W
        else:
            elig = np.ones(len(lab), dtype=bool)
        Xe = X[elig]
        ne = int(elig.sum())
        for rname, y_reg in regressors.items():
            m = ~np.isnan(y_reg)
            keep = m & elig
            y = y_reg[keep]
            Xm = Xe[m[elig]]
            nuis_f = nuis_full[keep]
            nuis_np = nuis_nopost[keep]
            for nset, nuis in (("full", nuis_f), ("no_postchoice", nuis_np)):
                for control in ["none", "target", "both"]:
                    Xc = Xm
                    yc = y
                    if control == "target":
                        yc = residualize(y, nuis)
                    elif control == "both":
                        yc = residualize(y, nuis)
                        Xc = residualize_features(Xm, nuis)
                    if np.std(yc) < 1e-12:
                        continue
                    corr_m, corr_s, corrs = cv_regression(
                        yc, Xc, N_FOLDS, N_REPEATS, RNG_SEED)
                    p = permutation_corr_p(yc, Xc, corr_m, N_PERM,
                                           PERM_N_FOLDS, RNG_SEED)
                    rows.append({
                        "regressor": rname, "rep": "rate", "window": wname,
                        "nuisance_set": nset, "control": control,
                        "n_trials": int(len(y)),
                        "corr_mean": corr_m, "corr_std": corr_s, "perm_p": p,
                    })
                    print(f"  {wname}/{rname}/{nset}/{control}: "
                          f"corr {corr_m:+.3f} (p {p:.3f}, n={len(y)})")

    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / "entry_locked_continuous_results.csv", index=False)
    print(f"\nSaved entry_locked_continuous_results.csv ({len(res)} rows)")

    print("\nSummary (approach window, rate rep, conflict_mag):")
    sub = res[(res["regressor"] == "conflict_mag")
              & (res["window"] == "approach")]
    print(sub.pivot_table(index="nuisance_set", columns="control",
                          values="corr_mean").round(3).to_string())


# %%
if __name__ == "__main__":
    main()

# %%
