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
# # Bursty-unit sensitivity of the continuous decoding results
#
# Perkel, Gerstein & Moore (1967) define the coefficient of variation of the
# interspike-interval distribution, CV = SD(ISI)/mean(ISI), as the basic
# measure of a unit's firing regularity (pacemaker ~0, Poisson ~1, bursty > 1).
# The decoding pipeline's unit QC (`segment_trials.load_units`) keeps every
# unit with firing_rate_hz >= 0.5 and never inspects ISI structure, so units
# with CV > 1.3 (bursty / irregular) are included as features even though their
# per-trial rate estimates are noisier than a Poisson assumption implies.
#
# This script is the sensitivity check that question calls for: recompute CV
# from the raw spike times, drop the bursty units, and re-run the *headline*
# continuous-decoding result (`conflict_mag`, which survives full nuisance
# residualization in `neural_continuous_decoding.py`) on the reduced set. It is
# deliberately self-contained — it re-derives the rate matrix, the regressors,
# the nuisances, and its own ridge CV from the on-disk outputs, so re-running
# this one file alone reproduces the comparison without depending on any
# precomputed intermediate.
#
# Outputs (analysis/neural_outputs/):
#   bursty_unit_cv_screen.csv             per-unit CV, n spikes, flag
#   bursty_sensitivity_conflict_mag.csv   conflict_mag decode, all vs no-bursty
#   bursty_sensitivity_decoding.csv       same for every continuous regressor
#
# Run with:
#   C:\Users\manik\AppData\Local\Programs\Python\Python311\python.exe analysis\neural_bursty_unit_sensitivity.py

# %%
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# ---------------------------- Configuration ----------------------------
from neural_common import get_run, out_dir
_RUN = get_run()
OUT_DIR = out_dir(_RUN.run_id)
BINNED = OUT_DIR / "segmented_spikes_binned.npz"
SPIKES_UNITS = OUT_DIR / "spikes_units.csv"
UNIT_META = OUT_DIR / "unit_metadata.csv"
TRIAL_TABLE = OUT_DIR / "trial_table.csv"
TRIAL_LABELS = OUT_DIR / "trial_labels.csv"

CV_THRESHOLD = 1.3          # units with CV above this are "bursty"
MIN_ISI_N = 50              # units with fewer ISIs get CV = NaN (not flagged)
WINDOW = "post"             # [0, +1000] ms relative to choice
REPS = ["rate", "pca"]      # per-unit firing-rate features / PCA-reduced counts
PCA_COMPONENTS = 30

RIDGE_ALPHA = 1.0
N_FOLDS = 10
N_REPEATS = 3
N_PERM = 150
PERM_N_FOLDS = 5
RNG_SEED = 42

REGRESSORS = [
    "greedy_gap", "planning_gap", "model_conflict",
    "conflict_mag", "chosen_greedy_cost", "chosen_planning_cost",
]
NUISANCE_COLS = ["rt_ms", "ball_time_ms", "trial_duration_ms",
                 "block_index", "side", "move_dir"]
# -----------------------------------------------------------------------


# %%
def compute_unit_cv():
    """Per-unit CV from raw ISIs (spikes_units.csv), restricted to the units
    actually present in the binned decoding array.

    CV = SD(ISI)/mean(ISI) over the full session. Returns (df, flagged_ids).
    """
    z = np.load(BINNED, allow_pickle=True)
    binned_ids = {int(u) for u in z["unit_ids"]}
    spikes = pd.read_csv(SPIKES_UNITS)
    meta = pd.read_csv(UNIT_META).set_index("unit_id")

    rows = []
    for uid, grp in spikes.groupby("unit_id"):
        uid = int(uid)
        if uid not in binned_ids:
            continue
        t = np.sort(grp["spike_time_behavioral_ms"].to_numpy())
        isi = np.diff(t)
        isi = isi[isi > 0.0]
        fr = float(meta.loc[uid, "firing_rate_hz"]) if uid in meta.index else np.nan
        if len(isi) < MIN_ISI_N:
            rows.append({"unit_id": uid, "n_isi": int(len(isi)),
                         "cv": np.nan, "firing_rate_hz": fr, "flagged": False})
            continue
        cv = float(isi.std() / isi.mean())
        rows.append({"unit_id": uid, "n_isi": int(len(isi)), "cv": cv,
                     "firing_rate_hz": fr, "flagged": bool(cv > CV_THRESHOLD)})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "bursty_unit_cv_screen.csv", index=False)
    flagged = set(df.loc[df["flagged"], "unit_id"].astype(int))
    return df, flagged


# %%
def build_rate_matrix(unit_mask=None):
    """(n_trials, n_units) sqrt-transformed mean firing rate (Hz) in the
    choice-anchored window, optionally masked to a unit subset. Mirrors
    neural_lda_decoding.rate_features but applies the unit mask first so the
    retained features are a strict subset of the same representation."""
    z = np.load(BINNED, allow_pickle=True)
    binned, bin_centers = z["binned"], z["bin_centers"]
    lo, hi = (-1000.0, 0.0) if WINDOW == "pre" else (
        (0.0, 1000.0) if WINDOW == "post" else (None, None))
    if lo is None:
        mask = np.ones(len(bin_centers), dtype=bool)
    else:
        mask = (bin_centers >= lo) & (bin_centers <= hi)
    if unit_mask is not None:
        binned = binned[unit_mask]
    sel = binned[:, :, mask]
    valid = ~np.isnan(sel)
    counts = np.where(valid, sel, 0.0)
    n_valid = valid.sum(axis=2)
    rate = counts.sum(axis=2) / (n_valid * 0.025)
    rate = np.where(n_valid > 0, rate, 0.0)
    return np.sqrt(np.maximum(rate, 0.0)).T


# %%
def build_pca_matrix(unit_mask=None):
    """(n_trials, PCA_COMPONENTS) PCA-reduced time-resolved counts in the
    window. Unsupervised PCA fit once on all trials (the CV loop below only
    re-standardizes per training fold), matching neural_lda_decoding.pca_features
    but with the unit mask applied before feature construction."""
    z = np.load(BINNED, allow_pickle=True)
    binned, bin_centers = z["binned"], z["bin_centers"]
    lo, hi = (-1000.0, 0.0) if WINDOW == "pre" else (
        (0.0, 1000.0) if WINDOW == "post" else (None, None))
    if lo is None:
        mask = np.ones(len(bin_centers), dtype=bool)
    else:
        mask = (bin_centers >= lo) & (bin_centers <= hi)
    if unit_mask is not None:
        binned = binned[unit_mask]
    sel = binned[:, :, mask]                     # (n_units, n_trials, n_bins)
    X = np.where(np.isnan(sel), 0.0, sel)
    X = np.moveaxis(X, 0, 1).reshape(binned.shape[1], -1)  # (n_trials, feats)
    sc = StandardScaler().fit(X)
    pca = PCA(n_components=min(PCA_COMPONENTS, X.shape[1]),
              random_state=RNG_SEED).fit(sc.transform(X))
    return pca.transform(sc.transform(X))


# %%
def build_regressors():
    """{regressor: (n_trials,) array} from trial_labels.csv / trial_table.csv."""
    lab = pd.read_csv(TRIAL_LABELS)
    tt = pd.read_csv(TRIAL_TABLE)
    gL = lab["greedy_cost_L"].to_numpy(float)
    gR = lab["greedy_cost_R"].to_numpy(float)
    pL = lab["planning_cost_L"].to_numpy(float)
    pR = lab["planning_cost_R"].to_numpy(float)
    ch = lab["choice_hole"].to_numpy()

    def holes(s):
        try:
            return json.loads(s)
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


def build_nuisances():
    """(n_trials, n_nuisance+1) matrix (intercept column first)."""
    tt = pd.read_csv(TRIAL_TABLE)
    lab = pd.read_csv(TRIAL_LABELS)
    df = tt[["trial_id", "block_index", "choice_hole"]].copy()
    df["rt_ms"] = tt["choice_time_ms"] - tt["entry_time_ms"]
    df["ball_time_ms"] = tt["exit_time_ms"] - tt["choice_time_ms"]
    df["trial_duration_ms"] = tt["exit_time_ms"] - tt["trial_start_ms"]
    df["side"] = (tt["choice_hole"] >= 6).astype(int)
    df = df.merge(lab[["trial_id", "entry_hole"]], on="trial_id", how="left")
    df["move_dir"] = np.sign(df["choice_hole"] - df["entry_hole"]).to_numpy()
    Xn = df[NUISANCE_COLS].to_numpy(float)
    Xn = np.nan_to_num(Xn, nan=0.0)
    return np.column_stack([np.ones(len(Xn)), Xn])


def residualize(target, nuisance):
    beta, *_ = np.linalg.lstsq(nuisance, target, rcond=None)
    return target - nuisance @ beta


# %%
def cv_ridge_corr(y, X, n_folds, n_repeats, seed):
    """Mean CV Pearson correlation from repeated k-fold ridge regression."""
    corrs = []
    for rep in range(n_repeats):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed + rep)
        for tr, te in kf.split(X):
            sc = StandardScaler().fit(X[tr])
            model = Ridge(alpha=RIDGE_ALPHA).fit(sc.transform(X[tr]), y[tr])
            pred = model.predict(sc.transform(X[te]))
            if np.std(pred) > 1e-12 and np.std(y[te]) > 1e-12:
                corrs.append(float(np.corrcoef(y[te], pred)[0, 1]))
            else:
                corrs.append(np.nan)
    corrs = np.asarray(corrs)
    return float(np.nanmean(corrs))


def permutation_corr_p(y, X, obs_corr, n_perm, n_folds, seed):
    """Two-sided p: fraction of shuffled-regressor CV correlations >= observed."""
    rng = np.random.default_rng(seed)
    nulls = np.empty(n_perm)
    for i in range(n_perm):
        ysh = rng.permutation(y)
        nulls[i] = cv_ridge_corr(ysh, X, n_folds, 1, seed + i)
    return float(np.mean(nulls >= obs_corr))


# %%
def decode_one(y, X, nuisance, n_folds, n_repeats, n_perm, seed):
    """{control: (corr, p)} for the three control schemes."""
    out = {}
    for control in ["none", "target", "both"]:
        Xc, yc = X, y
        if control == "target":
            yc = residualize(y, nuisance)
        elif control == "both":
            yc = residualize(y, nuisance)
            Xc = np.column_stack([residualize(X[:, j], nuisance)
                                  for j in range(X.shape[1])])
        if np.std(yc) < 1e-12:
            continue
        corr = cv_ridge_corr(yc, Xc, n_folds, n_repeats, seed)
        p = permutation_corr_p(yc, Xc, corr, n_perm, PERM_N_FOLDS, seed)
        out[control] = (corr, p)
    return out


# %%
def main():
    print("Computing per-unit CV from raw spike times ...")
    screen, flagged = compute_unit_cv()
    n_all = len(screen)
    n_flagged = len(flagged)
    print(f"  {n_all} units in decoding array; {n_flagged} flagged CV>{CV_THRESHOLD}")

    z = np.load(BINNED, allow_pickle=True)
    all_ids = np.asarray(z["unit_ids"])
    keep_all = np.ones(len(all_ids), dtype=bool)
    keep_no_bursty = np.array([int(u) not in flagged for u in all_ids])

    print("Building features and regressors ...")
    regressors = build_regressors()
    nuisance = build_nuisances()

    X_by_rep = {}
    for rep in REPS:
        X_all = (build_rate_matrix(unit_mask=keep_all)
                 if rep == "rate" else build_pca_matrix(unit_mask=keep_all))
        X_trim = (build_rate_matrix(unit_mask=keep_no_bursty)
                  if rep == "rate" else build_pca_matrix(unit_mask=keep_no_bursty))
        X_by_rep[rep] = {"all": X_all, "no_bursty": X_trim}

    sets = {"all": keep_all, "no_bursty": keep_no_bursty}
    rows = []
    for rep in REPS:
        for rname, y_reg in regressors.items():
            m = ~np.isnan(y_reg)
            y = y_reg[m]
            for setname, mask in sets.items():
                X = X_by_rep[rep][setname]
                dec = decode_one(y, X[m], nuisance[m], N_FOLDS, N_REPEATS,
                                 N_PERM, RNG_SEED)
                for control, (corr, p) in dec.items():
                    rows.append({
                        "rep": rep,
                        "unit_set": setname,
                        "n_units": int(mask.sum()),
                        "regressor": rname,
                        "control": control,
                        "corr_mean": corr,
                        "perm_p": p,
                    })
                    print(f"  {rep}/{setname}/{rname}/{control}: "
                          f"corr {corr:+.3f} (p {p:.3f})")

    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / "bursty_sensitivity_decoding.csv", index=False)

    # headline: conflict_mag, all controls, both reps
    head = res[res["regressor"] == "conflict_mag"]
    head.to_csv(OUT_DIR / "bursty_sensitivity_conflict_mag.csv", index=False)

    for rep in REPS:
        print(f"\nconflict_mag ({WINDOW} window, {rep} rep) — all vs no-bursty:")
        sub = head[head["rep"] == rep]
        piv = sub.pivot_table(index="control", columns="unit_set",
                              values="corr_mean")
        print(piv.round(3).to_string())
        print("\nperm_p:")
        print(sub.pivot_table(index="control", columns="unit_set",
                              values="perm_p").round(3).to_string())
    print("\nSaved bursty_sensitivity_decoding.csv and "
          "bursty_sensitivity_conflict_mag.csv")


# %%
if __name__ == "__main__":
    main()
