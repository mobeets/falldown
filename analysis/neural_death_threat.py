# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#     jupytext_version: 1.19.4
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Death / threat neural analysis
#
# The EMU sessions are all-drift (camera mode 1), so the participant is under
# continuous time pressure and can die when the ball falls behind. This script
# asks two questions the earlier pipeline did not:
#
#   1. Does the population distinguish genuine **death moments** from normal
#      play? (death-locked decoding; only sessions with >= 5 deaths)
#   2. Is the continuous **proximity to death** (`ball_y - camera_y` at the
#      choice pass; small = close to death) represented in the population, and
#      does threat modulate the planning/conflict representation?
#
# Threat decoding reuses the ridge CV + nuisance-residualization machinery from
# `neural_continuous_decoding.py` (`none` / `target` / `both` controls), so the
# proximity signal is separated from reaction time, kinematics, position,
# block, and steering direction.
#
# Outputs (analysis/neural_outputs/<run>/):
#   death_decoding.csv              death-vs-normal LDA (rate, pre-death windows)
#   threat_continuous_decoding.csv  ridge decode of ball_y - camera_y
#   threat_modulation.csv           planning_vs_greedy decode near death vs far
#   threat_bally_profile.csv        planning_vs_greedy decode across rel_y bins
#
# Run with:
#   C:\Users\manik\AppData\Local\Programs\Python\Python311\python.exe analysis\neural_death_threat.py --run yga_1

# %%
import json
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd
from pathlib import Path

from neural_common import get_run, out_dir
from neural_lda_decoding import (
    RNG_SEED, PERM_FOLDS, load_binned, rate_features, pca_features,
    decode_balanced_accuracy, permutation_p, make_transform,
    build_label_vectors, add_move_dir_vx,
)
from neural_continuous_decoding import (
    build_regressors, build_nuisances, residualize, residualize_features,
    cv_regression, permutation_corr_p, N_FOLDS, N_REPEATS, N_PERM,
)

# ---------------------------- Configuration ----------------------------
_RUN = get_run()
OUT_DIR = out_dir(_RUN.run_id)
BINNED = OUT_DIR / "segmented_spikes_binned.npz"
SPIKES_UNITS = OUT_DIR / "spikes_units.csv"
UNIT_META = OUT_DIR / "unit_metadata.csv"
TRIAL_TABLE = OUT_DIR / "trial_table.csv"
TRIAL_LABELS = OUT_DIR / "trial_labels.csv"
DEATH_TIMES = OUT_DIR / "death_times.csv"
BEHAVIOR_PATH = _RUN.behavior_path

DEATH_WINDOWS = [(-1000.0, 0.0), (-500.0, 0.0)]
MIN_DEATHS = 5
N_DEATH_PERM = 300
# -----------------------------------------------------------------------


# %%
def per_unit_spike_times():
    """{unit_id: sorted behavioral-ms spike times} for QC-passed units."""
    units = pd.read_csv(UNIT_META)
    keep = set(units["unit_id"])
    spikes = pd.read_csv(SPIKES_UNITS)
    spikes = spikes[spikes["unit_id"].isin(keep)]
    return {int(u): np.sort(g["spike_time_behavioral_ms"].to_numpy())
            for u, g in spikes.groupby("unit_id")}


def _count(times, center, lo, hi):
    a = np.searchsorted(times, center + lo, side="left")
    b = np.searchsorted(times, center + hi, side="left")
    return b - a


def rate_matrix(times_by_unit, unit_ids, centers, lo, hi):
    """(n_events, n_units) sqrt-Hz firing rate in [center+lo, center+hi)."""
    dur_s = (hi - lo) / 1000.0
    X = np.zeros((len(centers), len(unit_ids)))
    for j, uid in enumerate(unit_ids):
        ts = times_by_unit[uid]
        for i, c in enumerate(centers):
            X[i, j] = _count(ts, c, lo, hi) / dur_s
    return np.sqrt(np.maximum(X, 0.0))


# %%
def build_threat_regressor(trial_table):
    """`ball_y - camera_y` (screen y) at each trial's choice pass.

    Small values = the ball is close to the top/death line; this is the same
    quantity the game uses to trigger death (`ball.y - cameraY < 0`).
    """
    with open(BEHAVIOR_PATH, encoding="utf-8") as fh:
        data = json.load(fh)
    t, rel = [], []
    for b in data.get("blocks", []):
        bi = b.get("block_index")
        if bi is None or bi < 4:
            continue
        gs = b.get("game_states") or {}
        if not gs:
            continue
        tt = np.asarray(gs["time"], dtype=float)
        by = np.asarray(gs["ball_y"], dtype=float)
        cy = np.asarray(gs["camera_y"], dtype=float)
        t.append(tt)
        rel.append(by - cy)
    if not t:
        return np.full(len(trial_table), np.nan)
    t = np.concatenate(t)
    rel = np.concatenate(rel)
    o = np.argsort(t)
    t, rel = t[o], rel[o]
    choice = trial_table["choice_time_ms"].to_numpy(float)
    idx = np.searchsorted(t, choice, side="left")
    idx = np.clip(idx, 0, len(t) - 1)
    return rel[idx]


# %%
def death_decoding():
    """LDA death vs normal on pre-death windows (rate rep)."""
    deaths = pd.read_csv(DEATH_TIMES)
    if "death_time_ms" not in deaths.columns or len(deaths) < MIN_DEATHS:
        return pd.DataFrame()
    tt = pd.read_csv(TRIAL_TABLE)
    times_by_unit = per_unit_spike_times()
    unit_ids = sorted(times_by_unit)
    d_times = deaths["death_time_ms"].to_numpy(float)
    choice = tt["choice_time_ms"].to_numpy(float)
    transform = make_transform("rate")

    rows = []
    for lo, hi in DEATH_WINDOWS:
        Xd = rate_matrix(times_by_unit, unit_ids, d_times, lo, hi)
        Xn = rate_matrix(times_by_unit, unit_ids, choice, lo, hi)
        X = np.vstack([Xd, Xn])
        y = np.r_[np.ones(len(Xd), int), np.zeros(len(Xn), int)]
        acc_m, acc_s = decode_balanced_accuracy(y, X, transform, N_FOLDS,
                                                N_REPEATS, RNG_SEED)
        p = permutation_p(y, X, transform, acc_m, n_perm=N_DEATH_PERM,
                          n_folds=PERM_FOLDS, seed=RNG_SEED)
        rows.append({"window_lo_ms": lo, "window_hi_ms": hi,
                     "n_deaths": len(Xd), "n_normal": len(Xn),
                     "acc_mean": acc_m, "acc_std": acc_s,
                     "chance": 0.5, "perm_p": p})
        print(f"  death [{lo:.0f},{hi:.0f}] rate: acc {acc_m:.3f} "
              f"(p {p:.3f}, n_death={len(Xd)})")
    return pd.DataFrame(rows)


# %%
def threat_continuous():
    """Ridge decode of ball_y - camera_y with none/target/both controls."""
    binned, unit_ids, bin_centers = load_binned()
    tt = pd.read_csv(TRIAL_TABLE)
    threat = build_threat_regressor(tt)
    nuisance = build_nuisances()

    Xfeats = {
        rep: {"pre": rate_features(binned, bin_centers, -1000.0, 0.0),
              "post": rate_features(binned, bin_centers, 0.0, 1000.0)}
        for rep in ["rate"]
    }
    Xfeats["pca"] = {"post": pca_features(binned, bin_centers, 0.0, 1000.0)}

    rows = []
    for rep, wins in Xfeats.items():
        for win, X in wins.items():
            m = ~np.isnan(threat)
            y = threat[m]
            Xm = X[m]
            for control in ["none", "target", "both"]:
                Xc, yc = Xm, y
                if control == "target":
                    yc = residualize(y, nuisance[m])
                elif control == "both":
                    yc = residualize(y, nuisance[m])
                    Xc = residualize_features(Xm, nuisance[m])
                (r2_m, r2_s, corr_m, corr_s), _ = cv_regression(
                    yc, Xc, N_FOLDS, N_REPEATS, RNG_SEED)
                p = permutation_corr_p(yc, Xc, corr_m, N_PERM, PERM_FOLDS,
                                       RNG_SEED)
                rows.append({"regressor": "threat_rel_y", "rep": rep,
                             "window": win, "control": control,
                             "n_trials": int(m.sum()), "corr_mean": corr_m,
                             "corr_std": corr_s, "perm_p": p})
                print(f"  threat {rep}/{win}/{control}: corr {corr_m:+.3f} "
                      f"(p {p:.3f})")
    return pd.DataFrame(rows)


# %%
def threat_modulation():
    """planning_vs_greedy decode near death vs far (rate, post).

    `rel_y = ball_y - camera_y` is *small* when the ball is close to the death
    line (`classify_trials.find_deaths`): so LOW rel_y = near death. The median
    split therefore puts the low-rel_y half in `near` and the high-rel_y half in
    `far`.
    """
    binned, unit_ids, bin_centers = load_binned()
    tt = pd.read_csv(TRIAL_TABLE)
    labels = pd.read_csv(TRIAL_LABELS)
    threat = build_threat_regressor(tt)
    lv = build_label_vectors(add_move_dir_vx(labels))
    y, idx = lv["planning_vs_greedy"]
    base = np.flatnonzero(idx if idx is not None else np.ones(len(y), bool))
    X = rate_features(binned, bin_centers, 0.0, 1000.0)
    transform = make_transform("rate")

    med = np.nanmedian(threat[base])
    near = base[threat[base] <= med]   # low rel_y = close to death
    far = base[threat[base] > med]

    def acc(sel):
        if len(sel) < 30 or len(np.unique(y[sel])) < 2:
            return np.nan, np.nan
        m, s = decode_balanced_accuracy(y[sel], X[sel], transform, N_FOLDS,
                                        N_REPEATS, RNG_SEED)
        p = permutation_p(y[sel], X[sel], transform, m, n_perm=200,
                          n_folds=PERM_FOLDS, seed=RNG_SEED)
        return m, p

    a_near, p_near = acc(near)
    a_far, p_far = acc(far)
    diff = (a_near - a_far if np.isfinite(a_near) and np.isfinite(a_far)
            else np.nan)

    # permutation for the difference: shuffle the near/far assignment
    rng = np.random.default_rng(RNG_SEED)
    null = []
    pool = np.concatenate([near, far])
    for _ in range(200):
        perm = rng.permutation(pool)
        n = perm[:len(near)]
        f = perm[len(near):]
        mn, _ = decode_balanced_accuracy(y[n], X[n], transform, PERM_FOLDS, 1,
                                         RNG_SEED)
        mf, _ = decode_balanced_accuracy(y[f], X[f], transform, PERM_FOLDS, 1,
                                         RNG_SEED)
        null.append(mn - mf)
    null = np.asarray(null)
    p_diff = float(np.mean(np.abs(null) >= abs(diff))) if np.isfinite(diff) else np.nan

    return pd.DataFrame([{
        "median_rel_y": float(med),
        "n_near_death": len(near), "n_far": len(far),
        "acc_near_death": a_near, "p_near_death": p_near,
        "acc_far": a_far, "p_far": p_far,
        "acc_diff_near_minus_far": diff, "perm_p_diff": p_diff,
    }])


# %%
THREAT_BINS = 5


def threat_bally_profile():
    """planning_vs_greedy decode across 5 bins of rel_y (near death -> far)."""
    binned, unit_ids, bin_centers = load_binned()
    tt = pd.read_csv(TRIAL_TABLE)
    labels = pd.read_csv(TRIAL_LABELS)
    threat = build_threat_regressor(tt)
    lv = build_label_vectors(add_move_dir_vx(labels))
    y, idx = lv["planning_vs_greedy"]
    base = np.flatnonzero(idx if idx is not None else np.ones(len(y), bool))
    X = rate_features(binned, bin_centers, 0.0, 1000.0)
    transform = make_transform("rate")

    tv = threat[base]
    bins = pd.qcut(pd.Series(tv), q=THREAT_BINS, labels=False,
                   duplicates="drop")
    rows = []
    for bin_id in sorted(bins.dropna().astype(int).unique()):
        sel = base[(bins == bin_id).to_numpy()]
        if len(sel) < 30 or len(np.unique(y[sel])) < 2:
            continue
        m, s = decode_balanced_accuracy(y[sel], X[sel], transform, N_FOLDS,
                                        N_REPEATS, RNG_SEED)
        p = permutation_p(y[sel], X[sel], transform, m, n_perm=150,
                          n_folds=PERM_FOLDS, seed=RNG_SEED)
        rows.append({"bin": int(bin_id), "n": int(len(sel)),
                     "rel_y_center": float(np.nanmean(tv[(bins == bin_id).to_numpy()])),
                     "acc_mean": m, "acc_std": s, "perm_p": p})
    return pd.DataFrame(rows)


# %%
def main():
    print(f"Run {_RUN.run_id} ({_RUN.participant})")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1] Death vs normal population decoding")
    dd = death_decoding()
    if not dd.empty:
        dd.to_csv(OUT_DIR / "death_decoding.csv", index=False)
        print(f"  saved death_decoding.csv")
    else:
        print(f"  skipped (fewer than {MIN_DEATHS} deaths)")

    print("\n[2] Continuous threat decoding (ball_y - camera_y)")
    tc = threat_continuous()
    tc.to_csv(OUT_DIR / "threat_continuous_decoding.csv", index=False)
    print("  saved threat_continuous_decoding.csv")

    print("\n[3] Threat modulation of planning_vs_greedy (near death vs far)")
    tm = threat_modulation()
    tm.to_csv(OUT_DIR / "threat_modulation.csv", index=False)
    print("  " + tm.round(3).to_string(index=False))

    print("\n[4] planning_vs_greedy decode across rel_y bins")
    tp = threat_bally_profile()
    tp.to_csv(OUT_DIR / "threat_bally_profile.csv", index=False)
    print(f"  saved threat_bally_profile.csv ({len(tp)} bins)")


if __name__ == "__main__":
    main()
