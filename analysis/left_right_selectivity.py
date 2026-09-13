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
# # Neural selectivity for movement direction (LEFT vs RIGHT)
#
# Tests whether any unit's firing rate is sensitive to whether the participant
# was going LEFT or RIGHT, using two per-trial operationalizations:
#
#   move_dir     : sign(choice_hole - entry_hole) -- the direction the ball had
#                  to travel to reach the chosen hole (geometric, from the
#                  level design).
#   move_dir_vx  : dominant sign(ball_vx) over [choice-1000, choice] ms from
#                  the game-state trajectory (motion-based). The game is
#                  digital-input, so ball_vx is almost always 0 or +/-max and
#                  the sign is unambiguous when there is motion. This labels
#                  909/910 trials and agrees with move_dir on 92% of them.
#
# For every unit, per-trial firing rates are computed in the standard windows
# (pre [-1000,0] ms, post [0,+1000] ms, whole trial) and tested with exactly
# the same machinery as neural_selectivity.py:
#   * effect size: modulation index MI = (mean_R - mean_L)/(mean_R + mean_L)
#   * two-sided permutation test (5000 label shuffles, seed 42)
#   * FDR (Benjamini-Hochberg) across units, per (contrast, window)
#
# Extra outputs beyond the fixed windows:
#   * TIME-RESOLVED left-vs-right MI on the 25 ms binned grid
#     (segmented_spikes_binned.npz) with per-element permutation p and FDR
#     across (unit x bin) -- locates WHEN direction selectivity appears
#     (approach phase before choice, or after passing the hole).
#   * CONFOUND CONTROL (the important one). Left-going and right-going trials
#     differ in horizontal position (ball_x), so a naive left/right contrast
#     could just be position tuning. Three checks:
#       (1) stratified test matching the geometric label on entry_hole and the
#           vx label on choice_hole (within-stratum permutation),
#       (2) a POSITION-CONTROLLED continuous test: rate during leftward vs
#           rightward motion within matched (ball_x x trial-phase) bins, with a
#           within-bin spike-direction resampling null,
#       (3) overlap with the 18 ball_x-tuned units from spatial_tuning.py.
#
# Outputs (CSV only; plot_* functions are for you to call in Jupyter):
#   left_right_selectivity_results.csv       fixed-window per-unit contrasts
#   left_right_selectivity_timeresolved.csv  per-unit x bin MI + p/q
#   left_right_confound_control.csv          stratified + continuous + overlap
#
# Run with:
#   C:\Users\manik\AppData\Local\Programs\Python\Python311\python.exe analysis\left_right_selectivity.py
#
# ## Why the modulation index (MI) as the test statistic?
#
# MI = (mean_L - mean_R)/(mean_L + mean_R) is used as the effect size, but
# SIGNIFICANCE comes from a nonparametric permutation null (label shuffle).
# That separation is the key to why MI is the right statistic here:
#
#   1. No distributional assumptions. Per-trial firing rates are skewed and
#      heteroscedastic (near-Poisson: variance grows with the mean). t-tests
#      and ANOVA assume normality + homoscedasticity and are invalid on this
#      data. MI makes no distributional assumptions because it is only ever
#      compared against its own label-shuffle null.
#   2. Scale-free and bounded. MI is invariant to a unit's overall firing
#      rate: a 30 Hz and a 0.3 Hz unit with the same fractional left/right
#      preference get the same MI, so units spanning orders of magnitude in
#      rate can be compared and FDR-corrected together. Alternatives that
#      normalize by the pooled SD (Cohen's d) are NOT rate-invariant for count
#      data because the noise floor grows with the mean.
#   3. Keeps magnitude and sign. Rank-based statistics (Mann-Whitney U / AUC)
#      discard magnitude -- they say "distributions differ" but not by how much
#      or which way. MI is signed (+ = prefers right) and reads directly as the
#      fractional rate modulation between directions: the standard
#      direction-selectivity index in the hippocampal/MTL and sensorimotor
#      literatures.
#   4. Becomes the natural null statistic. With the nearly balanced groups we
#      have (move_dir 470/440, move_dir_vx ~460/449), a mean-difference-based
#      statistic is unbiased under label shuffling. (The statistic that
#      misbehaves under permutation is a t-statistic that pools unequal
#      variances, not a mean-difference ratio.)
#   5. Comparable with the rest of the repo. neural_selectivity.py already uses
#      MI for the planning/greedy/lapse contrasts, so left/right results can be
#      compared directly in effect size and the identical FDR pipeline applies.
#
# Honest caveats: MI is a trial-mean statistic; a strongly nonlinear or
# bimodal direction response would be understated by MI, but for "does mean
# firing differ left vs right" (the question asked) MI is exactly the quantity
# of interest. Unequal group sizes would pull a pooled MI toward the larger
# group; our groups are balanced, and the stratified/position-controlled tests
# in the confound section remove that risk at the source.

# %%
import json
import numpy as np
import pandas as pd
from pathlib import Path

from neural_selectivity import (count_in_window, fdr_bh, permutation_test,
                                unit_channel_labels, whole_trial_rates)
from spatial_tuning import trial_phase_of
from neural_common import get_run, out_dir
_RUN = get_run()

# ---------------------------- Configuration ----------------------------
OUT_DIR = out_dir(_RUN.run_id)
BEHAVIOR_PATH = _RUN.behavior_path
SPIKES_UNITS = OUT_DIR / "spikes_units.csv"
UNIT_META = OUT_DIR / "unit_metadata.csv"
TRIAL_TABLE = OUT_DIR / "trial_table.csv"
TRIAL_LABELS = OUT_DIR / "trial_labels.csv"
BINNED = OUT_DIR / "segmented_spikes_binned.npz"
SPATIAL_RESULTS = OUT_DIR / "spatial_tuning_results.csv"

WINDOWS = {"pre": (-1000.0, 0.0), "post": (0.0, 1000.0)}
INCLUDE_WHOLE_TRIAL = True
MIN_EXPERIMENT_BLOCK = 4

N_PERM = 5000             # fixed-window contrasts (matches neural_selectivity)
N_PERM_TIMERES = 500      # time-resolved per-element p (FDR handles screening)
N_PERM_STRAT = 1000       # stratified (confound) test
N_PERM_CONT = 1000        # position-controlled continuous test
RNG_SEED = 42

VX_WINDOW_LO, VX_WINDOW_HI = -1000.0, 0.0   # ms relative to choice
VX_MIN_COVERAGE_MS = 250.0                   # min moving-time to classify a trial

CONT_MIN_OCC_S = 0.5       # position x phase bins need this much occupancy in
                           # BOTH directions before they count (as spatial_tuning)
CONT_X_BINS = 12
CONT_PHASE_BIN_MS = 50.0
# -----------------------------------------------------------------------


# %%
def load_track():
    """Sorted (time, ball_x, ball_vx) across experiment blocks (behavioral
    clock, ms)."""
    with open(BEHAVIOR_PATH, encoding="utf-8") as fh:
        data = json.load(fh)
    t, x, vx = [], [], []
    for b in data["blocks"]:
        bi = b.get("block_index")
        if bi is None or bi < MIN_EXPERIMENT_BLOCK:
            continue
        gs = b.get("game_states", {})
        if not gs:
            continue
        t.append(np.asarray(gs["time"]))
        x.append(np.asarray(gs["ball_x"]))
        vx.append(np.asarray(gs["ball_vx"]))
    t = np.concatenate(t)
    x = np.concatenate(x)
    vx = np.concatenate(vx)
    o = np.argsort(t)
    return t[o], x[o], vx[o]


def add_vx_label(table):
    """Add a 'move_dir_vx' column (+1 = going right, -1 = going left,
    0 = unclassified) to a trial table.

    The per-trial direction is the time-weighted dominant sign of ball_vx over
    [choice-1000, choice] ms. Trials with less than VX_MIN_COVERAGE_MS of
    moving samples (|vx| > 1e-9) are left unclassified.
    """
    t_all, _, vx_all = load_track()
    table = table.copy()
    dirs = np.zeros(len(table), dtype=int)
    choice = table["choice_time_ms"].to_numpy()
    for i, c in enumerate(choice):
        a = np.searchsorted(t_all, c + VX_WINDOW_LO, side="left")
        b = np.searchsorted(t_all, c + VX_WINDOW_HI, side="left")
        if a >= b:
            continue
        tseg, vseg = t_all[a:b], vx_all[a:b]
        dt = np.diff(np.concatenate([tseg, [c + VX_WINDOW_HI]]))
        dt = np.maximum(dt, 0.0)
        moving = np.abs(vseg) > 1e-9
        cov = dt[moving].sum()
        if cov < VX_MIN_COVERAGE_MS:
            continue
        s = np.sum(np.sign(vseg) * dt) / cov
        dirs[i] = 1 if s > 0 else -1
    table["move_dir_vx"] = dirs
    return table


def add_move_dir(table):
    """Add a 'move_dir' column: sign(choice_hole - entry_hole)."""
    table = table.copy()
    table["move_dir"] = np.sign(
        table["choice_hole"].to_numpy() - table["entry_hole"].to_numpy()
    ).astype(int)
    return table


def load_data():
    """(units, table, times_by_unit) with left/right labels attached.

    table merges trial_table + trial_labels (choice_hole is in both, so the
    duplicate is dropped) and carries move_dir and move_dir_vx.
    """
    units = pd.read_csv(UNIT_META)
    trials = pd.read_csv(TRIAL_TABLE)
    labels = pd.read_csv(TRIAL_LABELS).drop(columns=["choice_hole"])
    spikes = pd.read_csv(SPIKES_UNITS)
    keep_ids = set(units["unit_id"])
    spikes = spikes[spikes["unit_id"].isin(keep_ids)]
    times_by_unit = {}
    for uid, grp in spikes.groupby("unit_id"):
        times_by_unit[int(uid)] = grp["spike_time_behavioral_ms"].to_numpy()

    table = trials.merge(labels, on=["trial_id", "block_index", "sequence_index"])
    table = add_move_dir(table)
    table = add_vx_label(table)
    return units, table, times_by_unit


def label_sides(table, label):
    """(ia, ib): trial indexes of LEFT and RIGHT trials for a label column
    whose values are +1 = right, -1 = left, 0 = unclassified (excluded)."""
    v = table[label].to_numpy()
    return np.flatnonzero(v < 0), np.flatnonzero(v > 0)


# %%
def trial_rates(times_by_unit, table, window_lo, window_hi):
    """Per-unit per-trial firing rate (Hz) in the given window."""
    choice = table["choice_time_ms"].to_numpy()
    dur_s = (window_hi - window_lo) / 1000.0
    rates = {}
    for uid, ts in times_by_unit.items():
        rates[uid] = np.array(
            [count_in_window(ts, c, window_lo, window_hi) / dur_s for c in choice])
    return rates


def run_contrast(rows, contrast, window_name, unit_ids, rates, ia, ib,
                 chan_labels=None):
    """MI + permutation p for one left/right contrast across all units."""
    for uid in unit_ids:
        a = rates[uid][ia]
        b = rates[uid][ib]
        mi, p = permutation_test(a, b, n_perm=N_PERM, seed=RNG_SEED)
        rows.append({
            "unit_id": uid,
            "channel": chan_labels.get(uid, "") if chan_labels else "",
            "contrast": contrast,
            "window": window_name,
            "n_A": len(a),
            "n_B": len(b),
            "mean_rate_A_Hz": float(np.nanmean(a)) if len(a) else np.nan,
            "mean_rate_B_Hz": float(np.nanmean(b)) if len(b) else np.nan,
            "modulation_index": mi,
            "p_perm": p,
        })
    return rows


def run_window(rows, window_name, unit_ids, rates, table, chan_labels):
    """All left/right contrasts for a per-trial rate array."""
    for label in ("move_dir", "move_dir_vx"):
        ia, ib = label_sides(table, label)
        rows = run_contrast(rows, f"left_vs_right_{label}", window_name,
                            unit_ids, rates, ia, ib, chan_labels=chan_labels)
    return rows


# %%
def time_resolved_rows(unit_ids, table, chan_labels):
    """Per-unit x per-bin left-vs-right MI with per-element permutation p.

    Uses the 25 ms binned grid (segmented_spikes_binned.npz). A bin counts
    only where it is covered (non-NaN) in at least one trial of each group;
    the permutation shuffles direction labels across the pooled trials.
    """
    z = np.load(BINNED, allow_pickle=True)
    binned, trial_ids, bin_centers = z["binned"], z["trial_ids"], z["bin_centers"]
    n_units, n_trials, n_bins = binned.shape
    pos = np.searchsorted(trial_ids, table["trial_id"].to_numpy())
    rng = np.random.default_rng(RNG_SEED)
    rows = []

    for label in ("move_dir", "move_dir_vx"):
        ia, ib = label_sides(table, label)
        a, b = pos[ia], pos[ib]
        A = binned[:, a, :]                                  # (U, nA, B)
        B = binned[:, b, :]                                  # (U, nB, B)
        covA, covB = ~np.isnan(A), ~np.isnan(B)
        nA = np.maximum(covA.sum(1), 1.0)                    # (U, B) covered trials
        nB = np.maximum(covB.sum(1), 1.0)
        meanA = np.nansum(np.where(covA, A, 0.0), 1) / nA    # (U, B) count avg
        meanB = np.nansum(np.where(covB, B, 0.0), 1) / nB
        ok = (covA.sum(1) >= 1) & (covB.sum(1) >= 1)         # both groups present
        denom = meanA + meanB
        mi = np.full_like(meanA, np.nan)
        m = ok & (denom != 0)
        mi[m] = (meanB[m] - meanA[m]) / denom[m]

        pool = np.concatenate([a, b])
        nA_p = len(a)
        null_abs = np.zeros((N_PERM_TIMERES, n_units, n_bins))
        for k in range(N_PERM_TIMERES):
            perm = rng.permutation(len(pool))
            pa, pb = pool[perm[:nA_p]], pool[perm[nA_p:]]
            PA = binned[:, pa, :]
            PB = binned[:, pb, :]
            cPA, cPB = ~np.isnan(PA), ~np.isnan(PB)
            nPA = np.maximum(cPA.sum(1), 1.0)
            nPB = np.maximum(cPB.sum(1), 1.0)
            mPA = np.nansum(np.where(cPA, PA, 0.0), 1) / nPA
            mPB = np.nansum(np.where(cPB, PB, 0.0), 1) / nPB
            d = mPA + mPB
            pm = np.where(d != 0, (mPB - mPA) / np.where(d == 0, 1, d), np.nan)
            null_abs[k] = np.abs(pm)
        ge = (null_abs >= np.abs(mi)[None, :, :]).astype(float)
        ge[np.isnan(null_abs)] = np.nan
        p = np.nanmean(ge, axis=0)                           # (U, B) two-sided p

        for u_i, uid in enumerate(unit_ids):
            for b_i, bc in enumerate(bin_centers):
                if np.isnan(mi[u_i, b_i]):
                    continue
                rows.append({
                    "unit_id": uid,
                    "channel": chan_labels.get(uid, "") if chan_labels else "",
                    "label": label,
                    "bin_center_ms": float(bc),
                    "n_left": int(covA.sum(1)[u_i, b_i]),
                    "n_right": int(covB.sum(1)[u_i, b_i]),
                    "mean_rate_left_Hz": float(meanA[u_i, b_i]),
                    "mean_rate_right_Hz": float(meanB[u_i, b_i]),
                    "modulation_index": float(mi[u_i, b_i]),
                    "p_perm": float(p[u_i, b_i]),
                })

    res = pd.DataFrame(rows)
    res["q_fdr"] = np.nan
    for label, grp in res.groupby("label"):
        q = fdr_bh(grp["p_perm"].to_numpy())
        res.loc[grp.index, "q_fdr"] = q
    res["significant"] = res["q_fdr"] < 0.05
    return res


# %%
# ----------------------- Confound-control helpers ----------------------

def stratified_contrast(unit_ids, rates_mat, strata, ia, ib,
                        n_perm=N_PERM_STRAT, seed=RNG_SEED):
    """Weighted within-stratum MI + within-stratum label permutation.

    rates_mat : (n_trials, n_units) per-trial rates.
    ia, ib    : left/right trial indexes.
    strata    : (n_trials,) integer stratum per trial (e.g. entry_hole).
    Statistic : weighted mean over strata of within-stratum MI, weighted by
                the number of classified trials in the stratum. The null
                shuffles direction labels WITHIN each stratum (keeping each
                stratum's left/right counts fixed).
    Returns per-unit (obs_mi, p).
    """
    strata = np.asarray(strata)
    n_trials, n_units = rates_mat.shape
    sA, sB = strata[ia], strata[ib]
    valid_s = np.intersect1d(np.unique(sA), np.unique(sB))
    S = len(valid_s)
    if S == 0:
        return {u: (np.nan, np.nan) for u in unit_ids}

    # eligible trials: classified AND in a stratum that has both directions
    ia_v = ia[np.isin(sA, valid_s)]
    ib_v = ib[np.isin(sB, valid_s)]
    elig = np.zeros(n_trials, dtype=bool)
    elig[ia_v] = True
    elig[ib_v] = True

    code = np.full(n_trials, -1, dtype=int)
    code[ia_v] = np.searchsorted(valid_s, sA[np.isin(sA, valid_s)])
    code[ib_v] = np.searchsorted(valid_s, sB[np.isin(sB, valid_s)])
    K = (code[:, None] == np.arange(S)[None, :])            # (n_trials, S)
    Rm = np.where(elig[:, None], rates_mat, 0.0)

    cR = np.bincount(code[ib_v], minlength=S)               # right counts / stratum
    cL = np.bincount(code[ia_v], minlength=S)
    elig_idx = np.flatnonzero(elig)
    elig_code = code[elig_idx]

    def _stat(selR):
        """Weighted within-stratum MI from a right-trial boolean mask."""
        sr = (K * selR[:, None]).T @ Rm                      # (S, n_units) sums
        sl = (K * (~selR & elig)[:, None]).T @ Rm
        cr = (K * selR[:, None]).sum(0)
        cl = (K * (~selR & elig)[:, None]).sum(0)
        cr = np.where(cr == 0, np.nan, cr)
        cl = np.where(cl == 0, np.nan, cl)
        mR = sr / cr[:, None]
        mL = sl / cl[:, None]
        d = mR + mL
        stat = np.where(d != 0, (mR - mL) / np.where(d == 0, 1, d), np.nan)
        w = np.where(np.isnan(stat), 0.0, cL[:, None] + cR[:, None])
        wsum = w.sum(0)
        return np.where(wsum != 0, np.nansum(stat * w, axis=0) / wsum, np.nan)

    selR_obs = np.zeros(n_trials, dtype=bool)
    selR_obs[ib_v] = True
    obs = _stat(selR_obs)

    rng = np.random.default_rng(seed)
    nulls = np.full((n_perm, n_units), np.nan)
    for k in range(n_perm):
        selR = np.zeros(n_trials, dtype=bool)
        for s in range(S):
            idx_s = elig_idx[elig_code == s]
            n_r = int(cR[s])
            if len(idx_s) == 0 or n_r == 0:
                continue
            pick = rng.choice(len(idx_s), size=n_r, replace=False)
            selR[idx_s[pick]] = True
        nulls[k] = _stat(selR)

    out = {}
    for j, uid in enumerate(unit_ids):
        o = obs[j]
        if np.isnan(o):
            out[uid] = (np.nan, np.nan)
            continue
        nn = nulls[:, j]
        nn = nn[~np.isnan(nn)]
        if len(nn) == 0:
            out[uid] = (o, np.nan)
            continue
        out[uid] = (o, float(np.mean(np.abs(nn) >= abs(o))))
    return out


def continuous_direction_test(unit_ids, times_by_unit,
                              n_perm=N_PERM_CONT, seed=RNG_SEED):
    """Position-controlled direction selectivity.

    For each unit, build rate maps over (ball_x bin x trial-phase bin) split by
    whether the ball was moving LEFT or RIGHT (sign of ball_vx). The pooled
    statistic is the MI between occupancy-normalized rates during leftward vs
    rightward motion, restricted to (pos, phase) bins that have >=
    CONT_MIN_OCC_S of occupancy in BOTH directions.

    Null: within each (pos, phase) bin, resample which spikes fell in the
    left-moving vs right-moving half, conditional on the bin's observed spike
    total and the bin's observed occupancy split (Binomial(occ_L/(occ_L+occ_R))
    draw). This preserves the unit's spatial and trial-phase structure exactly
    and only breaks the spike <-> motion-direction link, so a significant
    result means genuine direction selectivity above and beyond ball_x tuning.
    """
    t_all, x_all, vx_all = load_track()
    phase = trial_phase_of(t_all)
    in_trial = phase >= 0
    t_all, x_all, vx_all, phase = (t_all[in_trial], x_all[in_trial],
                                   vx_all[in_trial], phase[in_trial])
    x_edges = np.linspace(x_all.min(), x_all.max(), CONT_X_BINS + 1)
    n_phase = int(phase.max()) + 1
    dir_all = np.sign(vx_all).astype(int)
    moving = np.abs(vx_all) > 1e-9
    dir_all[~moving] = 0
    has_dir = dir_all != 0

    pos_bin = np.clip(np.digitize(x_all, x_edges[1:-1]), 0, CONT_X_BINS - 1)
    dt = np.diff(np.concatenate([t_all, [t_all[-1] + (t_all[-1] - t_all[-2])]]))
    dt = np.maximum(dt, 0.0)
    keep = has_dir
    # occupancy (ms) per (phase, pos, dir)
    occ = np.zeros((n_phase, CONT_X_BINS, 2))
    np.add.at(occ, (phase[keep], pos_bin[keep],
                    np.where(dir_all[keep] > 0, 1, 0)), dt[keep])

    min_occ = CONT_MIN_OCC_S * 1000.0
    valid = (occ[:, :, 0] >= min_occ) & (occ[:, :, 1] >= min_occ)  # (P, X)
    p_pp = valid.ravel()
    pp_flat = np.argwhere(valid.ravel()).ravel()
    occL = occ[:, :, 0].ravel()[pp_flat]
    occR = occ[:, :, 1].ravel()[pp_flat]
    rng = np.random.default_rng(seed)

    out = {}
    for uid in unit_ids:
        st = times_by_unit[uid]
        st = st[(st >= t_all[0]) & (st <= t_all[-1])]
        obs_mi = np.nan
        p_val = np.nan
        if len(st):
            # per-spike pos bin, phase bin, direction
            sx = np.interp(st, t_all, x_all)
            spb = np.clip(np.digitize(sx, x_edges[1:-1]), 0, CONT_X_BINS - 1)
            sph = trial_phase_of(st)
            svx = np.interp(st, t_all, vx_all)
            sdir = np.where(np.abs(svx) > 1e-9, np.sign(svx).astype(int), 0)
            inbin = (sph >= 0) & (sph < n_phase) & (sdir != 0)
            sph, spb, sdir = sph[inbin], spb[inbin], sdir[inbin]
            if len(sph):
                sd = np.where(sdir > 0, 1, 0)
                cnt = np.zeros((n_phase, CONT_X_BINS, 2))
                np.add.at(cnt, (sph, spb, sd), 1)
                cntL = cnt[:, :, 0].ravel()[pp_flat]
                cntR = cnt[:, :, 1].ravel()[pp_flat]
                meanL = cntL / (occL / 1000.0)
                meanR = cntR / (occR / 1000.0)
                denom = meanL + meanR
                obs_mi = np.sum(meanR - meanL) / np.sum(denom) if denom.sum() else np.nan

                if np.isfinite(obs_mi):
                    total = cntL + cntR
                    nv = len(pp_flat)
                    null_mi = np.full(n_perm, np.nan)
                    for k in range(n_perm):
                        p_split = occL / (occL + occR)
                        cntLp = rng.binomial(total.astype(np.int64), p_split)
                        cntLp = cntLp.astype(float)
                        cntRp = total - cntLp
                        mL = cntLp / (occL / 1000.0)
                        mR = cntRp / (occR / 1000.0)
                        d = mL + mR
                        if d.sum() == 0:
                            continue
                        null_mi[k] = np.sum(mR - mL) / np.sum(d)
                    nn = null_mi[~np.isnan(null_mi)]
                    if len(nn):
                        p_val = float(np.mean(np.abs(nn) >= abs(obs_mi)))
        out[uid] = (float(obs_mi), p_val)
    return out


def load_x_tuned_units():
    """Set of unit_ids with significant ball_x tuning (spatial_tuning.py)."""
    if not SPATIAL_RESULTS.exists():
        return set()
    s = pd.read_csv(SPATIAL_RESULTS)
    return set(s[(s["axis"] == "x") & s["significant"]]["unit_id"].astype(int))


# %%
def main():
    print("Loading data ...")
    units, table, times_by_unit = load_data()
    unit_ids = sorted(times_by_unit.keys())
    chan_labels = unit_channel_labels(units)
    n_move = label_sides(table, "move_dir")
    n_vx = label_sides(table, "move_dir_vx")
    print(f"  {len(unit_ids)} units, {len(table)} trials")
    print(f"  move_dir:    {len(n_move[0])} left / {len(n_move[1])} right")
    print(f"  move_dir_vx: {len(n_vx[0])} left / {len(n_vx[1])} right")

    # ---------------- fixed-window contrasts ----------------
    rows = []
    for window_name, (lo, hi) in WINDOWS.items():
        print(f"\nWindow {window_name} [{lo:+.0f},{hi:+.0f}] ms")
        rates = trial_rates(times_by_unit, table, lo, hi)
        rows = run_window(rows, window_name, unit_ids, rates, table, chan_labels)
    if INCLUDE_WHOLE_TRIAL:
        print("\nWindow whole_trial [trial_start_ms, exit_time_ms]")
        rates = whole_trial_rates(times_by_unit, table)
        rows = run_window(rows, "whole_trial", unit_ids, rates, table, chan_labels)

    res = pd.DataFrame(rows)
    res["q_fdr"] = np.nan
    for key, grp in res.groupby(["contrast", "window"]):
        q = fdr_bh(grp["p_perm"].to_numpy())
        res.loc[grp.index, "q_fdr"] = q
    res["significant"] = res["q_fdr"] < 0.05
    res.to_csv(OUT_DIR / "left_right_selectivity_results.csv", index=False)
    print("\nSaved left_right_selectivity_results.csv")
    for (contrast, window), grp in res.groupby(["contrast", "window"]):
        sig = grp[grp["significant"]]
        if len(sig):
            labels = sorted(set(sig["channel"]))
            print(f"  {contrast} / {window}: {len(sig)} significant -> "
                  f"{', '.join(labels)}")

    # ---------------- time-resolved ----------------
    print("\nTime-resolved (25 ms bins) ...")
    tr = time_resolved_rows(unit_ids, table, chan_labels)
    tr.to_csv(OUT_DIR / "left_right_selectivity_timeresolved.csv", index=False)
    print(f"  saved left_right_selectivity_timeresolved.csv "
          f"({len(tr)} unit-bins)")
    for label, grp in tr.groupby("label"):
        print(f"  {label}: {int(grp['significant'].sum())} significant "
              f"unit-bin pairs")

    # ---------------- confound control ----------------
    print("\nConfound control ...")
    x_tuned = load_x_tuned_units()
    crows = []

    # (1) stratified, whole-trial rates
    whole = whole_trial_rates(times_by_unit, table)
    whole_mat = np.column_stack([whole[u] for u in unit_ids])   # (n_trials, U)
    for label, stratum_col in (("move_dir", "entry_hole"),
                               ("move_dir_vx", "choice_hole")):
        ia, ib = label_sides(table, label)
        strata = table[stratum_col].to_numpy()
        res_ = stratified_contrast(unit_ids, whole_mat, strata, ia, ib)
        for uid in unit_ids:
            mi, p = res_[uid]
            crows.append({
                "test": "stratified",
                "label": label,
                "stratum": stratum_col,
                "window": "whole_trial",
                "unit_id": uid,
                "channel": chan_labels.get(uid, ""),
                "n_left": len(ia),
                "n_right": len(ib),
                "modulation_index": mi,
                "p_perm": p,
            })

    # (2) position-controlled continuous
    cont = continuous_direction_test(unit_ids, times_by_unit)
    for uid in unit_ids:
        mi, p = cont[uid]
        crows.append({
            "test": "continuous_position_controlled",
            "label": "left_vs_right_motion",
            "stratum": "ball_x x trial_phase",
            "window": "all_in_trial",
            "unit_id": uid,
            "channel": chan_labels.get(uid, ""),
            "n_left": np.nan,
            "n_right": np.nan,
            "modulation_index": mi,
            "p_perm": p,
        })

    cc = pd.DataFrame(crows)
    cc["q_fdr"] = np.nan
    for key, grp in cc.groupby(["test", "label"]):
        q = fdr_bh(grp["p_perm"].to_numpy())
        cc.loc[grp.index, "q_fdr"] = q
    cc["significant"] = cc["q_fdr"] < 0.05
    cc["is_x_tuned"] = cc["unit_id"].isin(x_tuned)
    cc.to_csv(OUT_DIR / "left_right_confound_control.csv", index=False)
    print("  saved left_right_confound_control.csv")
    for (test, label), grp in cc.groupby(["test", "label"]):
        sig = grp[grp["significant"]]
        print(f"  {test}/{label}: {len(sig)} significant "
              f"(of which x-tuned: {int(sig['is_x_tuned'].sum())})")

    print("\nOverlap with ball_x-tuned units "
          f"({len(x_tuned)} x-tuned of {len(unit_ids)}):")
    fw = res[res["significant"]]
    if len(fw):
        fw_x = fw[fw["unit_id"].isin(x_tuned)]
        print(f"  fixed-window significant: {len(fw)} total, "
              f"{len(fw_x)} also x-tuned")


# %%
# ----------------------- Plot functions (not saved) ---------------------

def load_results():
    return pd.read_csv(OUT_DIR / "left_right_selectivity_results.csv")


def plot_bar_by_label(ax=None):
    """Significant units per (label, window) for the fixed-window results."""
    import matplotlib.pyplot as plt
    res = load_results()
    if ax is None:
        ax = plt.gca()
    res["label"] = res["contrast"].str.replace("left_vs_right_", "")
    piv = res[res["significant"]].groupby(["label", "window"]).size().unstack(fill_value=0)
    piv.plot(kind="bar", ax=ax)
    ax.set_ylabel("significant units (FDR<0.05)")
    ax.set_title("Direction-selective units by label")
    ax.legend(title="window")
    return ax


def plot_significant_units(label="move_dir", window="pre"):
    """Print the significant units for a label/window."""
    res = load_results()
    sub = res[(res["contrast"] == f"left_vs_right_{label}")
              & (res["window"] == window)]
    sig = sub[sub["significant"]].sort_values("modulation_index")
    print(f"left_vs_right_{label} / {window}: {len(sig)} significant units")
    return sig


def plot_timeresolved_psth(unit_id, label="move_dir", ax=None):
    """Left/right PSTH (Hz) with FDR-significant bins highlighted."""
    import matplotlib.pyplot as plt
    tr = pd.read_csv(OUT_DIR / "left_right_selectivity_timeresolved.csv")
    sub = tr[(tr["unit_id"] == unit_id) & (tr["label"] == label)]
    if ax is None:
        ax = plt.gca()
    ax.plot(sub["bin_center_ms"], sub["mean_rate_left_Hz"], "b-",
            label="left")
    ax.plot(sub["bin_center_ms"], sub["mean_rate_right_Hz"], "r-",
            label="right")
    sig = sub[sub["significant"]]
    ax.scatter(sig["bin_center_ms"], sig["mean_rate_right_Hz"], s=8,
               color="k", zorder=5, label="FDR sig")
    ax.axvline(0, color="k", ls=":")
    ax.set_xlabel("ms from choice")
    ax.set_ylabel("Hz")
    ax.set_title(f"unit {unit_id} ({label})")
    ax.legend()
    return ax


if __name__ == "__main__":
    main()

# %%
