# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#     jupytext_version: 1.19.4
#   kernelspec:
#     display_name: analysis
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Cross-dataset planning: cloud study x EMU neural
#
# Combines the online cloud-study behavioural data (large N, three
# environments) with the four EMU intracranial sessions (2 participants) to:
#
#   Part 1  behavioural comparability: is the EMU sample on the cloud manifold?
#   Part 2  neural-behavioural linkage within EMU (out-of-fold scores)
#   Part 3  cross-dataset model transfer (cloud-fitted models -> EMU choices)
#   Part 4  threat: cloud behavioural effect vs EMU neural result
#   Part 5  environmental statistics: within-subject planning ~ block conflict
#   Part 6  individual differences: EMU in the cloud trait space
#
# Baseline: cloud all-drift cohort (`short_trials_experiment-7-10`).
#
# Run under the pinned analysis env (numpy<2 + ssm):
#   .venv-analysis\Scripts\python.exe analysis\cross_dataset_planning.py

# %%
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from neural_common import RUN_ORDER, RUNS, out_dir
from neural_lda_decoding import (
    rate_features, pca_features, build_label_vectors, make_transform,
    decode_balanced_accuracy, RNG_SEED,
)
from neural_entry_locked import rate_features_entry
from neural_confound_audit import (
    build_regressors as emu_regressors, build_nuisances as emu_nuisances,
    load_labels as emu_labels, load_binned as emu_load_binned,
    build_threat as emu_threat,
)
from neural_region_decoding import oof_scores
import planning_experiments as pe

_REPO = Path(__file__).resolve().parent.parent
CLOUD_TABLE = _REPO / "analysis" / "planning_outputs" / "sequence_table.csv"
OUT = _REPO / "analysis" / "cross_dataset_outputs"
OUT.mkdir(parents=True, exist_ok=True)

CLOUD_ALDRIFT = "exp3_alldrift"
EMU_RUNS = list(RUN_ORDER)
# Ball-height analyses are reported separately for two trial sets:
#   "conflict" = conflict (greedy-vs-planning) trials only
#   "all"      = every experimental drift trial (conflict + agreement)
TRIAL_SETS = ("conflict", "all")
# -----------------------------------------------------------------------


# %% [markdown]
# ## Part 0 - Harmonization

# %%
def rel_ball_y_at(run, times):
    """`ball_y - camera_y` at the given behavioral-clock times (nearest frame)."""
    with open(RUNS[run].behavior_path, encoding="utf-8") as fh:
        data = json.load(fh)
    t, rel = [], []
    for b in data.get("blocks", []):
        if b.get("block_index", 0) < 4:
            continue
        gs = b.get("game_states") or {}
        if not gs:
            continue
        t.append(np.asarray(gs["time"], float))
        rel.append(np.asarray(gs["ball_y"], float) - np.asarray(gs["camera_y"], float))
    if not t:
        return np.full(len(times), np.nan)
    t = np.concatenate(t); rel = np.concatenate(rel)
    o = np.argsort(t); t, rel = t[o], rel[o]
    idx = np.clip(np.searchsorted(t, np.asarray(times, float)), 0, len(t) - 1)
    return rel[idx]


def harmonize_emu(run):
    """EMU trial_labels + trial_table -> cloud-like sequence schema."""
    out = out_dir(run)
    lab = pd.read_csv(out / "trial_labels.csv")
    tt = pd.read_csv(out / "trial_table.csv")
    tt_cols = ["trial_id", "block_index", "sequence_index", "hole_locations",
               "trial_start_ms", "entry_time_ms", "choice_time_ms", "exit_time_ms"]
    m = lab.merge(tt[tt_cols], on=["trial_id", "block_index", "sequence_index"],
                  how="left")
    holes = m["hole_locations"].map(json.loads)
    hL, hR = holes.str[0].to_numpy(), holes.str[1].to_numpy()
    chosen = m["choice_hole"].to_numpy()
    unchosen = np.where(chosen == hL, hR, hL)
    entry = m["entry_hole"].to_numpy(); goal = m["goal_hole"].to_numpy()

    c1 = np.abs(chosen - entry); u1 = np.abs(unchosen - entry)
    c2 = c1 + np.abs(chosen - goal); u2 = u1 + np.abs(unchosen - goal)

    def cost_for(hole, cL, cR):
        return np.where(hole == hL, cL, cR)

    plan_g = cost_for(m["greedy_optimal_hole"].to_numpy(),
                      m["planning_cost_L"].to_numpy(), m["planning_cost_R"].to_numpy())
    plan_p = cost_for(m["planning_optimal_hole"].to_numpy(),
                      m["planning_cost_L"].to_numpy(), m["planning_cost_R"].to_numpy())
    gre_g = cost_for(m["greedy_optimal_hole"].to_numpy(),
                     m["greedy_cost_L"].to_numpy(), m["greedy_cost_R"].to_numpy())
    gre_p = cost_for(m["planning_optimal_hole"].to_numpy(),
                     m["greedy_cost_L"].to_numpy(), m["greedy_cost_R"].to_numpy())

    df = pd.DataFrame({
        "participant": f"{RUNS[run].participant}_{run}",
        "source": "emu", "dataset": run, "experiment": "emu",
        "block_number": m["block_index"], "sequence_index": m["sequence_index"],
        "is_experimental": True, "block_drift": 1,
        "entry_hole": entry, "goal_hole": goal,
        "chosen_hole": chosen, "unchosen_hole": unchosen,
        "greedy_optimal_hole": m["greedy_optimal_hole"],
        "planning_optimal_hole": m["planning_optimal_hole"],
        "conflict": (~m["agree"].astype(bool)).astype(int),
        "chose_greedy": (chosen == m["greedy_optimal_hole"].to_numpy()).astype(int),
        "chose_planning": (chosen == m["planning_optimal_hole"].to_numpy()).astype(int),
        "condition": m["condition"],
        "chosen_1step_dist": c1, "unchosen_1step_dist": u1,
        "chosen_2step_dist": c2, "unchosen_2step_dist": u2,
        "chosen_left": (chosen < unchosen).astype(int),
        "plan_advantage": plan_g - plan_p,
        "greedy_advantage": gre_p - gre_g,
        "observed_rt": m["exit_time_ms"] - m["entry_time_ms"],
        "rt_decision": m["choice_time_ms"] - m["entry_time_ms"],
        "rt_exec": m["exit_time_ms"] - m["choice_time_ms"],
        "ball_y_at_top": rel_ball_y_at(run, m["entry_time_ms"].to_numpy()),
        "environment": "balanced", "threat": "drift",
    })
    df = df.sort_values(["block_number", "sequence_index"]).reset_index(drop=True)
    prev_goal = df["goal_hole"].shift(1)
    valid = (df["block_number"].shift(1) == df["block_number"]) & \
            (df["sequence_index"].shift(1) + 1 == df["sequence_index"])
    df["incoming_direction"] = np.where(valid, -np.sign(prev_goal - df["entry_hole"]), np.nan)
    env = df.groupby("block_number").agg(
        block_conflict_rate=("conflict", "mean"),
        block_plan_advantage=("plan_advantage", "mean"))
    return df.merge(env, on="block_number", how="left")


def harmonize_cloud():
    df = pd.read_csv(CLOUD_TABLE)
    df = df[df["is_experimental"]].copy()
    df["source"] = "cloud"
    return df


def build_combined():
    cloud = harmonize_cloud()
    emu = pd.concat([harmonize_emu(r) for r in EMU_RUNS], ignore_index=True)
    common = [c for c in cloud.columns if c in emu.columns]
    combined = pd.concat([cloud[common], emu[common]], ignore_index=True)
    combined = pe.add_features(combined)
    combined.to_csv(OUT / "cross_dataset_trials.csv", index=False)
    return combined


# %% [markdown]
# ## Model-based planning weight (logistic coefficient space)
#
# `P(plan | conflict)` treats every conflict trial as equivalent. The canonical
# cloud decision model instead separates the greedy and planning geometry:
#
#     logit P(choose left) = b_1step * z(diff_1step)
#                          + b_plan  * z(diff_planning)
#                          + incoming-direction terms
#
# fit per participant on **all choice trials** (the `evaluate_logistic_baseline`
# specification; drift interaction added only when drift varies). The
# **normalized planning weight** is the share of the geometry log-odds carried
# by the 2-step (planning) difference:
#
#     planning_weight = |b_plan| / (|b_plan| + |b_1step|)
#
# Sign convention: choosing the left hole is favoured by negative coefficients
# (left closer -> diff_1step < 0), so magnitudes are used; the weight lies in
# [0, 1] (0 = purely greedy geometry, 1 = purely planning geometry).

# %%
def _zscore(v):
    v = np.asarray(v, float)
    s = v.std()
    return (v - v.mean()) / (s if s > 0 else 1.0)


def _planning_design(d, extra_cols=()):
    """Design matrix for the per-participant choice logistic.

    Distance features are z-scored (`_z1`/`_zp` if precomputed on the caller's
    full frame, so that moderator terciles share one standardization);
    `incoming_direction` is added as in the cloud baseline, plus its drift
    interaction when drift varies. `extra_cols` are appended.
    """
    if "_z1" in d.columns and "_zp" in d.columns:
        z1, zp = d["_z1"].values, d["_zp"].values
    else:
        z1 = _zscore(d["diff_1step"].values)
        zp = _zscore(d["diff_planning"].values)
    X = np.column_stack([z1, zp, d["incoming_direction"].values])
    if d["block_drift"].nunique() > 1:
        X = np.column_stack([X, d["incoming_direction"].values
                             * d["block_drift"].values])
    for c in extra_cols:
        X = np.column_stack([X, d[c].values])
    return X


def _fit_choice_logistic(d, extra_cols=()):
    d = d.dropna(subset=["diff_1step", "diff_planning", "incoming_direction",
                         "chosen_left", "block_drift"]).copy()
    if len(d) < 30 or d["chosen_left"].nunique() < 2:
        return None
    X = _planning_design(d, extra_cols)
    m = LogisticRegression(penalty=None, max_iter=2000).fit(
        X, d["chosen_left"].astype(int).to_numpy())
    return m, d


def planning_weight(d, extra_cols=()):
    """Per-participant normalized planning weight (+ raw coefficients)."""
    fit = _fit_choice_logistic(d, extra_cols)
    if fit is None:
        return None
    m, dd = fit
    b1, bp = float(m.coef_[0][0]), float(m.coef_[0][1])
    den = abs(b1) + abs(bp)
    return {"beta_1step": b1, "beta_plan": bp,
            "planning_weight": abs(bp) / den if den > 1e-9 else np.nan,
            "n_choice": int(len(dd))}


def _weight_table(combined):
    rows = []
    for (src, pid), g in combined.groupby(["source", "participant"]):
        w = planning_weight(g)
        if w is None:
            continue
        w.update({"source": src, "participant": pid,
                  "dataset": g["dataset"].iloc[0]})
        rows.append(w)
    return pd.DataFrame(rows)


def _planning_weight_moderated(d, z_col, extra_cols=()):
    """Planning weight in the low/high terciles of `z_col`, plus the
    `z_col x planning` interaction coefficient and its Wald p-value.

    For `z_col="ball_y_at_top"` (screen y from the top; small = ball high on
    screen = near death), `pw_low` is the near-death weight and `pw_high` the
    safe weight. For `z_col="block_conflict_rate"`, low/high are quiet/rich
    environments.
    """
    d = d.dropna(subset=["diff_1step", "diff_planning", "incoming_direction",
                         "chosen_left", "block_drift", z_col]).copy()
    if len(d) < 60:
        return None
    # one standardization of the geometry across all terciles
    d["_z1"] = _zscore(d["diff_1step"].values)
    d["_zp"] = _zscore(d["diff_planning"].values)
    q1, q2 = d[z_col].quantile([1 / 3, 2 / 3])
    wl = planning_weight(d[d[z_col] <= q1])
    wh = planning_weight(d[d[z_col] >= q2])
    if wl is None or wh is None:
        return None
    d["_mod"] = _zscore(d[z_col].values)
    d["_mz1"] = d["_mod"] * d["_z1"]
    d["_mzp"] = d["_mod"] * d["_zp"]
    fit = _fit_choice_logistic(d, extra_cols=("_mod", "_mz1", "_mzp"))
    if fit is None:
        return None
    m, dfit = fit
    nbase = 3 + (1 if dfit["block_drift"].nunique() > 1 else 0)
    idx = nbase + 2
    X = _planning_design(dfit, extra_cols=("_mod", "_mz1", "_mzp"))
    try:
        sfit = sm.Logit(dfit["chosen_left"].astype(float).values,
                        sm.add_constant(X)).fit(disp=0)
        p = float(np.asarray(sfit.pvalues)[idx + 1])
    except Exception:
        p = np.nan
    # conditional raw coefficients at the low/high tercile means:
    # beta_plan(z) = beta_plan_0 + slope * z.  Planning => beta_plan < 0, so a
    # negative slope means the planning coefficient grows (more planning) as
    # the moderator rises.
    coef = m.coef_[0]
    zl = float(d.loc[d[z_col] <= q1, "_mod"].mean())
    zh = float(d.loc[d[z_col] >= q2, "_mod"].mean())
    b1_0, bp_0 = float(coef[0]), float(coef[1])
    c1, cp = float(coef[idx - 1]), float(coef[idx])
    return {"pw_low": wl["planning_weight"], "pw_high": wh["planning_weight"],
            "pw_diff_high_minus_low": wh["planning_weight"] - wl["planning_weight"],
            "beta_1step_low": b1_0 + c1 * zl, "beta_1step_high": b1_0 + c1 * zh,
            "beta_plan_low": bp_0 + cp * zl, "beta_plan_high": bp_0 + cp * zh,
            "inter_mod_x_plan": cp, "p_inter": p,
            "n_low": wl["n_choice"], "n_high": wh["n_choice"]}


# %% [markdown]
# ## Part 1 - Behavioural comparability

# %%
def part1_behavior(combined):
    rows = []
    for (src, pid), g in combined.groupby(["source", "participant"]):
        conf = g[g["conflict"] == 1]
        agree = g[g["conflict"] == 0]
        p = conf[conf["chose_planning"] == 1]
        gr = conf[conf["chose_planning"] == 0]
        rows.append({
            "source": src, "participant": pid,
            "dataset": g["dataset"].iloc[0],
            "n_conflict": len(conf), "n_agree": len(agree),
            "p_plan": conf["chose_planning"].mean(),
            "p_optimal": agree["chose_greedy"].mean(),
            "p_lapse": (agree["condition"] == "lapse").mean(),
            "rt_dec_plan": p["rt_decision"].mean(),
            "rt_dec_greedy": gr["rt_decision"].mean(),
            "rt_exec_plan": p["rt_exec"].mean(),
            "rt_exec_greedy": gr["rt_exec"].mean(),
        })
    summ = pd.DataFrame(rows)

    # model-based planning weight (coefficient space), side-by-side with p_plan
    wdf = _weight_table(combined)
    wdf.to_csv(OUT / "xd_planning_weight.csv", index=False)
    summ = summ.merge(
        wdf[["source", "participant", "beta_1step", "beta_plan",
             "planning_weight", "n_choice"]],
        on=["source", "participant"], how="left")
    summ.to_csv(OUT / "xd_behavioral_summary.csv", index=False)
    _ok = summ.dropna(subset=["p_plan", "planning_weight"])
    if len(_ok) > 2:
        r = np.corrcoef(_ok["p_plan"], _ok["planning_weight"])[0, 1]
        print(f"  Part1: corr(p_plan, planning_weight) = {r:+.3f} "
              f"(n={len(_ok)})")

    # mixture model per participant/run
    mix = []
    for (src, pid), g in combined.groupby(["source", "participant"]):
        fit = pe.fit_mixture_model(g, use_threat=True)
        if fit:
            fit.update({"source": src, "participant": pid,
                        "dataset": g["dataset"].iloc[0]})
            mix.append(fit)
    mix = pd.DataFrame(mix)
    mix.to_csv(OUT / "xd_mixture.csv", index=False)

    # GLM-HMM BIC selection for EMU
    emu = combined[combined["source"] == "emu"]
    sel, Kstar, per = pe.exp1_hmm_model_selection(emu, experiment="emu")
    sel.to_csv(OUT / "xd_hmm_selection.csv", index=False)
    per.to_csv(OUT / "xd_hmm_optimal_k.csv", index=False)
    print(f"  Part1: behavioral summary {len(summ)} rows, mixture {len(mix)}, "
          f"EMU HMM K*={Kstar}")
    return summ, mix


# %% [markdown]
# ## Part 2 - Neural-behavioural linkage (EMU)

# %%
def oof_ridge(y, X, n_folds=10, n_repeats=3, seed=RNG_SEED):
    preds = np.full((n_repeats, len(y)), np.nan)
    for rep in range(n_repeats):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed + rep)
        for tr, te in kf.split(X):
            sc = StandardScaler().fit(X[tr])
            model = Ridge(alpha=1.0).fit(sc.transform(X[tr]), y[tr])
            preds[rep, te] = model.predict(sc.transform(X[te]))
    return np.nanmean(preds, axis=0)


def _cv_logit_ll(y, X, n_folds=10, seed=RNG_SEED):
    """Out-of-sample log-likelihood/accuracy of a logistic model (CV)."""
    from sklearn.model_selection import StratifiedKFold
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    lls, accs = [], []
    for rep in range(2):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                              random_state=seed + rep)
        for tr, te in skf.split(X, y):
            m = LogisticRegression(penalty=None, max_iter=2000).fit(X[tr], y[tr])
            p = np.clip(m.predict_proba(X[te])[:, 1], 1e-9, 1 - 1e-9)
            lls.append(np.mean(y[te] * np.log(p) + (1 - y[te]) * np.log(1 - p)))
            accs.append(np.mean((p >= 0.5).astype(int) == y[te]))
    return float(np.mean(lls)), float(np.mean(accs))


def part2_linkage(combined):
    trial_frames, rows = [], []
    for run in EMU_RUNS:
        out = out_dir(run)
        binned, unit_ids, bc = emu_load_binned(out)
        lv, lab = emu_labels(out)
        X = rate_features(binned, bc, 0.0, 1000.0)
        Xp = pca_features(binned, bc, 0.0, 1000.0)
        n = len(lab)
        # out-of-fold planning score
        y, idx = lv["planning_vs_greedy"]
        keep = np.isin(np.arange(n), idx) if idx is not None else np.ones(n, bool)
        score = np.full(n, np.nan)
        score[keep] = oof_scores(y[keep], X[keep])
        # out-of-fold conflict_mag prediction (pca representation)
        regs = emu_regressors(out)
        cm = regs["conflict_mag"]
        mcm = ~np.isnan(cm)
        cm_pred = np.full(n, np.nan)
        cm_pred[mcm] = oof_ridge(cm[mcm], Xp[mcm])
        # entry-anchored pre-decision score
        z = np.load(out / "segmented_spikes_entrylocked_binned.npz", allow_pickle=True)
        Xe = rate_features_entry(z["binned"], z["bin_centers"], 0.0, 250.0)
        tt = pd.read_csv(out / "trial_table.csv")
        rt = (tt["choice_time_ms"] - tt["entry_time_ms"]).to_numpy(float)
        pre = np.full(n, np.nan)
        kk = keep & (rt >= 250.0)
        if kk.sum() >= 40:
            pre[kk] = oof_scores(y[kk], Xe[kk])

        # merge harmonized geometry (diff_planning, chosen_1step_dist, RT)
        eh = combined[combined["dataset"] == run][
            ["block_number", "sequence_index", "diff_planning",
             "chosen_1step_dist", "rt_decision", "rt_exec",
             "chosen_2step_dist", "ball_y_at_top"]].rename(
                 columns={"block_number": "block_index"})
        d = lab.merge(eh, on=["block_index", "sequence_index"], how="left")
        d["run_id"] = run
        d["neural_plan_score"] = score
        d["neural_conflict_pred"] = cm_pred
        d["neural_pre_score"] = pre
        trial_frames.append(d)

        # incremental choice information from the neural score (CV)
        sub = d[d["condition"].isin(["planning", "greedy"])].dropna(
            subset=["neural_plan_score", "diff_planning"])
        yc = (sub["condition"] == "planning").astype(int).to_numpy()
        Xg = sub[["diff_planning"]].to_numpy(float)
        Xgn = np.column_stack([Xg, sub["neural_plan_score"].to_numpy(float)])
        ll_g, acc_g = _cv_logit_ll(yc, Xg)
        ll_n, acc_n = _cv_logit_ll(yc, Xgn)

        # neural conflict_mag vs RT residual (kinematic-controlled)
        rr = d.dropna(subset=["neural_conflict_pred", "chosen_1step_dist",
                              "rt_decision"])
        cmv = rr["neural_conflict_pred"].to_numpy(float)
        b = np.polyfit(rr["chosen_1step_dist"].to_numpy(float),
                       rr["rt_decision"].to_numpy(float), 1)
        rres = rr["rt_decision"].to_numpy(float) - (
            b[0] * rr["chosen_1step_dist"].to_numpy(float) + b[1])
        corr_rt = (float(np.corrcoef(cmv, rres)[0, 1])
                   if np.std(cmv) > 1e-12 and np.std(rres) > 1e-12 else np.nan)

        # pre-decision incremental (CV)
        subp = d[d["condition"].isin(["planning", "greedy"])].dropna(
            subset=["neural_pre_score", "diff_planning"])
        yp = (subp["condition"] == "planning").astype(int).to_numpy()
        Xpg = subp[["diff_planning"]].to_numpy(float)
        Xpgn = np.column_stack([Xpg, subp["neural_pre_score"].to_numpy(float)])
        ll_pg, acc_pg = _cv_logit_ll(yp, Xpg)
        ll_pn, acc_pn = _cv_logit_ll(yp, Xpgn)

        rows.append({
            "run_id": run, "participant": RUNS[run].participant,
            "n_conflict": int(len(yc)),
            "ll_geometry": ll_g, "ll_geometry_neural": ll_n,
            "acc_geometry": acc_g, "acc_geometry_neural": acc_n,
            "corr_conflict_rt_resid": corr_rt,
            "ll_pre_geometry": ll_pg, "ll_pre_neural": ll_pn,
            "acc_pre_geometry": acc_pg, "acc_pre_neural": acc_pn,
        })
        print(f"  Part2 {run}: acc geom {acc_g:.3f} -> +neural {acc_n:.3f}; "
              f"pre {acc_pg:.3f} -> {acc_pn:.3f}; corr(cm,rt_resid)={corr_rt:+.3f}")

    linkage = pd.DataFrame(rows)
    linkage.to_csv(OUT / "xd_neural_linkage.csv", index=False)
    pd.concat(trial_frames, ignore_index=True).to_csv(
        OUT / "xd_neural_trials.csv", index=False)
    return linkage


# %% [markdown]
# ## Part 2b - Neural planning profiles across ball height
#
# Collects the per-run outputs of `neural_planning_profiles.py` (planning /
# value decoding by ball-y bin, the OOF neural-score x ball_y choice
# interaction, region profiles) and `neural_death_threat.py` (rel_y profile)
# into cross-dataset tables. EMU is all-drift, so there is no neural
# drift-vs-follow contrast here.

# %%
# per-run file -> combined cross-dataset output
NEURAL_PROFILE_FILES = {
    "planning_bally_profile.csv": "xd_neural_planning_bally.csv",
    "value_bally_profile.csv": "xd_neural_value_bally.csv",
    "neural_choice_bally.csv": "xd_neural_choice_bally.csv",
    "region_bally_profile.csv": "xd_neural_region_bally.csv",
    "threat_bally_profile.csv": "xd_neural_threat_bally.csv",
    "planning_bally_controlled.csv": "xd_neural_planning_controlled.csv",
    "planning_bally_clusters.csv": "xd_neural_planning_clusters.csv",
    "planning_cross_bally.csv": "xd_neural_cross_bally.csv",
    "unit_threat_planning.csv": "xd_neural_unit_threat.csv",
    "planning_history.csv": "xd_neural_history.csv",
    "rsa_bally.csv": "xd_neural_rsa.csv",
    "dpca_threat.csv": "xd_neural_dpca_threat.csv",
}


def part2b_neural_profiles():
    written = {}
    for per_run, combined_name in NEURAL_PROFILE_FILES.items():
        frames = []
        for run in EMU_RUNS:
            p = out_dir(run) / per_run
            if not p.exists():
                continue
            d = pd.read_csv(p)
            d["run_id"] = run
            d["participant"] = RUNS[run].participant
            frames.append(d)
        if frames:
            out = pd.concat(frames, ignore_index=True)
            out.to_csv(OUT / combined_name, index=False)
            written[combined_name] = len(out)
    print(f"  Part2b: neural profile tables {written}")
    return written


# %% [markdown]
# ## Part 3 - Cross-dataset model transfer

# %%
def part3_transfer(combined):
    cloud = combined[(combined["source"] == "cloud")
                     & (combined["dataset"] == CLOUD_ALDRIFT)]
    emu = combined[combined["source"] == "emu"]
    rows = []

    def feats(d):
        return d[["diff_1step", "diff_planning"]].to_numpy(float)

    def fit_eval(train, test, label):
        ytr = (train["chose_planning"] == 1).astype(int).to_numpy()
        yte = (test["chose_planning"] == 1).astype(int).to_numpy()
        m = LogisticRegression(penalty=None, max_iter=2000).fit(feats(train), ytr)
        p = np.clip(m.predict_proba(feats(test))[:, 1], 1e-9, 1 - 1e-9)
        ll = float(np.mean(yte * np.log(p) + (1 - yte) * np.log(1 - p)))
        acc = float(np.mean((p >= 0.5).astype(int) == yte))
        rows.append({"model": label, "n_test": len(yte), "loglik": ll, "acc": acc})

    cloud_conf = cloud[cloud["conflict"] == 1]
    emu_conf = emu[emu["conflict"] == 1]
    # cloud-fitted -> EMU
    fit_eval(cloud_conf, emu_conf, "cloud->EMU")
    # EMU-fitted -> EMU (block-held-out)
    lls, accs = [], []
    for blk, g in emu_conf.groupby("block_number"):
        tr = emu_conf[emu_conf["block_number"] != blk]
        if len(g) < 5 or len(tr) < 20:
            continue
        m = LogisticRegression(penalty=None, max_iter=2000).fit(
            feats(tr), (tr["chose_planning"] == 1).astype(int))
        p = np.clip(m.predict_proba(feats(g))[:, 1], 1e-9, 1 - 1e-9)
        yte = (g["chose_planning"] == 1).astype(int).to_numpy()
        lls.append(float(np.mean(yte * np.log(p) + (1 - yte) * np.log(1 - p))))
        accs.append(float(np.mean((p >= 0.5).astype(int) == yte)))
    rows.append({"model": "EMU(block-CV)->EMU", "n_test": int(emu_conf.shape[0]),
                 "loglik": float(np.mean(lls)), "acc": float(np.mean(accs))})
    transfer = pd.DataFrame(rows)
    transfer.to_csv(OUT / "xd_transfer.csv", index=False)

    # neural quantities vs model-internal variables (per EMU run)
    mn = []
    for run in EMU_RUNS:
        d = pd.read_csv(OUT / "xd_neural_trials.csv")
        d = d[d["run_id"] == run]
        lab = pd.read_csv(out_dir(run) / "trial_labels.csv")
        mc = np.abs(lab["planning_cost_L"] - lab["planning_cost_R"]
                    - (lab["greedy_cost_L"] - lab["greedy_cost_R"])).to_numpy(float)
        cm = d["neural_conflict_pred"].to_numpy(float)
        m = np.isfinite(cm) & np.isfinite(mc)
        r = float(np.corrcoef(cm[m], mc[m])[0, 1]) if m.sum() > 20 and \
            np.std(cm[m]) > 1e-12 else np.nan
        mn.append({"run_id": run, "corr_neural_cm_model_conflict": r,
                   "n": int(m.sum())})
    mn = pd.DataFrame(mn)
    mn.to_csv(OUT / "xd_model_neural.csv", index=False)
    print(f"  Part3: transfer rows {len(transfer)}; model-neural {len(mn)}")
    return transfer, mn


# %% [markdown]
# ## Part 4 - Threat
#
# The ball-y (screen position) analyses below use **drift trials only**
# (`block_drift == 1`), matching `exploratory_data_analysis`. This guarantees
# every contributing trial comes from a participant exposed to drift: the
# all-follow `exp1_nodrift` cohort drops out entirely, and mixed `exp3_altdrift`
# participants contribute only their drift blocks.
#
# Both outputs are reported for two trial sets (column `trials`), because the
# choice statistic means different things on each:
#   `conflict` = conflict (greedy-vs-planning) trials only -> P(plan);
#   `all`      = every experimental drift trial -> agreement trials, where the
#                greedy and planning options coincide, are included and the
#                rate becomes P(choose optimal).

# %%
def _trial_subset(g, trials):
    """Drift trials for the requested trial set.

    "conflict" keeps only greedy-vs-planning conflict trials; "all" keeps every
    experimental drift trial (conflict + agreement). Both require `block_drift
    == 1`, which drops participants never exposed to drift (exp1_nodrift) and
    excludes follow-block trials from mixed (exp3_altdrift) participants.
    """
    d = g[g["block_drift"] == 1]
    if trials == "conflict":
        d = d[d["conflict"] == 1]
    return d


def part4_threat(combined):
    rows = []
    for trials in TRIAL_SETS:
        # EMU behavioral: choice rate by ball_y tercile + slope (all trials are
        # drift for EMU; the filter is explicit for consistency with cloud)
        for run in EMU_RUNS:
            d = combined[(combined["dataset"] == run)
                         & (combined["block_drift"] == 1)]
            if trials == "conflict":
                d = d[d["conflict"] == 1]
            d = d.dropna(subset=["ball_y_at_top"])
            if len(d) < 30:
                continue
            q1, q2 = d["ball_y_at_top"].quantile([1 / 3, 2 / 3])
            near = d[d["ball_y_at_top"] <= q1]; safe = d[d["ball_y_at_top"] >= q2]
            slope, intercept, r, p, se = stats.linregress(
                d["ball_y_at_top"], d["chose_planning"])
            rows.append({"source": "emu", "trials": trials,
                         "participant": f"{RUNS[run].participant}_{run}",
                         "dataset": run, "n": len(d),
                         "p_plan_near": near["chose_planning"].mean(),
                         "p_plan_safe": safe["chose_planning"].mean(),
                         "slope_ball_y": slope, "p_slope": p})
        # cloud per-participant ball_y slope
        for ds in ["exp3_alldrift", "exp3_altdrift", "exp1_nodrift"]:
            d = combined[(combined["source"] == "cloud")
                         & (combined["dataset"] == ds)]
            for pid, g in d.groupby("participant"):
                conf = _trial_subset(g, trials).dropna(
                    subset=["ball_y_at_top"])
                if len(conf) < 30:
                    continue
                slope, intercept, r, p, se = stats.linregress(
                    conf["ball_y_at_top"], conf["chose_planning"])
                rows.append({"source": "cloud", "trials": trials,
                             "participant": pid, "dataset": ds,
                             "n": len(conf), "p_plan_near": np.nan,
                             "p_plan_safe": np.nan, "slope_ball_y": slope,
                             "p_slope": p})
    # cloud drift-vs-follow (exp3_altdrift within-subject)
    alt = combined[(combined["source"] == "cloud")
                   & (combined["dataset"] == "exp3_altdrift") & (combined["conflict"] == 1)]
    dfv = []
    for pid, g in alt.groupby("participant"):
        f = g[g["block_drift"] == 0]["chose_planning"].mean()
        dr = g[g["block_drift"] == 1]["chose_planning"].mean()
        if np.isfinite(f) and np.isfinite(dr):
            dfv.append({"participant": pid, "follow": f, "drift": dr})
    dfv = pd.DataFrame(dfv)
    if len(dfv) > 1:
        t, p = stats.ttest_rel(dfv["drift"], dfv["follow"])
        print(f"  Part4 cloud drift-vs-follow: drift {dfv['drift'].mean():.3f} "
              f"vs follow {dfv['follow'].mean():.3f}, p={p:.3f}")

    threat = pd.DataFrame(rows)
    threat.to_csv(OUT / "xd_threat_behavior.csv", index=False)

    # neural threat modulation (already computed per run)
    nm = []
    for run in EMU_RUNS:
        pth = out_dir(run) / "threat_modulation.csv"
        if pth.exists():
            t = pd.read_csv(pth)
            t["run_id"] = run
            nm.append(t)
    if nm:
        pd.concat(nm, ignore_index=True).to_csv(
            OUT / "xd_threat_neural.csv", index=False)
    print(f"  Part4: threat behavior rows {len(threat)}")
    return threat


def part4_threat_weight(combined):
    """Coefficient-space threat effect on the planning weight.

    For each participant/run and each trial set, the planning weight in the
    near-death and safe thirds of `ball_y_at_top`, and the `ball_y x planning`
    interaction from the per-trial choice logistic (drift blocks only; conflict
    trials only for `trials="conflict"`, conflict + agreement for "all").
    """
    rows = []
    for trials in TRIAL_SETS:
        for (src, pid), g in combined.groupby(["source", "participant"]):
            dg = _trial_subset(g, trials)
            w = _planning_weight_moderated(dg, "ball_y_at_top")
            if w is None:
                continue
            w.update({"source": src, "trials": trials, "participant": pid,
                      "dataset": g["dataset"].iloc[0]})
            rows.append(w)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "xd_threat_weight.csv", index=False)
    print(f"  Part4w: threat-weight rows {len(out)}")
    return out


# %% [markdown]
# ## Part 4c - Coefficient profile across ball-y positions
#
# The same choice logistic as above, but instead of a single number per
# participant we trace **all three coefficients** (greedy 1-step, planning
# 2-step, incoming direction) across ball-y positions. Per participant and
# trial set, drift blocks only, the participant's trials are split into
# `COEF_BINS` quantile bins of `ball_y_at_top`; the geometry terms are z-scored
# **once** on the participant's full frame (to avoid per-bin rescaling), then a
# separate logistic is fit in each bin. The normalized share
# `|b_plan| / (|b_plan| + |b_greedy|)` is reported alongside the raw betas.
#
# `part4_coef_profile_drift` does the same for the within-subject drift vs
# follow contrast, but **without a ball-y axis** (follow blocks have no
# near-death range), restricted to `exp3_altdrift` (the only cohort with both).

# %%
COEF_BINS = 5
COEF_MIN_BIN = 20
# reject quasi-separated fits (coefficient/SE blow-up)
COEF_MAX_BETA = 12.0
COEF_MAX_SE = 8.0


def _zscore_vec(v):
    v = np.asarray(v, float)
    s = v.std()
    return (v - v.mean()) / (s if s > 0 else 1.0)


def _coef_share(m):
    den = abs(m.params["greedy"]) + abs(m.params["planning"])
    return abs(m.params["planning"]) / den if den > 1e-9 else np.nan


def _coef_rows(m, n, extra=None):
    """Long-format rows for the three coefficients of a fitted Logit."""
    share = _coef_share(m)
    rows = []
    for term in ["greedy", "planning", "incoming"]:
        if term == "planning":
            sh = share
        elif term == "greedy":
            sh = 1.0 - share if np.isfinite(share) else np.nan
        else:
            sh = np.nan
        row = {"term": term, "beta": float(m.params[term]),
               "se": float(m.bse[term]), "share": sh, "n": int(n)}
        if extra:
            row.update(extra)
        rows.append(row)
    return rows


def _fit_coef_model(d):
    X = pd.DataFrame({
        "greedy": d["_z1"].values,
        "planning": d["_zp"].values,
        "incoming": d["incoming_direction"].values.astype(float)})
    try:
        m = sm.Logit(d["chosen_left"].astype(float).values,
                     sm.add_constant(X)).fit(disp=0)
    except Exception:
        return None
    params = np.asarray(m.params, float)
    ses = np.asarray(m.bse, float)
    if (not np.all(np.isfinite(params)) or not np.all(np.isfinite(ses))
            or np.nanmax(np.abs(params)) > COEF_MAX_BETA
            or np.nanmax(ses) > COEF_MAX_SE):
        return None
    return m


def part4_coef_profile(combined):
    """Greedy/planning/incoming coefficients across ball-y bins."""
    rows = []
    for trials in TRIAL_SETS:
        for (src, pid), g in combined.groupby(["source", "participant"]):
            d = _trial_subset(g, trials).dropna(subset=[
                "diff_1step", "diff_planning", "incoming_direction",
                "chosen_left", "ball_y_at_top"]).copy()
            if len(d) < COEF_BINS * 10:
                continue
            d["_z1"] = _zscore_vec(d["diff_1step"].values)
            d["_zp"] = _zscore_vec(d["diff_planning"].values)
            try:
                d["_bin"] = pd.qcut(d["ball_y_at_top"], q=COEF_BINS,
                                    labels=False, duplicates="drop")
            except ValueError:
                continue
            for b, gb in d.groupby("_bin"):
                if len(gb) < COEF_MIN_BIN or gb["chosen_left"].nunique() < 2:
                    continue
                m = _fit_coef_model(gb)
                if m is None:
                    continue
                rows += _coef_rows(m, len(gb), {
                    "source": src, "trials": trials, "participant": pid,
                    "dataset": g["dataset"].iloc[0], "bin": int(b),
                    "y_center": float(gb["ball_y_at_top"].mean())})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "xd_coef_profile.csv", index=False)
    print(f"  Part4c: coefficient profile rows {len(out)}")
    return out


def part4_coef_profile_drift(combined):
    """Within-subject drift vs follow coefficients (exp3_altdrift only)."""
    d = combined[(combined["source"] == "cloud")
                 & (combined["dataset"] == "exp3_altdrift")]
    rows = []
    for trials in TRIAL_SETS:
        for pid, g in d.groupby("participant"):
            for cond, name in [(1, "drift"), (0, "follow")]:
                sub = g[g["block_drift"] == cond]
                if trials == "conflict":
                    sub = sub[sub["conflict"] == 1]
                sub = sub.dropna(subset=[
                    "diff_1step", "diff_planning", "incoming_direction",
                    "chosen_left"]).copy()
                if len(sub) < 30 or sub["chosen_left"].nunique() < 2:
                    continue
                sub["_z1"] = _zscore_vec(sub["diff_1step"].values)
                sub["_zp"] = _zscore_vec(sub["diff_planning"].values)
                m = _fit_coef_model(sub)
                if m is None:
                    continue
                rows += _coef_rows(m, len(sub), {
                    "source": "cloud", "trials": trials, "participant": pid,
                    "dataset": "exp3_altdrift", "condition": name})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "xd_coef_profile_drift.csv", index=False)
    print(f"  Part4cd: drift-vs-follow coefficient rows {len(out)}")
    return out


# %% [markdown]
# ## Part 5 - Environmental statistics

# %%
def part5_environment(combined):
    rows = []
    for (src, pid), g in combined[combined["conflict"] == 1].groupby(
            ["source", "participant"]):
        be = g.groupby("block_number").agg(
            cr=("block_conflict_rate", "first"),
            pr=("chose_planning", "mean"), n=("chose_planning", "size"))
        be = be[be["n"] >= 5]
        if len(be) < 6 or be["cr"].std() < 1e-9:
            continue
        slope, intercept, r, p, se = stats.linregress(be["cr"], be["pr"])
        rows.append({"source": src, "participant": pid, "n_blocks": len(be),
                     "slope": slope, "r": r, "p": p})
    env = pd.DataFrame(rows)
    env.to_csv(OUT / "xd_environment.csv", index=False)

    # per-block neural score vs block conflict rate
    nt = pd.read_csv(OUT / "xd_neural_trials.csv")
    bn = []
    for run in EMU_RUNS:
        d = nt[nt["run_id"] == run].dropna(subset=["neural_plan_score"])
        bcr = (combined[combined["dataset"] == run]
               .groupby("block_number")["block_conflict_rate"].first())
        g = d.groupby("block_index").agg(score=("neural_plan_score", "mean"))
        g["cr"] = bcr.reindex(g.index).to_numpy()
        g = g.dropna()
        if len(g) >= 6 and g["cr"].std() > 1e-9:
            r, p = stats.pearsonr(g["cr"], g["score"])
            bn.append({"run_id": run, "r_score_conflict": r, "p": p,
                       "n_blocks": len(g)})
    if bn:
        pd.DataFrame(bn).to_csv(OUT / "xd_block_neural.csv", index=False)
    print(f"  Part5: environment slopes {len(env)}")
    return env


def part5_environment_weight(combined):
    """Coefficient-space environment effect on the planning weight.

    Planning weight in low/high block-conflict-rate thirds and the
    `block_conflict_rate x planning` interaction (conflict trials).
    """
    rows = []
    for (src, pid), g in combined.groupby(["source", "participant"]):
        w = _planning_weight_moderated(g[g["conflict"] == 1],
                                       "block_conflict_rate")
        if w is None:
            continue
        w.update({"source": src, "participant": pid,
                  "dataset": g["dataset"].iloc[0]})
        rows.append(w)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "xd_environment_weight.csv", index=False)
    print(f"  Part5w: environment-weight rows {len(out)}")
    return out


# %% [markdown]
# ## Part 6 - Individual differences

# %%
def part6_individual(summ, threat):
    """Place EMU participants in the cloud trait space.

    The mixture's `p_plan_base` saturates at its 0.98 bound for most
    participants, so we use model-free quantities instead: the conflict-trial
    planning rate and the per-participant slope of planning on `ball_y`
    (threat sensitivity).
    """
    # the individual trait space stays on the conflict-trial definition
    slopes = (threat[threat["trials"] == "conflict"]
              .drop_duplicates("participant").set_index("participant")
              ["slope_ball_y"])
    s = summ.copy()
    s["slope_ball_y"] = s["participant"].map(slopes)
    cloud = s[s["source"] == "cloud"].dropna(subset=["p_plan", "slope_ball_y"])
    emu = s[s["source"] == "emu"].dropna(subset=["p_plan", "slope_ball_y"])
    cloud_w = cloud.dropna(subset=["planning_weight"])
    rows = []
    for r in emu.itertuples():
        rows.append({
            "participant": r.participant, "p_plan": r.p_plan,
            "planning_weight": r.planning_weight,
            "slope_ball_y": r.slope_ball_y,
            "pctile_p_plan_vs_cloud": float((cloud["p_plan"] < r.p_plan).mean()),
            "pctile_planning_weight_vs_cloud": (
                float((cloud_w["planning_weight"] < r.planning_weight).mean())
                if len(cloud_w) and np.isfinite(r.planning_weight) else np.nan),
            "pctile_slope_vs_cloud": float(
                (cloud["slope_ball_y"] < r.slope_ball_y).mean()),
            "n_cloud": len(cloud),
        })
    ind = pd.DataFrame(rows)
    ind.to_csv(OUT / "xd_individual.csv", index=False)
    print(f"  Part6: individual rows {len(ind)}")
    return ind


# %%
def main():
    print("Harmonizing cloud + EMU ...")
    combined = build_combined()
    print(f"  combined: {combined['source'].value_counts().to_dict()}")
    print("\nPart 1 ..."); summ, mix = part1_behavior(combined)
    print("\nPart 2 ..."); part2_linkage(combined)
    print("\nPart 2b ..."); part2b_neural_profiles()
    print("\nPart 3 ..."); part3_transfer(combined)
    print("\nPart 4 ..."); threat = part4_threat(combined)
    part4_threat_weight(combined)
    part4_coef_profile(combined)
    part4_coef_profile_drift(combined)
    print("\nPart 5 ..."); part5_environment(combined)
    part5_environment_weight(combined)
    print("\nPart 6 ..."); part6_individual(summ, threat)
    print(f"\nDone. Outputs in {OUT}")


if __name__ == "__main__":
    main()
