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
# # Neural planning profiles across ball height
#
# Directly relate population activity to the behavioural planning questions:
#
#   A  planning_bally_profile.csv   planning_vs_greedy decode within ball-y bins
#                                   (choice `post` [0,1000] and entry `pre`)
#   B  value_bally_profile.csv      ridge decode of greedy_gap / planning_gap /
#                                   conflict_mag within ball-y bins
#   C  neural_choice_bally.csv      OOF neural planning score x ball_y on choice
#   D  region_bally_profile.csv     A repeated per mesial-temporal region
#   E  (entry-anchored windows are included in A/B/C as `window="entry"`)
#
# Ball height is the entry-frame `ball_y - camera_y` (`ball_y_at_top`), matching
# the behavioural analyses. Small values = ball high on screen = near death
# (see `classify_trials.find_deaths`).
#
# NOTE: the EMU sessions are all-drift, so there is no neural drift-vs-follow
# contrast; these analyses characterize threat modulation within drift.
#
# Outputs (analysis/neural_outputs/<run>/):
#   planning_bally_profile.csv
#   value_bally_profile.csv
#   neural_choice_bally.csv
#   region_bally_profile.csv
#
# Run with:
#   .venv-analysis\Scripts\python.exe analysis\neural_planning_profiles.py --run yfz_1

# %%
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score

from neural_common import get_run, out_dir, unit_regions
from neural_lda_decoding import (
    RNG_SEED, PERM_FOLDS, rate_features, pca_features,
    decode_balanced_accuracy, permutation_p, make_transform,
    build_label_vectors,
)
from neural_confound_audit import (
    load_binned, build_regressors, build_nuisances,
    leave_block_out_acc, build_position,
)
from neural_decoding_confounds import (
    build_nuisance_table, match_schemes, stratify_match_indices,
)
from neural_selectivity import (
    load_data as sel_load_data, trial_rates, modulation_index, fdr_bh,
)
from neural_region_decoding import oof_scores, region_unit_masks
from neural_entry_locked import rate_features_entry, pca_features_entry
from neural_continuous_decoding import (
    residualize, residualize_features, cv_regression, permutation_corr_p,
)

# ---------------------------- Configuration ----------------------------
_RUN = get_run()
OUT_DIR = out_dir(_RUN.run_id)
TRIAL_TABLE = OUT_DIR / "trial_table.csv"
TRIAL_LABELS = OUT_DIR / "trial_labels.csv"
ENTRY_NPZ = OUT_DIR / "segmented_spikes_entrylocked_binned.npz"
BEHAVIOR_PATH = _RUN.behavior_path

POST = (0.0, 1000.0)      # choice-anchored post-choice
ENTRY = (0.0, 250.0)      # entry-anchored pre-decision
N_BINS = 5
PERM_BIN = 150            # reduced permutation budget for bin-wise tests
PROF_FOLDS = 5
PROF_REPEATS = 2
PROF_PERM = 100
TARGETS = ["greedy_gap", "planning_gap", "conflict_mag"]
CONTROLS = ["none", "target", "both"]
# -----------------------------------------------------------------------


# %%
def rel_y_at(times, behavior_path):
    """`ball_y - camera_y` at the given behavioral-clock times (nearest frame)."""
    with open(behavior_path, encoding="utf-8") as fh:
        data = json.load(fh)
    t, rel = [], []
    for b in data.get("blocks", []):
        if b.get("block_index", 0) < 4:
            continue
        gs = b.get("game_states") or {}
        if not gs:
            continue
        t.append(np.asarray(gs["time"], float))
        rel.append(np.asarray(gs["ball_y"], float)
                   - np.asarray(gs["camera_y"], float))
    if not t:
        return np.full(len(times), np.nan)
    t = np.concatenate(t); rel = np.concatenate(rel)
    o = np.argsort(t)
    t, rel = t[o], rel[o]
    idx = np.clip(np.searchsorted(t, np.asarray(times, float)), 0,
                  len(t) - 1)
    return rel[idx]


def keep_index(idx, n):
    if idx is None:
        return np.ones(n, bool)
    idx = np.asarray(idx)
    if idx.dtype == bool:
        return idx
    m = np.zeros(n, bool)
    m[idx] = True
    return m


def _z(v):
    v = np.asarray(v, float)
    s = v.std()
    return (v - v.mean()) / (s if s > 0 else 1.0)


def load_entry_binned():
    z = np.load(ENTRY_NPZ, allow_pickle=True)
    return z["binned"], z["bin_centers"]


def build_geometry():
    """Per-trial geometry aligned to the binned trial axis (trial_labels order)."""
    lab = pd.read_csv(TRIAL_LABELS)
    tt = pd.read_csv(TRIAL_TABLE)
    lab = lab.merge(tt[["trial_id", "hole_locations", "entry_time_ms"]],
                    on="trial_id", how="left")
    holes = lab["hole_locations"].map(json.loads)
    hL, hR = holes.str[0].to_numpy(), holes.str[1].to_numpy()
    chosen = lab["choice_hole"].to_numpy()
    unchosen = np.where(chosen == hL, hR, hL)
    entry = lab["entry_hole"].to_numpy()
    goal = lab["goal_hole"].to_numpy()

    c1 = np.abs(chosen - entry); u1 = np.abs(unchosen - entry)
    c2 = c1 + np.abs(chosen - goal); u2 = u1 + np.abs(unchosen - goal)

    d = pd.DataFrame({
        "trial_id": lab["trial_id"].to_numpy(),
        "block_index": lab["block_index"].to_numpy(),
        "sequence_index": lab["sequence_index"].to_numpy(),
        "entry_hole": entry,
        "goal_hole": goal,
        "entry_time_ms": lab["entry_time_ms"].to_numpy(),
        "diff_1step": c1 - u1,
        "diff_planning": c2 - u2,
        "chose_planning": (chosen == lab["planning_optimal_hole"].to_numpy()).astype(int),
        "conflict": (~lab["agree"].astype(bool)).astype(int),
    })
    d["ball_y_entry"] = rel_y_at(d["entry_time_ms"].to_numpy(), BEHAVIOR_PATH)

    # incoming direction computed in block/sequence order, then restored
    s = d.sort_values(["block_index", "sequence_index"]).reset_index(drop=True)
    prev_goal = s["goal_hole"].shift(1)
    valid = ((s["block_index"].shift(1) == s["block_index"])
             & (s["sequence_index"].shift(1) + 1 == s["sequence_index"]))
    s["incoming_direction"] = np.where(
        valid, -np.sign(prev_goal - s["entry_hole"]), np.nan)
    order = {tid: i for i, tid in enumerate(d["trial_id"].to_numpy())}
    s["_ord"] = s["trial_id"].map(order)
    return s.sort_values("_ord").reset_index(drop=True)


def bin_by_ball_y(vals, n_bins=N_BINS):
    """Integer quantile bins of `vals` (0 = lowest ball_y = nearest death)."""
    return pd.qcut(pd.Series(vals), q=n_bins, labels=False, duplicates="drop")


# %%
def planning_bally_profile():
    """A + E: planning_vs_greedy decode within ball-y bins."""
    binned, unit_ids, bin_centers = load_binned(OUT_DIR)
    ebinned, ecenters = load_entry_binned()
    lab = pd.read_csv(TRIAL_LABELS)
    lab["move_dir_vx"] = 0
    lv = build_label_vectors(lab)
    y, idx = lv["planning_vs_greedy"]
    n = len(y)
    base_all = keep_index(idx, n)
    geom = build_geometry()
    ball_y = geom["ball_y_entry"].to_numpy()

    feats = {
        ("post", "rate"): rate_features(binned, bin_centers, *POST),
        ("post", "pca"): pca_features(binned, bin_centers, *POST),
        ("entry", "rate"): rate_features_entry(ebinned, ecenters, *ENTRY),
        ("entry", "pca"): pca_features_entry(ebinned, ecenters, *ENTRY),
    }

    rows = []
    for (window, rep), X in feats.items():
        transform = make_transform(rep)
        valid = base_all & np.isfinite(ball_y)
        bins = bin_by_ball_y(ball_y[valid], N_BINS)
        bin_ids = sorted(set(bins.dropna().astype(int)))
        vals = ball_y[valid]
        for bid in bin_ids:
            sel_mask = (bins == bid).to_numpy()
            sel = np.flatnonzero(valid)[sel_mask]
            if len(sel) < 30 or len(np.unique(y[sel])) < 2:
                continue
            acc_m, acc_s = decode_balanced_accuracy(
                y[sel], X[sel], transform, PROF_FOLDS, PROF_REPEATS, RNG_SEED)
            p = permutation_p(y[sel], X[sel], transform, acc_m,
                              n_perm=PERM_BIN, n_folds=PERM_FOLDS,
                              seed=RNG_SEED)
            rows.append({"window": window, "rep": rep, "bin": int(bid),
                         "ball_y_center": float(np.nanmean(vals[sel_mask])),
                         "n_trials": int(len(sel)),
                         "acc_mean": acc_m, "acc_std": acc_s,
                         "chance": 0.5, "perm_p": p})
            print(f"  planning {window}/{rep} bin {bid}: acc {acc_m:.3f} "
                  f"(p {p:.3f}, n={len(sel)})")
    return pd.DataFrame(rows)


# %%
def value_bally_profile():
    """B + E: ridge decode of model values within ball-y bins."""
    binned, unit_ids, bin_centers = load_binned(OUT_DIR)
    ebinned, ecenters = load_entry_binned()
    regressors = build_regressors(OUT_DIR)
    nuisance, _ = build_nuisances(OUT_DIR)
    geom = build_geometry()
    ball_y = geom["ball_y_entry"].to_numpy()

    feats = {
        "post": rate_features(binned, bin_centers, *POST),
        "entry": rate_features_entry(ebinned, ecenters, *ENTRY),
    }

    rows = []
    for window, X in feats.items():
        for target in TARGETS:
            y_reg = regressors[target]
            valid = ~np.isnan(y_reg) & np.isfinite(ball_y)
            vals = ball_y[valid]
            bins = bin_by_ball_y(vals, N_BINS)
            for bid in sorted(set(bins.dropna().astype(int))):
                sel_mask = (bins == bid).to_numpy()
                sel = np.flatnonzero(valid)[sel_mask]
                if len(sel) < 30:
                    continue
                yb = y_reg[sel]
                Xb = X[sel]
                nu = nuisance[sel]
                for control in CONTROLS:
                    yc, Xc = yb, Xb
                    if control in ("target", "both"):
                        yc = residualize(yb, nu)
                    if control == "both":
                        Xc = residualize_features(Xb, nu)
                    if np.std(yc) < 1e-12:
                        continue
                    (r2_m, r2_s, corr_m, corr_s), _ = cv_regression(
                        yc, Xc, PROF_FOLDS, PROF_REPEATS, RNG_SEED)
                    p = permutation_corr_p(yc, Xc, corr_m, PROF_PERM,
                                           PROF_FOLDS, RNG_SEED)
                    rows.append({"window": window, "target": target,
                                 "bin": int(bid),
                                 "ball_y_center": float(np.nanmean(
                                     ball_y[valid][sel_mask])),
                                 "control": control, "n_trials": int(len(sel)),
                                 "r2_mean": r2_m, "corr_mean": corr_m,
                                 "corr_std": corr_s, "perm_p": p})
            print(f"  value {window}/{target}: done "
                  f"({sum(r['window']==window and r['target']==target for r in rows)} cells)")
    return pd.DataFrame(rows)


# %%
def neural_choice_bally(n_perm=200):
    """C + E: does the OOF neural planning score's effect on choice depend on
    ball height?"""
    binned, unit_ids, bin_centers = load_binned(OUT_DIR)
    ebinned, ecenters = load_entry_binned()
    lab = pd.read_csv(TRIAL_LABELS)
    lab["move_dir_vx"] = 0
    lv = build_label_vectors(lab)
    y, idx = lv["planning_vs_greedy"]
    n = len(y)
    base_pg = keep_index(idx, n)
    geom = build_geometry()
    conf = geom["conflict"].to_numpy().astype(bool)

    feats = {
        "post": rate_features(binned, bin_centers, *POST),
        "entry": rate_features_entry(ebinned, ecenters, *ENTRY),
    }

    rows = []
    for window, X in feats.items():
        base = conf & base_pg & np.isfinite(geom["ball_y_entry"].to_numpy())
        base = base & geom[["diff_1step", "diff_planning",
                            "incoming_direction"]].notna().all(axis=1).to_numpy()
        sel = np.flatnonzero(base)
        if len(sel) < 60 or len(np.unique(y[sel])) < 2:
            continue
        s = oof_scores(y[sel], X[sel])
        d = geom.iloc[sel].copy()
        d["s"] = s
        d = d.dropna(subset=["diff_1step", "diff_planning",
                             "incoming_direction", "ball_y_entry", "s"])
        if len(d) < 60 or len(np.unique(d["chose_planning"])) < 2:
            continue
        z1 = _z(d["diff_1step"].values)
        zp = _z(d["diff_planning"].values)
        zi = _z(d["incoming_direction"].values)
        zb = _z(d["ball_y_entry"].values)
        zs = _z(d["s"].values)
        yv = d["chose_planning"].astype(int).values
        terms = ["greedy", "planning", "incoming", "neural",
                 "neural_x_ball_y"]

        def fit(zbv, zsv):
            Xm = np.column_stack([z1, zp, zi, zsv, zsv * zbv])
            m = LogisticRegression(C=1.0, max_iter=5000).fit(Xm, yv)
            return m.coef_[0]

        obs = fit(zb, zs)
        # permutation nulls: shuffle ball_y for the interaction, shuffle the
        # neural score for its main effect (regularized fit avoids separation)
        rng = np.random.default_rng(RNG_SEED)
        null_int = np.array([fit(rng.permutation(zb), zs)[4]
                             for _ in range(n_perm)])
        null_neu = np.array([fit(zb, rng.permutation(zs))[3]
                             for _ in range(n_perm)])
        p_terms = {"neural": float(np.mean(np.abs(null_neu) >= abs(obs[3]))),
                   "neural_x_ball_y": float(
                       np.mean(np.abs(null_int) >= abs(obs[4])))}
        for i, term in enumerate(terms):
            rows.append({"window": window, "term": term,
                         "coef": float(obs[i]), "se": np.nan,
                         "p": p_terms.get(term, np.nan),
                         "n_trials": int(len(d))})
        print(f"  choice {window}: neural_x_ball_y {obs[4]:+.3f} "
              f"(perm p {p_terms['neural_x_ball_y']:.3f}); "
              f"neural {obs[3]:+.3f} (p {p_terms['neural']:.3f})")
    return pd.DataFrame(rows)


# %%
def region_bally_profile():
    """D: planning_vs_greedy decode within ball-y bins, per region."""
    binned, unit_ids, bin_centers = load_binned(OUT_DIR)
    lab = pd.read_csv(TRIAL_LABELS)
    lab["move_dir_vx"] = 0
    lv = build_label_vectors(lab)
    y, idx = lv["planning_vs_greedy"]
    n = len(y)
    base_all = keep_index(idx, n)
    geom = build_geometry()
    ball_y = geom["ball_y_entry"].to_numpy()
    X = rate_features(binned, bin_centers, *POST)
    transform = make_transform("rate")
    masks = region_unit_masks(unit_ids)

    rows = []
    for region, mask in masks.items():
        n_units = int(mask.sum())
        if n_units < 8:
            continue
        valid = base_all & np.isfinite(ball_y)
        bins = bin_by_ball_y(ball_y[valid], N_BINS)
        for bid in sorted(set(bins.dropna().astype(int))):
            sel = np.flatnonzero(valid)[(bins == bid).to_numpy()]
            if len(sel) < 30 or len(np.unique(y[sel])) < 2:
                continue
            Xr = X[sel][:, mask]
            acc_m, acc_s = decode_balanced_accuracy(
                y[sel], Xr, transform, PROF_FOLDS, PROF_REPEATS, RNG_SEED)
            p = permutation_p(y[sel], Xr, transform, acc_m, n_perm=PERM_BIN,
                              n_folds=PERM_FOLDS, seed=RNG_SEED)
            rows.append({"region": region, "n_units": n_units,
                         "bin": int(bid),
                         "ball_y_center": float(np.nanmean(
                             ball_y[valid][(bins == bid).to_numpy()])),
                         "n_trials": int(len(sel)),
                         "acc_mean": acc_m, "acc_std": acc_s,
                         "chance": 0.5, "perm_p": p})
            print(f"  region {region} bin {bid}: acc {acc_m:.3f} "
                  f"(p {p:.3f}, n={len(sel)}, {n_units} units)")
    return pd.DataFrame(rows)


# %% [markdown]
# ## Tier 1 - confound-controlled ball-height contrast
#
# Near vs far ball-y planning decode under: unmatched, nuisance-matched
# (side x RT-quartile x block-half), position+kinematic residualization, and
# block-held-out. Plus a cluster-based permutation across the 5 bins.

# %%
def _aligned_nuisance(geom):
    """build_nuisance_table rows reindexed to `geom` (trial_labels) order."""
    nuis = build_nuisance_table()
    return nuis.set_index("trial_id").reindex(geom["trial_id"]).reset_index()


def _near_far(ball_y, sel, pct=1 / 3):
    vals = ball_y[sel]
    q1, q2 = np.quantile(vals, [pct, 1 - pct])
    return sel[vals <= q1], sel[vals >= q2]


def _gacc(y, X, transform, sel):
    if len(sel) < 20 or len(np.unique(y[sel])) < 2:
        return np.nan
    m, _ = decode_balanced_accuracy(y[sel], X[sel], transform, PROF_FOLDS,
                                    PROF_REPEATS, RNG_SEED)
    return float(m)


def planning_bally_controlled(n_perm=200):
    binned, unit_ids, bc = load_binned(OUT_DIR)
    lab = pd.read_csv(TRIAL_LABELS); lab["move_dir_vx"] = 0
    lv = build_label_vectors(lab)
    y, idx = lv["planning_vs_greedy"]
    base = keep_index(idx, len(y))
    geom = build_geometry()
    ball_y = geom["ball_y_entry"].to_numpy()
    nuis = _aligned_nuisance(geom)
    pos = build_position(_RUN.run_id, OUT_DIR)
    X = rate_features(binned, bc, *POST)
    tr = make_transform("rate")
    rng = np.random.default_rng(RNG_SEED)
    sel_all = np.flatnonzero(base & np.isfinite(ball_y))
    near, far = _near_far(ball_y, sel_all)

    rows = [{"control": "unmatched", "rep": "rate",
             "n_near": len(near), "n_far": len(far),
             "near_acc": _gacc(y, X, tr, near), "far_acc": _gacc(y, X, tr, far)}]

    all_idx = np.concatenate([near, far])
    g = np.r_[np.zeros(len(near), int), np.ones(len(far), int)]
    strata = match_schemes(nuis)["matched_side_rt_block"].iloc[all_idx] \
        .reset_index(drop=True)
    keep = stratify_match_indices(g, strata, rng)
    if len(keep):
        nm = all_idx[keep]; gm = g[keep]
        rows.append({"control": "matched_side_rt_block", "rep": "rate",
                     "n_near": int((gm == 0).sum()),
                     "n_far": int((gm == 1).sum()),
                     "near_acc": _gacc(y, X, tr, nm[gm == 0]),
                     "far_acc": _gacc(y, X, tr, nm[gm == 1])})

    nmat = np.column_stack([
        np.ones(len(X)),
        np.nan_to_num(nuis[["rt_ms", "ball_time_ms", "trial_duration_ms",
                            "block_index", "side"]].to_numpy(float)),
        pos[:, [0, 2]]])  # ball_x, camera_y (exclude relative-y = ball_y)
    Xc = residualize_features(X, nmat)
    rows.append({"control": "position_resid", "rep": "rate",
                 "n_near": len(near), "n_far": len(far),
                 "near_acc": _gacc(y, Xc, tr, near),
                 "far_acc": _gacc(y, Xc, tr, far)})

    groups = geom["block_index"].to_numpy()
    rows.append({"control": "block_heldout", "rep": "rate",
                 "n_near": len(near), "n_far": len(far),
                 "near_acc": leave_block_out_acc(y[near].astype(int), X[near],
                                                 groups[near], tr),
                 "far_acc": leave_block_out_acc(y[far].astype(int), X[far],
                                                groups[far], tr)})

    out = pd.DataFrame(rows)
    out["diff"] = out["near_acc"] - out["far_acc"]
    obs = float(out.loc[out["control"] == "unmatched", "diff"].iloc[0])
    pooled = np.concatenate([near, far])
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(pooled)
        null[i] = _gacc(y, X, tr, perm[:len(near)]) - \
            _gacc(y, X, tr, perm[len(near):])
    out["perm_p_diff"] = np.where(out["control"] == "unmatched",
                                  float(np.mean(np.abs(null) >= abs(obs)))
                                  if np.isfinite(obs) else np.nan, np.nan)
    print("  controlled near/far: " +
          ", ".join(f"{r.control} d={r.diff:+.3f}" for r in out.itertuples()))
    return out


def _cluster_stat(stats):
    best, i, n = 0.0, 0, len(stats)
    while i < n:
        s = np.sign(stats[i])
        if s == 0:
            i += 1
            continue
        total, j = 0.0, i
        while j < n and np.sign(stats[j]) == s:
            total += stats[j]; j += 1
        best = max(best, abs(total)); i = j
    return best


def planning_bally_clusters(n_perm=200):
    binned, unit_ids, bc = load_binned(OUT_DIR)
    lab = pd.read_csv(TRIAL_LABELS); lab["move_dir_vx"] = 0
    lv = build_label_vectors(lab)
    y, idx = lv["planning_vs_greedy"]
    base = keep_index(idx, len(y))
    geom = build_geometry()
    ball_y = geom["ball_y_entry"].to_numpy()
    sel = np.flatnonzero(base & np.isfinite(ball_y))
    bids = bin_by_ball_y(ball_y[sel], N_BINS)
    bins = [sel[(bids == b).to_numpy()]
            for b in sorted(bids.dropna().astype(int).unique())]
    ebin, ec = load_entry_binned()
    feats = {"post": rate_features(binned, bc, *POST),
             "entry": rate_features_entry(ebin, ec, *ENTRY)}
    rows = []
    for window, X in feats.items():
        tr = make_transform("rate")

        def ab(yy):
            a = np.full(len(bins), np.nan)
            for k, bs in enumerate(bins):
                if len(bs) < 20 or len(np.unique(yy[bs])) < 2:
                    continue
                m, _ = decode_balanced_accuracy(yy[bs], X[bs], tr,
                                                PROF_FOLDS, 1, RNG_SEED)
                a[k] = m - 0.5
            return a

        obs = ab(y)
        obs_c = _cluster_stat(np.nan_to_num(obs))
        rng = np.random.default_rng(RNG_SEED)
        null = np.array([_cluster_stat(np.nan_to_num(ab(rng.permutation(y))))
                         for _ in range(n_perm)])
        p = float(np.mean(null >= obs_c))
        rows.append({"window": window, "rep": "rate", "n_bins": len(bins),
                     "cluster_stat": obs_c,
                     "max_bin_diff": float(np.nanmax(obs)),
                     "min_bin_diff": float(np.nanmin(obs)), "perm_p": p})
        print(f"  cluster {window}: stat {obs_c:.3f} (p {p:.3f})")
    return pd.DataFrame(rows)


# %% [markdown]
# ## Tier 2a - cross-decoding across ball height

# %%
def _train_test_acc(X, y, tr_idx, te_idx, transform):
    if (len(tr_idx) < 20 or len(te_idx) < 20
            or len(np.unique(y[tr_idx])) < 2
            or len(np.unique(y[te_idx])) < 2):
        return np.nan
    Xtr, Xte = transform(X[tr_idx], y[tr_idx], X[te_idx])
    lda = LinearDiscriminantAnalysis(solver="eigen",
                                     shrinkage="auto").fit(Xtr, y[tr_idx])
    return float(balanced_accuracy_score(y[te_idx], lda.predict(Xte)))


def planning_cross_bally():
    binned, unit_ids, bc = load_binned(OUT_DIR)
    lab = pd.read_csv(TRIAL_LABELS); lab["move_dir_vx"] = 0
    lv = build_label_vectors(lab)
    y, idx = lv["planning_vs_greedy"]
    base = keep_index(idx, len(y))
    geom = build_geometry()
    ball_y = geom["ball_y_entry"].to_numpy()
    sel = np.flatnonzero(base & np.isfinite(ball_y))
    near, far = _near_far(ball_y, sel)
    ebin, ec = load_entry_binned()
    feats = {
        ("post", "rate"): rate_features(binned, bc, *POST),
        ("post", "pca"): pca_features(binned, bc, *POST),
        ("entry", "rate"): rate_features_entry(ebin, ec, *ENTRY),
        ("entry", "pca"): pca_features_entry(ebin, ec, *ENTRY),
    }
    bids = bin_by_ball_y(ball_y[sel], N_BINS)
    binlist = [sel[(bids == b).to_numpy()]
               for b in sorted(bids.dropna().astype(int).unique())]
    rows = []
    for (window, rep), X in feats.items():
        tr = make_transform(rep)
        nf = np.nan
        for label, ti, te in [("near_to_far", near, far),
                              ("far_to_near", far, near)]:
            acc = _train_test_acc(X, y, ti, te, tr)
            if label == "near_to_far":
                nf = acc
            rows.append({"window": window, "rep": rep, "direction": label,
                         "n_train": len(ti), "n_test": len(te), "acc": acc})
        for i, bi in enumerate(binlist):
            for j, bj in enumerate(binlist):
                rows.append({"window": window, "rep": rep,
                             "direction": f"bin{i}_to_bin{j}",
                             "n_train": len(bi), "n_test": len(bj),
                             "acc": _train_test_acc(X, y, bi, bj, tr)})
        print(f"  cross {window}/{rep}: near->far {nf:.3f}")
    return pd.DataFrame(rows)


# %% [markdown]
# ## Tier 2b - single-unit threat x planning (per-unit interaction + FDR)

# %%
def unit_threat_planning(n_perm=1000):
    units, table, times_by_unit, _ = sel_load_data()
    rates = trial_rates(times_by_unit, table, *POST)
    ball_y = rel_y_at(table["entry_time_ms"].to_numpy(), BEHAVIOR_PATH)
    cond = table["condition"].to_numpy()
    meta = pd.read_csv(OUT_DIR / "unit_metadata.csv")
    reg = unit_regions(meta)
    conflict = np.isin(cond, ["planning", "greedy"]) & np.isfinite(ball_y)
    cy = ball_y[conflict]
    q1, q2 = np.quantile(cy, [1 / 3, 2 / 3])
    near = conflict & (ball_y <= q1)
    far = conflict & (ball_y >= q2)
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for uid in units["unit_id"]:
        uid = int(uid)
        r = rates.get(uid)
        if r is None:
            continue
        pn = r[near & (cond == "planning")]
        gn = r[near & (cond == "greedy")]
        pf = r[far & (cond == "planning")]
        gf = r[far & (cond == "greedy")]
        if min(len(pn), len(gn), len(pf), len(gf)) < 10:
            continue
        mi_n = modulation_index(pn, gn)
        mi_f = modulation_index(pf, gf)
        if not (np.isfinite(mi_n) and np.isfinite(mi_f)):
            continue
        obs = mi_n - mi_f
        pooln = np.concatenate([pn, gn]); poolf = np.concatenate([pf, gf])
        na, nb = len(pn), len(pf)
        null = np.empty(n_perm)
        for i in range(n_perm):
            a = rng.permutation(pooln); b = rng.permutation(poolf)
            null[i] = modulation_index(a[:na], a[na:]) - \
                modulation_index(b[:nb], b[nb:])
        rows.append({"unit_id": uid, "region": reg.get(uid, ""),
                     "mi_near_death": mi_n, "mi_far": mi_f,
                     "interaction": obs, "perm_p": float(np.mean(
                         np.abs(null) >= abs(obs))),
                     "n_near_plan": len(pn), "n_far_plan": len(pf)})
    out = pd.DataFrame(rows)
    if len(out):
        out["q_fdr"] = fdr_bh(out["perm_p"].to_numpy())
        out["significant"] = out["q_fdr"] < 0.05
    print(f"  unit threat x planning: {len(out)} units, "
          f"{int(out['significant'].sum()) if len(out) else 0} significant")
    return out


# %% [markdown]
# ## Tier 2d - trial-history effects

# %%
def planning_history(n_perm=200):
    binned, unit_ids, bc = load_binned(OUT_DIR)
    lab = pd.read_csv(TRIAL_LABELS); lab["move_dir_vx"] = 0
    lv = build_label_vectors(lab)
    y_pg, idx = lv["planning_vs_greedy"]
    base = keep_index(idx, len(y_pg))
    geom = build_geometry()
    nuis = _aligned_nuisance(geom)
    prev = nuis["prev_condition"].to_numpy()
    prev_plan = np.where(pd.isna(prev), np.nan,
                         (prev == "planning").astype(float))
    X = rate_features(binned, bc, *POST)
    ebin, ec = load_entry_binned()
    Xe = rate_features_entry(ebin, ec, *ENTRY)
    ok = (geom["conflict"].astype(bool).to_numpy() & base
          & np.isfinite(prev_plan)
          & geom[["diff_1step", "diff_planning", "incoming_direction",
                  "ball_y_entry"]].notna().all(axis=1).to_numpy())
    sel = np.flatnonzero(ok)
    z1 = _z(geom["diff_1step"].values[sel])
    zp = _z(geom["diff_planning"].values[sel])
    zi = _z(geom["incoming_direction"].values[sel])
    zb = _z(geom["ball_y_entry"].values[sel])
    zprev = prev_plan[sel]
    yv = y_pg[sel].astype(int)

    rows = []
    terms = ["greedy", "planning", "incoming", "threat", "prev_plan",
             "threat_x_prev"]
    Xb = np.column_stack([z1, zp, zi, zb, zprev, zb * zprev])
    m = LogisticRegression(C=1.0, max_iter=5000).fit(Xb, yv)
    coef = m.coef_[0]
    rng = np.random.default_rng(RNG_SEED)
    null_prev = np.array([LogisticRegression(C=1.0, max_iter=5000).fit(
        np.column_stack([z1, zp, zi, zb, rng.permutation(zprev),
                         zb * rng.permutation(zprev)]), yv).coef_[0][5]
        for _ in range(n_perm)])
    p_prev = float(np.mean(np.abs(null_prev) >= abs(coef[5])))
    for i, term in enumerate(terms):
        rows.append({"window": "behavioral", "model": "geometry+threat",
                     "term": term, "coef": float(coef[i]),
                     "p": p_prev if term == "threat_x_prev" else np.nan,
                     "n_trials": len(sel)})

    for window, Xw in [("post", X), ("entry", Xe)]:
        s = _z(oof_scores(y_pg[sel], Xw[sel]))
        Xn = np.column_stack([z1, zp, zi, zb, s, s * zb, s * zprev])
        mn = LogisticRegression(C=1.0, max_iter=5000).fit(Xn, yv)
        cn = mn.coef_[0]
        null_np = np.array([
            LogisticRegression(C=1.0, max_iter=5000).fit(
                np.column_stack([z1, zp, zi, zb,
                                 rng.permutation(s), s * zb,
                                 rng.permutation(s) * zprev]), yv).coef_[0][4]
            for _ in range(n_perm)])
        p_np = float(np.mean(np.abs(null_np) >= abs(cn[4])))
        for i, term in enumerate(["greedy", "planning", "incoming", "threat",
                                  "neural", "neural_x_threat",
                                  "neural_x_prev"]):
            rows.append({"window": window, "model": "geometry+neural",
                         "term": term, "coef": float(cn[i]),
                         "p": p_np if term == "neural_x_prev" else np.nan,
                         "n_trials": len(sel)})
    return pd.DataFrame(rows)


# %%
def main():
    print(f"Run {_RUN.run_id} ({_RUN.participant})")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[A/E] planning decode across ball-y bins")
    a = planning_bally_profile()
    a.to_csv(OUT_DIR / "planning_bally_profile.csv", index=False)
    print(f"  saved planning_bally_profile.csv ({len(a)} rows)")

    print("\n[B/E] value decode across ball-y bins")
    b = value_bally_profile()
    b.to_csv(OUT_DIR / "value_bally_profile.csv", index=False)
    print(f"  saved value_bally_profile.csv ({len(b)} rows)")

    print("\n[C/E] neural planning score x ball_y on choice")
    c = neural_choice_bally()
    c.to_csv(OUT_DIR / "neural_choice_bally.csv", index=False)
    print(f"  saved neural_choice_bally.csv ({len(c)} rows)")

    print("\n[D] region planning decode across ball-y bins")
    d = region_bally_profile()
    d.to_csv(OUT_DIR / "region_bally_profile.csv", index=False)
    print(f"  saved region_bally_profile.csv ({len(d)} rows)")

    print("\n[T1] confound-controlled near/far planning decode")
    t1 = planning_bally_controlled()
    t1.to_csv(OUT_DIR / "planning_bally_controlled.csv", index=False)
    t1c = planning_bally_clusters()
    t1c.to_csv(OUT_DIR / "planning_bally_clusters.csv", index=False)
    print(f"  saved planning_bally_controlled.csv ({len(t1)} rows), "
          f"planning_bally_clusters.csv ({len(t1c)} rows)")

    print("\n[T2a] cross-decoding across ball height")
    x = planning_cross_bally()
    x.to_csv(OUT_DIR / "planning_cross_bally.csv", index=False)
    print(f"  saved planning_cross_bally.csv ({len(x)} rows)")

    print("\n[T2b] single-unit threat x planning")
    u = unit_threat_planning()
    u.to_csv(OUT_DIR / "unit_threat_planning.csv", index=False)
    print(f"  saved unit_threat_planning.csv ({len(u)} rows)")

    print("\n[T2d] trial-history effects")
    h = planning_history()
    h.to_csv(OUT_DIR / "planning_history.csv", index=False)
    print(f"  saved planning_history.csv ({len(h)} rows)")


if __name__ == "__main__":
    main()
