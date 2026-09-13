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
# # Neural decoding confound diagnostics + matched re-decoding
#
# The LDA decoding in `neural_lda_decoding.py` reports balanced accuracy per
# hypothesis, but a high accuracy does not by itself prove the hypothesis is
# represented: trial labels (side, planning_vs_greedy, ...) are defined from the
# hole geometry and the participant's choice, so they are mechanically entangled
# with per-trial behavior (reaction time, ball timing, block/drift, previous
# trial) that the decoder could be reading instead. This script makes those
# confounds explicit and then re-tests the decoders on subsets that are matched
# on the nuisance variables.
#
# Part 1 — nuisance table + association tests.
#   Builds one row per trial with continuous (RT, ball time, trial duration,
#   chosen-hole greedy/planning costs, block index, position in block) and
#   categorical (side, move_dir, previous trial's condition) nuisances, then
#   tests every hypothesis x nuisance association (point-biserial r / eta^2 for
#   continuous, Cramér's V for categorical) against a label-shuffle null. This
#   is the "confound table": which nuisances correlate with which label.
#
# Part 2 — matched re-decoding.
#   Re-runs the post-window decoder (rate + pca) for the key binary hypotheses
#   on subsets where the two classes have been matched on (a) side only,
#   (b) side x RT quartile x block half, by keeping equal per-stratum counts.
#   If accuracy collapses under matching, the original number rode on the
#   confound.
#
# Statistics mirror neural_lda_decoding.py: repeated stratified k-fold CV,
# balanced accuracy, chance = 1/n_classes, empirical permutation p.
#
# Outputs:
#   decoding_nuisance_table.csv      per-trial nuisances (one row per trial)
#   decoding_label_association.csv   hypothesis x nuisance association table
#   decoding_matched_redecode.csv    matched re-decoding results
#
# Run with:
#   C:\Users\manik\AppData\Local\Programs\Python\Python311\python.exe analysis\neural_decoding_confounds.py

# %%
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd
from pathlib import Path

from neural_lda_decoding import (
    WINDOWS, REPS, RNG_SEED, PERM_FOLDS,
    load_binned, rate_features, pca_features, make_transform,
    build_label_vectors, decode_balanced_accuracy, permutation_p,
)

# ---------------------------- Configuration ----------------------------
from neural_common import get_run, out_dir
_RUN = get_run()
OUT_DIR = out_dir(_RUN.run_id)
BINNED = OUT_DIR / "segmented_spikes_binned.npz"
TRIAL_TABLE = OUT_DIR / "trial_table.csv"
TRIAL_LABELS = OUT_DIR / "trial_labels.csv"

CONFOUND_WINDOW = "post"
CONFOUND_HYPOTHESES = [
    "planning_vs_greedy", "planning_optimal", "agree",
    "agree_optimal_vs_lapse",
]
N_ASSOC_PERM = 300          # label shuffles per association test
N_MATCH_PERM = 300          # label shuffles per matched re-decode
MATCH_N_FOLDS = 5           # matched subsets are small -> fewer, larger folds
MATCH_N_REPEATS = 3
MIN_MATCH_N = 40            # matched subsets smaller than this are not decoded
# -----------------------------------------------------------------------


# %%
def point_biserial(y, z):
    """Signed point-biserial correlation between binary y and continuous z."""
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    n1 = y.sum()
    n0 = len(y) - n1
    if n0 == 0 or n1 == 0 or n0 + n1 < 3 or np.std(z) == 0:
        return 0.0
    z1 = z[y == 1].mean()
    z0 = z[y == 0].mean()
    return (z1 - z0) / np.std(z, ddof=1) * np.sqrt(n1 * n0 / (n1 + n0) ** 2)


def eta_squared(y, z):
    """One-way ANOVA eta^2 for continuous z grouped by y (>=2 groups)."""
    y = np.asarray(y)
    z = np.asarray(z, dtype=float)
    grand = z.mean()
    ss_between = 0.0
    ss_total = np.sum((z - grand) ** 2)
    if ss_total == 0:
        return 0.0
    for k in np.unique(y):
        zk = z[y == k]
        ss_between += len(zk) * (zk.mean() - grand) ** 2
    return ss_between / ss_total


def cramers_v(y, w):
    """Cramér's V between two categorical arrays (same length)."""
    y = np.asarray(y)
    w = np.asarray(w)
    if len(np.unique(y)) < 2 or len(np.unique(w)) < 2:
        return 0.0
    tab = pd.crosstab(y, w).to_numpy(dtype=float)
    chi2 = 0.0
    n = tab.sum()
    if n == 0:
        return 0.0
    r, c = tab.shape
    for i in range(r):
        for j in range(c):
            exp = tab[i].sum() * tab[:, j].sum() / n
            if exp > 0:
                chi2 += (tab[i, j] - exp) ** 2 / exp
    return float(np.sqrt(chi2 / (n * min(r - 1, c - 1)) + 1e-12))


def permutation_p_stat(stat_fn, y, z, obs, n_perm, seed):
    """Two-sided permutation p for a generic statistic fn(y, z)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    nulls = np.empty(n_perm)
    for i in range(n_perm):
        nulls[i] = stat_fn(rng.permutation(y), z)
    return float(np.mean(nulls >= obs))


# %%
def build_nuisance_table():
    """One row per trial with continuous + categorical nuisance variables."""
    tt = pd.read_csv(TRIAL_TABLE)
    lab = pd.read_csv(TRIAL_LABELS)

    df = tt[["trial_id", "block_index", "sequence_index", "choice_hole",
             "hole_locations"]].copy()
    df["rt_ms"] = tt["choice_time_ms"] - tt["entry_time_ms"]
    df["ball_time_ms"] = tt["exit_time_ms"] - tt["choice_time_ms"]
    df["trial_duration_ms"] = tt["exit_time_ms"] - tt["trial_start_ms"]

    df = df.merge(lab, on="trial_id", how="left",
                  suffixes=("", "_geom"))
    df["side"] = (df["choice_hole"] >= 6).astype(int)
    md = np.sign(df["choice_hole"] - df["entry_hole"])
    df["move_dir"] = md.to_numpy()
    df.loc[md == 0, "move_dir"] = np.nan

    # chosen-hole cost under each model: hole_locations is "[A, B]", A=L, B=R
    def parse_holes(s):
        import json as _json
        try:
            return _json.loads(s)
        except Exception:
            return [np.nan, np.nan]

    holes = df["hole_locations"].map(parse_holes)
    hL = holes.str[0].to_numpy()
    hR = holes.str[1].to_numpy()
    ch = df["choice_hole"].to_numpy()
    costL = df["greedy_cost_L"].to_numpy()
    costR = df["greedy_cost_R"].to_numpy()
    pL = df["planning_cost_L"].to_numpy()
    pR = df["planning_cost_R"].to_numpy()
    df["greedy_cost_chosen"] = np.where(ch == hL, costL, costR)
    df["planning_cost_chosen"] = np.where(ch == hL, pL, pR)

    # signed left-vs-right task-value gaps (positive = left hole is farther)
    df["greedy_gap"] = df["greedy_cost_L"] - df["greedy_cost_R"]
    df["planning_gap"] = df["planning_cost_L"] - df["planning_cost_R"]

    # previous trial's condition within the same block (autocorrelation)
    df = df.sort_values(["block_index", "sequence_index"]).reset_index(drop=True)
    prev_cond = df.groupby("block_index")["condition"].shift(1)
    df["prev_condition"] = prev_cond
    df["position_in_block"] = df["sequence_index"]
    return df


# %%
def label_vectors_for_association():
    """{hypothesis: (y, idx)} from build_label_vectors on the merged labels."""
    lab = pd.read_csv(TRIAL_LABELS)
    return build_label_vectors(lab)


def association_table():
    """Hypothesis x nuisance association statistics with permutation p."""
    nuis = build_nuisance_table()
    lv = label_vectors_for_association()

    cont_nuis = ["rt_ms", "ball_time_ms", "trial_duration_ms", "block_index",
                 "position_in_block", "greedy_gap", "planning_gap"]
    cat_nuis = ["side", "move_dir", "prev_condition"]

    rows = []
    for hname, (y, idx) in lv.items():
        y_sub = y if idx is None else y[idx]
        sub = nuis if idx is None else nuis.iloc[idx]
        n_classes = len(np.unique(y_sub))
        is_binary = n_classes == 2
        for c in cont_nuis:
            z = sub[c].to_numpy(dtype=float)
            m = ~np.isnan(z)
            if m.sum() < 20:
                continue
            if is_binary:
                fn = point_biserial
                stat_name = "point_biserial_r"
            else:
                fn = eta_squared
                stat_name = "eta_squared"
            obs = fn(y_sub[m], z[m])
            p = permutation_p_stat(fn, y_sub[m], z[m], obs, N_ASSOC_PERM,
                                   RNG_SEED)
            rows.append({"hypothesis": hname, "nuisance": c,
                         "nuisance_type": "continuous", "stat_name": stat_name,
                         "stat": obs, "p": p, "n": int(m.sum()),
                         "n_classes": n_classes})
        for c in cat_nuis:
            w = sub[c].to_numpy()
            m = ~pd.isna(w)
            if m.sum() < 20 or len(np.unique(w[m])) < 2:
                continue
            obs = cramers_v(y_sub[m], w[m])
            p = permutation_p_stat(cramers_v, y_sub[m], w[m], obs,
                                   N_ASSOC_PERM, RNG_SEED)
            rows.append({"hypothesis": hname, "nuisance": c,
                         "nuisance_type": "categorical",
                         "stat_name": "cramers_v", "stat": obs, "p": p,
                         "n": int(m.sum()), "n_classes": n_classes})
    return pd.DataFrame(rows)


# %%
def stratify_match_indices(y, strata_df, rng):
    """Indices to keep so each class appears equally often within each stratum
    (strata where either class is absent are dropped)."""
    y = np.asarray(y)
    keep = []
    strata = strata_df.to_numpy()
    for key in np.unique(strata, axis=0):
        mask = np.all(strata == key, axis=1)
        yi = y[mask]
        if len(np.unique(yi)) < 2:
            continue
        n0 = int(np.sum(yi == 0))
        n1 = int(np.sum(yi == 1))
        k = min(n0, n1)
        i0 = rng.choice(np.flatnonzero(yi == 0), size=k, replace=False)
        i1 = rng.choice(np.flatnonzero(yi == 1), size=k, replace=False)
        keep.append(np.flatnonzero(mask)[np.concatenate([i0, i1])])
    if not keep:
        return np.array([], dtype=int)
    return np.concatenate(keep)


def match_schemes(nuis):
    """{scheme_name: strata DataFrame over all 910 trials}."""
    rtq = pd.qcut(nuis["rt_ms"], 4, labels=False, duplicates="drop")
    btq = pd.qcut(nuis["ball_time_ms"], 4, labels=False, duplicates="drop")
    block_half = (nuis["sequence_index"] >= 17.5).astype(int)
    return {
        "matched_side": nuis[["side"]],
        "matched_side_rt_block": pd.DataFrame({
            "side": nuis["side"], "rt_quartile": rtq,
            "block_half": block_half}),
        "matched_side_rt_ball": pd.DataFrame({
            "side": nuis["side"], "rt_quartile": rtq,
            "ball_quartile": btq}),
        "matched_side_rt_ball_block": pd.DataFrame({
            "side": nuis["side"], "rt_quartile": rtq,
            "ball_quartile": btq, "block_half": block_half}),
    }


def matched_redecode(binned, bin_centers, nuis):
    """Post-window decoding on unmatched + side-matched + side/rt/block-matched
    subsets, for the key binary hypotheses."""
    lo, hi = WINDOWS[CONFOUND_WINDOW]
    lab = pd.read_csv(TRIAL_LABELS)
    lv = build_label_vectors(lab)

    Xfeats = {
        "rate": rate_features(binned, bin_centers, lo, hi),
        "pca": pca_features(binned, bin_centers, lo, hi),
    }
    transforms = {rep: make_transform(rep) for rep in REPS}
    schemes = match_schemes(nuis)

    rows = []
    rng = np.random.default_rng(RNG_SEED)
    for hname in CONFOUND_HYPOTHESES:
        y, idx = lv[hname]
        if idx is None:
            base = np.arange(len(y))
        else:
            i = np.asarray(idx)
            base = i if i.dtype.kind == "i" else np.flatnonzero(i)
        for rep in REPS:
            X = Xfeats[rep]
            for scheme_name, strata in [("unmatched", None)] + \
                    list(schemes.items()):
                if scheme_name == "unmatched":
                    sel = base
                else:
                    matched_pos = stratify_match_indices(
                        y[base], strata.iloc[base].reset_index(drop=True), rng)
                    sel = base[matched_pos]
                if len(sel) < MIN_MATCH_N:
                    rows.append({
                        "hypothesis": hname, "rep": rep,
                        "window": CONFOUND_WINDOW, "scheme": scheme_name,
                        "n_trials": int(len(sel)),
                        "n_A": int(np.sum(y[sel] == 0)) if len(sel) else np.nan,
                        "n_B": int(np.sum(y[sel] == 1)) if len(sel) else np.nan,
                        "acc_mean": np.nan, "acc_std": np.nan,
                        "chance": np.nan, "perm_p": np.nan,
                        "note": ("insufficient class overlap after matching "
                                 "(nuisance distributions too different)"),
                    })
                    print(f"  {hname}/{rep}/{scheme_name}: "
                          f"n={len(sel)} < {MIN_MATCH_N}, "
                          f"insufficient overlap -- skipped")
                    continue
                y_s = y[sel]
                X_s = X[sel]
                acc_m, acc_s = decode_balanced_accuracy(
                    y_s, X_s, transforms[rep], MATCH_N_FOLDS, MATCH_N_REPEATS,
                    RNG_SEED)
                chance = 1.0 / len(np.unique(y_s))
                p = permutation_p(y_s, X_s, transforms[rep], acc_m,
                                  n_perm=N_MATCH_PERM, n_folds=PERM_FOLDS,
                                  seed=RNG_SEED)
                rows.append({
                    "hypothesis": hname, "rep": rep,
                    "window": CONFOUND_WINDOW, "scheme": scheme_name,
                    "n_trials": len(y_s),
                    "n_A": int(np.sum(y_s == 0)),
                    "n_B": int(np.sum(y_s == 1)),
                    "acc_mean": acc_m, "acc_std": acc_s,
                    "chance": chance, "perm_p": p, "note": None,
                })
                print(f"  {hname}/{rep}/{scheme_name}: "
                      f"n={len(y_s)} acc {acc_m:.3f} ± {acc_s:.3f} "
                      f"(chance {chance:.2f}, p {p:.3f})")
    return pd.DataFrame(rows)


# %%
def main():
    print("Loading binned spikes ...")
    binned, unit_ids, bin_centers = load_binned()

    print("Building nuisance table ...")
    nuis = build_nuisance_table()
    nuis.to_csv(OUT_DIR / "decoding_nuisance_table.csv", index=False)
    print(f"  saved decoding_nuisance_table.csv ({len(nuis)} rows)")

    print("Running hypothesis x nuisance association tests ...")
    assoc = association_table()
    assoc.to_csv(OUT_DIR / "decoding_label_association.csv", index=False)
    print(f"  saved decoding_label_association.csv ({len(assoc)} rows)")

    print(f"Matched re-decoding ({CONFOUND_WINDOW} window) ...")
    matched = matched_redecode(binned, bin_centers, nuis)
    matched.to_csv(OUT_DIR / "decoding_matched_redecode.csv", index=False)
    print(f"\nSaved decoding_matched_redecode.csv ({len(matched)} rows)")

    print("\nSummary (matched re-decoding):")
    print(matched[["hypothesis", "rep", "scheme", "n_trials", "acc_mean",
                   "acc_std", "perm_p"]].to_string(index=False))


# %%
# ----------------------- Plot functions (not saved) ---------------------

def load_associations():
    return pd.read_csv(OUT_DIR / "decoding_label_association.csv")


def plot_association_heatmap(ax=None):
    """Pivot of association stats (hypothesis x nuisance), -log10(p) colored."""
    import matplotlib.pyplot as plt
    a = load_associations()
    a = a.copy()
    a["nlogp"] = -np.log10(np.clip(a["p"], 1e-6, 1))
    piv = a.pivot_table(index="hypothesis", columns="nuisance",
                        values="nlogp").fillna(0.0)
    if ax is None:
        ax = plt.gca()
    im = ax.imshow(piv.values, aspect="auto", cmap="Reds")
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=45, ha="right")
    ax.set_title("hypothesis x nuisance association, -log10(p)")
    ax.figure.colorbar(im, ax=ax)
    return ax


def plot_matched_redecode(ax=None):
    """Bar chart: acc_mean by scheme for each hypothesis, chance line at 0.5."""
    import matplotlib.pyplot as plt
    m = pd.read_csv(OUT_DIR / "decoding_matched_redecode.csv")
    m = m[m["rep"] == "rate"]
    if ax is None:
        ax = plt.gca()
    for i, h in enumerate(sorted(m["hypothesis"].unique())):
        sub = m[m["hypothesis"] == h]
        xs = np.arange(len(sub))
        ax.bar(xs + i * 0.28, sub["acc_mean"], width=0.25,
               yerr=sub["acc_std"], label=h)
    ax.axhline(0.5, color="k", ls="--")
    ax.set_xticks(range(len(m["scheme"].unique())))
    ax.set_xticklabels(m["scheme"].unique())
    ax.set_ylabel("balanced accuracy")
    ax.legend(fontsize=8)
    return ax


# %%
if __name__ == "__main__":
    main()
