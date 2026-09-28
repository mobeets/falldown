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
# # Region-resolved decoding (hippocampus vs amygdala)
#
# The 64-channel depth array is bilateral mesial temporal, with contacts in CA
# fields, amygdala, anterior hippocampus, and hippocampal body. This script
# asks whether the headline effects are carried by particular regions:
#
#   - per-region LDA (planning_vs_greedy, condition, move_dir) on that region's
#     units only;
#   - leave-one-region-out cross-decoding (fit on the other three regions,
#     test on the held-out region) to test cross-region generalization;
#   - per-region ridge decoding of `conflict_mag` (none/target/both controls);
#   - single-unit selectivity counts by region (join with selectivity_results).
#
# Region is assigned from the NS5 channel -> lead table in DATA_STRUCTURE.md
# (no patient-specific electrode localization). Units are not matched across
# runs; per-session decoders remain the replication unit.
#
# Outputs (analysis/neural_outputs/<run>/):
#   region_decoding.csv            per-region + leave-one-region-out LDA
#   region_continuous.csv          per-region conflict_mag ridge decoding
#   region_selectivity_counts.csv  significant units per region/contrast/window
#
# Run with:
#   C:\Users\manik\AppData\Local\Programs\Python\Python311\python.exe analysis\neural_region_decoding.py --run yfz_1

# %%
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

from neural_common import (get_run, out_dir, unit_regions,
                           region_from_label, REGION_BY_LEAD)
from neural_lda_decoding import (
    RNG_SEED, load_binned, rate_features, pca_features, make_transform,
    build_label_vectors, add_move_dir_vx, decode_balanced_accuracy,
    permutation_p, N_FOLDS, N_REPEATS, PERM_FOLDS,
)
from neural_continuous_decoding import (
    build_regressors, build_nuisances, residualize, residualize_features,
    cv_regression, permutation_corr_p, N_PERM,
)

# ---------------------------- Configuration ----------------------------
_RUN = get_run()
OUT_DIR = out_dir(_RUN.run_id)
UNIT_META = OUT_DIR / "unit_metadata.csv"
TRIAL_LABELS = OUT_DIR / "trial_labels.csv"
SELECTIVITY = OUT_DIR / "selectivity_results.csv"

WINDOW = (0.0, 1000.0)      # post-choice, the window where effects live
HYPOTHESES = ["planning_vs_greedy", "condition", "move_dir"]
MIN_UNITS = 8               # a region needs at least this many units to decode
# -----------------------------------------------------------------------


# %%
def region_unit_masks(unit_ids):
    """{region: boolean mask over the binned unit axis}."""
    meta = pd.read_csv(UNIT_META)
    reg = unit_regions(meta)
    out = {}
    for region in sorted(set(REGION_BY_LEAD.values())):
        out[region] = np.array([reg.get(int(u), "") == region for u in unit_ids])
    return out


def oof_scores(y, X, n_folds=N_FOLDS, n_repeats=N_REPEATS, seed=RNG_SEED):
    """Out-of-fold LDA decision scores (mean over repeats).

    Cross-region *decoding* is not defined with disjoint unit sets (a decoder
    trained on region A's units cannot be applied to region B's units). Instead
    we compare regions by correlating their out-of-fold decision scores: if two
    regions carry the same information about the label, their single-trial
    scores should agree.
    """
    scores = np.full((n_repeats, len(y)), np.nan)
    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                              random_state=seed + rep)
        for tr, te in skf.split(X, y):
            sc = StandardScaler().fit(X[tr])
            lda = LinearDiscriminantAnalysis(
                solver="eigen", shrinkage="auto").fit(sc.transform(X[tr]),
                                                      y[tr])
            if len(np.unique(y)) == 2:
                val = lda.decision_function(sc.transform(X[te]))
            else:
                # multiclass: scalar = probability of the true class
                proba = lda.predict_proba(sc.transform(X[te]))
                val = proba[np.arange(len(te)), y[te]]
            scores[rep, te] = val
    return np.nanmean(scores, axis=0)


# %%
def main():
    print(f"Run {_RUN.run_id} ({_RUN.participant})")
    binned, unit_ids, bin_centers = load_binned()
    labels = pd.read_csv(TRIAL_LABELS)
    lv = build_label_vectors(add_move_dir_vx(labels))
    lo, hi = WINDOW
    X_rate = rate_features(binned, bin_centers, lo, hi)
    X_pca = pca_features(binned, bin_centers, lo, hi)
    masks = region_unit_masks(unit_ids)
    transform = make_transform("rate")

    # ---------------- per-region LDA ----------------
    rows = []
    for region, mask in masks.items():
        n_units = int(mask.sum())
        if n_units < MIN_UNITS:
            continue
        for hname in HYPOTHESES:
            y, idx = lv[hname]
            keep = np.ones(len(y), bool) if idx is None else (
                idx if idx.dtype == bool else np.isin(np.arange(len(y)), idx))
            if keep.sum() < 30 or len(np.unique(y[keep])) < 2:
                continue
            Xr = X_rate[keep][:, mask]
            acc_m, acc_s = decode_balanced_accuracy(y[keep], Xr, transform,
                                                    N_FOLDS, N_REPEATS,
                                                    RNG_SEED)
            p = permutation_p(y[keep], Xr, transform, acc_m, n_perm=300,
                              n_folds=PERM_FOLDS, seed=RNG_SEED)
            rows.append({"analysis": "per_region", "region": region,
                         "n_units": n_units, "hypothesis": hname,
                         "n_trials": int(keep.sum()),
                         "acc_mean": acc_m, "acc_std": acc_s,
                         "chance": 1.0 / len(np.unique(y[keep])),
                         "perm_p": p})
            print(f"  per-region {region}/{hname}: acc {acc_m:.3f} "
                  f"(p {p:.3f}, {n_units} units)")

    # ---------------- cross-region decision-score correlation ----------------
    # Disjoint unit sets make cross-region *decoding* ill-defined; instead test
    # whether regions carry the same information by correlating their
    # out-of-fold LDA decision scores.
    for hname in ["planning_vs_greedy", "condition"]:
        y, idx = lv[hname]
        keep = np.ones(len(y), bool) if idx is None else (
            idx if idx.dtype == bool else np.isin(np.arange(len(y)), idx))
        if keep.sum() < 30 or len(np.unique(y[keep])) < 2:
            continue
        scores = {}
        for region, mask in masks.items():
            if mask.sum() < MIN_UNITS:
                continue
            scores[region] = oof_scores(y[keep], X_rate[keep][:, mask])
        regions = sorted(scores)
        for i, a in enumerate(regions):
            for b in regions[i + 1:]:
                sa, sb = scores[a], scores[b]
                valid = np.isfinite(sa) & np.isfinite(sb)
                if valid.sum() < 30 or np.std(sa[valid]) < 1e-12 \
                        or np.std(sb[valid]) < 1e-12:
                    r = np.nan
                else:
                    r = float(np.corrcoef(sa[valid], sb[valid])[0, 1])
                rows.append({"analysis": "score_correlation", "region": f"{a}|{b}",
                             "n_units": np.nan, "hypothesis": hname,
                             "n_trials": int(valid.sum()), "acc_mean": r,
                             "acc_std": np.nan, "chance": np.nan,
                             "perm_p": np.nan})
                print(f"  score-corr {a} vs {b}/{hname}: r {r:+.3f}")

    dec = pd.DataFrame(rows)
    dec["run_id"] = _RUN.run_id
    dec["participant_id"] = _RUN.participant
    dec.to_csv(OUT_DIR / "region_decoding.csv", index=False)
    print("  saved region_decoding.csv")

    # ---------------- per-region continuous conflict_mag ----------------
    regressors = build_regressors()
    nuisance = build_nuisances()
    y_reg = regressors["conflict_mag"]
    m = ~np.isnan(y_reg)
    rows = []
    for region, mask in masks.items():
        if mask.sum() < MIN_UNITS:
            continue
        for rep, Xall in [("rate", X_rate), ("pca", X_pca)]:
            if rep == "pca":
                # PCA components are not unit-aligned; skip per-region PCA
                continue
            X = Xall[m][:, mask]
            y = y_reg[m]
            for control in ["none", "target", "both"]:
                Xc, yc = X, y
                if control == "target":
                    yc = residualize(y, nuisance[m])
                elif control == "both":
                    yc = residualize(y, nuisance[m])
                    Xc = residualize_features(X, nuisance[m])
                _, _, corr_m, corr_s = cv_regression(
                    yc, Xc, N_FOLDS, N_REPEATS, RNG_SEED)[0]
                p = permutation_corr_p(yc, Xc, corr_m, N_PERM, PERM_FOLDS,
                                       RNG_SEED)
                rows.append({"region": region, "n_units": int(mask.sum()),
                             "rep": rep, "regressor": "conflict_mag",
                             "control": control, "corr_mean": corr_m,
                             "corr_std": corr_s, "perm_p": p})
                print(f"  cont {region}/conflict_mag/{rep}/{control}: "
                      f"corr {corr_m:+.3f} (p {p:.3f})")
    cont = pd.DataFrame(rows)
    cont["run_id"] = _RUN.run_id
    cont["participant_id"] = _RUN.participant
    cont.to_csv(OUT_DIR / "region_continuous.csv", index=False)
    print("  saved region_continuous.csv")

    # ---------------- region selectivity counts ----------------
    if SELECTIVITY.exists():
        sel = pd.read_csv(SELECTIVITY)
        sel["region"] = sel["channel"].map(
            lambda s: region_from_label(str(s)))
        counts = (sel[sel["significant"]]
                  .groupby(["region", "contrast", "window"]).size()
                  .reset_index(name="n_significant"))
        totals = (sel.groupby(["region", "contrast", "window"]).size()
                  .reset_index(name="n_units_tested"))
        counts = counts.merge(totals, on=["region", "contrast", "window"],
                              how="outer").fillna(0)
        counts["run_id"] = _RUN.run_id
        counts["participant_id"] = _RUN.participant
        counts.to_csv(OUT_DIR / "region_selectivity_counts.csv", index=False)
        print("  saved region_selectivity_counts.csv")


if __name__ == "__main__":
    main()
