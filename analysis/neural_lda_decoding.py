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
# # Neural LDA decoding: LDA on tuning hypotheses
#
# Asks which of several trial-level properties the 122-unit mesial-temporal
# population can linearly discriminate, and whether different properties are
# carried by shared or distinct population axes.
#
# Two feature representations per trial, for each of three choice-anchored
# windows (pre [-1000,0], post [0,+1000], whole trial):
#   rep 'rate' : per-unit mean firing rate (Hz) in the window -> 122-dim vector
#                (sqrt-transformed to stabilize spike-count variance).
#   rep 'pca'  : per-unit time-resolved counts (25 ms bins) concatenated across
#                the window -> high-dim vector, PCA-reduced to PCA_COMPONENTS
#                (30). The PCA is fit once on all trials (unsupervised; the
#                per-fold CV step only re-standardizes within each training
#                fold, so the null/observed comparison stays honest).
#
# Tuning hypotheses (trial labels), all from trial_labels.csv:
#   side             choice_hole in the left (0-5) vs right (6-11) screen half
#   agree            greedy & planning optima agree (no conflict) vs disagree
#   planning_optimal chose the 2-step-optimal hole vs not (efficiency)
#   move_dir         ball moved left vs right between entry and choice levels
#   condition        4-class: planning / greedy / agree_optimal / lapse
#   planning_vs_greedy        within-disagree choice policy
#   agree_optimal_vs_lapse    within-agree decision quality
#
# Statistics: LDA (Ledoit-Wolf shrunk within-class covariance; solver='eigen',
# needed because p > n for the small lapse/planning classes), repeated
# stratified 10-fold CV (x5 repeats), balanced accuracy (imbalance-proof),
# chance = 1/n_classes (expected balanced accuracy under label shuffling), and
# an empirical permutation p (label shuffle through the same CV).
#
# Interpretation: standardized LDA loadings per unit (rate rep), cosine
# similarity between loading vectors (are the axes shared or distinct?), and
# cross-decoding (does an LDA trained on hypothesis A's labels also separate
# hypothesis B's labels?).
#
# Outputs: neural_lda_decoding_results.csv, neural_lda_decoding_cross.csv,
#          neural_lda_decoding_loadings.csv, neural_lda_decoding_similarity.csv
#          (NO images; plot_* functions are provided for you to call yourself).
#
# Run with:
#   C:\Users\manik\AppData\Local\Programs\Python\Python311\python.exe analysis\neural_lda_decoding.py

# %%
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# ---------------------------- Configuration ----------------------------
OUT_DIR = Path(r"C:\Users\manik\Desktop\Obsidian\General Thoughts\Z Images and Files\Hennig Lab Project\falldown\analysis\neural_outputs")
BINNED = OUT_DIR / "segmented_spikes_binned.npz"
TRIAL_TABLE = OUT_DIR / "trial_table.csv"
TRIAL_LABELS = OUT_DIR / "trial_labels.csv"
UNIT_META = OUT_DIR / "unit_metadata.csv"

WINDOWS = {"pre": (-1000.0, 0.0), "post": (0.0, 1000.0), "whole": None}
REPS = ["rate", "pca"]
PCA_COMPONENTS = 30
N_FOLDS = 10
N_REPEATS = 5
N_PERM = 500
PERM_FOLDS = 5
RNG_SEED = 42
CROSS_WINDOW = "post"
# -----------------------------------------------------------------------


# %%
def load_binned():
    z = np.load(BINNED, allow_pickle=True)
    return z["binned"], z["unit_ids"], z["bin_centers"]


def channel_label(source_file):
    """Electrode label from a source_file name, e.g. 'times_mLF1aCa01_2285.mat'
    -> 'mLF1aCa01'."""
    stem = Path(source_file).stem          # times_mLF1aCa01_2285
    stem = stem[len("times_"):] if stem.startswith("times_") else stem
    return stem.rsplit("_", 1)[0]


def unit_channel_labels(units):
    """{unit_id: electrode label} from unit_metadata."""
    return {int(r["unit_id"]): channel_label(r["source_file"])
            for _, r in units.iterrows()}


# %%
def _window_bounds(wh):
    """Unpack a window spec ('whole' -> (None, None))."""
    return (None, None) if wh is None else wh


def _window_bin_mask(bin_centers, lo, hi):
    """Boolean mask over bins whose center lies in [lo, hi]; whole window if
    lo is None."""
    if lo is None:
        return np.ones(len(bin_centers), dtype=bool)
    return (bin_centers >= lo) & (bin_centers <= hi)


def rate_features(binned, bin_centers, lo, hi):
    """(n_trials, n_units) sqrt-transformed mean firing rate (Hz) in the window.

    NaN bins (outside the trial's actual span) are excluded from the mean.
    A trial with zero covered bins gets 0 Hz (never happens here: min coverage
    is ~21% of the window).
    """
    mask = _window_bin_mask(bin_centers, lo, hi)
    sel = binned[:, :, mask]                     # (n_units, n_trials, n_bins)
    valid = ~np.isnan(sel)
    counts = np.where(valid, sel, 0.0)
    n_valid = valid.sum(axis=2)
    rate = counts.sum(axis=2) / (n_valid * 0.025)   # Hz
    rate = np.where(n_valid > 0, rate, 0.0)
    return np.sqrt(np.maximum(rate, 0.0)).T         # (n_trials, n_units)


def pca_features(binned, bin_centers, lo, hi, n_components=PCA_COMPONENTS):
    """(n_trials, n_components) PCA-reduced time-resolved counts in the window.

    Counts are concatenated per unit across the window's bins (NaN -> 0),
    standardized, then PCA-reduced. The PCA is unsupervised and fit on all
    trials once; the CV loop below only re-standardizes per training fold.
    """
    mask = _window_bin_mask(bin_centers, lo, hi)
    sel = binned[:, :, mask]                     # (n_units, n_trials, n_bins)
    X = np.where(np.isnan(sel), 0.0, sel)
    n_trials = binned.shape[1]
    X = np.moveaxis(X, 0, 1).reshape(n_trials, -1)  # (n_trials, feats)
    sc = StandardScaler().fit(X)
    pca = PCA(n_components=n_components, random_state=RNG_SEED).fit(sc.transform(X))
    return pca.transform(sc.transform(X))


# %%
def build_label_vectors(labels):
    """{name: (y, idx)}. y is always full trial length; positions outside a
    hypothesis's trial subset hold -1. idx selects the hypothesis's rows
    (None = all trials). Callers use y[idx] / X[idx]."""
    n = len(labels)
    out = {}
    out["side"] = (labels["choice_hole"] >= 6).astype(int).to_numpy(), None
    out["agree"] = labels["agree"].astype(int).to_numpy(), None
    out["planning_optimal"] = (
        (labels["choice_hole"] == labels["planning_optimal_hole"]).astype(int).to_numpy(),
        None)

    md = np.sign(labels["choice_hole"] - labels["entry_hole"]).to_numpy()
    keep = md != 0
    y = np.full(n, -1, dtype=int)
    y[keep] = ((md[keep] + 1) // 2).astype(int)
    out["move_dir"] = y, keep

    cond_order = {"planning": 0, "greedy": 1, "agree_optimal": 2, "lapse": 3}
    out["condition"] = labels["condition"].map(cond_order).to_numpy(), None

    for name, cA, cB in (("planning_vs_greedy", "planning", "greedy"),
                         ("agree_optimal_vs_lapse", "agree_optimal", "lapse")):
        idx = np.flatnonzero(labels["condition"].isin([cA, cB]).to_numpy())
        y = np.full(n, -1, dtype=int)
        y[idx] = labels["condition"].isin([cB]).to_numpy()[idx].astype(int)
        out[name] = y, idx
    return out


# %%
def _zscore_transform(Xtr, ytr, Xte):
    """Standardize on the training fold only (rate rep)."""
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte)


def _identity_transform(Xtr, ytr, Xte):
    return Xtr, Xte


def make_transform(rep):
    return _zscore_transform if rep == "rate" else _identity_transform


# %%
def decode_balanced_accuracy(y, X, transform, n_folds, n_repeats, seed):
    """Repeated stratified k-fold CV balanced accuracy (per-fold values)."""
    accs = []
    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                              random_state=seed + rep)
        for tr, te in skf.split(X, y):
            Xtr, Xte = transform(X[tr], y[tr], X[te])
            lda = LinearDiscriminantAnalysis(
                solver="eigen", shrinkage="auto").fit(Xtr, y[tr])
            accs.append(balanced_accuracy_score(y[te], lda.predict(Xte)))
    accs = np.asarray(accs)
    return float(accs.mean()), float(accs.std())


def permutation_p(y, X, transform, obs_mean, n_perm=N_PERM, n_folds=PERM_FOLDS,
                  seed=RNG_SEED):
    """Two-sided p: fraction of shuffled-label balanced accuracies >= observed."""
    rng = np.random.default_rng(seed)
    nulls = np.empty(n_perm)
    for i in range(n_perm):
        ysh = rng.permutation(y)
        m, _ = decode_balanced_accuracy(ysh, X, transform, n_folds, 1,
                                        seed + i)
        nulls[i] = m
    return float(np.mean(nulls >= obs_mean))


# %%
def threshold_balanced_accuracy(s_tr, b_tr, s_te, b_te):
    """Best-threshold (fit in fold) balanced accuracy separating b by s."""
    cands = np.quantile(s_tr, np.linspace(0, 1, 25))
    best_t, best = None, -1.0
    for t in cands:
        a = balanced_accuracy_score(b_tr, (s_tr >= t).astype(int))
        if a > best:
            best, best_t = a, t
    return balanced_accuracy_score(b_te, (s_te >= best_t).astype(int))


def cross_decode_accuracy(X, yA, yB, transform, n_folds=N_FOLDS,
                          n_repeats=N_REPEATS, seed=RNG_SEED):
    """Balanced accuracy separating yB using the projection of an LDA trained
    on yA (both binary). Threshold fit inside each training fold.

    Returns (nan, nan) when either label set is constant in the subset (the
    LDA would have a single class and no discriminant to project onto).
    """
    if len(np.unique(yA)) < 2 or len(np.unique(yB)) < 2:
        return np.nan, np.nan
    accs = []
    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                              random_state=seed + rep)
        for tr, te in skf.split(X, yA):
            Xtr, Xte = transform(X[tr], yA[tr], X[te])
            lda = LinearDiscriminantAnalysis(
                solver="eigen", shrinkage="auto").fit(Xtr, yA[tr])
            s_tr = lda.transform(Xtr)[:, 0]
            s_te = lda.transform(Xte)[:, 0]
            accs.append(threshold_balanced_accuracy(s_tr, yB[tr], s_te, yB[te]))
    accs = np.asarray(accs)
    return float(accs.mean()), float(accs.std())


# %%
def fit_loadings(X, y):
    """Standardized LDA coefficients (z-scored features) on the full data.

    Returns coef (n_discriminants, n_features); each entry is the unit's
    contribution in per-standard-deviation-of-feature units.
    """
    sc = StandardScaler().fit(X)
    lda = LinearDiscriminantAnalysis(
        solver="eigen", shrinkage="auto").fit(sc.transform(X), y)
    return lda.coef_


def cosine_sim(u, v):
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))


# %%
def main():
    print("Loading binned spike data ...")
    binned, unit_ids, bin_centers = load_binned()
    labels = pd.read_csv(TRIAL_LABELS)
    meta = pd.read_csv(UNIT_META)
    chan = unit_channel_labels(meta)
    label_dict = build_label_vectors(labels)

    transforms = {rep: make_transform(rep) for rep in REPS}
    rows = []
    for window_name, wh in WINDOWS.items():
        lo, hi = _window_bounds(wh)
        X_feats = {
            "rate": rate_features(binned, bin_centers, lo, hi),
            "pca": pca_features(binned, bin_centers, lo, hi),
        }
        for rep in REPS:
            X = X_feats[rep]
            for hname, (y, idx) in label_dict.items():
                y_sub = y if idx is None else y[idx]
                X_sub = X if idx is None else X[idx]
                acc_m, acc_s = decode_balanced_accuracy(
                    y_sub, X_sub, transforms[rep], N_FOLDS, N_REPEATS, RNG_SEED)
                chance = 1.0 / len(np.unique(y_sub))
                p = permutation_p(y_sub, X_sub, transforms[rep], acc_m)
                n_classes = len(np.unique(y_sub))
                n_a = int(np.sum(y_sub == 0))
                n_b = int(np.sum(y_sub == 1))
                rows.append({
                    "rep": rep,
                    "window": window_name,
                    "hypothesis": hname,
                    "n_trials": len(y_sub),
                    "n_classes": n_classes,
                    "n_A": n_a if n_classes == 2 else np.nan,
                    "n_B": n_b if n_classes == 2 else np.nan,
                    "acc_mean": acc_m,
                    "acc_std": acc_s,
                    "chance": chance,
                    "perm_p": p,
                })
                print(f"  {window_name}/{rep}/{hname}: "
                      f"acc {acc_m:.3f} ± {acc_s:.3f} "
                      f"(chance {chance:.2f}, p {p:.3f})")

    res = pd.DataFrame(rows)
    res["above_chance"] = res["acc_mean"] > res["chance"]
    res.to_csv(OUT_DIR / "neural_lda_decoding_results.csv", index=False)
    print(f"\nSaved neural_lda_decoding_results.csv ({len(res)} rows)")

    # ---------------- cross-decoding (one window to bound runtime) ----------
    binary = ["side", "agree", "planning_optimal", "move_dir",
              "planning_vs_greedy", "agree_optimal_vs_lapse"]
    lo, hi = _window_bounds(WINDOWS[CROSS_WINDOW])
    X_feats = {
        "rate": rate_features(binned, bin_centers, lo, hi),
        "pca": pca_features(binned, bin_centers, lo, hi),
    }
    cross_rows = []
    for i, a in enumerate(binary):
        for b in binary[i + 1:]:
            ya, ia = label_dict[a]
            yb, ib = label_dict[b]
            if ia is None and ib is None:
                common = None
            elif ia is None:
                common = ib
            elif ib is None:
                common = ia
            else:
                common = np.intersect1d(ia, ib)
            if common is not None and len(common) < 10:
                print(f"  cross {a}->{b}: no overlapping trials, skipped")
                continue
            for rep in REPS:
                X = X_feats[rep]
                if common is None:
                    Xc, yAc, yBc = X, ya, yb
                else:
                    Xc, yAc, yBc = X[common], ya[common], yb[common]
                aob, _ = cross_decode_accuracy(Xc, yAc, yBc, transforms[rep])
                boa, _ = cross_decode_accuracy(Xc, yBc, yAc, transforms[rep])
                if np.isnan(aob) or np.isnan(boa):
                    print(f"  cross {rep}/{a}<->{b}: label constant in "
                          f"overlap, skipped")
                    continue
                cross_rows.append({
                    "rep": rep,
                    "window": CROSS_WINDOW,
                    "hypothesis_A": a,
                    "hypothesis_B": b,
                    "n_trials": len(Xc),
                    "acc_A_on_B": aob,
                    "acc_B_on_A": boa,
                    "chance": 0.5,
                })
                print(f"  cross {rep}/{a}->{b}: A_on_B {aob:.3f}, "
                      f"B_on_A {boa:.3f}")
    cross = pd.DataFrame(cross_rows)
    cross.to_csv(OUT_DIR / "neural_lda_decoding_cross.csv", index=False)
    print(f"\nSaved neural_lda_decoding_cross.csv ({len(cross)} rows)")

    # ---------------- per-unit loadings (rate rep, all windows) -------------
    loading_rows = []
    for window_name, wh in WINDOWS.items():
        lo, hi = _window_bounds(wh)
        X = rate_features(binned, bin_centers, lo, hi)
        for hname, (y, idx) in label_dict.items():
            y_sub = y if idx is None else y[idx]
            X_sub = X if idx is None else X[idx]
            coef = fit_loadings(X_sub, y_sub)
            for d in range(coef.shape[0]):
                for j, uid in enumerate(unit_ids):
                    loading_rows.append({
                        "window": window_name,
                        "hypothesis": hname,
                        "discriminant": d,
                        "unit_id": int(uid),
                        "channel": chan.get(int(uid), ""),
                        "loading": float(coef[d, j]),
                    })
    loadings = pd.DataFrame(loading_rows)
    loadings.to_csv(OUT_DIR / "neural_lda_decoding_loadings.csv", index=False)
    print(f"\nSaved neural_lda_decoding_loadings.csv ({len(loadings)} rows)")

    # ---------------- weight-vector similarity (rate rep, cross window) -----
    lo, hi = _window_bounds(WINDOWS[CROSS_WINDOW])
    X = rate_features(binned, bin_centers, lo, hi)
    vecs = {}
    for hname in binary:
        y, idx = label_dict[hname]
        Xs = X if idx is None else X[idx]
        vecs[hname] = fit_loadings(Xs, y)[0]
    sim_rows = []
    names = list(vecs)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            sim_rows.append({
                "hypothesis_A": a,
                "hypothesis_B": b,
                "cosine_sim": cosine_sim(vecs[a], vecs[b]),
            })
    sim = pd.DataFrame(sim_rows)
    sim.to_csv(OUT_DIR / "neural_lda_decoding_similarity.csv", index=False)
    print(f"\nSaved neural_lda_decoding_similarity.csv ({len(sim)} rows)")

    print("\nSummary (accuracy vs chance, post window, rate rep):")
    sub = res[(res["rep"] == "rate") & (res["window"] == "post")]
    print(sub[["hypothesis", "acc_mean", "acc_std", "chance", "perm_p"]]
          .to_string(index=False))


# %%
# ----------------------- Plot functions (not saved) ---------------------

def load_decoding_results():
    return pd.read_csv(OUT_DIR / "neural_lda_decoding_results.csv")


def plot_decoding_bar(rep="rate", window="post", ax=None):
    """Bar chart: balanced accuracy by hypothesis with chance line."""
    import matplotlib.pyplot as plt
    res = load_decoding_results()
    sub = res[(res["rep"] == rep) & (res["window"] == window)]
    sub = sub.sort_values("acc_mean")
    if ax is None:
        ax = plt.gca()
    ax.bar(sub["hypothesis"], sub["acc_mean"],
           yerr=sub["acc_std"], color="steelblue")
    ax.axhline(sub["chance"].iloc[0], color="k", ls="--",
               label=f"chance = {sub['chance'].iloc[0]:.2f}")
    ax.set_ylabel("balanced accuracy")
    ax.set_title(f"{rep} / {window}")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    return ax


def plot_confusion(rep="rate", window="post", hypothesis="condition",
                   ax=None):
    """Confusion matrix (aggregated across CV folds) for a hypothesis."""
    import matplotlib.pyplot as plt
    z = np.load(BINNED, allow_pickle=True)
    binned, unit_ids, bin_centers = z["binned"], z["unit_ids"], z["bin_centers"]
    labels = pd.read_csv(TRIAL_LABELS)
    y, idx = build_label_vectors(labels)[hypothesis]
    lo, hi = _window_bounds(WINDOWS[window])
    X = rate_features(binned, bin_centers, lo, hi)
    X, y = (X, y) if idx is None else (X[idx], y[idx])
    classes = np.unique(y)
    agg = np.zeros((len(classes), len(classes)), dtype=float)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RNG_SEED)
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        lda = LinearDiscriminantAnalysis(
            solver="eigen", shrinkage="auto").fit(sc.transform(X[tr]), y[tr])
        cm = confusion_matrix(y[te], lda.predict(sc.transform(X[te])),
                              labels=classes)
        agg += cm / cm.sum(axis=1, keepdims=True)
    agg /= N_FOLDS
    if ax is None:
        ax = plt.gca()
    im = ax.imshow(agg, cmap="Blues")
    ax.set_xticks(range(len(classes)), [str(c) for c in classes])
    ax.set_yticks(range(len(classes)), [str(c) for c in classes])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"{hypothesis} ({window}, {rep})")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{agg[i, j]:.2f}", ha="center", va="center")
    ax.figure.colorbar(im, ax=ax)
    return ax


def plot_loading_heatmap(window="post", hypothesis=None, ax=None):
    """Heatmap of per-unit loadings (rate rep) across hypotheses or units."""
    import matplotlib.pyplot as plt
    load = pd.read_csv(OUT_DIR / "neural_lda_decoding_loadings.csv")
    load = load[(load["window"] == window) & (load["discriminant"] == 0)]
    if hypothesis is not None:
        load = load[load["hypothesis"] == hypothesis]
    piv = load.pivot_table(index="unit_id", columns="hypothesis",
                           values="loading").fillna(0.0)
    if ax is None:
        ax = plt.gca()
    im = ax.imshow(piv.values, aspect="auto", cmap="RdBu_r")
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index.astype(int))
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=45)
    ax.set_ylabel("unit_id")
    ax.figure.colorbar(im, ax=ax)
    return ax


# %%
if __name__ == "__main__":
    main()

# %%
