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
# # Re-run headline decodes with 100 ms bins (vs 25 ms)
#
# Rebuilds choice-anchored windows at 100 ms bin width from the raw spike times
# (`segmented_spikes_raw.pkl`) and re-runs the headline continuous decodes plus
# the planning/greedy classifier, under:
#   truncated (NaN outside span, zero-padded PCA), spliced (continuous global
#   extraction), and coverage-controlled variants.
#
# Outputs: rebin100.csv, figures/fig_rebin100.png

# %%
import pickle
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score

from neural_common import RUN_ORDER, AGG_DIR, out_dir
from neural_lda_decoding import rate_features, pca_features, RNG_SEED
from neural_pass_timing import ridge_cv
from neural_decoder_geometry import load, targets

# %%
BIN = 100.0
LO, HI = -2000.0, 2000.0
WIN = (0.0, 1000.0)
TARGETS = ["conflict_mag", "chosen_cost", "step2", "rt_exec"]


def extract_window(V, centers, lo, hi):
    m = (centers >= lo) & (centers <= hi)
    return V[:, :, m], centers[m]


def rebin_nan(arr, factor=4):
    u, t, b = arr.shape
    b2 = b // factor
    a = arr[:, :, :b2 * factor].reshape(u, t, b2, factor)
    with np.errstate(invalid="ignore"):
        cnt = np.sum(~np.isnan(a), axis=3)
        s = np.nansum(a, axis=3)
    s[cnt == 0] = np.nan
    return s


def build_global100(run):
    d = load(run)
    tt = d["tt"]
    raw = pickle.load(open(out_dir(run) / "segmented_spikes_raw.pkl", "rb"))
    npz = np.load(out_dir(run) / "segmented_spikes_binned.npz", allow_pickle=True)
    uid2row = {int(u): i for i, u in enumerate(npz["unit_ids"])}
    n_units = len(npz["unit_ids"])
    choice = tt["choice_time_ms"].to_numpy(float)
    gmin = choice.min() + LO - BIN
    gmax = choice.max() + HI + BIN
    n_bins = int(np.ceil((gmax - gmin) / BIN))
    G = np.zeros((n_units, n_bins))
    for (u, t), rel in raw.items():
        r = uid2row.get(int(u))
        if r is None or t >= len(choice):
            continue
        idx = ((choice[t] + rel - gmin) / BIN).astype(int)
        idx = idx[(idx >= 0) & (idx < n_bins)]
        if len(idx):
            G[r] += np.bincount(idx, minlength=n_bins)
    return d, G, gmin, choice


def extract_global(G, gmin, choice, rel_centers):
    n_units, n_trials, n_bins = G.shape[0], len(choice), len(rel_centers)
    W = np.zeros((n_units, n_trials, n_bins))
    for t in range(n_trials):
        idx = ((choice[t] + rel_centers - gmin) / BIN).astype(int)
        ok = (idx >= 0) & (idx < G.shape[1])
        W[:, t, ok] = G[:, idx[ok]]
    return W


def grouped_lda_acc(X, y, g):
    ng = min(5, len(np.unique(g)))
    if len(np.unique(y)) < 2 or ng < 2:
        return np.nan
    a = []
    for tr, te in GroupKFold(n_splits=ng).split(X, y, g):
        sc = StandardScaler().fit(X[tr])
        m = LDA(solver="svd").fit(sc.transform(X[tr]), y[tr])
        a.append(balanced_accuracy_score(y[te], m.predict(sc.transform(X[te]))))
    return float(np.mean(a)) if a else np.nan


def main():
    rows = []
    for run in RUN_ORDER:
        print(f"[{run}] 100 ms")
        d, G, gmin, choice = build_global100(run)
        T = targets(d)
        g = d["tt"]["block_index"].to_numpy()
        cond = d["lab"]["condition"].to_numpy()
        yplan = (cond == "planning").astype(int)
        conflict = np.isin(cond, ["planning", "greedy"])

        # truncated: rebin the existing 25 ms npz to 100 ms
        b25 = d["binned"]
        b100 = rebin_nan(b25, 4)
        c100 = d["bc"][: (b25.shape[2] // 4) * 4].reshape(-1, 4).mean(1)
        Wt, ct = extract_window(b100, c100, *WIN)
        # spliced: continuous global 100 ms
        rel_centers = np.arange(LO + BIN / 2, HI, BIN)
        Ws_full = extract_global(G, gmin, choice, rel_centers)
        Ws, cs = extract_window(Ws_full, rel_centers, *WIN)
        # coverage of the truncated window (valid bins per trial)
        cov = np.isfinite(Wt).all(axis=0).sum(axis=1).astype(float)
        A = np.column_stack([np.ones(len(cov)), cov])

        for rep, V in [("truncated", Wt), ("spliced", Ws)]:
            Xp = pca_features(V, ct, *WIN)
            for tname in TARGETS:
                y = T[tname].to_numpy(float)
                rows.append({"run_id": run, "bin_ms": 100, "variant": rep,
                             "target": tname, "metric": "pca",
                             "corr": ridge_cv(Xp, y, g)})
            rows.append({"run_id": run, "bin_ms": 100, "variant": rep,
                         "target": "planning_vs_greedy", "metric": "lda",
                         "corr": grouped_lda_acc(pca_features(V, ct, *WIN)[conflict],
                                                 yplan[conflict], g[conflict])})
        # coverage-controlled truncated
        Xp = pca_features(Wt, ct, *WIN)
        Xc = Xp - A @ np.linalg.lstsq(A, Xp, rcond=None)[0]
        for tname in TARGETS:
            y = T[tname].to_numpy(float)
            yc = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
            rows.append({"run_id": run, "bin_ms": 100, "variant": "covctrl",
                         "target": tname, "metric": "pca", "corr": ridge_cv(Xc, yc, g)})
            rows.append({"run_id": run, "bin_ms": 100, "variant": "cov_only",
                         "target": tname, "metric": "r", "corr": np.corrcoef(cov, y)[0, 1]})
    df = pd.DataFrame(rows)
    df.to_csv(AGG_DIR / "rebin100.csv", index=False)

    piv = df.pivot_table(index="target", columns="variant", values="corr", aggfunc="mean")
    print(piv.round(3).to_string())

    # figure
    variants = [v for v in ["truncated", "spliced", "covctrl", "cov_only"] if v in set(df.variant)]
    tgts = TARGETS + ["planning_vs_greedy"]
    fig, axes = plt.subplots(1, len(RUN_ORDER), figsize=(4.6 * len(RUN_ORDER), 3.8), sharey=True)
    x = np.arange(len(tgts)); w = 0.2
    for ax, run in zip(axes, RUN_ORDER):
        s = df[df.run_id == run]
        for i, v in enumerate(variants):
            vals = [s[(s.variant == v) & (s.target == t)]["corr"].mean() for t in tgts]
            ax.bar(x + (i - len(variants) / 2) * w, vals, w, label=(v if run == RUN_ORDER[0] else None))
        ax.axhline(0, color="k", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(tgts, rotation=45, ha="right", fontsize=7)
        ax.set_title(run, fontsize=10)
    axes[0].set_ylabel("decode (100 ms bins)")
    axes[0].legend(fontsize=7)
    fig.suptitle("100 ms-bin re-run: truncated vs spliced vs coverage-controlled", fontsize=12)
    fig.tight_layout()
    fig.savefig(AGG_DIR / "figures" / "fig_rebin100.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote rebin100.csv + fig_rebin100.png")


if __name__ == "__main__":
    main()
