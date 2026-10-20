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
# # Coverage / censoring control for fixed-window decoding
#
# A fixed post-choice window (e.g. `[0,1000]` ms) processed with `pca_features`
# pads NaN->0, so a trial that exits early has mostly zero bins. The decoder can
# then read *how long the trial was still in flight* -- which is `rt_exec`.
# This script re-decodes the key targets with the coverage confound controlled:
#
#   raw           - pca [0,1000], all trials (original)
#   inflight      - pca [0,1000], only trials with rt_exec >= 1000
#   rate          - duration-normalized rate features (no zero padding)
#   covctrl       - pca [0,1000], target and features residualized on coverage
#   cov_only      - decode the target from the coverage scalar alone (artifact floor)
#   short_inflight- pca [0,250], only trials with rt_exec >= 250 (censoring-free)
#
# Outputs (aggregate/): coverage_control.csv, figures/fig_coverage_control.png

# %%
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neural_common import RUN_ORDER, AGG_DIR
from neural_lda_decoding import rate_features, pca_features
from neural_pass_timing import ridge_cv
from neural_decoder_geometry import load, targets

# %%
RIDGE_ALPHA = 1.0
TARGETS = ["conflict_mag", "step1", "step2", "chosen_cost", "rt_exec"]
WIN_LO, WIN_HI = 0.0, 1000.0


def coverage(d, lo, hi):
    bc = d["bc"]
    valid = ~np.isnan(d["binned"]).all(axis=0)
    mask = (bc >= lo) & (bc <= hi)
    return valid[:, mask].sum(1).astype(float)


def residue_cols(X, c):
    A = np.column_stack([np.ones(len(c)), c])
    beta = np.linalg.lstsq(A, X, rcond=None)[0]
    return X - A @ beta


def residue_vec(y, c):
    A = np.column_stack([np.ones(len(c)), c])
    b = np.linalg.lstsq(A, y, rcond=None)[0]
    return y - A @ b


def main():
    rows = []
    for run in RUN_ORDER:
        print(f"[{run}] coverage control")
        d = load(run)
        T = targets(d)
        g = d["tt"]["block_index"].to_numpy()
        cov1000 = coverage(d, WIN_LO, WIN_HI)
        cov250 = coverage(d, 0.0, 250.0)
        Xp = pca_features(d["binned"], d["bc"], WIN_LO, WIN_HI)
        Xr = rate_features(d["binned"], d["bc"], WIN_LO, WIN_HI)
        Xp250 = pca_features(d["binned"], d["bc"], 0.0, 250.0)
        rt_exec = (d["tt"]["exit_time_ms"] - d["tt"]["choice_time_ms"]).to_numpy(float)
        infl = rt_exec >= WIN_HI
        infl250 = rt_exec >= 250.0
        for tname in TARGETS:
            y = T[tname].to_numpy(float)
            rec = {"run_id": run, "target": tname, "n": int(len(y))}
            rec["raw"] = ridge_cv(Xp, y, g)
            rec["inflight"] = ridge_cv(Xp[infl], y[infl], g[infl])
            rec["rate"] = ridge_cv(Xr, y, g)
            rec["covctrl"] = ridge_cv(residue_cols(Xp, cov1000), residue_vec(y, cov1000), g)
            rec["cov_only"] = np.corrcoef(cov1000, y)[0, 1]
            rec["short_inflight"] = ridge_cv(Xp250[infl250], y[infl250], g[infl250])
            rec["cov_target_r"] = np.corrcoef(cov1000, y)[0, 1]
            rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(AGG_DIR / "coverage_control.csv", index=False)

    # figure: raw vs inflight vs covctrl per target, per run
    cols = ["raw", "inflight", "rate", "covctrl", "short_inflight", "cov_target_r"]
    ncol = len(RUN_ORDER)
    fig, axes = plt.subplots(1, ncol, figsize=(5.0 * ncol, 3.8), sharey=True)
    x = np.arange(len(TARGETS)); w = 0.16
    colors = ["0.75", "tab:blue", "tab:green", "tab:red", "tab:purple", "k"]
    for ax, run in zip(axes, RUN_ORDER):
        s = df[df.run_id == run].set_index("target")
        for i, (c, col) in enumerate(zip(cols, colors)):
            ax.bar(x + (i - 2.5) * w, [s.loc[t, c] for t in TARGETS], w, color=col,
                   edgecolor="k" if c != "cov_target_r" else "none",
                   label=(c if run == RUN_ORDER[0] else None))
        ax.axhline(0, color="k", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(TARGETS, rotation=45, ha="right", fontsize=7)
        ax.set_title(run, fontsize=10)
    axes[0].set_ylabel("ridge CV corr")
    axes[0].legend(fontsize=7)
    fig.suptitle("Coverage/censoring control: raw vs corrected decoding", fontsize=12)
    fig.tight_layout()
    fig.savefig(AGG_DIR / "figures" / "fig_coverage_control.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    print("\nmean over runs:")
    print(df.groupby("target")[cols].mean().round(3).to_string())
    print(f"\nwrote coverage_control.csv + fig_coverage_control.png")


if __name__ == "__main__":
    main()
