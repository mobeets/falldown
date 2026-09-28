# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_version: '1.3'
#     jupytext_version: 1.19.4
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tier 1.3 — fine-grained entry-anchored pre-decision decoding
#
# The existing entry sliding trace (`neural_entry_decoding.py`) is 150 ms / 50 ms
# with no permutation on the trace and binary labels only. This script runs a
# finer 50 ms window / 25 ms step from the entry pass, for:
#   * binary `planning_vs_greedy` (LDA, balanced accuracy);
#   * continuous `conflict_mag` (ridge, CV correlation);
# with permutation p per time bin and BH-FDR across time, plus an entry-frame
# temporal-generalization matrix. Windows never cross the choice pass (trials
# require `rt_ms >= window end`).
#
# Outputs:
#   analysis/neural_outputs/<run>/entry_sliding_fine.csv
#   analysis/neural_outputs/<run>/entry_sliding_continuous.csv
#   analysis/neural_outputs/<run>/entry_temporal_generalization.csv
#   analysis/neural_outputs/aggregate/entry_sliding_fine_summary.csv
#   analysis/neural_outputs/aggregate/figures/fig_neural_entry_sliding_fine.png

# %%
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

from neural_common import RUN_ORDER, out_dir, AGG_DIR
from neural_lda_decoding import rate_features, build_label_vectors, RNG_SEED

WIN = 50.0
STEP = 50.0
MAX_START = 400.0
N_FOLDS = 5
N_REPEATS = 2
N_PERM = 30
ALPHA = 1.0
TG_STARTS = [0.0, 100.0, 200.0, 300.0, 400.0]
TG_LEN = 100.0


def load_run(run_id):
    out = out_dir(run_id)
    z = np.load(out / "segmented_spikes_entrylocked_binned.npz", allow_pickle=True)
    tids = z["trial_ids"]
    lab = pd.read_csv(out / "trial_labels.csv").set_index("trial_id").loc[tids].reset_index()
    tt = pd.read_csv(out / "trial_table.csv").set_index("trial_id").loc[tids].reset_index()
    rt = (tt["choice_time_ms"] - tt["entry_time_ms"]).to_numpy(float)
    return out, z["binned"], z["bin_centers"], lab, rt


def binary_cv(y, X, seed=RNG_SEED, n_repeats=N_REPEATS):
    accs = []
    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed + rep)
        for tr, te in skf.split(X, y):
            sc = StandardScaler().fit(X[tr])
            lda = LinearDiscriminantAnalysis(solver="svd").fit(
                sc.transform(X[tr]), y[tr])
            accs.append(balanced_accuracy_score(y[te], lda.predict(sc.transform(X[te]))))
    return float(np.mean(accs))


def continuous_cv(y, X, seed=RNG_SEED, n_repeats=N_REPEATS):
    corrs = []
    for rep in range(n_repeats):
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed + rep)
        for tr, te in kf.split(X):
            sc = StandardScaler().fit(X[tr])
            m = Ridge(alpha=ALPHA).fit(sc.transform(X[tr]), y[tr])
            pred = m.predict(sc.transform(X[te]))
            if np.std(pred) > 1e-12 and np.std(y[te]) > 1e-12:
                corrs.append(float(np.corrcoef(y[te], pred)[0, 1]))
    return float(np.nanmean(corrs)) if corrs else np.nan


def analyse_run(run_id):
    out, binned, bc, lab, rt = load_run(run_id)
    lab = lab.copy(); lab["move_dir_vx"] = 0
    lv = build_label_vectors(lab)
    y_all, idx = lv["planning_vs_greedy"]
    conflict_mag = np.abs((lab["planning_cost_L"] - lab["planning_cost_R"])
                          - (lab["greedy_cost_L"] - lab["greedy_cost_R"])).to_numpy()

    rows, crows = [], []
    starts = np.arange(0.0, MAX_START + 1e-9, STEP)
    for s in starts:
        end = s + WIN
        elig = np.isfinite(rt) & (rt >= end)
        if idx is not None:
            elig = elig & np.isin(np.arange(len(lab)), idx)
        if elig.sum() < 60 or len(np.unique(y_all[elig])) < 2:
            continue
        X = rate_features(binned, bc, s, end)[elig]
        y = y_all[elig]
        acc = binary_cv(y, X)
        rng = np.random.default_rng(RNG_SEED + int(s))
        null = np.array([binary_cv(rng.permutation(y), X,
                                   seed=RNG_SEED + int(s) + i, n_repeats=1)
                         for i in range(N_PERM)])
        rows.append({"run_id": run_id, "start_ms": s, "end_ms": end,
                     "n": int(elig.sum()), "acc": acc, "chance": 0.5,
                     "perm_p": float(np.mean(null >= acc))})
        # continuous conflict_mag
        yc = conflict_mag[elig]
        if np.isfinite(yc).sum() > 60:
            cm = continuous_cv(yc, X)
            nullc = np.array([continuous_cv(rng.permutation(yc), X,
                                            seed=RNG_SEED + int(s) + i,
                                            n_repeats=1)
                              for i in range(N_PERM)])
            crows.append({"run_id": run_id, "start_ms": s, "end_ms": end,
                          "n": int(elig.sum()), "cv_corr": cm,
                          "perm_p": float(np.mean(np.abs(nullc) >= abs(cm)))})
        print(f"  {run_id} {s:.0f}-{end:.0f} ms n={int(elig.sum())} acc={acc:.3f}")
    df = pd.DataFrame(rows)
    if len(df):
        df["q_fdr"] = multipletests(df["perm_p"], method="fdr_bh")[1]
    cdf = pd.DataFrame(crows)
    if len(cdf):
        cdf["q_fdr"] = multipletests(cdf["perm_p"], method="fdr_bh")[1]

    # temporal generalization (binary, 100 ms windows)
    tg = []
    for s_tr in TG_STARTS:
        e_tr = s_tr + TG_LEN
        for s_te in TG_STARTS:
            e_te = s_te + TG_LEN
            elig = np.isfinite(rt) & (rt >= max(e_tr, e_te))
            if idx is not None:
                elig = elig & np.isin(np.arange(len(lab)), idx)
            if elig.sum() < 60:
                continue
            Xtr = rate_features(binned, bc, s_tr, e_tr)[elig]
            Xte = rate_features(binned, bc, s_te, e_te)[elig]
            y = y_all[elig]
            accs = []
            for rep in range(N_REPEATS):
                skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                      random_state=RNG_SEED + rep)
                for tr, te in skf.split(Xtr, y):
                    sc = StandardScaler().fit(Xtr[tr])
                    lda = LinearDiscriminantAnalysis(solver="svd").fit(
                        sc.transform(Xtr[tr]), y[tr])
                    accs.append(balanced_accuracy_score(
                        y[te], lda.predict(sc.transform(Xte[te]))))
            tg.append({"run_id": run_id, "train_ms": s_tr, "test_ms": s_te,
                       "n": int(elig.sum()), "acc": float(np.mean(accs))})
    df.to_csv(out / "entry_sliding_fine.csv", index=False)
    cdf.to_csv(out / "entry_sliding_continuous.csv", index=False)
    pd.DataFrame(tg).to_csv(out / "entry_temporal_generalization.csv", index=False)
    return df, cdf, pd.DataFrame(tg)


def make_figure(agg, cagg, tgagg):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    ax = axes[0]
    for run, g in agg.groupby("run_id"):
        g = g.sort_values("start_ms")
        ax.plot(g["start_ms"], g["acc"], "-o", ms=3, alpha=0.6, label=run)
    mean = agg.groupby("start_ms")["acc"].mean()
    ax.plot(mean.index, mean.values, "-k", lw=2, label="mean")
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.set_xlabel("entry-anchored window start (ms)")
    ax.set_ylabel("balanced accuracy (planning_vs_greedy)")
    ax.set_title("A. Fine entry sliding (50 ms / 25 ms)")
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.axhline(0, color="k", lw=1)
    if not cagg.empty:
        for run, g in cagg.groupby("run_id"):
            g = g.sort_values("start_ms")
            ax.plot(g["start_ms"], g["cv_corr"], "-o", ms=3, alpha=0.6, label=run)
        cm = cagg.groupby("start_ms")["cv_corr"].mean()
        ax.plot(cm.index, cm.values, "-k", lw=2)
    ax.set_xlabel("entry-anchored window start (ms)")
    ax.set_ylabel("ridge CV correlation")
    ax.set_title("B. Continuous conflict_mag")
    ax.legend(fontsize=7)

    fig2data = tgagg
    ax = axes[2]
    piv = fig2data.pivot_table(index="train_ms", columns="test_ms", values="acc")
    im = ax.imshow(piv.values, cmap="viridis", vmin=0.45, vmax=0.75, aspect="auto")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    ax.set_xlabel("test window (ms)"); ax.set_ylabel("train window (ms)")
    ax.set_title("C. Entry temporal generalization")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Entry-anchored pre-decision decoding (never crosses choice)",
                 fontsize=13)
    fig.tight_layout()
    figdir = AGG_DIR / "figures"; figdir.mkdir(parents=True, exist_ok=True)
    out = figdir / "fig_neural_entry_sliding_fine.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    frames, cframes, tframes = [], [], []
    for run_id in RUN_ORDER:
        print(f"[entry-sliding] {run_id}")
        try:
            df, cdf, tg = analyse_run(run_id)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            continue
        frames.append(df); cframes.append(cdf); tframes.append(tg)
    if not frames:
        print("no results"); return
    agg = pd.concat(frames, ignore_index=True)
    cagg = pd.concat(cframes, ignore_index=True)
    tagg = pd.concat(tframes, ignore_index=True)
    agg.to_csv(AGG_DIR / "entry_sliding_fine.csv", index=False)
    cagg.to_csv(AGG_DIR / "entry_sliding_continuous.csv", index=False)
    tagg.to_csv(AGG_DIR / "entry_temporal_generalization.csv", index=False)
    summ = agg.groupby("start_ms")["acc"].agg(["mean", "size"]).reset_index()
    summ.to_csv(AGG_DIR / "entry_sliding_fine_summary.csv", index=False)
    fig = make_figure(agg, cagg, tagg)
    peak = summ.loc[summ["mean"].idxmax()]
    print(f"\npeak mean acc = {peak['mean']:.3f} at {peak['start_ms']:.0f} ms "
          f"(chance 0.5); n significant bins (FDR<.05) = "
          f"{int((agg['q_fdr'] < 0.05).sum())}")
    print(f"wrote entry_sliding_fine.csv, summaries, {fig}")


if __name__ == "__main__":
    main()
