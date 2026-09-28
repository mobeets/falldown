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
# # Tier 1.2 — value vs place / successor axis alignment
#
# Is the value/`conflict_mag` population axis the *same* axis as a place or
# successor-representation axis? Per run:
#   * OOF ridge predictions of `conflict_mag`, `planning_gap`, `entry_hole`
#     (place), and the first principal component of the successor features
#     `M[entry]`;
#   * alignment = correlation between the OOF prediction vectors, plus the
#     cosine between the full-data fitted weight vectors.
#
# Outputs:
#   analysis/neural_outputs/<run>/axis_alignment.csv
#   analysis/neural_outputs/aggregate/axis_alignment_summary.csv
#   analysis/neural_outputs/aggregate/figures/fig_neural_axis_alignment.png

# %%
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from neural_common import RUN_ORDER, out_dir, AGG_DIR
from neural_lda_decoding import rate_features, RNG_SEED
from neural_successor_geometry import (
    run_arrays, transition_matrix, successor, N_SEGMENTS,
)

WINDOW = (0.0, 1000.0)
GAMMA = 0.9
N_FOLDS = 5
N_REPEATS = 3
ALPHA = 1.0


def oof_predictions(Y, X, n_folds=N_FOLDS, n_repeats=N_REPEATS, seed=RNG_SEED):
    """(n_trials, n_targets) mean out-of-fold ridge predictions."""
    Y = np.asarray(Y, float)
    if Y.ndim == 1:
        Y = Y[:, None]
    preds = np.full((n_repeats, len(Y), Y.shape[1]), np.nan)
    for rep in range(n_repeats):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed + rep)
        for tr, te in kf.split(X):
            sc = StandardScaler().fit(X[tr])
            model = Ridge(alpha=ALPHA).fit(sc.transform(X[tr]), Y[tr])
            pred = np.asarray(model.predict(sc.transform(X[te])))
            preds[rep, te] = pred.reshape(len(te), Y.shape[1])
    return np.nanmean(preds, axis=0)


def axis_vectors(Y, X):
    """Standardized ridge weight vector(s) on the full data (features x targets)."""
    Y = np.asarray(Y, float)
    if Y.ndim == 1:
        Y = Y[:, None]
    sc = StandardScaler().fit(X)
    model = Ridge(alpha=ALPHA).fit(sc.transform(X), Y)
    coef = np.asarray(model.coef_)
    if coef.ndim == 1:
        coef = coef[None, :]
    return coef  # (n_targets, n_features)


def cosine(u, v):
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))


def analyse_run(run_id):
    out = out_dir(run_id)
    z = np.load(out / "segmented_spikes_binned.npz", allow_pickle=True)
    binned, bc, tids = z["binned"], z["bin_centers"], z["trial_ids"]
    lab = pd.read_csv(out / "trial_labels.csv").set_index("trial_id").loc[tids]
    tt = pd.read_csv(out / "trial_table.csv").set_index("trial_id").loc[tids]
    arr = run_arrays(lab, tt)
    T = transition_matrix(arr)
    M = successor(T, GAMMA)
    entry = arr["entry"]
    ok = np.isfinite(entry) & np.isfinite(arr["conflict_mag"])
    entry_i = np.where(ok, entry, 0).astype(int)

    X = rate_features(binned, bc, *WINDOW)
    X = X[ok]
    targets = {
        "conflict_mag": arr["conflict_mag"][ok],
        "planning_gap": arr["planning_gap"][ok],
        "place_entry": entry[ok],
        "sr": M[entry_i[ok]],
    }
    # SR reduced to its first PC across trials
    sr = targets["sr"]
    sr = sr - sr.mean(0)
    u, s, vt = np.linalg.svd(sr, full_matrices=False)
    sr_pc1 = u[:, 0] * s[0]

    P = {}
    for name, y in [("value", targets["conflict_mag"]),
                    ("planning_gap", targets["planning_gap"]),
                    ("place", targets["place_entry"])]:
        P[name] = oof_predictions(y, X)[:, 0]
    P["sr"] = oof_predictions(sr_pc1, X)[:, 0]

    rows = []
    def corr(a, b):
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])
    for a, b in [("value", "sr"), ("value", "place"),
                 ("value", "planning_gap"), ("sr", "place")]:
        rows.append({"run_id": run_id, "analysis": "oof_pred_corr",
                     "pair": f"{a}|{b}", "value": corr(P[a], P[b])})

    # fitted axis cosines
    av = axis_vectors(targets["conflict_mag"], X)[0]
    ap = axis_vectors(targets["place_entry"], X)[0]
    asr = axis_vectors(targets["sr"], X)          # (12, n_features)
    # dominant SR direction via SVD of the weight matrix
    _, _, vt_sr = np.linalg.svd(asr, full_matrices=False)
    asr_dir = vt_sr[0]
    rows.append({"run_id": run_id, "analysis": "axis_cosine",
                 "pair": "value|sr", "value": cosine(av, asr_dir)})
    rows.append({"run_id": run_id, "analysis": "axis_cosine",
                 "pair": "value|place", "value": cosine(av, ap)})
    return pd.DataFrame(rows)


def make_figure(agg):
    fig, ax = plt.subplots(figsize=(8, 5))
    pairs = ["value|sr", "value|place", "value|planning_gap", "sr|place"]
    x = np.arange(len(pairs))
    oof = [agg[(agg.analysis == "oof_pred_corr") & (agg.pair == p)]["value"].mean()
           for p in pairs]
    cos = [agg[(agg.analysis == "axis_cosine") & (agg.pair == p)]["value"].mean()
           for p in pairs]
    ax.bar(x - 0.2, oof, width=0.4, label="OOF prediction corr", color="#2a9d8f")
    ax.bar(x + 0.2, cos, width=0.4, label="axis cosine", color="#e07a5f")
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(pairs, rotation=20, ha="right", fontsize=8)
    ax.set_title("Value axis vs place / successor axis")
    ax.legend()
    fig.tight_layout()
    figdir = AGG_DIR / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    out = figdir / "fig_neural_axis_alignment.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for run_id in RUN_ORDER:
        print(f"[axis-align] {run_id}")
        try:
            df = analyse_run(run_id)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            continue
        df.to_csv(out_dir(run_id) / "axis_alignment.csv", index=False)
        frames.append(df)
    if not frames:
        print("no results"); return
    agg = pd.concat(frames, ignore_index=True)
    agg.to_csv(AGG_DIR / "axis_alignment.csv", index=False)
    summ = agg.groupby(["analysis", "pair"])["value"].agg(
        ["mean", "size"]).reset_index()
    summ.to_csv(AGG_DIR / "axis_alignment_summary.csv", index=False)
    fig = make_figure(agg)
    print(summ.round(3).to_string(index=False))
    print(f"\nwrote axis_alignment.csv per run/summary and {fig}")


if __name__ == "__main__":
    main()
