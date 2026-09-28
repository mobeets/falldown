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
# # Tier 1.2 — successor / place geometry RSA
#
# Does the mesial-temporal population represent a **successor/place** geometry
# rather than (or in addition to) raw Euclidean path geometry and value?
#
# Per run, build an empirical transition matrix `T` over the 12 horizontal
# segments from the 1-2-1 hole sequence (entry -> chosen -> goal), the successor
# matrix `M = (I - gamma T)^-1`, and model RDMs:
#   * `sr`        cosine distance between successor rows `M[entry]`
#   * `place`     one-hot segment identity
#   * `entry_dist`|entry_i - entry_j|  (raw Euclidean geometry)
#   * `value`     |conflict_mag_i - conflict_mag_j|, planning/greedy gaps,
#                 chosen costs
# Neural RDM = 1 - correlation between z-scored per-unit rates. Spearman
# association + label-permutation p, plus partial Spearman controlling geometry
# and value (is the SR link unique?).
#
# Outputs:
#   analysis/neural_outputs/<run>/sr_rsa.csv
#   analysis/neural_outputs/aggregate/sr_rsa_aggregate.csv
#   analysis/neural_outputs/aggregate/figures/fig_neural_sr_rsa.png
#
# Run with:
#   python analysis/neural_successor_geometry.py

# %%
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata

from neural_common import RUN_ORDER, out_dir, AGG_DIR
from neural_lda_decoding import rate_features, RNG_SEED

N_SEGMENTS = 12
GAMMAS = (0.5, 0.9)
WINDOWS = {"post": (0.0, 1000.0)}
N_PERM = 100
ROW_MODELS = ["sr", "place"]
SCALAR_MODELS = ["conflict_mag", "planning_gap", "greedy_gap",
                 "chosen_planning_cost", "chosen_greedy_cost"]
C_SR = "#2a9d8f"


# %%
def load_run(run_id):
    out = out_dir(run_id)
    z = np.load(out / "segmented_spikes_binned.npz", allow_pickle=True)
    binned, unit_ids, bc, tids = (z["binned"], z["unit_ids"],
                                  z["bin_centers"], z["trial_ids"])
    lab = pd.read_csv(out / "trial_labels.csv").set_index("trial_id").loc[tids]
    tt = pd.read_csv(out / "trial_table.csv").set_index("trial_id").loc[tids]
    return out, binned, bc, lab, tt


def run_arrays(lab, tt):
    entry = lab["entry_hole"].to_numpy(float)
    goal = lab["goal_hole"].to_numpy(float)
    chosen = lab["choice_hole"].to_numpy(float)
    gL = lab["greedy_cost_L"].to_numpy(float)
    gR = lab["greedy_cost_R"].to_numpy(float)
    pL = lab["planning_cost_L"].to_numpy(float)
    pR = lab["planning_cost_R"].to_numpy(float)

    def holes(s):
        try:
            return json.loads(s)
        except Exception:
            return [np.nan, np.nan]
    hl = tt["hole_locations"].map(holes)
    hA, hB = hl.str[0].to_numpy(), hl.str[1].to_numpy()
    isA = chosen == hA
    greedy_gap = gL - gR
    planning_gap = pL - pR
    return {
        "entry": entry, "goal": goal, "chosen": chosen,
        "conflict_mag": np.abs(planning_gap - greedy_gap),
        "planning_gap": planning_gap, "greedy_gap": greedy_gap,
        "chosen_planning_cost": np.where(isA, pL, pR),
        "chosen_greedy_cost": np.where(isA, gL, gR),
        "condition": lab["condition"].to_numpy(),
    }


def transition_matrix(arr):
    C = np.zeros((N_SEGMENTS, N_SEGMENTS))
    for i, j in zip(arr["entry"], arr["chosen"]):
        if np.isfinite(i) and np.isfinite(j):
            C[int(i), int(j)] += 1
    for i, j in zip(arr["chosen"], arr["goal"]):
        if np.isfinite(i) and np.isfinite(j):
            C[int(i), int(j)] += 1
    row = C.sum(1, keepdims=True)
    T = np.divide(C, row, out=np.zeros_like(C), where=row > 0)
    return T


def successor(T, gamma):
    return np.linalg.inv(np.eye(N_SEGMENTS) - gamma * T)


# %%
def _rank(x):
    return rankdata(x)


def _pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 0 else np.nan


def _condensed(D):
    iu = np.triu_indices(D.shape[0], 1)
    return D[iu]


def _vector_rdm(V):
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    return 1.0 - Vn @ Vn.T


def _spearman(a, b):
    return _pearson(_rank(a), _rank(b))


def _partial_spearman(a, b, controls):
    ra, rb = _rank(a), _rank(b)
    C = np.column_stack([_rank(c) for c in controls] + [np.ones(len(a))])
    ra = ra - C @ np.linalg.lstsq(C, ra, rcond=None)[0]
    rb = rb - C @ np.linalg.lstsq(C, rb, rcond=None)[0]
    return _pearson(ra, rb)


# %% [markdown]
# ## Main

# %%
def analyse_run(run_id):
    out, binned, bc, lab, tt = load_run(run_id)
    arr = run_arrays(lab, tt)
    T = transition_matrix(arr)
    valid = np.isfinite(arr["entry"]) & np.isfinite(arr["conflict_mag"])
    conflict = np.isin(arr["condition"], ["planning", "greedy"])
    rows = []
    for wname, (lo, hi) in WINDOWS.items():
        X = rate_features(binned, bc, lo, hi)
        Xz = (X - X.mean(0)) / (X.std(0) + 1e-9)
        for subset, mask in [("all", valid), ("conflict", valid & conflict)]:
            sel = np.flatnonzero(mask)
            if len(sel) < 40:
                continue
            D = 1.0 - np.corrcoef(Xz[sel])
            dvec = _condensed(D)
            rd = _rank(dvec)
            print(f"  {wname}/{subset}: n={len(sel)} pairs={len(dvec)}")
            M50 = successor(T, GAMMAS[0])
            M90 = successor(T, GAMMAS[1])
            models = {
                f"sr_g{GAMMAS[0]}": _condensed(_vector_rdm(M50[arr["entry"][sel].astype(int)])),
                f"sr_g{GAMMAS[1]}": _condensed(_vector_rdm(M90[arr["entry"][sel].astype(int)])),
                "place": _condensed(1.0 - np.equal.outer(
                    arr["entry"][sel], arr["entry"][sel]).astype(float)),
                "entry_dist": _condensed(np.abs(
                    arr["entry"][sel][:, None] - arr["entry"][sel][None, :])),
            }
            for name in SCALAR_MODELS:
                v = arr[name][sel]
                models[name] = _condensed(np.abs(v[:, None] - v[None, :]))
            ctrl_geo = models["entry_dist"]
            ctrl_val = [models["conflict_mag"], models["planning_gap"]]
            rng = np.random.default_rng(RNG_SEED)

            def sp(mv):
                return _pearson(rd, _rank(mv))

            for name, mvec in models.items():
                r = sp(mvec)
                null = np.empty(N_PERM)
                for k in range(N_PERM):
                    if name.startswith("sr"):
                        g = float(name.split("_g")[1])
                        vp = rng.permutation(arr["entry"][sel].astype(int))
                        Dp = _vector_rdm(successor(T, g)[vp])
                        null[k] = sp(_condensed(Dp))
                    elif name == "place":
                        vp = rng.permutation(arr["entry"][sel])
                        null[k] = sp(_condensed(
                            1.0 - np.equal.outer(vp, vp).astype(float)))
                    else:
                        base = arr["entry"] if name == "entry_dist" else arr[name]
                        vv = rng.permutation(base[sel])
                        null[k] = sp(_condensed(
                            np.abs(vv[:, None] - vv[None, :])))
                rows.append({
                    "run_id": run_id, "window": wname, "subset": subset,
                    "model": name, "n": len(sel), "spearman_r": r,
                    "perm_p": float(np.mean(np.abs(null) >= abs(r))),
                    "partial_geom": _partial_spearman(dvec, mvec, [ctrl_geo]),
                    "partial_value": _partial_spearman(dvec, mvec, ctrl_val),
                })
    return pd.DataFrame(rows)


def make_figure(agg):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, subset in zip(axes, ["all", "conflict"]):
        a = agg[(agg.window == "post") & (agg.subset == subset)]
        ms = [m for m in a["model"].unique()]
        x = np.arange(len(ms))
        mean = [a[a.model == m]["spearman_r"].mean() for m in ms]
        colors = [C_SR if m.startswith("sr") else "#e07a5f" if m == "place"
                  else "#3d5a80" for m in ms]
        ax.bar(x, mean, color=colors)
        ax.axhline(0, color="k", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(ms, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("mean Spearman r (neural vs model RDM)")
        ax.set_title(f"post window, subset={subset}")
    fig.suptitle("Neural RDM vs successor / place / geometry / value RDMs", fontsize=13)
    fig.tight_layout()
    figdir = AGG_DIR / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    out = figdir / "fig_neural_sr_rsa.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for run_id in RUN_ORDER:
        print(f"[sr-rsa] {run_id}")
        try:
            df = analyse_run(run_id)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            continue
        df.to_csv(out_dir(run_id) / "sr_rsa.csv", index=False)
        all_rows.append(df)
        print(df.round(3).to_string(index=False))
    if not all_rows:
        print("no results")
        return
    agg = pd.concat(all_rows, ignore_index=True)
    # aggregate sign consistency across runs
    g = (agg.groupby(["window", "subset", "model"])
         .agg(mean_r=("spearman_r", "mean"), n_runs=("spearman_r", "size"),
              n_same_sign=("spearman_r", lambda s: int(max((s > 0).sum(),
                                                           (s < 0).sum()))),
              mean_partial_geom=("partial_geom", "mean"),
              mean_partial_value=("partial_value", "mean")).reset_index())
    agg.to_csv(AGG_DIR / "sr_rsa_aggregate.csv", index=False)
    g.to_csv(AGG_DIR / "sr_rsa_summary.csv", index=False)
    fig = make_figure(agg)
    print("\nAggregate:")
    print(g.round(3).to_string(index=False))
    print(f"\nwrote sr_rsa.csv per run, sr_rsa_aggregate.csv, {fig}")


if __name__ == "__main__":
    main()
