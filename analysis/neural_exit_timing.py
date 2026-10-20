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
# # Exit-hole targeting: timing (Test 2) and path geometry/value (Test 3)
#
# Test 2 (prospective exit anticipation): at a fixed 50 ms window after the
# entry pass (pre-choice only) or after the choice pass, decode (a) the exit
# time and (b) the exit direction (which side the goal is on relative to the
# ball). Test 3 (chosen-path-to-exit value/geometry): decode `conflict_mag`,
# signed `model_conflict`, `plan_advantage`, `greedy_advantage`,
# `chosen_2step_dist`, and remaining path length, with controls and a
# within-fixed-geometry residual test.
#
# Outputs (aggregate/):
#   exit_time_sweep.csv, exit_geometry.csv
#   figures/fig_exit_prospective.png, fig_exit_geometry.png

# %%
import argparse
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

from neural_common import RUNS, RUN_ORDER, AGG_DIR
from neural_lda_decoding import rate_features, pca_features, RNG_SEED
from neural_pass_timing import load_anchor, ridge_cv, residualize, residualize_features

# %%
WIN = 50.0
STEP = 50.0
TMAX = 1900.0
N_PERM = 20
MIN_TRIALS = 80
TARGETS_T2 = ["time_to_exit", "total"]

# %% -------------------------------------------------------------- helpers
_BALL = {}


def ball_sampler(run_id):
    if run_id in _BALL:
        return _BALL[run_id]
    d = json.load(open(RUNS[run_id].behavior_path, encoding="utf-8"))
    t, bx = [], []
    for b in d.get("blocks", []):
        if b.get("block_index", 0) < 4:
            continue
        gs = b.get("game_states") or {}
        if not gs:
            continue
        t.append(np.asarray(gs["time"], float))
        bx.append(np.asarray(gs["ball_x"], float))
    if not t:
        _BALL[run_id] = (lambda times: np.full(len(times), np.nan), 0.0, 1.0)
        return _BALL[run_id]
    t = np.concatenate(t)
    bx = np.concatenate(bx)
    o = np.argsort(t)
    t, bx = t[o], bx[o]
    gi = d.get("gameInfo") or {}
    gap = gi.get("levelWidth", 1.0) / 12.0
    fn = lambda times: bx[np.clip(np.searchsorted(t, np.asarray(times, float)), 0, len(t) - 1)]
    _BALL[run_id] = (fn, gi.get("levelStartX", 0.0), gap)
    return _BALL[run_id]


def grouped_lda_acc(X, y, groups, n_splits=5, seed=RNG_SEED):
    ng = len(np.unique(groups))
    n_splits = min(n_splits, ng)
    if len(np.unique(y)) < 2 or n_splits < 2:
        return np.nan
    accs = []
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        lda = LDA(solver="svd").fit(sc.transform(X[tr]), y[tr])
        accs.append(balanced_accuracy_score(y[te], lda.predict(sc.transform(X[te]))))
    return float(np.mean(accs)) if accs else np.nan


def sweep_exit_time(run, anchor, n_perm=N_PERM):
    """Entry(anchor=entry, pre-choice) or choice(anchor=choice) exit-time + direction."""
    d = load_anchor(run, anchor)
    binned, bc, first, last = d["binned"], d["bc"], d["first"], d["last"]
    rt, rt_exec = d["rt"], d["rt_exec"]
    observed = rt + rt_exec
    lab, tt = d["lab"], d["tt"]
    blocks = tt["block_index"].to_numpy()
    tids = d["tids"]
    anchor_t = (tt["entry_time_ms"] if anchor == "entry"
                else tt["choice_time_ms"]).to_numpy(float)
    fn, lsx, gap = ball_sampler(run)
    goal_x = lsx + (lab["goal_hole"].to_numpy() + 0.5) * gap

    rows = []
    for tau in np.arange(0.0, TMAX + 1e-9, STEP):
        lo, hi = tau, tau + WIN
        span = (first <= lo) & (last >= hi)
        if anchor == "entry":
            elig = span & (rt >= hi)
            tte, total = observed - tau, observed
        else:
            elig = span & (rt_exec >= hi)
            tte, total = rt_exec - tau, rt_exec
        if elig.sum() < MIN_TRIALS:
            continue
        m = elig
        X = rate_features(binned, bc, lo, hi)[m]
        xb = fn(anchor_t[m] + tau)
        ydir = (goal_x[m] > xb).astype(int)
        ytte, ytot = tte[m], total[m]
        blk = blocks[m]

        rec = {"run_id": run, "anchor": anchor, "tau_ms": float(tau),
               "n": int(m.sum()),
               "mean_lag_to_exit": float(np.nanmean(tte[m]))}
        for tname, y in [("time_to_exit", ytte), ("total", ytot)]:
            obs = ridge_cv(X, y, blk)
            rng = np.random.default_rng(RNG_SEED + int(tau))
            null = np.array([ridge_cv(X, rng.permutation(y), blk, seed=RNG_SEED + i)
                             for i in range(n_perm)])
            rec[f"{tname}_corr"] = obs
            rec[f"{tname}_p"] = float(np.mean(np.abs(null) >= abs(obs))) if np.isfinite(obs) else np.nan
        acc = grouped_lda_acc(X, ydir, tids[m])
        rec["direction_acc"] = acc
        rows.append(rec)
    return pd.DataFrame(rows)


# %% -------------------------------------------------------------- Test 2 fig
def fig_prospective(sweep, path):
    runs = RUN_ORDER
    fig, axes = plt.subplots(1, len(runs), figsize=(4.4 * len(runs), 3.6), sharey=True)
    for ax, run in zip(axes, runs):
        s = sweep[sweep["run_id"] == run]
        e = s[s["anchor"] == "entry"].sort_values("tau_ms")
        c = s[s["anchor"] == "choice"].sort_values("tau_ms")
        ax.plot(e["tau_ms"], e["time_to_exit_corr"], "-o", ms=3, color="tab:red",
                label="entry (pre-choice)")
        ax.plot(c["tau_ms"], c["time_to_exit_corr"], "-o", ms=3, color="tab:blue",
                label="choice")
        ax.axhline(0, color="k", lw=1)
        ax.set_title(run, fontsize=10)
        ax.set_xlabel("window offset τ (ms, rel. anchor)")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("ridge CV corr (time-to-exit)")
    fig.suptitle("Test 2: exit time decode — entry (pre-choice) vs choice anchored", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_direction(sweep, path):
    runs = RUN_ORDER
    fig, axes = plt.subplots(1, len(runs), figsize=(4.4 * len(runs), 3.6), sharey=True)
    for ax, run in zip(axes, runs):
        s = sweep[sweep["run_id"] == run]
        for anchor, col in [("entry", "tab:red"), ("choice", "tab:blue")]:
            g = s[s["anchor"] == anchor].sort_values("tau_ms")
            ax.plot(g["tau_ms"], g["direction_acc"], "-o", ms=3, color=col, label=anchor)
        ax.axhline(0.5, color="k", ls="--", lw=1)
        ax.set_title(run, fontsize=10)
        ax.set_xlabel("window offset τ (ms, rel. anchor)")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("exit-direction balanced acc")
    fig.suptitle("Test 2: exit direction decode (goal side vs ball x)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# %% -------------------------------------------------------------- Test 3
def geometry_targets(d):
    lab, tt = d["lab"], d["tt"]
    holes = tt["hole_locations"].map(json.loads)
    hA, hB = holes.str[0].to_numpy(), holes.str[1].to_numpy()
    gL, gR = lab["greedy_cost_L"].to_numpy(float), lab["greedy_cost_R"].to_numpy(float)
    pL, pR = lab["planning_cost_L"].to_numpy(float), lab["planning_cost_R"].to_numpy(float)
    chosen = lab["choice_hole"].to_numpy()
    ch1 = np.where(chosen == hA, gL, gR)
    ch2 = np.where(chosen == hA, pL, pR)
    mc = (pL - pR) - (gL - gR)
    gopt = lab["greedy_optimal_hole"].to_numpy()
    popt = lab["planning_optimal_hole"].to_numpy()
    plan_adv = np.where(gopt == hA, pL, pR) - np.where(popt == hA, pL, pR)
    greedy_adv = np.where(popt == hA, gL, gR) - np.where(gopt == hA, gL, gR)
    return pd.DataFrame({
        "conflict_mag": np.abs(mc), "model_conflict": mc,
        "plan_advantage": plan_adv, "greedy_advantage": greedy_adv,
        "chosen_1step": ch1, "chosen_2step": ch2, "remaining": ch2 - ch1,
    })


def sweep_geometry(run, n_perm=N_PERM):
    d = load_anchor(run, "choice")            # post-choice: the chosen path to exit
    binned, bc, first, last = d["binned"], d["bc"], d["first"], d["last"]
    rt_exec = d["rt_exec"]
    tt = d["tt"]
    blocks = tt["block_index"].to_numpy()
    G = geometry_targets(d)
    # nuisances: time-in-trial + position/movement/geometry
    K = np.column_stack([rt_exec, G["chosen_1step"], G["chosen_2step"],
                         tt["sequence_index"].to_numpy(float)])
    rows = []
    windows = {"pre": (-1000.0, 0.0), "post": (0.0, 1000.0), "whole": None}
    for wname, wh in windows.items():
        if wh is None:
            lo, hi, elig = -1000.0, 1000.0, (first <= -1000.0) & (last >= 1000.0)
        else:
            lo, hi = wh
            elig = (first <= lo) & (last >= hi)
            if wname == "pre":
                elig &= (tt["choice_time_ms"] - tt["entry_time_ms"]).to_numpy() >= -lo
            else:
                elig &= (rt_exec >= hi)
        if elig.sum() < MIN_TRIALS:
            continue
        Xp = pca_features(binned, bc, lo, hi)
        Xr = rate_features(binned, bc, lo, hi)
        for rep, X in [("pca", Xp), ("rate", Xr)]:
            for tname in G.columns:
                y = G[tname].to_numpy(float)
                specs = [("none", y, X),
                         ("target", residualize(y, K), X),
                         ("both", residualize(y, K), residualize_features(X, K))]
                for ctrl, yy, XX in specs:
                    corr = ridge_cv(XX[elig], yy[elig], blocks[elig])
                    p = np.nan
                    if ctrl == "both":
                        rng = np.random.default_rng(RNG_SEED + hash((wname, tname)) % 1000)
                        null = np.array([ridge_cv(XX[elig], rng.permutation(yy[elig]),
                                                  blocks[elig], seed=RNG_SEED + i)
                                         for i in range(n_perm)])
                        p = float(np.mean(np.abs(null) >= abs(corr))) if np.isfinite(corr) else np.nan
                    rows.append({"run_id": run, "rep": rep, "window": wname,
                                 "target": tname, "control": ctrl, "n": int(elig.sum()),
                                 "corr": corr, "perm_p": p})
            ycm = G["conflict_mag"].to_numpy(float)
            cm_res = residualize(ycm, G[["chosen_1step", "chosen_2step"]].to_numpy(float))
            corr = ridge_cv(X[elig], cm_res[elig], blocks[elig])
            rows.append({"run_id": run, "rep": rep, "window": wname,
                         "target": "conflict_mag|geom", "control": "fixed-geom",
                         "n": int(elig.sum()), "corr": corr, "perm_p": np.nan})
    return pd.DataFrame(rows)


def fig_lag_matched(sweep, path):
    runs = RUN_ORDER
    fig, axes = plt.subplots(1, len(runs), figsize=(4.4 * len(runs), 3.6), sharey=True)
    for ax, run in zip(axes, runs):
        s = sweep[(sweep["run_id"] == run) & (sweep["time_to_exit_corr"].notna())]
        for anchor, col in [("entry", "tab:red"), ("choice", "tab:blue")]:
            g = s[s["anchor"] == anchor].sort_values("mean_lag_to_exit")
            ax.plot(g["mean_lag_to_exit"], g["time_to_exit_corr"], "-o", ms=3,
                    color=col, label=anchor)
        ax.axhline(0, color="k", lw=1)
        ax.set_title(run, fontsize=10)
        ax.set_xlabel("mean time-to-exit at the window (ms)")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("ridge CV corr (time-to-exit)")
    fig.suptitle("Test 2 lag-to-exit-matched: entry (pre-choice) vs choice", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_geometry(geom, path, rep="pca", window="whole"):
    runs = RUN_ORDER
    tgts = ["conflict_mag", "model_conflict", "plan_advantage",
            "greedy_advantage", "chosen_2step", "remaining"]
    fig, axes = plt.subplots(1, len(runs), figsize=(5.0 * len(runs), 3.8), sharey=True)
    x = np.arange(len(tgts))
    w = 0.27
    for ax, run in zip(axes, runs):
        s = geom[(geom["run_id"] == run) & (geom["rep"] == rep) & (geom["window"] == window)]
        for i, (ctrl, col) in enumerate([("none", "0.75"), ("target", "tab:orange"),
                                         ("both", "tab:red")]):
            vals = [s[(s["target"] == t) & (s["control"] == ctrl)]["corr"].mean()
                    for t in tgts]
            ax.bar(x + (i - 1) * w, vals, w, color=col, edgecolor="k", label=ctrl)
        fg = [s[(s["target"] == "conflict_mag|geom")]["corr"].mean()]
        ax.scatter([0], fg, marker="*", s=120, color="tab:purple", zorder=5,
                   label="conflict_mag | geom")
        ax.axhline(0, color="k", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(tgts, rotation=45, ha="right", fontsize=7)
        ax.set_title(run, fontsize=10)
        ax.legend(fontsize=7)
    axes[0].set_ylabel(f"ridge CV corr ({rep}, {window})")
    fig.suptitle("Test 3: chosen-path-to-exit value/geometry", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# %%
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=RUN_ORDER)
    ap.add_argument("--perm", type=int, default=N_PERM)
    args = ap.parse_args()

    t2, t3 = [], []
    for run in args.runs:
        print(f"[{run}] Test 2 exit-time sweep")
        for anchor in ["entry", "choice"]:
            t2.append(sweep_exit_time(run, anchor, args.perm))
        print(f"[{run}] Test 3 geometry sweep")
        t3.append(sweep_geometry(run, args.perm))

    t2 = pd.concat(t2, ignore_index=True)
    t3 = pd.concat(t3, ignore_index=True)
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    (AGG_DIR / "figures").mkdir(parents=True, exist_ok=True)
    t2.to_csv(AGG_DIR / "exit_time_sweep.csv", index=False)
    t3.to_csv(AGG_DIR / "exit_geometry.csv", index=False)

    fig_prospective(t2, AGG_DIR / "figures" / "fig_exit_prospective.png")
    fig_direction(t2, AGG_DIR / "figures" / "fig_exit_direction.png")
    fig_lag_matched(t2, AGG_DIR / "figures" / "fig_exit_lag_matched.png")
    fig_geometry(t3, AGG_DIR / "figures" / "fig_exit_geometry.png", rep="pca")

    print("\nTest 2 peaks (mean over runs, time_to_exit):")
    m = t2.groupby(["anchor", "tau_ms"])["time_to_exit_corr"].mean().reset_index()
    for a in ["entry", "choice"]:
        s = m[m["anchor"] == a]
        if len(s):
            r = s.loc[s["time_to_exit_corr"].abs().idxmax()]
            print(f"  {a:6s} peak |corr| {abs(r['time_to_exit_corr']):.3f} @ {r['tau_ms']:+.0f} ms")
    print("\nTest 3 (mean over runs, pca, whole, control=both):")
    m = t3[(t3["control"] == "both") & (t3["rep"] == "pca") & (t3["window"] == "whole")]
    for t in m["target"].unique():
        print(f"  {t:20s} corr {m[m['target'] == t]['corr'].mean():+.3f}")
    print(f"\nwrote exit_time_sweep.csv, exit_geometry.csv + figures to {AGG_DIR}")


if __name__ == "__main__":
    main()
