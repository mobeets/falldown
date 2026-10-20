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
# # Pass-timing decode (Part 2)
#
# Independently of the comparison sweep, ask whether the population can tell
# **when the ball passes a layer**.
#
# * 2A pass-event detection: is this 25 ms bin within ±50 ms of a layer pass?
#   one sample per (trial, bin); GroupKFold by trial; permutation null;
#   temporal-generalization (lag x lag) matrix.
# * 2B pass-time regression:
#     B1 per-bin ridge trace: predict the pass time from each bin's activity
#        (leave-block-out CV), for the non-anchored passes.
#     B2 whole-window ridge: reuse the Part 1 250 ms windows, predict rt and
#        rt_resid (kinematic residual), controls none/target/both.
#
# Both existing binnings are used (entry-anchored and choice-anchored).
#
# Outputs (aggregate/):
#   pass_event_detection.csv, pass_event_transfer.csv, pass_time_regression.csv
#   figures/fig_pass_event_detection.png, fig_pass_event_transfer.png,
#           fig_pass_time_regression.png, fig_pass_time_controls.png
#
# Run:  .venv-analysis/bin/python analysis/neural_pass_timing.py

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

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from neural_common import RUN_ORDER, RUNS, AGG_DIR, out_dir
from neural_lda_decoding import rate_features, RNG_SEED

# %%
BIN_MS = 25.0
TOL_MS = 50.0
N_FOLDS = 5
N_PERM_EVENT = 100
N_PERM_TRACE = 30
N_PERM_WIN = 50
RIDGE_ALPHA = 1.0
TRACE_STEP = 2                       # every 2nd bin for B1 (speed)
TRACE_LO, TRACE_HI = -400.0, 1400.0  # restrict B1 to the informative span

WIN = 250.0
CENTERS = np.arange(-300.0, 1000.0 + 1e-9, 50.0)
TG_OFFSETS = [-100.0, -50.0, 0.0, 50.0, 100.0]
FAR_MS = 300.0                       # |rel| > this = "non-pass" bin

PASSES = {"entry": ["choice", "exit", "entry"],
          "choice": ["entry", "exit", "choice"]}
TRACE_TARGETS = {"entry": ["choice", "exit"],       # rt, rt+rt_exec
                 "choice": ["entry", "exit"]}       # -rt, rt_exec


# %%
def load_anchor(run_id, anchor):
    out = out_dir(run_id)
    fname = ("segmented_spikes_entrylocked_binned.npz" if anchor == "entry"
             else "segmented_spikes_binned.npz")
    z = np.load(out / fname, allow_pickle=True)
    tids = z["trial_ids"]
    lab = pd.read_csv(out / "trial_labels.csv").set_index("trial_id").loc[tids].reset_index()
    tt = pd.read_csv(out / "trial_table.csv").set_index("trial_id").loc[tids].reset_index()
    binned, bc = z["binned"], z["bin_centers"]
    valid = ~np.isnan(binned).all(axis=0)
    first = np.array([bc[np.flatnonzero(valid[i])[0]] if valid[i].any() else np.nan
                      for i in range(len(valid))])
    last = np.array([bc[np.flatnonzero(valid[i])[-1]] if valid[i].any() else np.nan
                     for i in range(len(valid))])
    rt = (tt["choice_time_ms"] - tt["entry_time_ms"]).to_numpy(float)
    rt_exec = (tt["exit_time_ms"] - tt["choice_time_ms"]).to_numpy(float)
    return dict(run_id=run_id, anchor=anchor, binned=binned, rate=binned / 0.025,
                bc=bc, valid=valid, first=first, last=last, tt=tt, lab=lab,
                rt=rt, rt_exec=rt_exec, tids=tids)


def pass_time(data, anchor, which):
    if anchor == "entry":
        return {"entry": np.zeros_like(data["rt"]),
                "choice": data["rt"],
                "exit": data["rt"] + data["rt_exec"]}[which]
    return {"entry": -data["rt"], "choice": np.zeros_like(data["rt"]),
            "exit": data["rt_exec"]}[which]


# =============================================================== 2A
def event_index(data):
    """Flat per-(trial,bin) index over valid bins + time relative to each pass."""
    ti, bi = np.where(data["valid"])
    return ti, bi


def rel_to_pass(data, anchor, which, ti, bi, p):
    return data["bc"][bi] - p[ti]


def build_xy(data, ti, bi, rel, rng, tol=TOL_MS, balance=True):
    y = (np.abs(rel) <= tol).astype(int)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if balance and len(pos) and len(neg):
        neg = rng.choice(neg, size=min(len(pos), len(neg)), replace=False)
    idx = np.concatenate([pos, neg]) if balance else np.arange(len(y))
    X = data["rate"][:, ti[idx], bi[idx]].T
    return X, y[idx], ti[idx]


def grouped_lda_acc(X, y, groups, n_splits=N_FOLDS, seed=RNG_SEED):
    n_groups = len(np.unique(groups))
    n_splits = min(n_splits, n_groups)
    if len(np.unique(y)) < 2 or n_splits < 2:
        return np.nan
    accs = []
    gk = GroupKFold(n_splits=n_splits)
    for tr, te in gk.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        lda = LinearDiscriminantAnalysis(solver="svd").fit(sc.transform(X[tr]), y[tr])
        accs.append(balanced_accuracy_score(y[te], lda.predict(sc.transform(X[te]))))
    return float(np.mean(accs)) if accs else np.nan


def event_detection(data, anchor, which, n_perm=N_PERM_EVENT):
    ti, bi = event_index(data)
    p = pass_time(data, anchor, which)
    rel = rel_to_pass(data, anchor, which, ti, bi, p)
    rng = np.random.default_rng(RNG_SEED)
    X, y, g = build_xy(data, ti, bi, rel, rng)
    obs = grouped_lda_acc(X, y, g)
    null = np.empty(n_perm)
    rng2 = np.random.default_rng(RNG_SEED + 1)
    for i in range(n_perm):
        p_sh = p[rng2.permutation(len(p))]
        rel_sh = rel_to_pass(data, anchor, which, ti, bi, p_sh)
        Xs, ys, gs = build_xy(data, ti, bi, rel_sh, np.random.default_rng(RNG_SEED + i))
        null[i] = grouped_lda_acc(Xs, ys, gs)
    perm_p = float(np.mean(null >= obs)) if np.isfinite(obs) else np.nan
    return obs, perm_p, int((np.abs(rel) <= TOL_MS).sum())


def event_transfer(data, anchor, which):
    ti, bi = event_index(data)
    p = pass_time(data, anchor, which)
    rel = rel_to_pass(data, anchor, which, ti, bi, p)
    R, _ = data["rate"], None
    rows = []
    sets = {}
    for off in TG_OFFSETS:
        pos = np.flatnonzero(np.abs(rel - off) <= BIN_MS / 2)
        far = np.flatnonzero(np.abs(rel) > FAR_MS)
        if len(pos) < 20 or len(far) < 20:
            sets[off] = None
            continue
        rng = np.random.default_rng(RNG_SEED)
        far = rng.choice(far, size=min(len(pos), len(far)), replace=False)
        idx = np.concatenate([pos, far])
        yy = np.concatenate([np.ones(len(pos), int), np.zeros(len(far), int)])
        X = R[:, ti[idx], bi[idx]].T
        sets[off] = (X, yy)
    for a in TG_OFFSETS:
        if sets[a] is None:
            continue
        Xa, ya = sets[a]
        sc = StandardScaler().fit(Xa)
        lda = LinearDiscriminantAnalysis(solver="svd").fit(sc.transform(Xa), ya)
        for b in TG_OFFSETS:
            if sets[b] is None:
                continue
            Xb, yb = sets[b]
            acc = balanced_accuracy_score(yb, lda.predict(sc.transform(Xb)))
            rows.append({"train_ms": a, "test_ms": b, "acc": float(acc)})
    return pd.DataFrame(rows)


# =============================================================== 2B
def ridge_cv(X, y, groups, alpha=RIDGE_ALPHA, n_splits=N_FOLDS, seed=RNG_SEED):
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y, groups = X[m], y[m], groups[m]
    n_groups = len(np.unique(groups))
    n_splits = min(n_splits, n_groups)
    if len(y) < 40 or n_splits < 2 or np.std(y) < 1e-12:
        return np.nan
    corrs = []
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        model = Ridge(alpha=alpha).fit(sc.transform(X[tr]), y[tr])
        pred = model.predict(sc.transform(X[te]))
        if np.std(pred) > 1e-9 and np.std(y[te]) > 1e-9:
            corrs.append(float(np.corrcoef(y[te], pred)[0, 1]))
    return float(np.mean(corrs)) if corrs else np.nan


def trace_pass_time(data, anchor, which, n_perm=N_PERM_TRACE):
    R, valid, bc, tt = data["rate"], data["valid"], data["bc"], data["tt"]
    target = pass_time(data, anchor, which)
    blocks = tt["block_index"].to_numpy()
    rows = []
    for j in range(0, R.shape[2], TRACE_STEP):
        if not (TRACE_LO <= bc[j] <= TRACE_HI):
            continue
        m = valid[:, j] & np.isfinite(target)
        if m.sum() < 100:
            continue
        X = R[:, m, j].T
        obs = ridge_cv(X, target[m], blocks[m])
        rng = np.random.default_rng(RNG_SEED + j)
        null = np.empty(n_perm)
        for i in range(n_perm):
            null[i] = ridge_cv(X, rng.permutation(target[m]), blocks[m],
                               n_splits=N_FOLDS, seed=RNG_SEED + i)
        rows.append({"bin_ms": float(bc[j]), "n": int(m.sum()), "corr": obs,
                     "perm_p": float(np.mean(np.abs(null) >= abs(obs)))
                     if np.isfinite(obs) else np.nan})
    return pd.DataFrame(rows)


# ----- kinematics + residual target ---------------------------------
def behavior_series(run_id, times):
    path = RUNS[run_id].behavior_path
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    keys = ["ball_x", "ball_y", "camera_y", "ball_vx", "scroll_speeds"]
    acc = {k: [] for k in keys}
    t = []
    for b in data.get("blocks", []):
        if b.get("block_index", 0) < 4:
            continue
        gs = b.get("game_states") or {}
        if not gs:
            continue
        t.append(np.asarray(gs["time"], float))
        for k in keys:
            acc[k].append(np.asarray(gs.get(k, np.full(len(gs["time"]), np.nan)), float))
    if not t:
        return {k: np.full(len(times), np.nan) for k in keys}
    t = np.concatenate(t)
    o = np.argsort(t)
    t = t[o]
    out = {}
    for k in keys:
        v = np.concatenate(acc[k])[o]
        idx = np.clip(np.searchsorted(t, np.asarray(times, float)), 0, len(t) - 1)
        out[k] = v[idx]
    out["rel_y"] = out["ball_y"] - out["camera_y"]
    return out


def kinematics(data):
    lab, tt = data["lab"], data["tt"]
    holes = lab["hole_locations"].map(json.loads) if "hole_locations" in lab else \
        tt["hole_locations"].map(json.loads)
    hA = holes.str[0].to_numpy()
    hB = holes.str[1].to_numpy()
    chosen = lab["choice_hole"].to_numpy()
    chosen1 = np.where(chosen == hA, lab["greedy_cost_L"].to_numpy(),
                       np.where(chosen == hB, lab["greedy_cost_R"].to_numpy(), np.nan))
    side = (chosen >= 6).astype(float)
    move_dir = np.sign(chosen - lab["entry_hole"].to_numpy()).astype(float)
    rt = data["rt"]
    order = np.lexsort((lab["sequence_index"].to_numpy(),
                        lab["block_index"].to_numpy()))
    prev_rt = np.full(len(rt), np.nan)
    seq = lab["sequence_index"].to_numpy()
    blk = lab["block_index"].to_numpy()
    for i in order:
        prev = np.where((blk == blk[i]) & (seq == seq[i] - 1))[0]
        if len(prev):
            prev_rt[i] = rt[prev[0]]
    beh = behavior_series(data["run_id"], tt["entry_time_ms"].to_numpy(float))
    return np.column_stack([
        chosen1, side, move_dir, blk, prev_rt,
        beh["rel_y"], beh["ball_vx"], beh["scroll_speeds"],
    ])


def residualize(y, N):
    m = np.all(np.isfinite(N), axis=1) & np.isfinite(y)
    out = np.full(len(y), np.nan)
    if m.sum() < 10:
        return out
    A = np.column_stack([np.ones(m.sum()), N[m]])
    beta, *_ = np.linalg.lstsq(A, y[m], rcond=None)
    out[m] = y[m] - A @ beta
    return out


def residualize_features(X, N):
    m = np.all(np.isfinite(N), axis=1)
    out = np.full_like(X, np.nan)
    A = np.column_stack([np.ones(m.sum()), N[m]])
    for j in range(X.shape[1]):
        x = X[:, j]
        mm = m & np.isfinite(x)
        if mm.sum() < 10:
            out[mm, j] = 0.0
            continue
        beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(mm.sum()), N[mm]]), x[mm],
                                   rcond=None)
        out[mm, j] = x[mm] - np.column_stack([np.ones(mm.sum()), N[mm]]) @ beta
    return out


def window_pass_time(data, n_perm=N_PERM_WIN):
    binned, bc = data["binned"], data["bc"]
    first, last = data["first"], data["last"]
    rt, rt_exec = data["rt"], data["rt_exec"]
    tt = data["tt"]
    blocks = tt["block_index"].to_numpy()
    K = kinematics(data)
    rt_res = residualize(rt, K)
    rows = []
    for c in CENTERS:
        lo, hi = c - WIN / 2, c + WIN / 2
        elig = (first <= lo) & (last >= hi)
        if elig.sum() < 40:
            continue
        X = rate_features(binned, bc, lo, hi)
        specs = [("rt", rt, ["none", "target", "both"]),
                 ("rt_resid", rt_res, ["none", "both"])]
        for tname, y, controls in specs:
            for control in controls:
                Xc = X.copy()
                yc = y.copy()
                if control in ("target", "both"):
                    yc = residualize(yc, K)
                if control == "both":
                    Xc = residualize_features(Xc, K)
                corr = ridge_cv(Xc[elig], yc[elig], blocks[elig])
                p = np.nan
                if tname == "rt_resid" and control == "both" and n_perm:
                    rng = np.random.default_rng(RNG_SEED + int(c) + 1000)
                    null = np.empty(n_perm)
                    for i in range(n_perm):
                        null[i] = ridge_cv(Xc[elig], rng.permutation(yc[elig]),
                                           blocks[elig], seed=RNG_SEED + i)
                    p = float(np.mean(np.abs(null) >= abs(corr)))
                rows.append({"center_ms": float(c), "target": tname,
                             "control": control, "n": int(elig.sum()),
                             "corr": corr, "perm_p": p})
    return pd.DataFrame(rows)


# %%
def make_figures(ev, tg, tr, win):
    figdir = AGG_DIR / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    # 2A detection bars
    fig, axes = plt.subplots(1, len(RUN_ORDER), figsize=(4.0 * len(RUN_ORDER), 3.4),
                             sharey=True)
    for ax, run in zip(axes, RUN_ORDER):
        sub = ev[ev["run_id"] == run]
        labels = [f"{r['anchor'][0]}:{r['target']}" for _, r in sub.iterrows()]
        colors = ["tab:red" if p < 0.05 else "tab:blue" for p in sub["perm_p"]]
        ax.bar(range(len(sub)), sub["acc"], color=colors)
        ax.axhline(0.5, color="k", ls="--", lw=1)
        ax.set_ylim(0.40, 0.68)
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_title(run, fontsize=10)
    axes[0].set_ylabel("pass-event balanced acc (−50 ms)")
    fig.suptitle("2A pass-event detection (e=entry anchor, c=choice anchor)", fontsize=12)
    fig.tight_layout()
    fig.savefig(figdir / "fig_pass_event_detection.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # 2A temporal transfer (entry anchor, choice target) per run
    fig, axes = plt.subplots(1, len(RUN_ORDER), figsize=(4.0 * len(RUN_ORDER), 3.6))
    for ax, run in zip(axes, RUN_ORDER):
        sub = tg[(tg["run_id"] == run) & (tg["anchor"] == "entry") &
                 (tg["target"] == "choice")]
        if not len(sub):
            ax.set_visible(False)
            continue
        piv = sub.pivot_table(index="train_ms", columns="test_ms", values="acc")
        im = ax.imshow(piv.values, cmap="viridis", vmin=0.45, vmax=0.65, aspect="auto")
        ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
        ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
        ax.set_xlabel("test lag (ms)"); ax.set_ylabel("train lag (ms)")
        ax.set_title(run, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("2A temporal generalization (entry anchor → choice pass)", fontsize=12)
    fig.tight_layout()
    fig.savefig(figdir / "fig_pass_event_transfer.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # 2B B1 traces
    targets = [("entry", "choice"), ("choice", "exit")]
    fig, axes = plt.subplots(1, len(RUN_ORDER), figsize=(4.2 * len(RUN_ORDER), 3.4),
                             sharey=True)
    for ax, run in zip(axes, RUN_ORDER):
        for (anchor, which), col in zip(targets, ["tab:red", "tab:blue"]):
            sub = tr[(tr["run_id"] == run) & (tr["anchor"] == anchor) &
                     (tr["target"] == which)].sort_values("bin_ms")
            if len(sub):
                ax.plot(sub["bin_ms"], sub["corr"], "-o", ms=2, color=col,
                        label=f"{anchor}->{which}")
        ax.axhline(0, color="k", lw=1)
        ax.set_title(run, fontsize=10)
        ax.set_xlabel("bin (ms rel. entry/choice)")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("ridge CV corr (pass time)")
    fig.suptitle("2B-B1 per-bin pass-time decoding", fontsize=12)
    fig.tight_layout()
    fig.savefig(figdir / "fig_pass_time_regression.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # 2B B2 controls
    fig, axes = plt.subplots(1, len(RUN_ORDER), figsize=(4.2 * len(RUN_ORDER), 3.4),
                             sharey=True)
    for ax, run in zip(axes, RUN_ORDER):
        for ctrl, col in [("none", "tab:gray"), ("target", "tab:orange"),
                          ("both", "tab:red")]:
            sub = win[(win["run_id"] == run) & (win["target"] == "rt_resid") &
                      (win["control"] == ctrl)].sort_values("center_ms")
            if len(sub):
                ax.plot(sub["center_ms"], sub["corr"], "-o", ms=3, color=col, label=ctrl)
        ax.axhline(0, color="k", lw=1)
        ax.set_title(run, fontsize=10)
        ax.set_xlabel("window center (ms rel. entry)")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("ridge CV corr (rt_resid)")
    fig.suptitle("2B-B2 window rt_resid decoding, controls", fontsize=12)
    fig.tight_layout()
    fig.savefig(figdir / "fig_pass_time_controls.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# %%
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=RUN_ORDER)
    ap.add_argument("--perm-event", type=int, default=N_PERM_EVENT)
    ap.add_argument("--perm-trace", type=int, default=N_PERM_TRACE)
    ap.add_argument("--perm-win", type=int, default=N_PERM_WIN)
    ap.add_argument("--skip-transfer", action="store_true")
    ap.add_argument("--figs-only", action="store_true")
    args = ap.parse_args()

    if args.figs_only:
        ev = pd.read_csv(AGG_DIR / "pass_event_detection.csv")
        tg = pd.read_csv(AGG_DIR / "pass_event_transfer.csv")
        tr = pd.read_csv(AGG_DIR / "pass_time_regression.csv")
        win = pd.read_csv(AGG_DIR / "pass_time_window.csv")
        make_figures(ev, tg, tr, win)
        print("regenerated figures")
        return

    ev_rows, tg_rows, tr_rows, win_rows = [], [], [], []
    for run in args.runs:
        for anchor in ["entry", "choice"]:
            data = load_anchor(run, anchor)
            for which in PASSES[anchor]:
                obs, p, npos = event_detection(data, anchor, which, args.perm_event)
                print(f"[{run}/{anchor}] 2A {which:6s} acc={obs:.3f} p={p:.3f} n={npos}")
                ev_rows.append({"run_id": run, "anchor": anchor, "target": which,
                                "n_pos": npos, "acc": obs, "perm_p": p})
                if not args.skip_transfer:
                    tg = event_transfer(data, anchor, which)
                    tg.insert(0, "target", which)
                    tg.insert(0, "anchor", anchor)
                    tg.insert(0, "run_id", run)
                    tg_rows.append(tg)
            for which in TRACE_TARGETS[anchor]:
                t = trace_pass_time(data, anchor, which, args.perm_trace)
                t.insert(0, "target", which)
                t.insert(0, "anchor", anchor)
                t.insert(0, "run_id", run)
                tr_rows.append(t)
                j = t["corr"].abs().idxmax() if len(t) else None
                if j is not None:
                    print(f"[{run}/{anchor}] 2B-B1 {which:6s} peak "
                          f"{t.loc[j,'corr']:+.3f} @ {t.loc[j,'bin_ms']:+.0f} ms "
                          f"(p={t.loc[j,'perm_p']:.3f})")
        w = window_pass_time(data, args.perm_win)
        w.insert(0, "run_id", run)
        w.insert(1, "anchor", "entry")
        win_rows.append(w)
        print(f"[{run}] 2B-B2 windows done")

    ev = pd.DataFrame(ev_rows)
    tg = pd.concat(tg_rows, ignore_index=True) if tg_rows else pd.DataFrame()
    tr = pd.concat(tr_rows, ignore_index=True) if tr_rows else pd.DataFrame()
    win = pd.concat(win_rows, ignore_index=True)

    AGG_DIR.mkdir(parents=True, exist_ok=True)
    ev.to_csv(AGG_DIR / "pass_event_detection.csv", index=False)
    if len(tg):
        tg.to_csv(AGG_DIR / "pass_event_transfer.csv", index=False)
    tr.to_csv(AGG_DIR / "pass_time_regression.csv", index=False)
    win.to_csv(AGG_DIR / "pass_time_window.csv", index=False)

    make_figures(ev, tg, tr, win)
    print(f"\nwrote Part 2 CSVs + figures to {AGG_DIR}")


if __name__ == "__main__":
    main()
