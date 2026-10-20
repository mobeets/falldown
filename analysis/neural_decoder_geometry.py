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
# # Decoder geometry: shared axes, step decomposition, timing/motion controls
#
# 1. Pairwise decoder-axis cosines (with a bootstrap null) and cross-decoding
#    among {conflict_mag, chosen_cost, step1, step2, rt_exec, rt_dec, exit_dir}.
# 2. Unique variance: does each target decode after partialling the other?
# 3. Step decomposition: decode chosen 1-step distance, remaining (2nd step) and
#    the full planning cost; then the same after partialling out overall timing
#    (rt_exec) -- does the planning-cost decoder reduce to timing?
# 4. Motion robustness: decode planning cost on monotone vs interrupted trials
#    (post-choice velocity sign changes / stops / tortuosity).
#
# Outputs (aggregate/): decoder_geometry.csv, step_decomposition.csv,
#   motion_robustness.csv, figures/fig_decoder_geometry.png,
#   fig_step_decomposition.png, fig_motion_robustness.png

# %%
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from scipy.stats import pearsonr

from neural_common import RUNS, RUN_ORDER, AGG_DIR, out_dir
from neural_lda_decoding import rate_features, pca_features

# %%
RIDGE_ALPHA = 1.0
N_BOOT = 200
POST = (0.0, 1000.0)
_T = {}


def tracks(run):
    if run in _T:
        return _T[run]
    d = json.load(open(RUNS[run].behavior_path, encoding="utf-8"))
    t, vx, x = [], [], []
    for b in d.get("blocks", []):
        if b.get("block_index", 0) < 4:
            continue
        gs = b.get("game_states") or {}
        if not gs:
            continue
        t.append(np.asarray(gs["time"], float))
        vx.append(np.asarray(gs["ball_vx"], float))
        x.append(np.asarray(gs["ball_x"], float))
    t = np.concatenate(t); vx = np.concatenate(vx); x = np.concatenate(x)
    o = np.argsort(t)
    gi = d.get("gameInfo") or {}
    _T[run] = (t[o], vx[o], x[o], gi.get("levelStartX", 0.0),
               gi.get("levelWidth", 1.0) / 12.0)
    return _T[run]


def load(run):
    o = out_dir(run)
    z = np.load(o / "segmented_spikes_binned.npz", allow_pickle=True)
    tids = z["trial_ids"]
    lab = pd.read_csv(o / "trial_labels.csv").set_index("trial_id").loc[tids].reset_index()
    tt = pd.read_csv(o / "trial_table.csv").set_index("trial_id").loc[tids].reset_index()
    binned, bc = z["binned"], z["bin_centers"]
    return dict(run=run, o=o, binned=binned, bc=bc, lab=lab, tt=tt)


def targets(d):
    lab, tt = d["lab"], d["tt"]
    chosen = lab["choice_hole"].to_numpy()
    hA = tt["hole_locations"].map(json.loads).str[0].to_numpy()
    ch1 = np.where(chosen == hA, lab["greedy_cost_L"], lab["greedy_cost_R"]).astype(float)
    ch2 = np.where(chosen == hA, lab["planning_cost_L"], lab["planning_cost_R"]).astype(float)
    g1 = (lab["greedy_cost_L"] - lab["greedy_cost_R"]).to_numpy(float)
    g2 = (lab["planning_cost_L"] - lab["planning_cost_R"]).to_numpy(float)
    rem = np.abs(chosen - lab["goal_hole"].to_numpy()).astype(float)
    rt_exec = (tt["exit_time_ms"] - tt["choice_time_ms"]).to_numpy(float)
    rt_dec = (tt["choice_time_ms"] - tt["entry_time_ms"]).to_numpy(float)
    t, vx, x, lsx, gap = tracks(d["run"])
    ci = np.clip(np.searchsorted(t, tt["choice_time_ms"].to_numpy(float)), 0, len(t) - 1)
    goal_x = lsx + (lab["goal_hole"].to_numpy() + 0.5) * gap
    exit_dir = (goal_x > x[ci]).astype(float)
    return pd.DataFrame({
        "conflict_mag": np.abs(g2 - g1), "chosen_cost": ch2,
        "step1": ch1, "step2": rem, "rt_exec": rt_exec, "rt_dec": rt_dec,
        "exit_dir": exit_dir,
    })


def motion_features(d):
    tt = d["tt"]
    t, vx, x, _, _ = tracks(d["run"])
    ns, sf, tort = [], [], []
    for r in tt.itertuples():
        i = np.searchsorted(t, r.choice_time_ms)
        j = np.searchsorted(t, r.exit_time_ms)
        z = vx[i:j]; xz = x[i:j]
        nz = z[np.abs(z) > 1e-9]
        s = np.sign(nz)
        ns.append(int(np.sum(np.diff(s) != 0)) if len(s) > 1 else 0)
        sf.append(float(np.mean(np.abs(z) < 1e-9)) if len(z) else 0.0)
        if len(xz) > 1:
            path = float(np.sum(np.abs(np.diff(xz))))
            straight = max(abs(xz[-1] - xz[0]), 1e-6)
            tort.append(min(path / straight, 10.0))
        else:
            tort.append(1.0)
    return pd.DataFrame({"n_sign": ns, "stop_frac": sf, "tort": tort})


# %%
def cv_corr(X, y, groups, seed=0):
    corrs = []
    for tr, te in GroupKFold(n_splits=min(5, len(np.unique(groups)))).split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        p = Ridge(RIDGE_ALPHA).fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[te]))
        if np.std(p) > 1e-9 and np.std(y[te]) > 1e-9:
            corrs.append(pearsonr(y[te], p)[0])
    return float(np.mean(corrs)) if corrs else np.nan


def decoder_w(X, y):
    Xz = StandardScaler().fit(X).transform(X)
    return Ridge(RIDGE_ALPHA).fit(Xz, (y - y.mean()) / (y.std() + 1e-9)).coef_


def residue(y, N):
    A = np.column_stack([np.ones(len(y))] + N)
    b = np.linalg.lstsq(A, y, rcond=None)[0]
    return y - A @ b


def geo_analysis(d, rep="rate"):
    T = targets(d)
    M = motion_features(d)
    Xf = rate_features(d["binned"], d["bc"], *POST) if rep == "rate" \
        else pca_features(d["binned"], d["bc"], *POST)
    # choose continuous targets (drop exit_dir for cosine; keep for cross)
    cols = ["conflict_mag", "chosen_cost", "step1", "step2", "rt_exec", "rt_dec"]
    W = {c: decoder_w(Xf, T[c].to_numpy(float)) for c in cols}
    cos = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for a in cols:
        for b in cols:
            cos.loc[a, b] = float(W[a] @ W[b] / (np.linalg.norm(W[a]) * np.linalg.norm(W[b]) + 1e-12))
    # cross-decode: train a, correlate cv prediction with b
    cross = pd.DataFrame(index=cols, columns=cols, dtype=float)
    groups = d["tt"]["block_index"].to_numpy()
    for a in cols:
        pred = np.full(len(T), np.nan)
        for tr, te in GroupKFold(n_splits=5).split(Xf, T[a].to_numpy(), groups):
            sc = StandardScaler().fit(Xf[tr])
            pred[te] = Ridge(RIDGE_ALPHA).fit(sc.transform(Xf[tr]), T[a].to_numpy(float)[tr]).predict(sc.transform(Xf[te]))
        for b in cols:
            cross.loc[a, b] = pearsonr(T[b].to_numpy(float), pred)[0]
    # unique variance
    cm = T["conflict_mag"].to_numpy(float); cp = T["chosen_cost"].to_numpy(float)
    uniq = {"cv_cm": cv_corr(Xf, cm, groups),
            "cv_cpc": cv_corr(Xf, cp, groups),
            "cv_cm_given_cpc": cv_corr(Xf, residue(cm, [cp]), groups),
            "cv_cpc_given_cm": cv_corr(Xf, residue(cp, [cm]), groups)}
    return T, M, Xf, cos, cross, uniq


# %%
def main():
    geo_rows, step_rows, motion_rows = [], [], []
    for run in RUN_ORDER:
        print(f"[{run}] decoder geometry")
        d = load(run)
        T, M, Xf, cos, cross, uniq = geo_analysis(d, "rate")
        groups = d["tt"]["block_index"].to_numpy()
        for a in cos.index:
            for b in cos.columns:
                geo_rows.append({"run_id": run, "rep": "rate", "a": a, "b": b,
                                 "cos": cos.loc[a, b], "cross": cross.loc[a, b]})
        geo_rows.append({"run_id": run, "rep": "rate", "a": "UNIQ", "b": "cm_given_cpc",
                         "cos": uniq["cv_cm_given_cpc"], "cross": uniq["cv_cm"]})
        geo_rows.append({"run_id": run, "rep": "rate", "a": "UNIQ", "b": "cpc_given_cm",
                         "cos": uniq["cv_cpc_given_cm"], "cross": uniq["cv_cpc"]})

        # step decomposition + timing control
        rte = T["rt_exec"].to_numpy(float); rtd = T["rt_dec"].to_numpy(float)
        step_rows.append({"run_id": run, "target": "step1", "control": "none",
                          "corr": cv_corr(Xf, T["step1"].to_numpy(float), groups)})
        step_rows.append({"run_id": run, "target": "step2", "control": "none",
                          "corr": cv_corr(Xf, T["step2"].to_numpy(float), groups)})
        step_rows.append({"run_id": run, "target": "chosen_cost", "control": "none",
                          "corr": cv_corr(Xf, T["chosen_cost"].to_numpy(float), groups)})
        step_rows.append({"run_id": run, "target": "rt_exec", "control": "none",
                          "corr": cv_corr(Xf, rte, groups)})
        step_rows.append({"run_id": run, "target": "chosen_cost", "control": "-rt_exec",
                          "corr": cv_corr(Xf, residue(T["chosen_cost"].to_numpy(float), [rte]), groups)})
        step_rows.append({"run_id": run, "target": "step1", "control": "-rt_exec",
                          "corr": cv_corr(Xf, residue(T["step1"].to_numpy(float), [rte]), groups)})
        step_rows.append({"run_id": run, "target": "step2", "control": "-rt_exec",
                          "corr": cv_corr(Xf, residue(T["step2"].to_numpy(float), [rte]), groups)})

        # motion interruption robustness
        cpc = T["chosen_cost"].to_numpy(float)
        inter = (M["n_sign"].to_numpy() >= 1) | (M["stop_frac"].to_numpy() > 0.1)
        motion_rows.append({"run_id": run, "subset": "all", "n": int(len(cpc)),
                            "corr": cv_corr(Xf, cpc, groups)})
        motion_rows.append({"run_id": run, "subset": "smooth", "n": int((~inter).sum()),
                            "corr": cv_corr(Xf[~inter], cpc[~inter], groups[~inter])})
        motion_rows.append({"run_id": run, "subset": "interrupted", "n": int(inter.sum()),
                            "corr": cv_corr(Xf[inter], cpc[inter], groups[inter])})
        Khat = np.column_stack([rte, M["n_sign"], M["stop_frac"], M["tort"]])
        motion_rows.append({"run_id": run, "subset": "cpc|-timing-motion",
                            "n": int(len(cpc)),
                            "corr": cv_corr(Xf, residue(cpc, [rte, M["n_sign"], M["stop_frac"], M["tort"]]), groups)})
        motion_rows.append({"run_id": run, "subset": "cpc|-rt_exec",
                            "n": int(len(cpc)),
                            "corr": cv_corr(Xf, residue(cpc, [rte]), groups)})

    geo = pd.DataFrame(geo_rows)
    step = pd.DataFrame(step_rows)
    motion = pd.DataFrame(motion_rows)
    geo.to_csv(AGG_DIR / "decoder_geometry.csv", index=False)
    step.to_csv(AGG_DIR / "step_decomposition.csv", index=False)
    motion.to_csv(AGG_DIR / "motion_robustness.csv", index=False)

    # figures
    fig, axes = plt.subplots(1, len(RUN_ORDER), figsize=(4.2 * len(RUN_ORDER), 3.8))
    for ax, run in zip(axes, RUN_ORDER):
        sub = geo[(geo.run_id == run) & (geo.a != "UNIQ")]
        piv = sub.pivot_table(index="a", columns="b", values="cos")
        im = ax.imshow(piv.values, cmap="bwr", vmin=-1, vmax=1)
        ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, rotation=90, fontsize=6)
        ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=6)
        ax.set_title(run, fontsize=10)
    fig.colorbar(im, ax=axes, fraction=0.02)
    fig.suptitle("Decoder-axis cosines (rate, post window)", fontsize=12)
    fig.savefig(AGG_DIR / "figures" / "fig_decoder_geometry.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(RUN_ORDER), figsize=(4.4 * len(RUN_ORDER), 3.6), sharey=True)
    tg = ["step1", "step2", "chosen_cost", "rt_exec"]
    for ax, run in zip(axes, RUN_ORDER):
        s = step[(step.run_id == run) & (step.control == "none")]
        ax.bar(range(len(tg)), [s[s.target == t]["corr"].mean() for t in tg],
               color="tab:blue", edgecolor="k")
        s2 = step[(step.run_id == run) & (step.control == "-rt_exec")]
        ax.scatter(range(len(tg)), [s2[s2.target == t]["corr"].mean() for t in tg],
                   color="tab:red", zorder=5, label="-rt_exec")
        ax.set_xticks(range(len(tg))); ax.set_xticklabels(tg, rotation=45, ha="right", fontsize=7)
        ax.axhline(0, color="k", lw=1); ax.set_title(run, fontsize=10)
    axes[0].set_ylabel("ridge CV corr")
    axes[0].legend(fontsize=7)
    fig.suptitle("Step decomposition (bars=none, dots=partialling rt_exec)", fontsize=12)
    fig.savefig(AGG_DIR / "figures" / "fig_step_decomposition.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(RUN_ORDER), figsize=(4.4 * len(RUN_ORDER), 3.6), sharey=True)
    subs = ["all", "smooth", "interrupted", "cpc|-rt_exec", "cpc|-timing-motion"]
    for ax, run in zip(axes, RUN_ORDER):
        s = motion[motion.run_id == run].set_index("subset")
        ax.bar(range(len(subs)), [s.loc[x, "corr"] if x in s.index else np.nan for x in subs],
               color="tab:green", edgecolor="k")
        for i, x in enumerate(subs):
            if x in s.index:
                ax.text(i, s.loc[x, "corr"], f"n={int(s.loc[x,'n'])}", ha="center", va="bottom", fontsize=6)
        ax.set_xticks(range(len(subs))); ax.set_xticklabels(subs, rotation=45, ha="right", fontsize=6)
        ax.axhline(0, color="k", lw=1); ax.set_title(run, fontsize=10)
    axes[0].set_ylabel("chosen_cost CV corr")
    fig.suptitle("Planning-cost decoding: motion interruption & timing control", fontsize=12)
    fig.savefig(AGG_DIR / "figures" / "fig_motion_robustness.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    print("\nstep decomposition (mean over runs):")
    print(step.groupby(["target", "control"])["corr"].mean().round(3).to_string())
    print("\nmotion robustness (mean over runs):")
    print(motion.groupby("subset")["corr"].mean().round(3).to_string())
    print(f"\nwrote decoder_geometry.csv, step_decomposition.csv, motion_robustness.csv + figures")


if __name__ == "__main__":
    main()
