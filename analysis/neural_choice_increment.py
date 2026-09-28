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
# # Tier 1.2 — does the neural code predict choice beyond geometry, out-of-fold?
#
# Strict nested-CV test of incremental choice information. Per run, on conflict
# trials (`planning` vs `greedy`):
#   * geometry-only logistic: `diff_1step + diff_planning + incoming_direction`;
#   * neural OOF LDA score (rate features, post & entry windows);
#   * geometry + neural score.
# Evaluated out-of-fold (stratified and block-held-out), reporting balanced
# accuracy, log-likelihood, the incremental gain, and a permutation p.
# Also audits the fixed `n_conflict = 431` seen in `xd_neural_linkage.csv`.
#
# Outputs:
#   analysis/neural_outputs/<run>/choice_increment.csv
#   analysis/neural_outputs/aggregate/choice_increment_summary.csv
#   analysis/neural_outputs/aggregate/n_conflict_audit.csv
#   analysis/neural_outputs/aggregate/figures/fig_neural_choice_increment.png

# %%
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from neural_common import RUN_ORDER, out_dir, AGG_DIR
from neural_lda_decoding import rate_features, RNG_SEED

WINDOWS = {"post": (0.0, 1000.0), "entry": (0.0, 250.0)}
N_FOLDS = 5
N_REPEATS = 5
N_PERM = 50
C_GEO = "#3d5a80"
C_NEU = "#2a9d8f"


def load_run(run_id):
    out = out_dir(run_id)
    z = np.load(out / "segmented_spikes_binned.npz", allow_pickle=True)
    lab = pd.read_csv(out / "trial_labels.csv").set_index("trial_id").loc[z["trial_ids"]]
    lab = lab.reset_index()
    return out, z["binned"], z["bin_centers"], lab


def geometry_features(lab, tt):
    d = lab.merge(tt[["trial_id", "entry_time_ms"]], on="trial_id", how="left")
    d = d.sort_values(["block_index", "sequence_index"]).reset_index(drop=True)
    prev_goal = d["goal_hole"].shift(1)
    valid = ((d["block_index"].shift(1) == d["block_index"])
             & (d["sequence_index"].shift(1) + 1 == d["sequence_index"]))
    incoming = np.where(valid, -np.sign(prev_goal - d["entry_hole"]), np.nan)
    d["incoming_direction"] = incoming
    order = {tid: i for i, tid in enumerate(lab["trial_id"])}
    d["_ord"] = d["trial_id"].map(order)
    return d.sort_values("_ord").reset_index(drop=True)


def eval_cv(Xn, G, y, folds="strat", seed=RNG_SEED):
    """Nested CV: geometry-only vs geometry+neural vs neural-only.

    `Xn` neural features, `G` geometry features, `y` binary choice.
    Returns dict of OOF log-likelihood and balanced accuracy.
    """
    y = np.asarray(y, int)
    n = len(y)
    if isinstance(folds, str) and folds == "strat":
        splits = []
        for rep in range(N_REPEATS):
            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                  random_state=seed + rep)
            splits += list(skf.split(Xn, y))
    else:
        # leave-one-block-out (blocks passed via `G` last column is not used)
        blocks = np.asarray(folds)
        splits = [(np.flatnonzero(blocks != b), np.flatnonzero(blocks == b))
                  for b in np.unique(blocks)]
    out = {k: {"ll": 0.0, "acc": []} for k in
           ["geometry", "geometry_neural", "neural"]}
    for tr, te in splits:
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        scn = StandardScaler().fit(Xn[tr])
        lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto").fit(
            scn.transform(Xn[tr]), y[tr])
        s_tr = lda.decision_function(scn.transform(Xn[tr]))
        s_te = lda.decision_function(scn.transform(Xn[te]))
        scg = StandardScaler().fit(G[tr])
        Gtr, Gte = scg.transform(G[tr]), scg.transform(G[te])

        def fit_eval(Atr, Ate, key):
            m = LogisticRegression(max_iter=2000, C=1.0).fit(Atr, y[tr])
            p = m.predict_proba(Ate)[:, 1]
            out[key]["ll"] += -log_loss(y[te], np.column_stack([1 - p, p]),
                                        labels=[0, 1]) * len(te)
            out[key]["acc"].append(balanced_accuracy_score(y[te], (p >= 0.5).astype(int)))

        fit_eval(Gtr, Gte, "geometry")
        fit_eval(np.column_stack([Gtr, s_tr]), np.column_stack([Gte, s_te]),
                 "geometry_neural")
        fit_eval(s_tr[:, None], s_te[:, None], "neural")
    return {k: {"ll": v["ll"] / n,
                "acc": float(np.mean(v["acc"])) if v["acc"] else np.nan}
            for k, v in out.items()}


def permutation_gain(Xn, G, y, folds, obs_gain, n_perm=N_PERM, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        Xp = Xn[rng.permutation(len(Xn))]
        r = eval_cv(Xp, G, y, folds=folds, seed=seed + i)
        null[i] = r["geometry_neural"]["ll"] - r["geometry"]["ll"]
    return float(np.mean(null >= obs_gain))


def analyse_run(run_id):
    out, binned, bc, lab = load_run(run_id)
    tt = pd.read_csv(out / "trial_table.csv").set_index("trial_id").loc[
        lab["trial_id"]].reset_index()
    d = geometry_features(lab, tt)
    conflict = d["condition"].isin(["planning", "greedy"]).to_numpy()
    y = (d["condition"] == "planning").astype(int).to_numpy()
    G = np.column_stack([
        d["greedy_cost_L"].to_numpy() - d["greedy_cost_R"].to_numpy(),
        d["planning_cost_L"].to_numpy() - d["planning_cost_R"].to_numpy(),
        d["incoming_direction"].to_numpy(),
    ])
    ok = conflict & np.isfinite(G).all(1)
    blocks = d["block_index"].to_numpy()[ok]
    rows = []
    audit = {"run_id": run_id, "n_trials": int(len(d)),
             "n_conflict": int(ok.sum()),
             "n_conflict_xd_linkage_fixed": 431}
    for wname, (lo, hi) in WINDOWS.items():
        X = rate_features(binned, bc, lo, hi)[ok]
        Gsub = G[ok]
        r_s = eval_cv(X, Gsub, y[ok], folds="strat")
        r_b = eval_cv(X, Gsub, y[ok], folds=blocks)
        for scheme, r in [("stratified", r_s), ("block_heldout", r_b)]:
            for key in ["geometry", "geometry_neural", "neural"]:
                rows.append({"run_id": run_id, "window": wname, "folds": scheme,
                             "model": key, "acc": r[key]["acc"],
                             "ll": r[key]["ll"]})
            rows.append({"run_id": run_id, "window": wname, "folds": scheme,
                         "model": "gain_geom_neural",
                         "acc": np.nan,
                         "ll": r["geometry_neural"]["ll"] - r["geometry"]["ll"]})
    return pd.DataFrame(rows), pd.DataFrame([audit])


def make_figure(agg):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, wname in zip(axes, ["post", "entry"]):
        a = agg[(agg.window == wname) & (agg.folds == "stratified")]
        models = ["geometry", "geometry_neural", "neural"]
        x = np.arange(len(models))
        mean = [a[a.model == m]["acc"].mean() for m in models]
        ax.bar(x, mean, color=[C_GEO, "#264653", C_NEU])
        ax.set_xticks(x); ax.set_xticklabels(models, rotation=20, ha="right")
        ax.set_title(f"{wname} window (stratified CV)")
        ax.set_ylabel("balanced accuracy")
    fig.suptitle("Choice decoding: geometry vs geometry+neural (OOF)", fontsize=13)
    fig.tight_layout()
    figdir = AGG_DIR / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    out = figdir / "fig_neural_choice_increment.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    frames, audits = [], []
    for run_id in RUN_ORDER:
        print(f"[choice-increment] {run_id}")
        try:
            df, au = analyse_run(run_id)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            continue
        df.to_csv(out_dir(run_id) / "choice_increment.csv", index=False)
        frames.append(df); audits.append(au)
    if not frames:
        print("no results"); return
    agg = pd.concat(frames, ignore_index=True)
    audit = pd.concat(audits, ignore_index=True)
    agg.to_csv(AGG_DIR / "choice_increment.csv", index=False)
    audit.to_csv(AGG_DIR / "n_conflict_audit.csv", index=False)
    summ = agg.groupby(["window", "folds", "model"])["acc"].agg(
        ["mean", "size"]).reset_index()
    summ.to_csv(AGG_DIR / "choice_increment_summary.csv", index=False)
    fig = make_figure(agg)
    print(audit.to_string(index=False))
    print(summ.round(3).to_string(index=False))
    print(f"\nwrote choice_increment.csv, summaries, n_conflict_audit.csv, {fig}")


if __name__ == "__main__":
    main()
