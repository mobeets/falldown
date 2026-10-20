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
# # Part 9 - Change of mind (CoM)
#
# A CoM is a `user_input` direction reversal; the **second** reversal within a
# layer is the change-of-mind event (reverse, then reverse back); the first
# reversal is logged as a higher-power contrast. Layers: approach
# (entry->choice) and traverse (choice->exit), run separately.
#
# Analyses: (1) event-locked population signal (+/-100 ms) with trial-grouped
# CV and permutation; (2) decode CoM presence (>=2 reversals) with coverage and
# kinematic controls; (3) per-run effects combined by a random-effects
# (DerSimonian-Laird) meta-analysis with I^2.
#
# Outputs (aggregate/): com_event.csv, com_presence.csv, com_meta.csv,
#   figures/fig_com_event.png, fig_com_presence.png

# %%
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score

from neural_common import RUNS, RUN_ORDER, AGG_DIR
from neural_lda_decoding import rate_features, pca_features, RNG_SEED
from neural_pass_timing import ridge_cv
from neural_decoder_geometry import load

# %%
TOL_MS = 100.0
FAR_MS = 400.0
N_PERM = 100
_UI = {}


def user_inputs(run):
    if run in _UI:
        return _UI[run]
    d = json.load(open(RUNS[run].behavior_path, encoding="utf-8"))
    T, I = [], []
    for b in d.get("blocks", []):
        if b.get("block_index", 0) < 4:
            continue
        ui = b.get("user_inputs") or {}
        if not ui:
            continue
        T.append(np.asarray(ui["time"], float))
        I.append(np.asarray(ui["input"], float))
    T = np.concatenate(T)
    I = np.concatenate(I)
    o = np.argsort(T)
    _UI[run] = (T[o], I[o])
    return _UI[run]


def reversals(run, a, b):
    """Times of direction reversals in (a, b) from user_input."""
    T, I = user_inputs(run)
    m = (T >= a) & (T < b)
    t = T[m]
    inp = I[m]
    nz = inp != 0
    t = t[nz]
    inp = inp[nz]
    if len(inp) < 2:
        return []
    flip = np.flatnonzero(np.diff(np.sign(inp)) != 0)
    return list(t[flip + 1])   # time of the reversal (new direction)


def layer_events(d):
    """Per-trial first/second reversal time for approach and traverse."""
    tt = d["tt"]
    out = {"approach": {"first": np.full(len(tt), np.nan),
                        "second": np.full(len(tt), np.nan)},
           "traverse": {"first": np.full(len(tt), np.nan),
                        "second": np.full(len(tt), np.nan)}}
    for i, r in enumerate(tt.itertuples()):
        for layer, (a, b) in [("approach", (r.entry_time_ms, r.choice_time_ms)),
                              ("traverse", (r.choice_time_ms, r.exit_time_ms))]:
            rv = reversals(d["run"], a, b)
            if len(rv) >= 1:
                out[layer]["first"][i] = rv[0]
            if len(rv) >= 2:
                out[layer]["second"][i] = rv[1]
    return out


def encode_event(d, layer, which):
    """(samples x units, labels, groups) for event-locked detection."""
    ev = layer_events(d)[layer][which]
    choice = d["tt"]["choice_time_ms"].to_numpy(float)
    ev_rel = ev - choice                      # choice-anchored event time
    valid = d["valid"] if "valid" in d else ~np.isnan(d["binned"]).all(axis=0)
    ti, bi = np.where(valid)
    have = ~np.isnan(ev_rel[ti])
    ti, bi = ti[have], bi[have]
    rel = d["bc"][bi] - ev_rel[ti]
    y = (np.abs(rel) <= TOL_MS).astype(int)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(np.abs(rel) > FAR_MS)
    if len(pos) < 20 or len(neg) < 20:
        return None
    rng = np.random.default_rng(RNG_SEED)
    neg = rng.choice(neg, size=min(len(pos), len(neg)), replace=False)
    idx = np.concatenate([pos, neg])
    X = d["binned"][:, ti[idx], bi[idx]].T
    return X, y[idx], ti[idx]


def grouped_acc(X, y, g, seed=RNG_SEED):
    ng = min(5, len(np.unique(g)))
    if len(np.unique(y)) < 2 or ng < 2:
        return np.nan
    a = []
    for tr, te in GroupKFold(n_splits=ng).split(X, y, g):
        sc = StandardScaler().fit(X[tr])
        m = LDA(solver="svd").fit(sc.transform(X[tr]), y[tr])
        a.append(balanced_accuracy_score(y[te], m.predict(sc.transform(X[te]))))
    return float(np.mean(a)) if a else np.nan


def event_test(d, layer, which, n_perm=N_PERM):
    enc = encode_event(d, layer, which)
    if enc is None:
        return np.nan, np.nan, 0
    X, y, g = enc
    obs = grouped_acc(X, y, g)
    rng = np.random.default_rng(RNG_SEED + 2)
    null = np.array([grouped_acc(X, rng.permutation(y), g, seed=RNG_SEED + i)
                     for i in range(n_perm)])
    p = float(np.mean(null >= obs)) if np.isfinite(obs) else np.nan
    return obs, p, int(len(np.unique(g)))


def presence(d, layer):
    """Decode CoM presence (>=2 reversals) from layer activity."""
    which = "second"
    ev = layer_events(d)[layer][which]
    y = (~np.isnan(ev)).astype(float)
    if y.sum() < 10:
        return None
    lo, hi = {"approach": (-1000.0, 0.0), "traverse": (0.0, 1000.0)}[layer]
    X = rate_features(d["binned"], d["bc"], lo, hi)
    Xp = pca_features(d["binned"], d["bc"], lo, hi)
    valid = ~np.isnan(d["binned"]).all(axis=0)
    mask = (d["bc"] >= lo) & (d["bc"] <= hi)
    cov = valid[:, mask].sum(1).astype(float)
    g = d["tt"]["block_index"].to_numpy()
    A = np.column_stack([np.ones(len(cov)), cov])
    beta = np.linalg.lstsq(A, Xp, rcond=None)[0]
    Xp_c = Xp - A @ beta
    y_c = y - np.column_stack([np.ones(len(y)), cov]) @ np.linalg.lstsq(
        np.column_stack([np.ones(len(y)), cov]), y, rcond=None)[0]
    return {
        "prev": float(y.mean()),
        "raw_rate": ridge_cv(X, y, g),
        "raw_pca": ridge_cv(Xp, y, g),
        "cov_pca": ridge_cv(Xp_c, y_c, g),
        "cov_target_r": float(np.corrcoef(cov, y)[0, 1]),
    }


# %% ------------------------------------------------------- DL meta
def dl_meta(effects):
    e = np.asarray([x for x in effects if np.isfinite(x)], float)
    k = len(e)
    if k < 2:
        return dict(mean=float(np.mean(e)) if k else np.nan, se=np.nan,
                    i2=np.nan, k=k)
    v = np.ones(k)            # unit within-run variance proxy (effects are corr)
    w = 1.0 / v
    mean_fe = np.sum(w * e) / np.sum(w)
    Q = np.sum(w * (e - mean_fe) ** 2)
    C = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (Q - (k - 1)) / C) if C > 0 else 0.0
    i2 = max(0.0, (Q - (k - 1)) / Q) * 100 if Q > 0 else 0.0
    wr = 1.0 / (v + tau2)
    mean_re = np.sum(wr * e) / np.sum(wr)
    se = float(np.sqrt(1.0 / np.sum(wr)))
    return dict(mean=float(mean_re), se=se, i2=float(i2), k=k)


# %%
def main():
    ev_rows, pres_rows, meta_rows = [], [], []
    for run in RUN_ORDER:
        print(f"[{run}] CoM")
        d = load(run)
        for layer in ["approach", "traverse"]:
            for which in ["first", "second"]:
                acc, p, n = event_test(d, layer, which)
                ev_rows.append({"run_id": run, "layer": layer, "event": which,
                                "acc": acc, "perm_p": p, "n_trials": n})
            pr = presence(d, layer)
            if pr:
                pres_rows.append({"run_id": run, "layer": layer, **pr})

    ev = pd.DataFrame(ev_rows)
    pres = pd.DataFrame(pres_rows)
    ev.to_csv(AGG_DIR / "com_event.csv", index=False)
    pres.to_csv(AGG_DIR / "com_presence.csv", index=False)

    for layer in ["approach", "traverse"]:
        for metric in ["raw_rate", "raw_pca", "cov_pca"]:
            m = dl_meta(pres[pres.layer == layer][metric].tolist())
            meta_rows.append({"layer": layer, "metric": metric, **m})
        for which in ["first", "second"]:
            s = ev[(ev.layer == layer) & (ev.event == which)]["acc"] - 0.5
            m = dl_meta(s.tolist())
            meta_rows.append({"layer": layer, "metric": f"event_{which}_acc-0.5", **m})
    meta = pd.DataFrame(meta_rows)
    meta.to_csv(AGG_DIR / "com_meta.csv", index=False)
    print("\nrandom-effects meta (mean, I^2):")
    print(meta.round(3).to_string(index=False))

    # figures
    fig, axes = plt.subplots(1, len(RUN_ORDER), figsize=(4.2 * len(RUN_ORDER), 3.6), sharey=True)
    for ax, run in zip(axes, RUN_ORDER):
        s = ev[ev.run_id == run]
        labels = [f"{l[:3]}:{e}" for l, e in zip(s.layer, s.event)]
        ax.bar(range(len(s)), s.acc - 0.5, color="tab:purple", edgecolor="k")
        ax.axhline(0, color="k", lw=1)
        ax.set_xticks(range(len(s))); ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_title(run, fontsize=10)
    axes[0].set_ylabel("event detection bal.acc - 0.5")
    fig.suptitle("Part 9: CoM event-locked population signal", fontsize=12)
    fig.tight_layout()
    fig.savefig(AGG_DIR / "figures" / "fig_com_event.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(RUN_ORDER), figsize=(4.2 * len(RUN_ORDER), 3.6), sharey=True)
    metrics = ["raw_rate", "raw_pca", "cov_pca", "cov_target_r"]
    for ax, run in zip(axes, RUN_ORDER):
        s = pres[pres.run_id == run]
        for layer, col in [("approach", "tab:red"), ("traverse", "tab:blue")]:
            g = s[s.layer == layer]
            ax.bar(np.arange(len(metrics)) + (-0.2 if layer == "approach" else 0.2),
                   [g[m].mean() for m in metrics], 0.4, color=col, edgecolor="k",
                   label=layer)
        ax.axhline(0, color="k", lw=1)
        ax.set_xticks(range(len(metrics))); ax.set_xticklabels(metrics, rotation=45, ha="right", fontsize=7)
        ax.set_title(run, fontsize=10); ax.legend(fontsize=7)
    axes[0].set_ylabel("CoM-presence decode (corr)")
    fig.suptitle("Part 9: CoM presence decoding + coverage control", fontsize=12)
    fig.tight_layout()
    fig.savefig(AGG_DIR / "figures" / "fig_com_presence.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote com_event.csv, com_presence.csv, com_meta.csv + figures")


if __name__ == "__main__":
    main()
