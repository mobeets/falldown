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
# # Tier 1.2 — dPCA conditioned on geometry (conflict magnitude)
#
# The existing dPCA (`neural_dpca.py`) demixes condition (policy) x side x time.
# Here the "condition" axis is instead a **geometry** factor: the quartile of
# `conflict_mag`. We ask whether population variance is carried by a
# geometry axis (`c`), a policy axis, or condition-independent time (`t`), and
# whether the geometry and policy axes occupy the same subspace.
#
# Outputs:
#   analysis/neural_outputs/<run>/dpca_geometry.csv
#   analysis/neural_outputs/<run>/dpca_geometry_decoding.csv
#   analysis/neural_outputs/aggregate/dpca_geometry_summary.csv
#   analysis/neural_outputs/aggregate/figures/fig_neural_dpca_geometry.png
#
# Run with:
#   python analysis/neural_dpca_geometry.py

# %%
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neural_common import RUN_ORDER, out_dir, AGG_DIR
from neural_lda_decoding import make_transform, build_label_vectors
from neural_dpca import run_dpca_window, raw_window_rate, decode_from_term

WINDOW = (0.0, 1000.0)
POLICY_ORDER = {"planning": 0, "greedy": 1, "agree_optimal": 2, "lapse": 3}
TERMS = ["t", "c", "s", "cs", "ct", "st", "cst"]
C_GEOM = "#2a9d8f"
C_POL = "#e07a5f"
C_TIME = "#3d5a80"


# %%
def load_run(run_id):
    out = out_dir(run_id)
    z = np.load(out / "segmented_spikes_binned.npz", allow_pickle=True)
    tids = z["trial_ids"]
    lab = pd.read_csv(out / "trial_labels.csv").set_index("trial_id").loc[tids]
    return out, z["binned"], z["bin_centers"], lab


def fit_and_report(binned, bc, cond_idx, side_idx, label_dict):
    cell = [(t, int(cond_idx[t]), int(side_idx[t])) for t in range(len(cond_idx))]
    dpca, X, bin_idx, mu = run_dpca_window(binned, bc, *WINDOW,
                                           labels="cst", cell_trial_ids=cell)
    var = {term: float(np.sum(dpca.explained_variance_ratio_.get(term, [0])))
           for term in TERMS}
    X_trial = raw_window_rate(binned, bc, bin_idx, *WINDOW)
    transform = make_transform("rate")
    dec = {}
    y, idx = label_dict["planning_vs_greedy"]
    for term in ["t", "c", "ct", "cst"]:
        acc_m, acc_s, chance, p = decode_from_term(
            dpca, term, X_trial, y, idx, mu, transform)
        dec[term] = {"acc": acc_m, "chance": chance, "p": p}
    return dpca, var, dec, X_trial, mu


def axis_cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# %% [markdown]
# ## Main

# %%
def analyse_run(run_id):
    out, binned, bc, lab = load_run(run_id)
    lab = lab.copy()
    lab["move_dir_vx"] = 0
    label_dict = build_label_vectors(lab)

    side_idx = (lab["choice_hole"] >= 6).astype(int).to_numpy()
    policy_idx = lab["condition"].map(POLICY_ORDER).to_numpy()
    conflict_mag = np.abs((lab["planning_cost_L"] - lab["planning_cost_R"])
                          - (lab["greedy_cost_L"] - lab["greedy_cost_R"])).to_numpy()
    geom_idx = pd.qcut(pd.Series(conflict_mag), 4, labels=False).to_numpy()
    keep = np.isfinite(geom_idx)
    label_dict["conflict_quartile"] = (geom_idx.astype(int), np.flatnonzero(keep))

    rows = []
    dpca_g, var_g, dec_g, _, _ = fit_and_report(binned, bc, geom_idx, side_idx, label_dict)
    dpca_p, var_p, dec_p, Xp, mu_p = fit_and_report(binned, bc, policy_idx, side_idx, label_dict)
    for term in TERMS:
        rows.append({"run_id": run_id, "fit": "geometry", "term": term,
                     "explained_variance": var_g[term]})
        rows.append({"run_id": run_id, "fit": "policy", "term": term,
                     "explained_variance": var_p[term]})
    rows.append({"run_id": run_id, "fit": "axis_similarity", "term": "c_cos",
                 "explained_variance": axis_cos(dpca_g.D["c"][:, 0], dpca_p.D["c"][:, 0])})
    rows.append({"run_id": run_id, "fit": "axis_similarity", "term": "ct_cos",
                 "explained_variance": axis_cos(dpca_g.D["ct"][:, 0], dpca_p.D["ct"][:, 0])})

    dec_rows = []
    for fit, dec in [("geometry", dec_g), ("policy", dec_p)]:
        for term, d in dec.items():
            dec_rows.append({"run_id": run_id, "fit": fit, "term": term,
                             "hypothesis": "planning_vs_greedy",
                             "acc_mean": d["acc"], "chance": d["chance"],
                             "perm_p": d["p"]})
    # cross: decode conflict quartile from the policy dPCA axes
    yq, idxq = label_dict["conflict_quartile"]
    transform = make_transform("rate")
    for term in ["t", "c", "ct", "cst"]:
        acc_m, acc_s, chance, p = decode_from_term(
            dpca_p, term, Xp, yq, idxq, mu_p, transform)
        dec_rows.append({"run_id": run_id, "fit": "policy", "term": term,
                         "hypothesis": "conflict_quartile", "acc_mean": acc_m,
                         "chance": chance, "perm_p": p})

    pd.DataFrame(rows).to_csv(out / "dpca_geometry.csv", index=False)
    pd.DataFrame(dec_rows).to_csv(out / "dpca_geometry_decoding.csv", index=False)
    return pd.DataFrame(rows), pd.DataFrame(dec_rows)


def make_figure(varagg):
    fig, ax = plt.subplots(figsize=(8, 5))
    terms = ["t", "c", "s", "cs", "ct", "st", "cst"]
    x = np.arange(len(terms))
    for fit, col in [("geometry", C_GEOM), ("policy", C_POL)]:
        vals = [varagg[(varagg.fit == fit) & (varagg.term == t)]["explained_variance"].mean()
                for t in terms]
        ax.bar(x + (0.2 if fit == "policy" else -0.2), vals, width=0.4,
               color=col, label=fit)
    ax.set_xticks(x); ax.set_xticklabels(terms)
    ax.set_ylabel("mean explained variance")
    ax.set_title("dPCA demixing: geometry-conditioned vs policy-conditioned")
    ax.legend()
    fig.tight_layout()
    figdir = AGG_DIR / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    out = figdir / "fig_neural_dpca_geometry.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    varlist, declist = [], []
    for run_id in RUN_ORDER:
        print(f"[dpca-geometry] {run_id}")
        try:
            v, d = analyse_run(run_id)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            continue
        varlist.append(v); declist.append(d)
    if not varlist:
        print("no results"); return
    var = pd.concat(varlist, ignore_index=True)
    dec = pd.concat(declist, ignore_index=True)
    summ = (var[var.fit.isin(["geometry", "policy"])]
            .groupby(["fit", "term"])["explained_variance"]
            .agg(["mean", "size"]).reset_index())
    summ.to_csv(AGG_DIR / "dpca_geometry_summary.csv", index=False)
    dec.to_csv(AGG_DIR / "dpca_geometry_decoding.csv", index=False)
    fig = make_figure(var)
    print(summ.round(4).to_string(index=False))
    print(f"\nwrote dpca_geometry.csv per run, dpca_geometry_summary.csv, {fig}")


if __name__ == "__main__":
    main()
