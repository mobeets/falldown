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
# # Neural representational geometry of planning across ball height
#
# Two analyses that relate the *geometry* of population activity to the
# planning questions (beyond decoding accuracy):
#
#   rsa_bally.csv     representational similarity: neural RDMs (correlation
#                    distance between trial population vectors) vs model RDMs
#                    (greedy_gap / planning_gap / conflict_mag / rel_y), overall
#                    and within near/far / ball-y bins.
#   dpca_threat.csv   threat-conditioned dPCA: explained variance per demixed
#                    term and near/far subspace similarity (principal angles).
#
# Outputs (analysis/neural_outputs/<run>/):
#   rsa_bally.csv
#   dpca_threat.csv
#
# Run with:
#   .venv-analysis\Scripts\python.exe analysis\neural_planning_geometry.py --run yfz_1

# %%
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

from neural_common import get_run, out_dir
from neural_lda_decoding import RNG_SEED, rate_features, build_label_vectors
from neural_confound_audit import load_binned, build_regressors
from neural_dpca import run_dpca_window

from neural_planning_profiles import (
    build_geometry, keep_index, _near_far, bin_by_ball_y, N_BINS,
)

# ---------------------------- Configuration ----------------------------
_RUN = get_run()
OUT_DIR = out_dir(_RUN.run_id)
TRIAL_LABELS = OUT_DIR / "trial_labels.csv"
POST = (0.0, 1000.0)
RSA_MODELS = ["greedy_gap", "planning_gap", "conflict_mag", "rel_y"]
DPCA_TERMS = ["c", "s", "cs"]
# -----------------------------------------------------------------------


# %%
def _condensed(D):
    iu = np.triu_indices(D.shape[0], 1)
    return D[iu]


def rsa_bally(n_perm=100):
    binned, unit_ids, bc = load_binned(OUT_DIR)
    lab = pd.read_csv(TRIAL_LABELS)
    lab["move_dir_vx"] = 0
    lv = build_label_vectors(lab)
    y, idx = lv["planning_vs_greedy"]
    base = keep_index(idx, len(y))
    geom = build_geometry()
    ball_y = geom["ball_y_entry"].to_numpy()
    regs = build_regressors(OUT_DIR)

    X = rate_features(binned, bc, *POST)
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-9)
    valid = base & np.isfinite(ball_y)
    sel_all = np.flatnonzero(valid)
    near, far = _near_far(ball_y, sel_all)
    subsets = {"all": sel_all, "near": near, "far": far}
    bids = bin_by_ball_y(ball_y[sel_all], N_BINS)
    for b in sorted(bids.dropna().astype(int).unique()):
        subsets[f"bin{int(b)}"] = sel_all[(bids == b).to_numpy()]

    models = {m: regs[m] for m in RSA_MODELS if m != "rel_y"}
    models["rel_y"] = ball_y

    rows = []
    for sname, s in subsets.items():
        if len(s) < 30:
            continue
        D = 1.0 - np.corrcoef(Xz[s])
        dvec = _condensed(D)
        rng = np.random.default_rng(RNG_SEED)
        for mname, mv in models.items():
            v = np.asarray(mv, float)[s]
            if not np.all(np.isfinite(v)):
                continue
            mvec = _condensed(np.abs(v[:, None] - v[None, :]))
            r = float(stats.spearmanr(dvec, mvec).correlation)
            null = np.empty(n_perm)
            for i in range(n_perm):
                vp = rng.permutation(v)
                null[i] = stats.spearmanr(
                    dvec, _condensed(np.abs(vp[:, None] - vp[None, :]))
                ).correlation
            rows.append({"subset": sname, "model": mname, "n": len(s),
                         "spearman_r": r,
                         "perm_p": float(np.mean(np.abs(null) >= abs(r)))})
        print(f"  RSA {sname} (n={len(s)}): " +
              ", ".join(f"{r['model']} {r['spearman_r']:+.3f}"
                        for r in rows[-len(models):]))
    return pd.DataFrame(rows)


# %%
def _subspace_sim(A, B):
    """Mean cosine of principal angles between column spaces of A and B."""
    if A.shape[1] == 0 or B.shape[1] == 0:
        return np.nan
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    sv = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return float(np.mean(sv))


def threat_dpca():
    binned, unit_ids, bc = load_binned(OUT_DIR)
    lab = pd.read_csv(TRIAL_LABELS)
    geom = build_geometry()
    ball_y = geom["ball_y_entry"].to_numpy()
    cond = lab["condition"].to_numpy()
    cond_idx = pd.Series(cond).map(
        {"planning": 0, "greedy": 1, "agree_optimal": 2, "lapse": 3}).to_numpy()
    side_idx = (lab["choice_hole"] >= 6).astype(int).to_numpy()
    # use all conditions so every (condition x side) cell is populated; the
    # near/far split is still by ball height
    base = np.isfinite(ball_y)
    sel = np.flatnonzero(base)
    near, far = _near_far(ball_y, sel)

    rows = []
    fitted = {}
    for name, idxs in [("near", near), ("far", far)]:
        cell = [(int(t), int(cond_idx[t]), int(side_idx[t])) for t in idxs]
        dpca, X, bin_idx, mu = run_dpca_window(
            binned, bc, POST[0], POST[1], labels="cst", cell_trial_ids=cell)
        fitted[name] = dpca
        for term, ev in dpca.explained_variance_ratio_.items():
            for k, v in enumerate(ev):
                rows.append({"analysis": "variance", "condition": name,
                             "term": term, "component": k,
                             "explained_variance": float(v)})
        print(f"  dPCA {name}: n_trials={len(idxs)}")

    for term in DPCA_TERMS:
        sim = _subspace_sim(fitted["near"].D[term], fitted["far"].D[term])
        rows.append({"analysis": "subspace_similarity", "condition": "near|far",
                     "term": term, "component": np.nan,
                     "explained_variance": sim})
        print(f"  dPCA subspace similarity {term}: {sim:.3f}")
    return pd.DataFrame(rows)


# %%
def main():
    print(f"Run {_RUN.run_id} ({_RUN.participant})")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[RSA] neural vs model representational similarity")
    rsa = rsa_bally()
    rsa.to_csv(OUT_DIR / "rsa_bally.csv", index=False)
    print(f"  saved rsa_bally.csv ({len(rsa)} rows)")

    print("\n[dPCA] threat-conditioned demixing")
    dp = threat_dpca()
    dp.to_csv(OUT_DIR / "dpca_threat.csv", index=False)
    print(f"  saved dpca_threat.csv ({len(dp)} rows)")


if __name__ == "__main__":
    main()
