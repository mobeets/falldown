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
# # Test 4 - per-unit / per-region selectivity: exit-pass vs planning vs side
#
# Exit-pass selectivity is computed here with the project's modulation-index
# machinery (MI = (A-B)/(A+B), permutation p, BH-FDR), contrasting activity in
# the bin at the exit pass vs far-from-exit bins. Planning and side counts are
# taken from the existing `selectivity_results.csv` and
# `left_right_selectivity_results.csv` (same machinery), so all three are
# directly comparable.
#
# Outputs (aggregate/): exit_selectivity.csv, exit_selectivity_summary.csv,
#   exit_selectivity_regions.csv, figures/fig_exit_selectivity.png

# %%
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests

from neural_common import (RUN_ORDER, AGG_DIR, out_dir, region_from_channel,
                           region_from_label)
from neural_lda_decoding import RNG_SEED

# %%
NEAR_MS = 25.0
FAR_MS = 400.0
N_PERM = 200
PLAN_CONTRAST, PLAN_WINDOW = "planning_vs_greedy", "post"
SIDE_CONTRAST, SIDE_WINDOW = "left_vs_right_move_dir", "post"


def load(run):
    out = out_dir(run)
    z = np.load(out / "segmented_spikes_binned.npz", allow_pickle=True)
    tids = z["trial_ids"]
    tt = pd.read_csv(out / "trial_table.csv").set_index("trial_id").loc[tids].reset_index()
    u = pd.read_csv(out / "unit_metadata.csv")
    binned, bc = z["binned"], z["bin_centers"]
    valid = ~np.isnan(binned).all(axis=0)
    rt_exec = (tt["exit_time_ms"] - tt["choice_time_ms"]).to_numpy(float)
    return dict(binned=binned, bc=bc, valid=valid, tt=tt, u=u, rt_exec=rt_exec)


def exit_pass_mi(d, n_perm=N_PERM):
    ti, bi = np.where(d["valid"])
    t = d["bc"][bi]
    re = d["rt_exec"][ti]
    near = np.flatnonzero(np.abs(t - re) <= NEAR_MS)
    far = np.flatnonzero(np.abs(t - re) > FAR_MS)
    if len(near) < 20 or len(far) < 20:
        return None, None, None
    rng = np.random.default_rng(RNG_SEED)
    far = rng.choice(far, size=min(len(near), len(far)), replace=False)
    idx = np.concatenate([near, far])
    X = d["binned"][:, ti[idx], bi[idx]].T          # (samples, units)
    y = np.concatenate([np.ones(len(near), int), np.zeros(len(far), int)])
    m1, m0 = X[y == 1].mean(0), X[y == 0].mean(0)
    mi = (m1 - m0) / (m1 + m0 + 1e-9)
    null = np.empty((n_perm, X.shape[1]))
    for k in range(n_perm):
        yp = rng.permutation(y)
        a, b = X[yp == 1].mean(0), X[yp == 0].mean(0)
        null[k] = (a - b) / (a + b + 1e-9)
    p = (np.abs(null) >= np.abs(mi)).mean(0)
    q = multipletests(p, method="fdr_bh")[1]
    return mi, p, q


def existing_stats(run, contrast, window, filename):
    """(n_sig, n_total, {region: count}) from a project selectivity file.

    Note: these files index units differently from the binned array, so counts
    come from the file itself and regions from its `channel` column.
    """
    p = out_dir(run) / filename
    if not p.exists():
        return None
    df = pd.read_csv(p)
    s = df[(df["contrast"] == contrast) & (df["window"] == window)]
    if not len(s):
        return None
    sig = s[s["significant"].astype(bool)]
    regs = sig["channel"].map(region_from_label).value_counts().to_dict()
    return int(len(sig)), int(len(s)), regs


# %%
def main():
    unit_rows, region_rows, summ_rows = [], [], []
    for run in RUN_ORDER:
        print(f"[{run}] Test 4 selectivity")
        d = load(run)
        regions = {int(r.unit_id): region_from_channel(r.channel) for r in d["u"].itertuples()}
        n_units = d["binned"].shape[0]

        mi, p, q = exit_pass_mi(d)
        exit_sig = set()
        if mi is not None:
            for j in range(n_units):
                sig = bool(q[j] < 0.05)
                unit_rows.append({"run_id": run, "unit_id": j,
                                  "region": regions.get(j, ""),
                                  "contrast": "exit_pass", "mi": mi[j],
                                  "p": p[j], "q": q[j], "significant": sig})
                if sig:
                    exit_sig.add(j)

        plan = existing_stats(run, PLAN_CONTRAST, PLAN_WINDOW,
                              "selectivity_results.csv") or (0, 0, {})
        side = existing_stats(run, SIDE_CONTRAST, SIDE_WINDOW,
                              "left_right_selectivity_results.csv") or (0, 0, {})

        summ_rows.append({"run_id": run, "contrast": "exit_pass",
                          "n_sig": len(exit_sig), "n_units": n_units})
        for u in exit_sig:
            region_rows.append({"run_id": run, "contrast": "exit_pass",
                                "region": regions.get(u, ""), "n": 1})
        for name, (ns, nt, regs) in [("planning", plan), ("side", side)]:
            summ_rows.append({"run_id": run, "contrast": name,
                              "n_sig": ns, "n_units": nt})
            for r, c in regs.items():
                region_rows.append({"run_id": run, "contrast": name,
                                    "region": r, "n": int(c)})

    units = pd.DataFrame(unit_rows)
    units.to_csv(AGG_DIR / "exit_selectivity.csv", index=False)
    summ = pd.DataFrame(summ_rows)
    summ.to_csv(AGG_DIR / "exit_selectivity_summary.csv", index=False)
    reg = (pd.DataFrame(region_rows).groupby(["run_id", "contrast", "region"])["n"]
           .sum().reset_index())
    reg.to_csv(AGG_DIR / "exit_selectivity_regions.csv", index=False)
    print(summ.to_string(index=False))

    # figure: significant-unit counts per contrast
    fig, axes = plt.subplots(1, len(RUN_ORDER), figsize=(4.4 * len(RUN_ORDER), 3.6),
                             sharey=True)
    contrasts = ["exit_pass", "planning", "side"]
    for ax, run in zip(axes, RUN_ORDER):
        s = summ[summ["run_id"] == run].set_index("contrast")
        vals = [int(s.loc[c, "n_sig"]) if c in s.index else 0 for c in contrasts]
        tot = int(s.loc["exit_pass", "n_units"]) if "exit_pass" in s.index else 0
        ax.bar(contrasts, vals, color=["tab:purple", "tab:red", "tab:blue"], edgecolor="k")
        for i, v in enumerate(vals):
            ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
        ax.set_title(f"{run}  (n={tot} units)", fontsize=10)
    axes[0].set_ylabel("selective units (FDR<0.05, post)")
    fig.suptitle("Test 4: per-unit selectivity — exit-pass vs planning vs side", fontsize=12)
    fig.tight_layout()
    fig.savefig(AGG_DIR / "figures" / "fig_exit_selectivity.png", dpi=140,
                bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote exit_selectivity*.csv + fig_exit_selectivity.png to {AGG_DIR}")


if __name__ == "__main__":
    main()
