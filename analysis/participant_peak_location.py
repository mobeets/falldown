# %% [markdown]
# # Per-participant ball-height peak location
#
# Question: is "people plan best somewhere in the middle of the screen" a
# generally true *rule*, allowing each participant a different best bin?
#
# `threat_bins_analysis.py` finds a pooled inverted-U that is between-block and
# does not survive centring. That pooled test fixes one common set of bins. If
# instead **each participant peaks in their own bin**, a common-bin average could
# flatten a real per-person peak. This script tests that directly:
#
#   * per-participant 5-bin profiles using each participant's own quantiles,
#   * the empirical argmax bin and quadratic curvature per participant,
#   * a within-participant permutation test of the proportion of **interior**
#     peaks (bins 1-3 of 0-4) against the chance rate,
#   * the dispersion of peak locations across participants.
#
# Both raw `ball_y_at_top` (between-block) and within-block-centred `byc`
# (trial-level) versions are reported, because raw height is camera/block tied.
#
# Outputs (analysis/planning_outputs/):
#   participant_peak_location.csv       per-participant peak table
#   participant_peak_summary.csv        group-level tests
#   figures/fig_participant_peak_location.png
#
# Run with:
#   python analysis/participant_peak_location.py
# (env: bash analysis/setup_env.sh)

# %%
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")

_REPO = Path(__file__).resolve().parent.parent
XD = _REPO / "analysis" / "cross_dataset_outputs"
PO = _REPO / "analysis" / "planning_outputs"

N_BINS = 5
MIN_TRIALS = 40
MIN_BIN_N = 3
N_PERM = 5000
RNG = np.random.default_rng(0)
C_RAW = "#e07a5f"
C_BYC = "#3d5a80"
C_MID = "#2a9d8f"


def load_trials():
    d = pd.read_csv(XD / "cross_dataset_trials.csv")
    d = d[(d["is_experimental"]) & (d["conflict"] == 1)].copy()
    d = d.dropna(subset=["ball_y_at_top", "chose_planning"])
    d["byc"] = d["ball_y_at_top"] - d.groupby(
        ["source", "participant", "block_number"])["ball_y_at_top"].transform("mean")
    return d


def participant_profile(p, col, n_bins=N_BINS):
    """Per-bin (center, p_plan, n) for one participant, or None if too sparse."""
    if len(p) < MIN_TRIALS:
        return None
    try:
        b = pd.qcut(p[col], n_bins, labels=False, duplicates="drop")
    except ValueError:
        return None
    if b.nunique() < n_bins:
        return None
    r = p.assign(bin=b).groupby("bin").agg(
        center=(col, "mean"), p_plan=("chose_planning", "mean"),
        n=("chose_planning", "size")).reindex(range(n_bins))
    if r["n"].isna().any() or (r["n"] < MIN_BIN_N).any():
        return None
    return r


def peak_row(p, col, binning):
    r = participant_profile(p, col)
    if r is None:
        return None
    x = r.index.values.astype(float)
    y = r["p_plan"].values
    w = r["n"].values.astype(float)
    # weighted quadratic on bin-index means; negative b2 => interior maximum
    b2 = np.polyfit(x, y, 2, w=w)[0]
    arg = int(r["p_plan"].idxmax())
    extremes = np.mean([r.loc[0, "p_plan"], r.loc[N_BINS - 1, "p_plan"]])
    mid = np.mean([r.loc[b, "p_plan"] for b in range(1, N_BINS - 1)])
    return {"binning": binning, "participant": None,
            "n": int(len(p)), "argmax_bin": arg,
            "argmax_center": float(r.loc[arg, "center"]),
            "interior": bool(arg in (1, 2, 3)),
            "curv_b2": float(b2), "inv_u": bool(b2 < 0),
            "mid_minus_extreme": float(mid - extremes),
            "p_bin0": float(r.loc[0, "p_plan"]), "p_bin1": float(r.loc[1, "p_plan"]),
            "p_bin2": float(r.loc[2, "p_plan"]), "p_bin3": float(r.loc[3, "p_plan"]),
            "p_bin4": float(r.loc[4, "p_plan"])}


def build_table(d, source="cloud", col="ball_y_at_top", binning="raw"):
    rows = []
    for pid, p in d[d.source == source].groupby("participant"):
        r = peak_row(p, col, binning)
        if r:
            r["participant"] = pid
            rows.append(r)
    return pd.DataFrame(rows)


def interior_permutation_test(d, col, n_perm=N_PERM):
    """Null: planning labels are exchangeable across a participant's bins.

    Reshuffle each participant's labels within participant, recompute argmax,
    and count interior peaks. Returns (observed, p_one_sided, null_counts).
    """
    profs = []
    for pid, p in d.groupby("participant"):
        if participant_profile(p, col) is None:
            continue
        profs.append((pid, p["chose_planning"].values,
                      pd.qcut(p[col], N_BINS, labels=False, duplicates="drop").values))
    if not profs:
        return None
    obs = sum(int(np.argmax([y[b == i].mean() for i in range(N_BINS)]) in (1, 2, 3))
              for _, y, b in profs)
    null = np.empty(n_perm, dtype=int)
    for k in range(n_perm):
        c = 0
        for _, y, b in profs:
            yp = y.copy()
            RNG.shuffle(yp)
            if int(np.argmax([yp[b == i].mean() for i in range(N_BINS)])) in (1, 2, 3):
                c += 1
        null[k] = c
    n = len(profs)
    p_one = (np.sum(null >= obs) + 1) / (n_perm + 1)
    return {"n_participants": n, "observed_interior": int(obs), "obs_prop": obs / n,
            "null_prop_mean": float(null.mean() / n), "p_one_sided": p_one}


def summarise(t, perm):
    if t.empty:
        return {}
    n = len(t)
    out = {"n_participants": n,
           "interior_prop": float(t["interior"].mean()),
           "argmax_bin0": int((t.argmax_bin == 0).sum()),
           "argmax_bin1": int((t.argmax_bin == 1).sum()),
           "argmax_bin2": int((t.argmax_bin == 2).sum()),
           "argmax_bin3": int((t.argmax_bin == 3).sum()),
           "argmax_bin4": int((t.argmax_bin == 4).sum()),
           "inv_u_count": int(t["inv_u"].sum()),
           "curvature_sign_p": float(stats.binomtest(int(t["inv_u"].sum()), n, 0.5,
                                                     alternative="two-sided").pvalue),
           "mid_minus_extreme_mean": float(t["mid_minus_extreme"].mean()),
           "mid_minus_extreme_p": float(stats.ttest_1samp(t["mid_minus_extreme"], 0).pvalue)
           if n > 1 else np.nan,
           "peak_center_std_px": float(t["argmax_center"].std(ddof=1)) if n > 1 else np.nan,
           "argmax_bin_std": float(t["argmax_bin"].std(ddof=1)) if n > 1 else np.nan}
    if perm:
        out.update({"perm_obs_interior": perm["observed_interior"],
                    "perm_null_prop": perm["null_prop_mean"],
                    "perm_p_one_sided": perm["p_one_sided"]})
    return out


# %% [markdown]
# ## Figure

# %%
def make_figure(d, tables, summary, perm):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # A/B: per-participant spaghetti, raw and within-block
    for ax, (col, binning, color, title) in zip(
            axes[0],
            [("ball_y_at_top", "raw", C_RAW,
              "A. Per-participant profiles (raw ball_y, own quantiles)"),
             ("byc", "byc", C_BYC,
              "B. Per-participant profiles (within-block, own quantiles)")]):
        t = tables[binning]
        profs = []
        for pid, p in d[d.source == "cloud"].groupby("participant"):
            r = participant_profile(p, col)
            if r is None:
                continue
            profs.append(r)
            ax.plot(r["center"], r["p_plan"], "-o", color=color, alpha=0.35,
                    lw=1, ms=3)
        if profs:
            centers = np.mean([r["center"].values for r in profs], axis=0)
            means = np.mean([r["p_plan"].values for r in profs], axis=0)
            ax.plot(centers, means, "-o", color="k", lw=2.6, ms=7,
                    label=f"group mean (N={len(profs)})")
            ax.legend(fontsize=8)
        ax.set_xlabel("ball_y_at_top (px)" if binning == "raw"
                      else "within-block-centred ball_y (px)")
        ax.set_ylabel("P(plan | conflict)")
        ax.set_title(title)

    # C: argmax-bin distribution, raw vs within-block
    ax = axes[1, 0]
    width = 0.38
    xs = np.arange(N_BINS)
    for off, binning, color, ls in [(-width / 2, "raw", C_RAW, ":"),
                                    (width / 2, "byc", C_BYC, "--")]:
        t = tables[binning]
        counts = [int((t.argmax_bin == b).sum()) for b in range(N_BINS)]
        ax.bar(xs + off, counts, width=width, color=color, label=binning)
        if len(t):
            ax.axhline(len(t) / N_BINS, color=color, ls=ls, lw=1.2,
                       label=f"{binning} uniform expectation")
    if perm:
        ax.set_title(f"C. Preferred bin per participant "
                     f"(interior {perm['observed_interior']}/{perm['n_participants']}, "
                     f"chance p={perm['p_one_sided']:.2f})")
    else:
        ax.set_title("C. Preferred bin per participant")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{b}\n({'near' if b == 0 else 'far' if b == 4 else 'mid'})"
                        for b in range(N_BINS)])
    ax.set_xlabel("ball-height bin (own quantiles)")
    ax.set_ylabel("participants with peak in bin")
    ax.legend(fontsize=8)

    # D: per-participant curvature (weighted quadratic b2)
    ax = axes[1, 1]
    t = tables["byc"].sort_values("curv_b2").reset_index(drop=True)
    tr = tables["raw"].sort_values("curv_b2").reset_index(drop=True)
    ax.scatter(np.zeros(len(tr)), tr["curv_b2"], color=C_RAW, s=28, label="raw")
    ax.scatter(np.ones(len(t)), t["curv_b2"], color=C_BYC, s=28,
               label="within-block")
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["raw", "within-block"])
    ax.set_ylabel("per-participant quadratic b2  (b2<0 = inverted-U)")
    ax.set_title("D. Individual curvature: no consistent inverted-U")
    ax.legend(fontsize=8)

    fig.suptitle("Per-participant ball-height peak location: peaks scatter "
                 "across bins; no shared 'middle' rule", fontsize=14, y=1.0)
    fig.tight_layout()
    out = PO / "figures" / "fig_participant_peak_location.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


# %% [markdown]
# ## Main

# %%
def main():
    PO.mkdir(parents=True, exist_ok=True)
    d = load_trials()
    cloud = d[d.source == "cloud"]

    tables = {"raw": build_table(cloud, col="ball_y_at_top", binning="raw"),
              "byc": build_table(cloud, col="byc", binning="byc")}
    perm = interior_permutation_test(cloud, "ball_y_at_top")
    perm_byc = interior_permutation_test(cloud, "byc")

    rows = pd.concat([t for t in tables.values() if not t.empty],
                     ignore_index=True)
    rows.to_csv(PO / "participant_peak_location.csv", index=False)

    summary = []
    for binning, t, perm_t in [("raw", tables["raw"], perm),
                               ("byc", tables["byc"], perm_byc)]:
        s = summarise(t, perm_t)
        s.update({"binning": binning, "source": "cloud"})
        summary.append(s)
    # drift-only (raw): usually too few participants for a per-person peak
    drift = cloud[cloud.threat == "drift"]
    t_drift = build_table(drift, col="ball_y_at_top", binning="drift_raw")
    if not t_drift.empty:
        s = summarise(t_drift, None)
        s.update({"binning": "drift_raw", "source": "cloud_drift"})
        summary.append(s)
    summ = pd.DataFrame(summary)
    summ.to_csv(PO / "participant_peak_summary.csv", index=False)

    fig = make_figure(cloud, tables, summ, perm)

    pd.set_option("display.width", 200)
    print("Per-participant peak summary:")
    print(summ.round(3).to_string(index=False))
    print("\nPer-participant table:")
    print(rows.round(3).to_string(index=False))
    print(f"\nwrote participant_peak_location.* and {fig}")


if __name__ == "__main__":
    main()
