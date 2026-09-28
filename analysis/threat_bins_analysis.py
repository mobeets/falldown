# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#     jupytext_version: 1.19.4
#   kernelspec:
#     display_name: analysis
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Ball-height (threat) analysis in 5 bins
#
# Segments `ball_y_at_top` into 5 bins and characterises the planning-rate
# profile for the cloud study and EMU. The pooled profile is an **inverted-U**:
# planning is most likely when the ball is in the middle of the screen, lower
# near death (low `ball_y_at_top`) and far/safe (high).
#
# Ball height in the cloud data is entangled with camera mode / block (drift
# blocks are high on screen and have less planning). So the analysis separates:
#   * **between-block** structure (pooled bins; block-mean ball height vs block
#     planning rate), and
#   * **within-block** structure (`byc` = ball_y minus the participant x block
#     mean), the pure trial-level test.
#
# The primary trial-level test is a within-block-centred GEE with a quadratic
# term; a negative quadratic => inverted-U, peak at `-b_lin / (2 b_quad)`.
#
# A **drift-only, non-centred** result is also overlaid on the figure: panel A
# draws the drift-only quadratic fit and panel C the drift-only block-mean
# regression, so the reader can see whether dropping centration rescues a
# trial-level height effect (it does not -- the drift-only inverted-U is still
# a between-block/camera-mode effect).
#
# Outputs (analysis/planning_outputs/):
#   threat_bins5_summary.csv      cohort x threat x binning x bin means
#   threat_bins5_participant.csv  per-participant per-bin planning rate
#   threat_bins5_tests.csv        quadratic GEEs, block-level test, peak vs extremes
#   threat_bins5_block.csv        block-level mean ball_y vs planning rate
#   threat_bins5_drift_pooled.csv drift-only, non-centred per-bin means + stats
#   figures/fig_threat_bins5.png
#
# Run with:
#   python analysis/threat_bins_analysis.py
# (env: bash analysis/setup_env.sh)

# %%
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")

_REPO = Path(__file__).resolve().parent.parent
XD = _REPO / "analysis" / "cross_dataset_outputs"
PO = _REPO / "analysis" / "planning_outputs"

N_BINS = 5
C_CLOUD = "0.45"
C_EMU = "#d1495b"
C_ALT = "#00798c"
C_DRIFT = "#e07a5f"
CLOUD_DRIFT = ["drift", "alternating"]


def load_trials():
    d = pd.read_csv(XD / "cross_dataset_trials.csv")
    d = d[(d["is_experimental"]) & (d["conflict"] == 1)].copy()
    d = d.dropna(subset=["ball_y_at_top", "chose_planning"])
    d["byc"] = d["ball_y_at_top"] - d.groupby(
        ["source", "participant", "block_number"])["ball_y_at_top"].transform("mean")
    return d


# %% [markdown]
# ## 5-bin profiles

# %%
def pooled_bins(g, col, n_bins=N_BINS):
    g = g.copy()
    try:
        g["bin"] = pd.qcut(g[col], q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    return g.groupby("bin").agg(
        center=(col, "mean"), p_plan=("chose_planning", "mean"),
        n=("chose_planning", "size"),
        n_part=("participant", "nunique")).reset_index()


def participant_bins(d, col="ball_y_at_top", n_bins=N_BINS):
    rows = []
    for (src, pid), g in d.groupby(["source", "participant"]):
        try:
            g = g.copy()
            g["bin"] = pd.qcut(g[col], q=n_bins, labels=False, duplicates="drop")
        except ValueError:
            continue
        if g["bin"].nunique() < n_bins:
            continue
        for b, gb in g.groupby("bin"):
            rows.append({"source": src, "participant": pid, "bin": int(b),
                         "center": gb[col].mean(),
                         "p_plan": gb["chose_planning"].mean(), "n": len(gb)})
    return pd.DataFrame(rows)


def build_summary(d):
    rows = []

    def add(coh, threat, binning, s):
        for _, r in s.iterrows():
            rows.append({"cohort": coh, "threat": threat, "binning": binning,
                         "bin": int(r["bin"]), "center": r["center"],
                         "p_plan": r["p_plan"], "n": int(r["n"]),
                         "n_part": int(r["n_part"])})

    cloud = d[d["source"] == "cloud"]
    for th, g in cloud.groupby("threat"):
        add("cloud", th, "pooled", pooled_bins(g, "ball_y_at_top"))
        add("cloud", th, "within_block", pooled_bins(g, "byc"))
    add("cloud", "all", "within_block", pooled_bins(cloud, "byc"))

    emu = d[d["source"] == "emu"]
    for ds, g in list(emu.groupby("dataset")) + [("ALL", emu)]:
        add("EMU", ds, "pooled", pooled_bins(g, "ball_y_at_top"))
    return pd.DataFrame(rows)


# %% [markdown]
# ## Tests

# %%
def gee_quadratic(g, x, groups):
    gg = g.dropna(subset=["chose_planning", x]).copy()
    if gg["chose_planning"].nunique() < 2 or len(gg) < 50:
        return None
    formula = f"chose_planning ~ {x} + I({x} ** 2) + block_drift"
    try:
        return smf.gee(formula, groups=groups, data=gg,
                       family=sm.families.Binomial(),
                       cov_struct=sm.cov_struct.Exchangeable()).fit()
    except Exception as exc:  # pragma: no cover
        print(f"    GEE failed ({groups}/{x}): {exc}")
        return None


def quadratic_row(g, x, groups, label):
    m = gee_quadratic(g, x, groups)
    if m is None:
        return None
    b1 = m.params.get(x, np.nan)
    b2 = m.params.get(f"I({x} ** 2)", np.nan)
    p2 = m.pvalues.get(f"I({x} ** 2)", np.nan)
    peak = (-b1 / (2 * b2)) if (np.isfinite(b2) and b2 < 0) else np.nan
    return {"test": label, "x": x, "groups": groups, "n": int(len(g)),
            "coef_linear": b1, "coef_quadratic": b2, "p_quadratic": p2,
            "inv_u": bool(np.isfinite(b2) and b2 < 0),
            "peak_location": peak, "mean_mid": np.nan, "mean_extreme": np.nan}


def block_level(d, threats=CLOUD_DRIFT):
    """Per (participant, block): mean ball_y and planning rate (cloud)."""
    cloud = d[(d["source"] == "cloud") & (d["threat"].isin(threats))]
    blk = cloud.groupby(["participant", "block_number", "dataset", "threat"]).agg(
        mean_ball_y=("ball_y_at_top", "mean"),
        planning_rate=("chose_planning", "mean"),
        n=("chose_planning", "size")).reset_index()
    blk = blk[blk["n"] >= 5]
    return blk


def block_level_test(blk, label="cloud_between_block_blockmean_bally"):
    if blk.empty:
        return None
    m = smf.gee("chose_planning ~ mean_ball_y" if False else
                "planning_rate ~ mean_ball_y",
                groups="participant", data=blk,
                family=sm.families.Gaussian(),
                cov_struct=sm.cov_struct.Exchangeable()).fit()
    # per-participant correlation
    rs = [np.corrcoef(g["mean_ball_y"], g["planning_rate"])[0, 1]
          for _, g in blk.groupby("participant") if len(g) >= 6]
    return {"test": label, "x": "mean_ball_y",
            "groups": "participant", "n": int(len(blk)),
            "coef_linear": m.params.get("mean_ball_y"),
            "coef_quadratic": np.nan, "p_quadratic": m.pvalues.get("mean_ball_y"),
            "inv_u": np.nan, "peak_location": np.nan,
            "mean_mid": float(np.mean(rs)) if rs else np.nan,
            "mean_extreme": np.nan}


def mid_vs_extreme(d, col="ball_y_at_top", n_bins=N_BINS):
    pb = participant_bins(d, col=col, n_bins=n_bins)
    rows = []
    for (src, pid), g in pb.groupby(["source", "participant"]):
        g = g.set_index("bin")
        if len(g) < n_bins:
            continue
        ext = np.nanmean([g.loc[0, "p_plan"], g.loc[n_bins - 1, "p_plan"]])
        mid = np.nanmean([g.loc[b, "p_plan"] for b in range(1, n_bins - 1)])
        rows.append({"source": src, "participant": pid, "mid": mid,
                     "extreme": ext, "diff": mid - ext})
    r = pd.DataFrame(rows)
    if r.empty:
        return r, {"n": 0}
    t, p = stats.ttest_rel(r["mid"], r["extreme"])
    return r, {"n": len(r), "mean_mid": r["mid"].mean(),
               "mean_extreme": r["extreme"].mean(), "mean_diff": r["diff"].mean(),
               "t": t, "p": p, "n_positive": int((r["diff"] > 0).sum())}


def build_tests(d):
    rows = []
    cloud = d[(d["source"] == "cloud") & (d["threat"].isin(CLOUD_DRIFT))]
    specs = [("ball_y_at_top", "cloud_pooled_ball_y", cloud),
             ("byc", "cloud_within_block_centered", cloud),
             ("ball_y_at_top", "cloud_drift_pooled", cloud[cloud.threat == "drift"]),
             ("byc", "cloud_drift_within_block", cloud[cloud.threat == "drift"]),
             ("ball_y_at_top", "cloud_alt_pooled", cloud[cloud.threat == "alternating"]),
             ("byc", "cloud_alt_within_block", cloud[cloud.threat == "alternating"])]
    for x, lab, sub in specs:
        r = quadratic_row(sub, x, "participant", lab)
        if r:
            rows.append(r)
    r = quadratic_row(d[d["source"] == "emu"], "ball_y_at_top", "dataset",
                      "EMU_pooled_ball_y")
    if r:
        rows.append(r)

    blk = block_level(d)
    bt = block_level_test(blk)
    if bt:
        rows.append(bt)
    blk_d = block_level(d, threats=["drift"])
    bt_d = block_level_test(blk_d, label="cloud_drift_between_block_blockmean_bally")
    if bt_d:
        rows.append(bt_d)

    for name, col in [("cloud_mid_vs_extreme_pooled_bins", "ball_y_at_top"),
                      ("cloud_mid_vs_extreme_within_block", "byc")]:
        _, s = mid_vs_extreme(cloud, col)
        rows.append({"test": name, "x": "", "groups": "participant",
                     "n": int(s.get("n", 0)), "coef_linear": s.get("mean_diff"),
                     "coef_quadratic": np.nan, "p_quadratic": s.get("p"),
                     "inv_u": bool(s.get("mean_diff", np.nan) > 0),
                     "peak_location": np.nan, "mean_mid": s.get("mean_mid"),
                     "mean_extreme": s.get("mean_extreme")})
    return pd.DataFrame(rows), blk


# %% [markdown]
# ## Figure

# %%
def drift_quadratic_fit(sub):
    """Pooled (non-centred) drift-only quadratic GEE plus a fitted curve."""
    m = gee_quadratic(sub, "ball_y_at_top", "participant")
    if m is None:
        return None
    b0 = m.params.get("Intercept", 0.0)
    b1 = m.params.get("ball_y_at_top", np.nan)
    b2 = m.params.get("I(ball_y_at_top ** 2)", np.nan)
    bd = m.params.get("block_drift", 0.0)
    x = np.linspace(sub["ball_y_at_top"].min(), sub["ball_y_at_top"].max(), 100)
    eta = b0 + b1 * x + b2 * x ** 2 + bd
    peak = (-b1 / (2 * b2)) if (np.isfinite(b2) and b2 < 0) else np.nan
    return {"x": x, "p": 1.0 / (1.0 + np.exp(-eta)), "b2": b2,
            "p2": m.pvalues.get("I(ball_y_at_top ** 2)", np.nan),
            "peak": peak, "n": int(len(sub))}


def make_figure(d, summ, per_within, blk, blk_d):
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5))

    # A: cloud pooled by threat + drift-only (non-centred) quadratic overlay
    ax = axes[0, 0]
    for th, col in [("alternating", C_ALT), ("drift", C_DRIFT)]:
        s = summ[(summ.cohort == "cloud") & (summ.threat == th)
                 & (summ.binning == "pooled")].sort_values("bin")
        ax.plot(s["center"], s["p_plan"], "-o", color=col, label=th)
    s_d = summ[(summ.cohort == "cloud") & (summ.threat == "drift")
               & (summ.binning == "pooled")]
    fit = drift_quadratic_fit(d[(d.source == "cloud") & (d.threat == "drift")])
    if fit:
        ax.plot(fit["x"], fit["p"], color=C_DRIFT, ls="--", lw=2,
                label="drift fit (drift only, no centring)")
        ax.text(0.02, 0.02,
                f"drift only, not centred:\n{len(s_d)} realized bins, n={fit['n']}\n"
                f"b2={fit['b2']:.2e}, p={fit['p2']:.3g}, peak={fit['peak']:.0f} px",
                transform=ax.transAxes, fontsize=7, va="bottom",
                bbox=dict(fc="white", ec="0.7", alpha=0.85))
    ax.set_xlabel("ball_y_at_top (px; low = near death)")
    ax.set_ylabel("P(plan | conflict)")
    ax.set_title("A. Cloud pooled 5 bins (+ drift-only fit)")
    ax.legend(fontsize=8)

    # B: cloud within-block-centred by threat
    ax = axes[0, 1]
    for th, col in [("all", C_CLOUD), ("drift", C_DRIFT),
                    ("alternating", C_ALT)]:
        s = summ[(summ.cohort == "cloud") & (summ.threat == th)
                 & (summ.binning == "within_block")].sort_values("bin")
        if s.empty:
            continue
        ax.plot(s["center"], s["p_plan"], "-o", color=col, label=th)
    ax.axhline(0, color="k", ls=":", lw=0.8)
    ax.set_xlabel("within-block-centred ball_y (px)")
    ax.set_ylabel("P(plan | conflict)")
    ax.set_title("B. Cloud within-block 5 bins (trial-level)")
    ax.legend(fontsize=8)

    # C: block-level mean ball_y vs planning + drift-only regression overlay
    ax = axes[0, 2]
    for th, col in [("drift", C_DRIFT), ("alternating", C_ALT)]:
        b = blk[blk.threat == th] if not blk.empty else pd.DataFrame()
        if b.empty:
            continue
        ax.scatter(b["mean_ball_y"], b["planning_rate"], color=col, s=22,
                   alpha=0.7, label=th)
    if not blk.empty:
        m, c = np.polyfit(blk["mean_ball_y"], blk["planning_rate"], 1)
        xs = np.linspace(blk["mean_ball_y"].min(), blk["mean_ball_y"].max(), 20)
        ax.plot(xs, m * xs + c, color="k", lw=1.5, label="drift+alt fit")
    if not blk_d.empty:
        md, cd = np.polyfit(blk_d["mean_ball_y"], blk_d["planning_rate"], 1)
        xd = np.linspace(blk_d["mean_ball_y"].min(), blk_d["mean_ball_y"].max(), 20)
        ax.plot(xd, md * xd + cd, color=C_DRIFT, lw=2.2, ls="--",
                label="drift-only fit")
        r = np.corrcoef(blk_d["mean_ball_y"], blk_d["planning_rate"])[0, 1]
        ax.text(0.02, 0.98,
                f"drift blocks n={len(blk_d)}\nslope={md:.2e}\nr={r:+.2f}",
                transform=ax.transAxes, fontsize=7, va="top",
                bbox=dict(fc="white", ec="0.7", alpha=0.85))
    ax.set_xlabel("block-mean ball_y (px)")
    ax.set_ylabel("block planning rate")
    ax.set_title("C. Between-block: height vs planning (+ drift-only)")
    ax.legend(fontsize=8)

    # D: EMU by run
    ax = axes[1, 0]
    for ds, g in summ[(summ.cohort == "EMU") & (summ.threat != "ALL")].groupby("threat"):
        g = g.sort_values("bin")
        ax.plot(g["center"], g["p_plan"], "-o", color=C_EMU, alpha=0.6, label=ds)
    s = summ[(summ.cohort == "EMU") & (summ.threat == "ALL")].sort_values("bin")
    ax.plot(s["center"], s["p_plan"], "-o", color="k", lw=2.5, label="mean")
    ax.set_xlabel("ball_y_at_top (px)")
    ax.set_ylabel("P(plan | conflict)")
    ax.set_title("D. EMU 5 bins (all-drift)")
    ax.legend(fontsize=7)

    # E: neural 5-bin decode
    ax = axes[1, 1]
    p = XD / "xd_neural_planning_bally.csv"
    if p.exists():
        nb = pd.read_csv(p)
        for rep, col in [("rate", "#2a9d8f"), ("pca", "#264653")]:
            sub = nb[(nb.window == "post") & (nb.rep == rep)]
            m = sub.groupby("bin")["acc_mean"].agg(["mean", "sem"])
            ax.errorbar(m.index, m["mean"], yerr=m["sem"], marker="o",
                        color=col, capsize=3, label=f"decode ({rep})")
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.set_xlabel("ball-y quintile (0 = near death)")
    ax.set_ylabel("planning_vs_greedy acc")
    ax.set_title("E. EMU neural decode by ball-y bin")
    ax.legend(fontsize=8)

    # F: per-participant mid vs extreme (within-block)
    ax = axes[1, 2]
    for _, g in per_within.groupby("participant"):
        g = g.set_index("bin")
        if len(g) < N_BINS:
            continue
        ext = np.nanmean([g.loc[0, "p_plan"], g.loc[N_BINS - 1, "p_plan"]])
        mid = np.nanmean([g.loc[b, "p_plan"] for b in range(1, N_BINS - 1)])
        ax.plot([0, 1], [ext, mid], color="0.7", lw=0.8, marker="o", ms=3)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["extreme\nbins", "middle\nbins"])
    ax.set_ylabel("P(plan | conflict)")
    ax.set_title("F. Per-participant mid vs extreme (within-block)")

    fig.suptitle("5-bin ball-height planning profile: pooled inverted-U, "
                 "drift-only overlay (not centred) still between-block",
                 fontsize=15, y=1.0)
    fig.tight_layout()
    out = PO / "figures" / "fig_threat_bins5.png"
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
    print(f"loaded {len(d)} conflict trials "
          f"({d[d.source=='cloud'].participant.nunique()} cloud participants, "
          f"{d[d.source=='emu'].dataset.nunique()} EMU runs)")

    summ = build_summary(d)
    per_pooled = participant_bins(d[d.source == "cloud"], "ball_y_at_top")
    per_within = participant_bins(d[d.source == "cloud"], "byc")
    tests, blk = build_tests(d)
    blk_d = block_level(d, threats=["drift"])

    drift_pooled = summ[(summ.cohort == "cloud") & (summ.threat == "drift")
                        & (summ.binning == "pooled")].copy()

    summ.to_csv(PO / "threat_bins5_summary.csv", index=False)
    pd.concat([per_pooled.assign(binning="pooled"),
               per_within.assign(binning="within_block")],
              ignore_index=True).to_csv(PO / "threat_bins5_participant.csv", index=False)
    tests.to_csv(PO / "threat_bins5_tests.csv", index=False)
    blk.to_csv(PO / "threat_bins5_block.csv", index=False)
    drift_pooled.to_csv(PO / "threat_bins5_drift_pooled.csv", index=False)
    fig = make_figure(d, summ, per_within, blk, blk_d)

    print("\nTests:")
    print(tests.round(4).to_string(index=False))
    print(f"\nwrote threat_bins5_*.csv and {fig}")


if __name__ == "__main__":
    main()
