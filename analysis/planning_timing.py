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
# # Trajectory-phase decomposition of planning vs execution time
#
# The game's ball falls continuously; `Level.passedThrough` registers the hole
# nearest the ball's x when it drops past a level (`app/static/objects.js:161`).
# So `rt_decision` (entry -> choice) and `rt_exec` (choice -> exit) are *travel*
# times, not deliberation. This script decomposes each interval frame-by-frame
# into physically meaningful phases and asks whether planning differs in
# transport (geometry) or in hold/hesitation (a decision pause).
#
# Phases are detected model-free from per-frame displacement (displacement
# plateau), not from a velocity threshold:
#   fall  = |dy| > EPS_DY                     (airborne / free vertical motion)
#   roll  = |dy| <= EPS_DY and |dx| > EPS_DX  (in platform contact, moving)
#   hold  = |dy| <= EPS_DY and |dx| <= EPS_DX (in contact, stationary)
#
# Runs on all 18 retained online participants (3 datasets) and all four EMU
# runs (all-drift). Outputs reuse the existing locations:
#   analysis/planning_outputs/               (online trial-level + summary CSV,
#                                            figures, geometry separation)
#   analysis/neural_outputs/aggregate/       (EMU trial-level + summaries)
#
# Run with:
#   .venv-analysis\Scripts\python.exe analysis\planning_timing.py

# %%
import json
import sys
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

import planning_experiments as pe

warnings.filterwarnings("ignore")

_REPO = Path(__file__).resolve().parent.parent
ONLINE_OUT = _REPO / "analysis" / "planning_outputs"
AGG_OUT = _REPO / "analysis" / "neural_outputs" / "aggregate"

RUN_ORDER = ["yfz_1", "yfz_2", "yga_1", "yga_2"]

EPS_DY = 0.05   # px of vertical motion that still counts as "in contact"
EPS_DX = 0.05   # px of horizontal motion that still counts as "stationary"

C_PLAN = "#d1495b"
C_GREED = "#00798c"
C_FALL = "#7fb3d5"
C_ROLL = "#f4b942"
C_HOLD = "#b0b0b0"


# %% [markdown]
# ## Trajectory loading and phase detection

# %%
def load_block_states(path):
    """{block_key: arrays} for every block with `game_states`.

    Keys are exposed as the raw `block_index` field (used by the EMU
    trial tables) and as the enumerate index (used by the online
    `build_sequence_table`).
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    out = {}
    for i, b in enumerate(data.get("blocks", [])):
        gs = b.get("game_states") or {}
        if "time" not in gs or len(gs["time"]) < 2:
            continue
        t = np.asarray(gs["time"], float)
        o = np.argsort(t)
        arr = {"time": t[o]}
        for k in ["ball_x", "ball_y", "ball_vx", "ball_vy", "camera_y"]:
            arr[k] = (np.asarray(gs[k], float)[o] if k in gs
                      else np.full(len(t), np.nan))
        out[("i", i)] = arr
        key = b.get("block_index", i)
        out[("b", key)] = arr
    return out


def phase_decompose(states, t0, t1, eps_dy=EPS_DY, eps_dx=EPS_DX):
    """Decompose [t0, t1] ms into fall / roll / hold time and transport stats."""
    t = states["time"]
    m = (t >= t0) & (t <= t1)
    if m.sum() < 3:
        return None
    tt = t[m]
    xx = states["ball_x"][m]
    yy = states["ball_y"][m]
    vx = states["ball_vx"][m]
    vy = states["ball_vy"][m]
    dt = np.diff(tt)
    dx = np.diff(xx)
    dy = np.diff(yy)
    good = dt > 0
    dt, dx, dy = dt[good], dx[good], dy[good]
    if len(dt) == 0 or dt.sum() <= 0:
        return None

    airborne = np.abs(dy) > eps_dy
    roll = (~airborne) & (np.abs(dx) > eps_dx)
    hold = (~airborne) & (~roll)

    roll_t = float(dt[roll].sum())
    roll_path = float(np.abs(dx[roll]).sum())
    contact = ~airborne
    n_contacts = int(np.sum(contact[1:] & airborne[:-1])) if len(dt) > 1 else 0
    return {
        "dur_ms": float(dt.sum()),
        "fall_ms": float(dt[airborne].sum()),
        "roll_ms": roll_t,
        "hold_ms": float(dt[hold].sum()),
        "net_dx": float(xx[-1] - xx[0]),
        "x_path": float(np.abs(dx).sum()),
        "vx_entry": float(vx[0]) if len(vx) else np.nan,
        "vy_entry": float(vy[0]) if len(vy) else np.nan,
        "roll_speed": (roll_path / roll_t) if roll_t > 0 else np.nan,
        "n_contacts": n_contacts,
    }


def _add_prefixed(row, prefix, d):
    for k, v in d.items():
        row[f"{prefix}_{k}"] = v


def covers(states, t0, t1, tol=1.0):
    """True if the block's frame clock fully spans [t0, t1] (ms).

    Some online blocks only log `game_states` for the tail of the block, so a
    trial can straddle the buffer edge; such trials must be skipped rather than
    decomposed on a truncated window.
    """
    t = states["time"]
    return len(t) > 1 and t[0] <= t0 + tol and t[-1] >= t1 - tol


# %% [markdown]
# ## Cohort builders

# %%
def online_file_map():
    files = {}
    for _, ds in pe.DATASETS.items():
        for fp in (pe._REPO_ROOT / ds.folder).glob("*_cleaned.json"):
            files[pe._participant_id(fp)] = (fp, ds.name)
    return files


def online_frames():
    tbl, _ = pe.load_all()
    exp = tbl[(tbl["is_experimental"]) & (tbl["conflict"] == 1)
              & (tbl["threat"].isin(["drift", "alternating"]))].copy()
    exp = exp.dropna(subset=["rt_decision", "rt_exec", "entry_time_ms",
                             "choice_time_ms", "exit_time_ms"])
    exp = exp[(exp["rt_decision"] > 0) & (exp["rt_exec"] > 0)].copy()

    files = online_file_map()

    rows = []
    n_skip = 0
    for pid, g in exp.groupby("participant"):
        fp, dsname = files[pid]
        states = load_block_states(fp)
        for r in g.itertuples():
            st = states.get(("i", int(r.block_number)))
            if st is None or not covers(st, r.entry_time_ms, r.exit_time_ms):
                n_skip += 1
                continue
            dec = phase_decompose(st, r.entry_time_ms, r.choice_time_ms)
            exe = phase_decompose(st, r.choice_time_ms, r.exit_time_ms)
            if dec is None or exe is None:
                n_skip += 1
                continue
            row = {
                "cohort": "online", "dataset": dsname, "participant": pid,
                "unit": pid,
                "block_number": int(r.block_number),
                "sequence_index": int(r.sequence_index),
                "chose_planning": int(r.chose_planning),
                "chosen_1step_dist": float(r.chosen_1step_dist),
                "chosen_2step_dist": float(r.chosen_2step_dist),
                "second_step": float(r.chosen_2step_dist - r.chosen_1step_dist),
                "rt_decision": float(r.rt_decision),
                "rt_exec": float(r.rt_exec),
                "entry_time_ms": float(r.entry_time_ms),
                "choice_time_ms": float(r.choice_time_ms),
                "exit_time_ms": float(r.exit_time_ms),
            }
            _add_prefixed(row, "dec", dec)
            _add_prefixed(row, "exec", exe)
            rows.append(row)
    print(f"  online: {len(rows)} trials with full trajectory coverage, "
          f"{n_skip} skipped")
    return pd.DataFrame(rows)


def emu_frames():
    from neural_common import RUNS
    rows = []
    for rid in RUN_ORDER:
        run = RUNS[rid]
        out = _REPO / "analysis" / "neural_outputs" / rid
        tt = pd.read_csv(out / "trial_table.csv")
        lab = pd.read_csv(out / "trial_labels.csv")
        df = tt.merge(lab, on="trial_id", suffixes=("", "_lab"), how="inner")
        df = df[df["block_index"] >= 4].copy()
        df = df[df["greedy_optimal_hole"] != df["planning_optimal_hole"]].copy()
        if df.empty:
            continue
        df["chose_planning"] = (df["choice_hole"]
                                == df["planning_optimal_hole"]).astype(int)
        df["chosen_1step_dist"] = (df["choice_hole"]
                                   - df["entry_hole"]).abs()
        df["chosen_2step_dist"] = (df["chosen_1step_dist"]
                                   + (df["choice_hole"] - df["goal_hole"]).abs())
        df["second_step"] = df["chosen_2step_dist"] - df["chosen_1step_dist"]
        df["rt_decision"] = df["choice_time_ms"] - df["entry_time_ms"]
        df["rt_exec"] = df["exit_time_ms"] - df["choice_time_ms"]
        df = df[(df["rt_decision"] > 0) & (df["rt_exec"] > 0)]

        states = load_block_states(run.behavior_path)
        for r in df.itertuples():
            st = states.get(("b", int(r.block_index)))
            if st is None or not covers(st, r.entry_time_ms, r.exit_time_ms):
                continue
            dec = phase_decompose(st, r.entry_time_ms, r.choice_time_ms)
            exe = phase_decompose(st, r.choice_time_ms, r.exit_time_ms)
            if dec is None or exe is None:
                continue
            row = {
                "cohort": "EMU", "dataset": rid, "participant": run.participant,
                "unit": rid,
                "block_number": int(r.block_index),
                "sequence_index": int(r.sequence_index),
                "chose_planning": int(r.chose_planning),
                "chosen_1step_dist": float(r.chosen_1step_dist),
                "chosen_2step_dist": float(r.chosen_2step_dist),
                "second_step": float(r.second_step),
                "rt_decision": float(r.rt_decision),
                "rt_exec": float(r.rt_exec),
                "entry_time_ms": float(r.entry_time_ms),
                "choice_time_ms": float(r.choice_time_ms),
                "exit_time_ms": float(r.exit_time_ms),
            }
            _add_prefixed(row, "dec", dec)
            _add_prefixed(row, "exec", exe)
            rows.append(row)
    print(f"  EMU: {len(rows)} trials with full trajectory coverage")
    return pd.DataFrame(rows)


# %% [markdown]
# ## Summaries and tests

# %%
def phase_summary(df):
    rows = []
    for cohort, g in df.groupby("cohort"):
        for interval in ["dec", "exec"]:
            for plan, gg in g.groupby("chose_planning"):
                dur = gg[f"{interval}_dur_ms"].mean()
                row = {
                    "cohort": cohort, "interval": interval,
                    "choice": "planning" if plan else "greedy",
                    "n": len(gg),
                    "dur_ms": dur,
                    "fall_ms": gg[f"{interval}_fall_ms"].mean(),
                    "roll_ms": gg[f"{interval}_roll_ms"].mean(),
                    "hold_ms": gg[f"{interval}_hold_ms"].mean(),
                    "x_path": gg[f"{interval}_x_path"].mean(),
                    "n_contacts": gg[f"{interval}_n_contacts"].mean(),
                    "net_dx": gg[f"{interval}_net_dx"].mean(),
                }
                for ph in ["fall", "roll", "hold"]:
                    row[f"{ph}_frac"] = (gg[f"{interval}_{ph}_ms"].mean() / dur
                                         if dur > 0 else np.nan)
                rows.append(row)
    return pd.DataFrame(rows)


def participant_means(df):
    rows = []
    for (cohort, unit), g in df.groupby(["cohort", "unit"]):
        for interval in ["dec", "exec"]:
            row = {"cohort": cohort, "unit": unit, "interval": interval}
            for plan in [0, 1]:
                gg = g[g["chose_planning"] == plan]
                tag = "plan" if plan else "greed"
                for ph in ["fall", "roll", "hold", "dur"]:
                    row[f"{tag}_{ph}_ms"] = (
                        gg[f"{interval}_{ph}_ms"].mean() if len(gg) else np.nan)
                row[f"{tag}_n"] = len(gg)
            rows.append(row)
    return pd.DataFrame(rows)


def dwell_tests(pm):
    """Paired planning-greedy hold-time difference per participant/interval."""
    rows = []
    for cohort, g in pm.groupby("cohort"):
        for interval in ["dec", "exec"]:
            gg = g[g["interval"] == interval].dropna(
                subset=[f"plan_hold_ms", f"greed_hold_ms"])
            d = gg[f"plan_hold_ms"] - gg[f"greed_hold_ms"]
            if len(d) >= 2:
                t, p = stats.ttest_rel(gg[f"plan_hold_ms"], gg[f"greed_hold_ms"])
            else:
                t, p = np.nan, np.nan
            rows.append({
                "cohort": cohort, "interval": interval, "n_participants": len(d),
                "plan_hold_ms": gg[f"plan_hold_ms"].mean(),
                "greed_hold_ms": gg[f"greed_hold_ms"].mean(),
                "diff_ms": d.mean(), "sd_ms": d.std(ddof=1),
                "t": t, "p": p,
                "n_positive": int((d > 0).sum()),
            })
    return pd.DataFrame(rows)


def speed_models(df):
    rows = []
    for cohort, g in df.groupby("cohort"):
        for interval in ["dec", "exec"]:
            gg = g.dropna(subset=[f"{interval}_roll_ms", f"{interval}_x_path",
                                  f"{interval}_vx_entry", "chose_planning"])
            gg = gg[gg[f"{interval}_roll_ms"] > 0].copy()
            if gg["chose_planning"].nunique() < 2 or len(gg) < 30:
                continue
            # transport model: rolling time from path length (+ entry speed)
            try:
                m = smf.gee(
                    f"{interval}_roll_ms ~ chose_planning + {interval}_x_path "
                    f"+ {interval}_vx_entry",
                    groups="unit", data=gg,
                    family=sm.families.Gaussian(),
                    cov_struct=sm.cov_struct.Exchangeable()).fit()
                rows.append({
                    "cohort": cohort, "interval": interval,
                    "outcome": f"{interval}_roll_ms", "n": len(gg),
                    "coef_planning": m.params.get("chose_planning"),
                    "p_planning": m.pvalues.get("chose_planning"),
                    "coef_x_path": m.params.get(f"{interval}_x_path"),
                    "coef_vx_entry": m.params.get(f"{interval}_vx_entry"),
                })
            except Exception as exc:  # pragma: no cover - numerical guard
                print(f"    speed model failed ({cohort}/{interval}): {exc}")
    return pd.DataFrame(rows)


def within_trial_corr(df):
    rows = []
    for (cohort, unit), g in df.groupby(["cohort", "unit"]):
        if len(g) < 20:
            continue
        r_all = np.corrcoef(g["rt_decision"], g["rt_exec"])[0, 1]
        row = {"cohort": cohort, "unit": unit, "n": len(g),
               "r_dec_exec_all": r_all}
        for plan in [0, 1]:
            gg = g[g["chose_planning"] == plan]
            row[f"r_dec_exec_{'plan' if plan else 'greed'}"] = (
                np.corrcoef(gg["rt_decision"], gg["rt_exec"])[0, 1]
                if len(gg) >= 15 else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def geometry_separation(df):
    rows = []
    for cohort, g in df.groupby("cohort"):
        cell = list(zip(g["chosen_1step_dist"].astype(int),
                        g["chosen_2step_dist"].astype(int)))
        tab = pd.DataFrame({"cell": cell,
                            "plan": g["chose_planning"].to_numpy()})
        n_cells = tab["cell"].nunique()
        both = tab.groupby("cell")["plan"].agg(["min", "max"])
        n_both = int(((both["min"] == 0) & (both["max"] == 1)).sum())
        # coarse bins: d1 in 2-px bands, second_step in 2-px bands
        d1b = (g["chosen_1step_dist"] // 2).astype(int)
        ssb = (g["second_step"] // 2).astype(int)
        ctab = pd.DataFrame({"d1": d1b, "ss": ssb,
                             "plan": g["chose_planning"].to_numpy()})
        cboth = ctab.groupby(["d1", "ss"])["plan"].agg(["min", "max"])
        n_cboth = int(((cboth["min"] == 0) & (cboth["max"] == 1)).sum())
        rows.append({"cohort": cohort, "n_cells": n_cells,
                     "cells_both": n_both,
                     "n_coarse_cells": int(len(cboth)),
                     "coarse_cells_both": n_cboth})
    return pd.DataFrame(rows)


# nested specifications: does the planning coefficient survive progressively
# stricter kinematic control? (velocity controls test momentum carryover)
CARRYOVER_SPECS = [
    ("M0 raw", "{y} ~ chose_planning", "ms"),
    ("M1 +path", "{y} ~ chose_planning + {i}_x_path", "ms"),
    ("M2 +vx", "{y} ~ chose_planning + {i}_x_path + {i}_vx_entry", "ms"),
    ("M3 +|vx|+vy", "{y} ~ chose_planning + {i}_x_path + {i}_vx_entry "
                    "+ abs_{i}_vx + {i}_vy_entry", "ms"),
    ("M4 +decision_roll", "{y} ~ chose_planning + {i}_x_path + {i}_vx_entry "
                          "+ {other}_roll_ms", "ms"),
    ("M5 log-log", "np.log({y}) ~ chose_planning + np.log({i}_x_path) "
                   "+ {i}_vx_entry", "log"),
    ("M6 spline path", "{y} ~ chose_planning + bs({i}_x_path, df=4) "
                       "+ {i}_vx_entry", "ms"),
    ("M7 +overshoot", "{y} ~ chose_planning + {i}_x_path + {i}_vx_entry + over",
     "ms"),
]


def carryover_models(df):
    d = df.copy()
    for i in ["dec", "exec"]:
        d[f"abs_{i}_vx"] = d[f"{i}_vx_entry"].abs()
    d["over"] = d["exec_x_path"] / d["exec_net_dx"].abs().clip(lower=1)
    rows = []
    for cohort, g in d.groupby("cohort"):
        for i in ["dec", "exec"]:
            other = "exec" if i == "dec" else "dec"
            y = f"{i}_roll_ms"
            gg = g[g[y] > 0].dropna(subset=[f"{i}_x_path", f"{i}_vx_entry",
                                            f"{i}_vy_entry", "chose_planning"])
            if gg["chose_planning"].nunique() < 2 or len(gg) < 30:
                continue
            base = None
            for name, tmpl, scale in CARRYOVER_SPECS:
                formula = tmpl.format(y=y, i=i, other=other)
                try:
                    m = smf.gee(formula, groups="unit", data=gg,
                                family=sm.families.Gaussian(),
                                cov_struct=sm.cov_struct.Exchangeable()).fit()
                    coef = m.params["chose_planning"]
                    p = m.pvalues["chose_planning"]
                except Exception as exc:  # pragma: no cover
                    print(f"    carryover {name} failed ({cohort}/{i}): {exc}")
                    continue
                if name == "M0 raw":
                    base = coef
                rows.append({
                    "cohort": cohort, "interval": i, "spec": name,
                    "scale": scale, "n": len(gg),
                    "coef_planning": coef, "p_planning": p,
                    "pct_explained": (np.nan if (scale == "log" or base in
                                                 (None, 0))
                                      else 100 * (1 - coef / base)),
                })
    return pd.DataFrame(rows)


def bin_overlap(df, n_bins=10):
    """Within path-length deciles, how much planning/greedy overlap exists?"""
    rows = []
    for cohort, g in df.groupby("cohort"):
        g = g.copy()
        g["pb"] = pd.qcut(g["exec_x_path"].rank(method="first"), n_bins,
                          labels=False, duplicates="drop")
        tab = g.groupby("pb")["chose_planning"].agg(["size", "mean"])
        both = tab[(tab["mean"] > 0) & (tab["mean"] < 1)].index
        diffs = []
        for b in both:
            gg = g[g["pb"] == b]
            diffs.append(gg[gg["chose_planning"] == 1]["exec_roll_ms"].mean()
                         - gg[gg["chose_planning"] == 0]["exec_roll_ms"].mean())
        rows.append({
            "cohort": cohort, "n_bins": n_bins,
            "overlap_bins": int(len(both)),
            "within_bin_diff_ms": float(np.mean(diffs)) if diffs else np.nan,
        })
    return pd.DataFrame(rows)


# %% [markdown]
# ## Figures

# %%
def _stacked_by_choice(ax, sub, interval):
    plans = ["greedy", "planning"]
    x = np.arange(2)
    bottoms = np.zeros(2)
    for ph, color in [("fall", C_FALL), ("hold", C_HOLD), ("roll", C_ROLL)]:
        vals = []
        for pl in plans:
            r = sub[(sub["interval"] == interval)
                    & (sub["choice"] == pl)]
            vals.append(float(r[f"{ph}_ms"].iloc[0]) if len(r) else np.nan)
        vals = np.array(vals)
        ax.bar(x, vals, bottom=bottoms, color=color, label=ph, edgecolor="k",
               linewidth=0.4)
        bottoms += vals
    for i, pl in enumerate(plans):
        tot = sub[(sub["interval"] == interval)
                  & (sub["choice"] == pl)]["dur_ms"]
        if len(tot):
            ax.text(i, bottoms[i], f"{tot.iloc[0]:.0f}", ha="center",
                    va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(plans)
    ax.set_ylabel(f"{'decision' if interval == 'dec' else 'execution'} time (ms)")
    ax.legend(fontsize=8)


def make_figures(df, summ, pm, wt, geo):
    # online + EMU phase composition
    for cohort, fname in [("online", "fig_timing_phases_online.png"),
                          ("EMU", "fig_timing_phases_emu.png")]:
        sub = summ[summ["cohort"] == cohort]
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
        _stacked_by_choice(axes[0], sub, "dec")
        _stacked_by_choice(axes[1], sub, "exec")
        axes[0].set_title("Decision interval (entry -> choice)")
        axes[1].set_title("Execution interval (choice -> exit)")
        fig.suptitle(f"{cohort}: trajectory-phase decomposition of the RT "
                     "trade-off", fontsize=12)
        fig.tight_layout()
        fig.savefig(ONLINE_OUT / "figures" / fname, dpi=140,
                    bbox_inches="tight")
        plt.close(fig)

    # combined 2x2
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for ax, cohort in [(axes[0, 0], "online"), (axes[0, 1], "EMU")]:
        _stacked_by_choice(ax, summ[summ["cohort"] == cohort], "dec")
        ax.set_title(f"{cohort}: decision phases")
    for ax, cohort in [(axes[1, 0], "online"), (axes[1, 1], "EMU")]:
        _stacked_by_choice(ax, summ[summ["cohort"] == cohort], "exec")
        ax.set_title(f"{cohort}: execution phases")
    fig.suptitle("Planning vs greedy timing is transport (roll), not dwell",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(ONLINE_OUT / "figures" / "fig_timing_phases_both.png", dpi=140,
                bbox_inches="tight")
    plt.close(fig)

    # diagnostic: per-participant hold difference + within-trial correlation
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for cohort, color in [("online", "0.4"), ("EMU", C_PLAN)]:
        g = pm[(pm["cohort"] == cohort) & (pm["interval"] == "dec")]
        if g.empty:
            continue
        d = g["plan_hold_ms"] - g["greed_hold_ms"]
        axes[0].scatter(np.full(len(d), 0.0 if cohort == "online" else 1.0)
                        + np.random.uniform(-0.06, 0.06, len(d)),
                        d, color=color, alpha=0.7,
                        label=cohort)
    axes[0].axhline(0, color="k", ls="--", lw=1)
    axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(["online", "EMU"])
    axes[0].set_ylabel("planning - greedy hold time (ms)")
    axes[0].set_title("Dwell difference per participant")
    axes[0].legend(fontsize=8)

    for cohort, color in [("online", "0.4"), ("EMU", C_PLAN)]:
        g = wt[wt["cohort"] == cohort]
        if g.empty:
            continue
        axes[1].scatter(g["r_dec_exec_all"],
                        g["r_dec_exec_plan"].astype(float), color=color,
                        alpha=0.7, label=cohort)
    axes[1].axhline(0, color="k", ls="--", lw=1)
    axes[1].set_xlabel("within-participant r(decision, execution), all trials")
    axes[1].set_ylabel("same, planning trials only")
    axes[1].set_title("No within-trial dec/exec trade-off")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ONLINE_OUT / "figures" / "fig_timing_diagnostics.png", dpi=140,
                bbox_inches="tight")
    plt.close(fig)


PHASE_COLORS = {"fall": C_FALL, "roll": C_ROLL, "hold": C_HOLD}
PHASE_ORDER = ["fall", "roll", "hold"]


def phase_labels(t, x, y, eps_dy=EPS_DY, eps_dx=EPS_DX):
    dy = np.diff(y)
    dx = np.diff(x)
    return np.where(np.abs(dy) > eps_dy, "fall",
                    np.where(np.abs(dx) > eps_dx, "roll", "hold"))


def make_example_figure(on):
    """Overlay detected phases on example planning/greedy trajectories."""
    from matplotlib.patches import Patch
    files = online_file_map()
    pid = on["unit"].value_counts().index[0]
    fp, _ = files[pid]
    states = load_block_states(fp)
    g = on[on["unit"] == pid]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for col, plan in enumerate([0, 1]):
        gg = g[g["chose_planning"] == plan]
        if gg.empty:
            continue
        r = gg.iloc[0]
        st = states[("i", int(r["block_number"]))]
        t, x, y = st["time"], st["ball_x"], st["ball_y"]
        m = (t >= r["entry_time_ms"] - 30) & (t <= r["exit_time_ms"] + 30)
        tt, xx, yy = t[m], x[m], y[m]
        lab = phase_labels(tt, xx, yy)
        for ax in (axes[0, col], axes[1, col]):
            for i, ph in enumerate(lab):
                ax.axvspan(tt[i], tt[i + 1], color=PHASE_COLORS[ph],
                           alpha=0.35, lw=0)
            for tv, c in [(r["entry_time_ms"], "k"),
                          (r["choice_time_ms"], "m"),
                          (r["exit_time_ms"], "k")]:
                ax.axvline(tv, color=c, ls="--", lw=1)
            ax.set_xlim(tt[0], tt[-1])
        axes[0, col].plot(tt, yy, color="k", lw=1)
        axes[1, col].plot(tt, xx, color="k", lw=1)
        axes[0, col].set_title(
            ("planning" if plan else "greedy")
            + f"  (dec {r['rt_decision']:.0f} ms, exec {r['rt_exec']:.0f} ms)")
        axes[1, col].set_xlabel("time (ms)")
    axes[0, 0].set_ylabel("ball_y (px)")
    axes[1, 0].set_ylabel("ball_x (px)")
    handles = [Patch(color=PHASE_COLORS[p], alpha=0.35, label=p)
               for p in PHASE_ORDER]
    axes[0, 0].legend(handles=handles, fontsize=8, loc="upper right")
    fig.suptitle("Phase detection on example conflict trials (participant "
                 f"{pid[:8]})", fontsize=12)
    fig.tight_layout()
    fig.savefig(ONLINE_OUT / "figures" / "fig_timing_example_trajectories.png",
                dpi=140, bbox_inches="tight")
    plt.close(fig)


def make_carryover_figure(carry):
    specs = [s[0] for s in CARRYOVER_SPECS if s[2] == "ms"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, interval in zip(axes, ["dec", "exec"]):
        for cohort, color, mk in [("online", "0.35", "o"), ("EMU", C_PLAN, "s")]:
            sub = carry[(carry["cohort"] == cohort)
                        & (carry["interval"] == interval)]
            if sub.empty:
                continue
            sub = sub.set_index("spec").reindex(specs)
            ax.plot(np.arange(len(specs)), sub["coef_planning"], color=color,
                    marker=mk, label=cohort)
        ax.axhline(0, color="k", ls="--", lw=1)
        ax.set_xticks(np.arange(len(specs)))
        ax.set_xticklabels([s.replace(" ", "\n") for s in specs], fontsize=7)
        ax.set_title(("decision" if interval == "dec" else "execution")
                     + " interval")
        ax.set_ylabel("planning coefficient on roll time (ms)")
    axes[1].legend(fontsize=8)
    fig.suptitle("Planning transport residual is specification-dependent",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(ONLINE_OUT / "figures" / "fig_timing_carryover.png", dpi=140,
                bbox_inches="tight")
    plt.close(fig)


# %% [markdown]
# ## Main

# %%
def main(from_cache=False):
    ONLINE_OUT.mkdir(parents=True, exist_ok=True)
    AGG_OUT.mkdir(parents=True, exist_ok=True)
    (ONLINE_OUT / "figures").mkdir(parents=True, exist_ok=True)

    on_cache = ONLINE_OUT / "timing_phases_online.csv"
    em_cache = AGG_OUT / "timing_phases_emu.csv"
    if from_cache and on_cache.exists() and em_cache.exists():
        print("Loading cached trajectory phases ...")
        on = pd.read_csv(on_cache)
        em = pd.read_csv(em_cache)
    else:
        print("Building online trajectory phases ...")
        on = online_frames()
        print(f"  {len(on)} online conflict trials, "
              f"{on['participant'].nunique()} participants")
        on.to_csv(on_cache, index=False)

        print("Building EMU trajectory phases ...")
        em = emu_frames()
        print(f"  {len(em)} EMU conflict trials, {em['dataset'].nunique()} runs")
        em.to_csv(em_cache, index=False)

    df = pd.concat([on, em], ignore_index=True)

    summ = phase_summary(df)
    pm = participant_means(df)
    dwell = dwell_tests(pm)
    speed = speed_models(df)
    wt = within_trial_corr(df)
    geo = geometry_separation(df)
    carry = carryover_models(df)
    bino = bin_overlap(df)

    summ.to_csv(ONLINE_OUT / "timing_phase_summary.csv", index=False)
    pm.to_csv(ONLINE_OUT / "timing_participant_means.csv", index=False)
    dwell.to_csv(ONLINE_OUT / "timing_dwell_tests.csv", index=False)
    speed.to_csv(ONLINE_OUT / "timing_speed_model.csv", index=False)
    wt.to_csv(ONLINE_OUT / "timing_within_trial_corr.csv", index=False)
    geo.to_csv(ONLINE_OUT / "timing_geometry_separation.csv", index=False)
    carry.to_csv(ONLINE_OUT / "timing_carryover_models.csv", index=False)
    bino.to_csv(ONLINE_OUT / "timing_bin_overlap.csv", index=False)

    make_figures(df, summ, pm, wt, geo)
    make_example_figure(on)
    make_carryover_figure(carry)

    print("\nPhase summary (ms):")
    print(summ.round(1).to_string(index=False))
    print("\nDwell tests (planning - greedy hold):")
    print(dwell.round(2).to_string(index=False))
    print("\nGeometry separation (cells containing both choices):")
    print(geo.to_string(index=False))
    print("\nCarryover models (planning coefficient by control set):")
    print(carry.round(3).to_string(index=False))
    print("\nPath-bin overlap (execution):")
    print(bino.round(1).to_string(index=False))
    print("\nWithin-trial dec/exec correlation by participant:")
    print(wt.round(3).to_string(index=False))
    print("\nWrote timing CSVs to", ONLINE_OUT, "and", AGG_OUT)


if __name__ == "__main__":
    main("--from-cache" in sys.argv)
