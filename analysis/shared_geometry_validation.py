# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format: .py
#     jupytext_version: 1.19.4
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tier 1.1 — shared-geometry design validation
#
# Validates the level pools produced by
# `tools/level_generation/generating_levels_sharedgeom.py`, and measures the
# baseline confound in the current level pool.
#
# For every conflict trial compute the `(1-step, 2-step)` distance profile of
# the greedy-optimal and planning-optimal choices. A pool is "shared geometry"
# to the extent those profile sets **overlap** (the property absent in current
# data: 0 shared cells).
#
# Outputs:
#   analysis/planning_outputs/shared_geometry_validation.csv
#   analysis/planning_outputs/figures/fig_shared_geometry_validation.png

# %%
import json
import random
import sys
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "tools" / "level_generation"))
from generating_levels_sharedgeom import (  # noqa: E402
    VARIANTS, greedy_cost, planning_cost, profile,
)

PO = _REPO / "analysis" / "planning_outputs"
CURRENT = _REPO / "data" / "generated_levels" / "trials_new.json"
N_GEN = 400


def _holes(level):
    if isinstance(level, dict):
        return level.get("holes", [])
    return list(level)


def trial_profiles(levels):
    """(greedy_profile, planning_profile) for a 1-2-1 level triple, or None."""
    try:
        entry = _holes(levels[0])[0]
        holes = _holes(levels[1])
        goal = _holes(levels[-1])[0]
    except (IndexError, TypeError, KeyError):
        return None
    if len(holes) != 2:
        return None
    a, b = holes
    ga, gb = greedy_cost(entry, a, goal), greedy_cost(entry, b, goal)
    pa, pb = planning_cost(entry, a, goal), planning_cost(entry, b, goal)
    if ga == gb:
        g_hole = a
    else:
        g_hole = a if ga < gb else b
    p_hole = a if pa < pb else b
    conflict = g_hole != p_hole
    return {"conflict": conflict, "entry": entry, "goal": goal, "a": a, "b": b,
            "greedy_profile": profile(entry, g_hole, goal),
            "planning_profile": profile(entry, p_hole, goal)}


def analyse_pool(trials):
    recs = [trial_profiles(t.get("levels")) for t in trials]
    recs = [r for r in recs if r is not None]
    conf = [r for r in recs if r["conflict"]]
    gcells = Counter(r["greedy_profile"] for r in conf)
    pcells = Counter(r["planning_profile"] for r in conf)
    shared = set(gcells) & set(pcells)
    # matched-pair check: greedy profile of one trial == planning profile of another
    pset = Counter(r["planning_profile"] for r in conf)
    g_match = sum(1 for r in conf if pset.get(r["greedy_profile"], 0) > 0)
    return {
        "n_trials": len(recs), "n_conflict": len(conf),
        "n_agreement": len(recs) - len(conf),
        "conflict_rate": len(conf) / len(recs) if recs else np.nan,
        "n_greedy_cells": len(gcells), "n_planning_cells": len(pcells),
        "n_shared_cells": len(shared),
        "frac_shared_of_union": len(shared) / len(set(gcells) | set(pcells))
        if (gcells or pcells) else np.nan,
        "frac_conflict_greedy_matched_to_some_planning": g_match / len(conf)
        if conf else np.nan,
    }


def make_figure(rows):
    fig, ax = plt.subplots(figsize=(9, 5))
    names = [r["pool"] for r in rows]
    x = np.arange(len(names))
    ax.bar(x, [r["n_shared_cells"] for r in rows], color="#2a9d8f")
    for i, r in enumerate(rows):
        ax.text(i, r["n_shared_cells"] + 0.2,
                f"{r['n_shared_cells']}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("shared (1-step, 2-step) geometry cells")
    ax.set_title("Shared geometry between greedy and planning chosen paths\n"
                 "(current pool vs design variants)")
    fig.tight_layout()
    PO.mkdir(parents=True, exist_ok=True)
    out = PO / "figures" / "fig_shared_geometry_validation.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    rows = []
    if CURRENT.exists():
        cur = json.load(open(CURRENT))
        r = analyse_pool(cur); r["pool"] = "current (trials_new.json)"
        rows.append(r)
    for name, fn in VARIANTS.items():
        rng = random.Random(0)
        trials = fn(N_GEN, rng)
        r = analyse_pool(trials); r["pool"] = name
        rows.append(r)
    df = pd.DataFrame(rows)[
        ["pool", "n_trials", "n_conflict", "n_agreement", "conflict_rate",
         "n_greedy_cells", "n_planning_cells", "n_shared_cells",
         "frac_shared_of_union",
         "frac_conflict_greedy_matched_to_some_planning"]]
    df.to_csv(PO / "shared_geometry_validation.csv", index=False)
    fig = make_figure(rows)
    pd.set_option("display.width", 220)
    print(df.round(3).to_string(index=False))
    print(f"\nwrote shared_geometry_validation.csv and {fig}")


if __name__ == "__main__":
    main()
