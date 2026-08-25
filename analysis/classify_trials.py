# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Classify trials: greedy/planning geometry, agreement, and conditions
#
# For every 1-2-1 sequence in the trial table, reconstruct the entry hole,
# choice holes, and goal/exit hole from the canonical `block_config.levels`,
# compute the greedy (1-step) and planning (2-step) costs for each choice hole,
# and label the trial's condition:
#
#   planning     : greedy and planning disagree, participant chose planning hole
#   greedy       : greedy and planning disagree, participant chose greedy hole
#   agree_optimal: greedy and planning agree,  participant chose the best hole
#   lapse        : greedy and planning agree,  participant chose the worse hole
#
# Deaths (trials with no pass-through event) are recovered from the RAW JSON
# (before keep-last dedup) and stamped with the time of the last event that
# preceded them.
#
# Output: analysis/neural_outputs/trial_labels.csv
#
# Run with:
#   C:\Users\manik\AppData\Local\Programs\Python\Python311\python.exe analysis\classify_trials.py

# %%
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------- Configuration ----------------------------
BEHAVIOR_PATH = Path(r"C:\Users\manik\Desktop\Obsidian\General Thoughts\Z Images and Files\Hennig Lab Project\falldown\data\emu\YFZ-2026-07-29T21-37-47-781Z-kdyd.json")
OUT_DIR = Path(r"C:\Users\manik\Desktop\Obsidian\General Thoughts\Z Images and Files\Hennig Lab Project\falldown\analysis\neural_outputs")
TRIAL_TABLE = OUT_DIR / "trial_table.csv"
MIN_EXPERIMENT_BLOCK = 4
# -----------------------------------------------------------------------


# %%
def greedy_cost(entry, hole):
    """1-step cost: distance from the entry hole to the choice hole."""
    return abs(entry - hole)


def planning_cost(entry, hole, goal):
    """2-step cost: entry->choice plus choice->goal."""
    return abs(entry - hole) + abs(hole - goal)


def build_condition_lookup():
    """Return dict keyed by (block_index, sequence_index) -> labels.

    Uses the canonical block_config.levels (3 levels per sequence) to get
    entry/choice/goal holes, and the trial table's choice_hole for what was
    actually chosen.
    """
    with open(BEHAVIOR_PATH, encoding="utf-8") as fh:
        data = json.load(fh)

    levels_by_block = {}
    for b in data["blocks"]:
        bi = b.get("block_index")
        if bi is None or bi < MIN_EXPERIMENT_BLOCK:
            continue
        levels_by_block.setdefault(bi, b.get("block_config", {}).get("levels", []))

    trials = pd.read_csv(TRIAL_TABLE)

    out = {}
    for _, row in trials.iterrows():
        bi = int(row["block_index"])
        s_global = int(row["sequence_index"])
        s_local = s_global % 35  # each experiment block has 35 sequences
        lv = levels_by_block[bi]
        entry = lv[3 * s_local][0]
        choice_holes = lv[3 * s_local + 1]
        goal = lv[3 * s_local + 2][0]
        chosen = int(row["choice_hole"])

        g_cost = {h: greedy_cost(entry, h) for h in choice_holes}
        p_cost = {h: planning_cost(entry, h, goal) for h in choice_holes}
        g_best = min(g_cost, key=g_cost.get)
        p_best = min(p_cost, key=p_cost.get)
        agree = g_best == p_best

        if agree:
            condition = "agree_optimal" if chosen == g_best else "lapse"
        else:
            condition = "planning" if chosen == p_best else (
                "greedy" if chosen == g_best else "other")

        out[(bi, s_global)] = {
            "entry_hole": entry,
            "goal_hole": goal,
            "choice_holes": choice_holes,
            "greedy_cost_L": g_cost[choice_holes[0]],
            "greedy_cost_R": g_cost[choice_holes[1]],
            "planning_cost_L": p_cost[choice_holes[0]],
            "planning_cost_R": p_cost[choice_holes[1]],
            "greedy_optimal_hole": g_best,
            "planning_optimal_hole": p_best,
            "agree": agree,
            "condition": condition,
        }
    return out


# %%
def find_deaths():
    """Detect genuine death moments from the ball trajectory (game_states).

    A death occurs when the ball falls to the bottom of the screen and the
    camera catches up: game-over triggers when `ball.y - cameraY < 0`
    (sketch.js). The trajectory signature is the ball freezing near the bottom
    while `ball.y - cameraY` approaches 0. We detect the moment this gap is
    minimal (< 5 px) — the death instant.

    IMPORTANT: the raw JSON lists 10 "death trials" (block 8, trials 54-63,
    empty events). These are NOT 10 deaths — they are 10 levels that scrolled
    past while the ball fell to the bottom during a single fatal fall. The
    trajectory shows exactly one death moment in this session (~758.9 s).

    Returns a list of dicts: block_index, death_time_ms, n_skipped_trials.
    """
    with open(BEHAVIOR_PATH, encoding="utf-8") as fh:
        data = json.load(fh)

    deaths = []
    for i, b in enumerate(data["blocks"]):
        bi = b.get("block_index")
        if bi is None or bi < MIN_EXPERIMENT_BLOCK:
            continue
        gs = b.get("game_states", {})
        if not gs:
            continue
        t = np.asarray(gs["time"])
        rel = np.asarray(gs["ball_y"]) - np.asarray(gs["camera_y"])
        if len(t) == 0:
            continue
        # ball frozen at/near the bottom while camera catches up => death
        # signature: rel reaches a small value and stays small to the end
        if rel[-1] < 5.0:
            k = int(np.argmin(rel))
            deaths.append({
                "block_index": bi,
                "death_time_ms": float(t[k]),
                "n_skipped_trials": int(sum(
                    1 for tt in b.get("trials", []) if not tt.get("events"))),
            })
    return deaths


# %%
def main():
    print("Classifying trials ...")
    lookup = build_condition_lookup()
    trials = pd.read_csv(TRIAL_TABLE)

    rows = []
    for _, row in trials.iterrows():
        bi = int(row["block_index"])
        s_global = int(row["sequence_index"])
        info = lookup[(bi, s_global)]
        rows.append({
            "trial_id": int(row["trial_id"]),
            "block_index": bi,
            "sequence_index": s_global,
            "entry_hole": info["entry_hole"],
            "goal_hole": info["goal_hole"],
            "choice_hole": int(row["choice_hole"]),
            "greedy_cost_L": info["greedy_cost_L"],
            "greedy_cost_R": info["greedy_cost_R"],
            "planning_cost_L": info["planning_cost_L"],
            "planning_cost_R": info["planning_cost_R"],
            "greedy_optimal_hole": info["greedy_optimal_hole"],
            "planning_optimal_hole": info["planning_optimal_hole"],
            "agree": info["agree"],
            "condition": info["condition"],
        })
    labels = pd.DataFrame(rows)

    deaths = find_deaths()
    death_df = pd.DataFrame(deaths)
    death_df.to_csv(OUT_DIR / "death_times.csv", index=False)

    labels.to_csv(OUT_DIR / "trial_labels.csv", index=False)

    print(f"{len(labels)} trials labeled")
    print(labels["condition"].value_counts().to_string())
    print(f"\n{len(death_df)} genuine death moment(s) found (see death_times.csv)")
    for _, r in death_df.iterrows():
        print(f"  block {r['block_index']}: death at {r['death_time_ms']:.0f} ms "
              f"({r['death_time_ms']/1000:.1f} s), "
              f"{int(r['n_skipped_trials'])} empty trials during the fatal fall")


if __name__ == "__main__":
    main()
