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
# # Shared-geometry level generator (Tier 1.1)
#
# The current 1-2-1 levels make planning and greedy choices geometrically
# distinct: in 0/48 `(1-step, 2-step)` distance cells do both policies' chosen
# paths appear, so choice and kinematics are collinear (the structural
# confound). This generator emits conflict levels under variants that *match*
# the two policies' chosen path geometry.
#
# Variants (see `docs/shared_geometry_design.md`):
#   * `matched_pairs` — pairs of conflict levels whose greedy-option profile in
#     one equals the planning-option profile in the other (and vice versa), so
#     across the pair the two policies' chosen paths are matched.
#   * `symmetric1step` — entry equidistant to both holes (1-step matched),
#     conflict from the 2-step; greedy requires a documented tie-break.
#   * `three_hop` — adds a hop so the policies prescribe equal-length paths.
#   * `cue_goal` — cue-coloured holes decouple policy from geometry (metadata).
#
# Level schema matches `generating_levels.py`: a trial dict with
# `levels` = [[entry], [holeA, holeB], [goal]] on the 0-11 segment grid.
#
# Run with:
#   python tools/level_generation/generating_levels_sharedgeom.py --variant matched_pairs --n 400

# %%
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

SEGMENTS = list(range(12))


def profile(entry, hole, goal):
    """(chosen 1-step, chosen 2-step) distance profile of a hole choice."""
    return (abs(entry - hole), abs(entry - hole) + abs(hole - goal))


def greedy_cost(entry, hole, goal, c=0):
    return abs(entry - hole)


def planning_cost(entry, hole, goal, c=0):
    return abs(entry - hole) + abs(hole - goal)


def make_trial(entry, a, b, goal, meta):
    return {"levels": [[entry], [a, b], [goal]], "metadata": meta}


# %%
def gen_matched_pairs(n, rng):
    """Generate matched conflict-level pairs.

    For each random (entry, a, b, goal) with the policies disagreeing, record
    its (greedy profile, planning profile). Then pair X, Y when
    X.greedy_profile == Y.planning_profile and X.planning_profile ==
    Y.greedy_profile, so the two policies' chosen paths are matched across the
    pair.
    """
    pool = []
    seen = set()
    tries = 0
    while len(pool) < n * 20 and tries < 200000:
        tries += 1
        entry = rng.randint(1, 10)
        goal = rng.randint(1, 10)
        a = rng.randint(0, 11)
        b = rng.randint(0, 11)
        if a == b:
            continue
        ga, gb = greedy_cost(entry, a, goal), greedy_cost(entry, b, goal)
        pa, pb = planning_cost(entry, a, goal), planning_cost(entry, b, goal)
        if ga == gb or pa == pb:
            continue
        greedy_hole = a if ga < gb else b
        planning_hole = a if pa < pb else b
        if greedy_hole == planning_hole:
            continue  # agreement, not conflict
        gp = profile(entry, greedy_hole, goal)
        pp = profile(entry, planning_hole, goal)
        key = (entry, a, b, goal)
        if key in seen:
            continue
        seen.add(key)
        pool.append({"entry": entry, "a": a, "b": b, "goal": goal,
                     "greedy_hole": greedy_hole, "planning_hole": planning_hole,
                     "greedy_profile": gp, "planning_profile": pp})

    by_gp = defaultdict(list)
    for t in pool:
        by_gp[t["greedy_profile"]].append(t)

    trials, used = [], set()
    for x in pool:
        if len(trials) >= n:
            break
        if (x["entry"], x["a"], x["b"], x["goal"]) in used:
            continue
        # find y whose planning profile matches x's greedy profile and whose
        # greedy profile matches x's planning profile
        cands = [y for y in by_gp.get(x["planning_profile"], [])
                 if y["planning_profile"] == x["greedy_profile"]
                 and (y["entry"], y["a"], y["b"], y["goal"]) not in used
                 and y is not x]
        if not cands:
            continue
        y = cands[0]
        used.add((x["entry"], x["a"], x["b"], x["goal"]))
        used.add((y["entry"], y["a"], y["b"], y["goal"]))
        meta = {"variant": "matched_pairs", "greedy_choice": x["greedy_hole"],
                "planner_choice": x["planning_hole"], "pair_id": len(trials) // 2,
                "greedy_profile": x["greedy_profile"],
                "planning_profile": x["planning_profile"]}
        trials.append(make_trial(x["entry"], x["a"], x["b"], x["goal"], meta))
        meta_y = dict(meta, greedy_choice=y["greedy_hole"],
                      planner_choice=y["planning_hole"],
                      greedy_profile=y["greedy_profile"],
                      planning_profile=y["planning_profile"])
        trials.append(make_trial(y["entry"], y["a"], y["b"], y["goal"], meta_y))
    for i, t in enumerate(trials):
        t["trial_id"] = i + 1
    return trials


def gen_symmetric1step(n, rng):
    """Entry equidistant to both holes; conflict from the 2-step geometry."""
    trials = []
    tries = 0
    while len(trials) < n and tries < 200000:
        tries += 1
        d = rng.randint(1, 5)
        entry = rng.randint(d, 11 - d)
        a, b = entry - d, entry + d
        if not (0 <= a and b <= 11):
            continue
        goal = rng.randint(0, 11)
        pa = abs(a - goal)
        pb = abs(b - goal)
        if pa == pb:
            continue  # 2-step tie -> no conflict
        planning_hole = a if pa < pb else b
        greedy_hole = a  # documented tie-break: leftmost
        meta = {"variant": "symmetric1step", "greedy_choice": greedy_hole,
                "planner_choice": planning_hole, "tie_break": "leftmost",
                "greedy_profile": profile(entry, greedy_hole, goal),
                "planning_profile": profile(entry, planning_hole, goal)}
        trials.append(make_trial(entry, a, b, goal, meta))
    for i, t in enumerate(trials):
        t["trial_id"] = i + 1
    return trials


def gen_three_hop(n, rng):
    """3-hop variant: holes at two choice levels so plan/greedy can prescribe
    equal-length paths. Emitted as an extended 4-level schema."""
    trials = []
    tries = 0
    while len(trials) < n and tries < 200000:
        tries += 1
        entry = rng.randint(1, 10)
        goal = rng.randint(1, 10)
        a = rng.randint(0, 11)
        b = rng.randint(0, 11)
        if a == b:
            continue
        # greedy = 1-step nearer; planning = full entry->a->goal vs entry->b->goal
        ga, gb = abs(entry - a), abs(entry - b)
        pa = abs(entry - a) + abs(a - goal)
        pb = abs(entry - b) + abs(b - goal)
        if ga == gb or pa == pb:
            continue
        g_hole = a if ga < gb else b
        p_hole = a if pa < pb else b
        if g_hole == p_hole:
            continue
        meta = {"variant": "three_hop", "greedy_choice": g_hole,
                "planner_choice": p_hole,
                "greedy_profile": profile(entry, g_hole, goal),
                "planning_profile": profile(entry, p_hole, goal)}
        trials.append({"levels": [[entry], [a, b], [goal]],
                       "metadata": meta, "trial_id": len(trials) + 1})
    return trials


def gen_cue_goal(n, rng):
    """Cue-goal variant: identical geometry, goal signalled by a cue (metadata
    only; requires app support for coloured/cued holes)."""
    trials = []
    tries = 0
    while len(trials) < n and tries < 200000:
        tries += 1
        entry = rng.randint(1, 10)
        d = rng.randint(2, 5)
        a, b = entry - d, entry + d
        if not (0 <= a and b <= 11):
            continue
        cue = rng.choice(["left", "right"])
        goal = rng.randint(0, 11)
        meta = {"variant": "cue_goal", "cue_side": cue,
                "greedy_choice": a, "planner_choice": a if cue == "left" else b,
                "note": "geometry identical; policy signalled by cue colour"}
        trials.append({"levels": [[entry], [a, b], [goal]],
                       "metadata": meta, "trial_id": len(trials) + 1})
    return trials


VARIANTS = {
    "matched_pairs": gen_matched_pairs,
    "symmetric1step": gen_symmetric1step,
    "three_hop": gen_three_hop,
    "cue_goal": gen_cue_goal,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS), default="matched_pairs")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    trials = VARIANTS[args.variant](args.n, rng)
    out = Path(args.out) if args.out else (
        Path("data/generated_levels") / f"sharedgeom_{args.variant}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(trials, f, indent=2)
    print(f"wrote {len(trials)} trials ({args.variant}) -> {out}")


if __name__ == "__main__":
    main()
