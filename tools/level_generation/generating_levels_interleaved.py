"""Generate balanced, interleaved one-maze-per-block experiments (stdlib only).

Run from the repo root: python tools/level_generation/generating_levels_interleaved.py
Counts must be multiples of 12: each shuffled packet contains four forced,
four two-path (2 agree / 2 conflict), and four randomized 1-2-2-1 trials (actual reachable path count is recorded).
"""
import argparse
from collections import Counter, defaultdict
from itertools import combinations, product
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[2]


def reachable_holes(holes, parent):
    left = [h for h in holes if h < parent]
    right = [h for h in holes if h > parent]
    return sorted(([max(left)] if left else []) + ([min(right)] if right else []) +
                  ([parent] if parent in holes else []))


def legal_paths(levels, parent=None):
    routes = [(h,) for h in levels[0]] if parent is None else [(parent,)]
    remaining = levels[1:] if parent is None else levels
    for holes in remaining:
        routes = [r + (h,) for r in routes for h in reachable_holes(holes, r[-1])]
    return routes if parent is None else [r[1:] for r in routes]


def describe(levels):
    """Enumerate paths; costs are horizontal travel in segment units.

    Only nearest-gap transitions count as physically valid paths.
    """
    paths = legal_paths(levels)
    costs = [sum(abs(b - a) for a, b in zip(p, p[1:])) for p in paths]
    best = min(costs)
    optimal = [list(p) for p, c in zip(paths, costs) if c == best]
    greedy = [levels[0][0]]
    greedy_tie = False
    for holes in levels[1:]:
        holes = reachable_holes(holes, greedy[-1])
        distances = [abs(h - greedy[-1]) for h in holes]
        choices = [h for h, d in zip(holes, distances) if d == min(distances)]
        greedy_tie |= len(choices) > 1
        greedy.append(min(choices))
    decisions = []
    for k in range(1, len(levels) - 1):
        if len(levels[k]) == 1:
            continue
        parent = optimal[0][k - 1]
        available = reachable_holes(levels[k], parent)
        immediate = [abs(h - parent) for h in available]
        totals = [min(c for p, c in zip(paths, costs)
                      if list(p[:k]) == optimal[0][:k] and p[k] == h)
                  for h in available]
        nearest = [h for h, d in zip(available, immediate) if d == min(immediate)]
        planned = [h for h, c in zip(available, totals) if c == min(totals)]
        decisions.append(dict(level_index=k, parent=parent,
                              greedy_holes=nearest, planning_holes=planned,
                              agreement=(nearest == planned),
                              reachable_holes=available,
                              greedy_margin=abs(immediate[0] - immediate[1]) if len(available) > 1 else None,
                              planning_margin=abs(totals[0] - totals[1]) if len(available) > 1 else None))
    greedy_cost = sum(abs(b - a) for a, b in zip(greedy, greedy[1:]))
    return dict(structure="-".join(str(len(h)) for h in levels),
                n_paths=len(paths), paths=[dict(holes=list(p), cost=c)
                                          for p, c in zip(paths, costs)],
                optimal_paths=optimal, optimal_cost=best,
                greedy_path=greedy, greedy_cost=greedy_cost,
                greedy_regret=greedy_cost - best, greedy_tie=greedy_tie,
                optimal_tie=len(optimal) > 1,
                greedy_path_optimal=greedy_cost == best,
                best_second_gap=sorted(costs)[1] - best if len(costs) > 1 else None,
                decisions_on_optimal_path=decisions)


def horizon_features(levels):
    states = []
    lookup = {}
    for k in range(1, len(levels)):
        for parent in levels[k - 1]:
            policies = {}
            for horizon in (1, 2, 3):
                values = {}
                for path in legal_paths(levels[k:k + horizon], parent):
                    cost = sum(abs(b - a) for a, b in zip((parent,) + path, path))
                    values[path[0]] = min(values.get(path[0], float('inf')), cost)
                best = min(values.values())
                actions = [a for a, cost in values.items() if cost == best]
                policies[str(horizon)] = dict(preferred_holes=actions,
                    action_excess_cost={str(a): cost - best for a, cost in values.items()})
                lookup[k, parent, horizon] = actions
            states.append(dict(level_index=k, parent=parent, horizons=policies))
    predictions = {}
    for horizon in (1, 2, 3):
        routes = [(levels[0][0],)]
        for k in range(1, len(levels)):
            routes = [r + (a,) for r in routes for a in lookup[k, r[-1], horizon]]
        predictions[str(horizon)] = [list(r) for r in routes]
    annotations = []
    for route in legal_paths(levels):
        choices = [dict(level_index=k, parent=route[k - 1], hole=route[k],
                        compatible_horizons=[h for h in (1, 2, 3)
                            if route[k] in lookup[k, route[k - 1], h]])
                   for k in range(1, len(levels))]
        annotations.append(dict(holes=list(route),
            compatible_horizons=[h for h in (1, 2, 3) if list(route) in predictions[str(h)]],
            choices=choices))
    return dict(horizon_predictions=predictions, horizon_states=states,
                route_horizon_annotations=annotations)


def condition(meta):
    decisions = meta['decisions_on_optimal_path']
    if not decisions:
        return 'forced'
    return '_'.join('agree' if d['agreement'] else 'conflict' for d in decisions)


def candidate_pools(segments, seed=20261116):
    pools = defaultdict(list)
    pairs = [(a, b) for a, b in combinations(range(segments), 2) if b - a >= 2]
    for entry in range(1, segments - 1):
        for goal in range(segments):
            # Avoid vertical drops in the forced baseline.
            for middle in range(segments):
                if entry != middle and middle != goal:
                    pools[('1-1-1', 'forced')].append([[entry], [middle], [goal]])
            for a, b in pairs:
                if not a < entry < b:
                    continue
                for second in (None,):
                    levels = [[entry], [a, b]]
                    if second is not None:
                        levels.append(list(second))
                    levels.append([goal])
                    meta = describe(levels)
                    # No arbitrary tie-breaking in scheduled policy conditions.
                    if meta['optimal_tie'] or meta['greedy_tie']:
                        continue
                    if any(len(d['greedy_holes']) != 1 or len(d['planning_holes']) != 1
                           for d in meta['decisions_on_optimal_path']):
                        continue
                    pools[(meta['structure'], condition(meta))].append(levels)
    # Independent hole pairs, without nesting or planning-distinctness filters.
    # A bounded seeded proposal pool avoids enumerating the entire geometry space.
    proposal_rng = random.Random(seed)
    seen = set()
    for _ in range(20000):
        lv = [[proposal_rng.randrange(1, segments - 1)],
              list(proposal_rng.choice(pairs)), list(proposal_rng.choice(pairs)),
              [proposal_rng.randrange(segments)]]
        key = tuple(tuple(hs) for hs in lv)
        if key not in seen:
            seen.add(key)
            if 2 <= len(legal_paths(lv)) <= 4:
                pools[('1-2-2-1', 'random')].append(lv)
    return pools


PACKET = [('1-1-1', 'forced')] * 4 + [('1-2-1', c) for c in ('agree', 'conflict') for _ in range(2)] + [
    ('1-2-2-1', c) for c in ('agree_agree', 'agree_conflict', 'conflict_agree', 'conflict_conflict')]


def generate(n, seed=20261116, segments=12):
    if n <= 0 or n % len(PACKET):
        raise ValueError('Trial count must be a positive multiple of 12 for exact balance.')
    rng = random.Random(seed)
    pools = candidate_pools(segments, seed)
    for key in pools:
        pools[key] = [lv for lv in pools[key] if not any(set(a) & set(b) for a, b in zip(lv, lv[1:])) and any(v not in sum(lv, []) for v in range(lv[0][0])) and any(v not in sum(lv, []) for v in range(lv[0][0]+1, segments))]
    packet_template = PACKET[:8] + [('1-2-2-1', 'random')] * 4
    required = Counter(packet_template)
    for key, count in required.items():
        needed = count * (n // len(PACKET))
        if len(pools[key]) < needed:
            raise ValueError(f'{key}: need {needed} unique trials, have {len(pools[key])}')
        rng.shuffle(pools[key])
    blocks = []
    for packet_index in range(n // len(PACKET)):
        packet = list(packet_template)
        rng.shuffle(packet)
        spawn_sides = {}
        for key in sorted(set(packet)):
            sides = [-1, 1] * (packet.count(key) // 2)
            rng.shuffle(sides)
            spawn_sides[key] = sides
        for key in packet:
            levels = pools[key].pop()
            meta = describe(levels)
            meta.update(horizon_features(levels))
            meta.update(condition=condition(meta), trial_id=len(blocks), packet_index=packet_index,
                        seed=seed, n_segments=segments, design_version='interleaved_v7')
            side = spawn_sides[key].pop()
            entry = levels[0][0]
            occupied = set(sum(levels, []))
            candidates = [v for v in range(segments) if v not in occupied and (v-entry)*side > 0]
            if not candidates:
                raise ValueError('No safe spawn on assigned side')
            spawn = min(candidates, key=lambda v: abs(v-entry))
            meta.update(spawn_segment=spawn, entry_segment=entry,
                        spawn_offset_segments=spawn-entry, intended_entry_direction='left' if side > 0 else 'right')
            blocks.append(dict(params=dict(startCameraMode=0), levels=levels, design=meta))
    return blocks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default=600)
    parser.add_argument('--seed', type=int, default=20261116)
    parser.add_argument('--segments', type=int, default=12)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    blocks = generate(args.n, args.seed, args.segments)
    # Retain the existing four tutorial blocks: their fixed indices are used
    # by the browser's instruction checks.
    source = ROOT / 'app/configs/emu_experiment_300_blocks.json'
    practice = json.loads(source.read_text())[:4]
    for block in practice:
        block['params']['startCameraMode'] = 0
    blocks[0]['params']['pre_instructions'] = [
        f'Now we can begin. There are {args.n} mazes in total.',
        'Mazes have different numbers of possible paths.',
        'Complete each maze as quickly as possible.']
    output = args.output or ROOT / f'app/configs/interleaved_experiment_{args.n}_blocks.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(practice + blocks, indent=2) + '\n')
    print(f'Wrote {output}: {dict(Counter((b["design"]["structure"], b["design"]["condition"]) for b in blocks))}')


if __name__ == '__main__':
    main()
