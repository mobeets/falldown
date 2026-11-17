# Interleaved trial-based experiment (v7)

The default `interleaved_experiment_600_blocks` has four tutorials followed by
600 test mazes: 200 each of 1-1-1, 1-2-1, and 1-2-2-1. Each shuffled packet of
12 contains four of each structure. Structure describes holes per platform,
not the number of physically reachable routes.

## Random placement and reachability

In 1-2-2-1 mazes, both hole pairs are independently proposed across the full
12-segment grid. Neither pair must straddle its parent or contain the other
pair. Layouts remain subject to no neighboring aligned holes, at least one solid segment between holes on each platform, safe spawn
columns on both sides of entry, uniqueness, and seeded sampling. Each 1-2-2-1 maze must have 2–4 reachable paths. There are no
quotas within that range or for distinct planning predictions. Two-path 1-2-1 mazes
retain their original straddling rule and 50/50 agreement/conflict balance.

A move reaches only the nearest gap in its direction; a farther gap behind it
is blocked. Paths, costs, and horizon predictions all use this rule. In the
default 600-trial configuration, the 200 randomized 1-2-2-1 trials contain:

| Reachable paths | Trials |
| --- | ---: |
| 2 | 131 |
| 3 | 55 |
| 4 | 14 |

Across all 600 trials, there are 200 one-path, 331 two-path, 55 three-path,
and 14 four-path mazes. These are realized seeded counts, not imposed quotas.

## Planning metadata

Each test block saves `block_config.design` with version `interleaved_v7`,
structure, actual `n_paths`, all reachable routes and costs, optimal paths,
greedy route and regret, ties, condition, and seed/trial/packet identifiers.
`horizon_predictions` lists complete routes predicted by one-, two-, and
three-transition lookahead, replanning at each platform. `horizon_states`
records preferred actions and excess costs for legal actions from each state.
`route_horizon_annotations` records whole-route and per-action compatibility.
Predictions may coincide or tie; three distinct routes are no longer required.
The descriptive agreement conditions compare greedy and full planning along
one recorded optimal route; inspect tie flags and actual reachable actions
before interpreting these as decision conflicts. Forced choices can agree
trivially. Horizontal travel cost is measured in segment units.

## Spawn and runtime behavior

Spawns use the nearest column clear of every hole on the assigned side of
entry. Side is balanced within packet and structure (and agreement condition
for 1-2-1). Actual offsets can exceed one segment and are logged. Retries keep
the assignment. Runtime verifies spawn clearance; legacy blocks using all
columns fall back to solid ground above entry. Spawn coordinates and first
actual horizontal movement are logged separately for each block attempt.

The original horizontal speed is restored. There is no airborne travel limiter.
Every block uses the follow camera, with drift and mode switching disabled.
Task scripts and configurations carry cache versions, and the server sends
no-store headers.

## Testing and regeneration

```sh
uv run python app/server.py
python3 tools/level_generation/generating_levels_interleaved.py
python3 tools/level_generation/generating_levels_interleaved.py --n 12
python3 -m unittest discover -s tools/level_generation/tests -p test_interleaved.py
```

[Short test: four tutorials plus 12 trials](http://localhost:8000/?participantId=TEST_RANDOM_LAYOUT&experiment=interleaved_experiment_12_blocks&params_name=emu_research_params&v=20261117g)

Use `interleaved_experiment_600_blocks` in the URL for the full experiment.
Counts must be positive multiples of 12. Historical configurations and logs
retain their original designs. Analyze movement onset separately from total
time, account for route lengths and spawn offsets, and handle retries explicitly.
Older analysis scripts assuming three levels per maze require adaptation.
