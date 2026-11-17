# Interleaved trial-based experiment (v8)

The default `interleaved_experiment_600_blocks` has six practice mazes followed by
600 test mazes: 200 each of 1-1-1, 1-2-1, and 1-2-2-1. Each shuffled packet of
12 contains four of each structure. Structure describes holes per platform,
not the number of physically reachable routes.

## Random placement and reachability

In 1-2-2-1 mazes, both hole pairs are independently proposed across the full
12-segment grid. Neither pair must straddle its parent or contain the other
pair. Layouts remain subject to no neighboring aligned holes, at least one solid segment between holes on each platform, safe spawn
columns on both sides of entry, uniqueness, and seeded sampling. Each 1-2-2-1 maze has three or four reachable paths, balanced exactly 50/50
overall and two each per packet. Distinct planning predictions are not required. Two-path 1-2-1 mazes
retain their original straddling rule and 50/50 agreement/conflict balance.

A move reaches only the nearest gap in its direction; a farther gap behind it
is blocked. Paths, costs, and horizon predictions all use this rule. In the
default 600-trial configuration, the 200 randomized 1-2-2-1 trials contain:

| Reachable paths | Trials |
| --- | ---: |
| 3 | 100 |
| 4 | 100 |

Across all 600 trials, there are 200 one-path, 200 two-path, 100 three-path,
and 100 four-path mazes. These counts are imposed quotas.

## Planning metadata

Each test block saves `block_config.design` with version `interleaved_v8`,
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
for 1-2-1, reachable path count for 1-2-2-1). Actual offsets can exceed one segment and are logged. Retries keep
the assignment. Runtime verifies spawn clearance; legacy blocks using all
columns fall back to solid ground above entry. Spawn coordinates and first
actual horizontal movement are logged separately for each block attempt.

Horizontal speed is 80% of the original (acceleration scale 8 rather than 10). There is no airborne travel limiter.
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

[Short test: six practices plus 12 trials](http://localhost:8000/?participantId=TEST_RANDOM_LAYOUT&experiment=interleaved_experiment_12_blocks&params_name=emu_research_params&v=20261117h)

Use `interleaved_experiment_600_blocks` in the URL for the full experiment.
Counts must be positive multiples of 12. Historical configurations and logs
retain their original designs. Analyze movement onset separately from total
time, account for route lengths and spawn offsets, and handle retries explicitly.
Older analysis scripts assuming three levels per maze require adaptation.

## Practice and initial visibility

The original four tutorials retain their positions and instruction checks. Two
additional 1-2-2-1 practices follow: one with three valid paths, one with four.
All six have `phase=practice`; measured blocks have `phase=test`. Added practice
metadata uses null test trial/packet IDs and a practice ID. Practice is excluded
from test quotas and can reuse a test geometry.

Six levels are visible with spacing `height / (6 + 2)`, starting at twice that
spacing. The follow camera cannot scroll above the initial origin. Consequently
all platforms in these three- and four-level mazes are visible at the beginning,
including the ready screen. Spawn height adapts to leave room above entry.

## Landing collision correction

Downward crossings of platform tops are resolved before side-overlap tests.
This preserves horizontal velocity during deep landings and prevents a platform
landing from being misclassified as a side collision. The JavaScript regression
in `app/tests/landing_regression.js` exercises both directions at four widths.
