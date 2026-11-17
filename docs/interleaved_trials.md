# Interleaved trial-based experiment

The default task now loads `interleaved_experiment_600_blocks.json`, with the
existing four tutorial blocks followed by 600 one-maze-per-block test trials.
Use the existing `emu_research_params` (12 segments). To select it explicitly:

`http://localhost:8000?participantId=SUBJECT_ID&experiment=interleaved_experiment_600_blocks&params_name=emu_research_params`

Each randomly shuffled 12-trial packet contains:

| Structure | Trials per packet | Policy conditions |
| --- | ---: | --- |
| 1-1-1 | 4 | One forced route; no choice comparison |
| 1-2-1 | 4 | Two agree, two conflict |
| 1-2-2-1 | 4 | Three distinct horizon predictions; balanced optimal first-choice direction |

Every four-path trial gives one-, two-, and three-level planning three distinct,
untied complete-route predictions. Each model minimizes horizontal travel over
its next H transitions and replans at each platform. The full planner's first
choice is balanced left/right, two each per packet. The previous four agreement
quotas are replaced: in the default seed, agreement conditions are 189
conflict/agree, 4 conflict/conflict, and 7 agree/conflict. These are descriptive
labels along the optimal route, not balanced factors.

Version `interleaved_v3` adds `horizon_predictions`, `horizon_states` (preferred
actions, ties, and action excess cost at every position), and
`route_horizon_annotations` (whole-route and per-action compatible horizons).
The fourth route matches none of the three fixed-horizon models. Its individual
choices remain labeled; this does not establish switching or a fourth depth.
One- and two-path trials also receive these fields, including shared predictions.

Greedy minimizes the next horizontal movement. Planning minimizes total
horizontal movement to the goal, in segment units. Scheduled choice trials
exclude tied optimal routes and greedy ties on both the greedy and optimal
routes. Unchosen alternatives can still tie in cost; every path and cost is
recorded. The second hole pair strictly surrounds the first pair so neither
intermediate gap blocks access to a purported alternative: all four routes
are physically available. Geometry is sampled without replacement within
each condition with a reproducible seed.

Each test block has a `design` object, saved automatically under
`block_config.design` in block logs and experiment JSON. It includes structure,
condition, trial/packet IDs, seed, path count, all path hole sequences and
costs, optimal routes, greedy route, greedy regret, best-versus-second cost
gap, tie flags, and per-choice policy holes and margins along the optimal route.
Existing level events retain timestamps, hole used, ball position, and block
and level indices; join these to the block design to analyze observed choices
and reaction times. Level indices in the metadata are zero-based.

The packets balance intended exposures; retries and back-navigation remain
recorded as additional block attempts and should be handled explicitly in
analysis. Historical analysis scripts that assume every trial spans three
levels need adaptation before analyzing this configuration.

Path count alone does not isolate evaluation time: four-path mazes have an
extra level and different travel demands. Analyze movement onset or choice
latency separately from total completion time, and control for path length,
initial ball-to-entry distance, choice distance, and policy cost gaps. This
generator records those geometries and costs but does not match their
distributions across structures. The forced baseline still requires motor
control and anticipation.

Regenerate with Python (standard library only):

```sh
python3 tools/level_generation/generating_levels_interleaved.py
# Exact balance requires a positive multiple of 12 (e.g. 504 rather than 500).
python3 tools/level_generation/generating_levels_interleaved.py --n 504 --seed 20261116
python3 -m unittest discover -s tools/level_generation/tests -p 'test_interleaved.py'
```

The generator preserves the original tutorial blocks and writes a separate
configuration, leaving historical experiment configurations available by URL.

## Alignment and spawn controls

Neighboring platforms never share a hole column in v3 test trials; columns may
recur farther apart. The grid remains 12 segments. Spawn is one segment left
or right of the entry center, balanced per packet within structure and policy
condition (optimal first-choice direction for four-path mazes). The assignment
is seeded and retained on retries. Tutorials and older configs retain their
existing spawn behavior.

Design records `spawn_segment`, `entry_segment`, `spawn_offset_segments`, and
`intended_entry_direction`. Each block attempt logs `spawn` with actual pixel
coordinates and timestamp, and `first_movement` with the first nonzero horizontal
displacement during active play. No movement leaves that field null. Dedicated
spawn and movement events also appear in JSONL logs. Movement direction is
measured independently of the intended direction toward entry.

For a short test (four tutorials plus twelve test trials), start the server with
`uv run python app/server.py`, then open:

[Short local test](http://localhost:8000/?participantId=TEST_INTERLEAVED&experiment=interleaved_experiment_12_blocks&params_name=emu_research_params)

## Spawn clearance and slower movement (v4)

Test spawns use the nearest unused column on the assigned side of entry, clear
of holes on every platform. Offsets may exceed one segment and remain logged;
side balance is preserved. Tutorials also select an unused column at runtime.
Horizontal speed in EMU parameters is halved (scale 5 instead of 10).
`requirePlatformLanding` limits cumulative airborne horizontal travel to half
the distance from takeoff to the nearest passable opening on the next platform.
The remaining movement must occur after landing. This geometry-based control
also covers the initial fall and resets on each block attempt. It changes
airborne steering, so timing analyses must distinguish these v4 sessions.
The regenerated 600-trial condition counts are 192 conflict/agree and 8
agree/conflict for four-path trials; horizon distinctions remain required.

## Runtime correction: follow camera and explicit speed

All loaded blocks, including tutorials, now use follow mode (0); mode switches
are disabled. Horizontal movement is explicitly limited to 0.75 segments per
second at the configured frame rate, independent of the legacy acceleration
scale. Spawn selection checks the actual entry-hole rectangles with ball radius
clearance. Generated trials avoid all hole columns; legacy blocks that use every
column select solid ground on the entry platform instead, or fail explicitly
if none exists. Config requests and scripts carry a new cache version, and the
server disables HTTP caching. Restart the server and reload an existing tab to
pick up the corrected runtime.

Horizontal speed has been restored to the original acceleration-based cap: `maxBallAccelScale=10` times `ballAccel`. The 0.75-segments/second override has been removed. Spawn clearance, airborne landing controls, and follow-only camera behavior remain enabled.

The airborne travel limiter has been removed to restore continuous horizontal movement during falls and landings. Original horizontal speed, safe spawns, and follow-only camera behavior remain. Landing before the next hole is no longer enforced.
