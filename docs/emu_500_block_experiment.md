# EMU 500-trial, one-trial-per-block experiment

Config: `app/configs/emu_experiment_500_blocks.json`

## Design

- **Training:** the first four blocks of `emu_experiment_7-29.json` are copied
  unchanged (practice + three training blocks).
- **Test:** 500 trials, each in its own block. Each block contains one 1-2-1
  trial set (three levels: one hole, two holes, one hole).
- **Trial pool:** sampled without replacement from the 910 existing 1-2-1 trial
  sets in the test phase of `emu_experiment_7-29.json`.
- **Sampling:** deterministic, seed `20261020`; selected trials are ordered by
  their original index.
- **Camera mode:** every test block uses `startCameraMode: 1` (drift), matching
  the existing EMU test blocks.
- **Instructions:** the first test block announces 500 mazes; later blocks have
  no pre-instructions.
- **Estimated duration:** at the observed ~2.44 s/block, 500 test blocks plus
  the 4 training blocks should take roughly 20 minutes.

## URLs

Patient YGD:

```
http://0.0.0.0:8000/?participantId=YGD&params_name=emu_research_params&experiment=emu_experiment_500_blocks
```

EMU host:

```
http://emu.rice.edu:8000/?participantId=YGD&params_name=emu_research_params&experiment=emu_experiment_500_blocks
```
