#%%

import json
trials_path = '../level_generation/trials_new.json'
out_path = '../configs/default_experiment-7-10.json'
trials = json.load(open(trials_path))

levels = []
for trial in trials:
    for level in trial['levels']:
        hole_indices = [int(hole) for hole in level['holes']]
        levels.append(hole_indices)

experiment_config = [{'params': {}, 'levels': levels}]
json.dump(experiment_config, open(out_path, 'w'))

# %%
