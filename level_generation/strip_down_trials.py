#%%

import json
trials_path = '../level_generation/trials_output.json'
out_path = '../configs/default_experiment.json'
trials = json.load(open(trials_path))

levels = []
for trial in trials:
    for level in trial['levels']:
        hole_indices = [int(12*hole) for hole in level['holes']]
        levels.append(hole_indices)

experiment_config = [{'params': {}, 'levels': levels}]
json.dump(experiment_config, open(out_path, 'w'))
