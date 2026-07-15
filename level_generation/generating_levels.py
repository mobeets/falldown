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

# %%
import numpy as np
import json
import random
import copy


# %%
def calculate_greedy_cost(p_prev, p_curr, p_next, c=0):
    return abs(p_prev - p_curr)
# revise based on previous level

def calculate_planning_cost(p_prev, p_curr, p_next, c=0):

    dist_step_1 = abs(p_prev - p_curr)
    dist_step_2 = abs(p_curr - p_next)
    

    v1 = p_curr - p_prev
    v2 = p_next - p_curr
    
    switch_penalty = c if (v1 * v2 < 0) else 0
    
    return dist_step_1 + dist_step_2 + switch_penalty


# %%
def generate_levels(num_levels=50, screen_width=600, 
                    greedy_func=calculate_greedy_cost, 
                    planning_func=calculate_planning_cost, 
                    c=0,
                    degree_of_conflict=1.5):
    # generate levels wo regard to screen size, scale up based 
    # set doc to 1
    experiment_configs = []
    
    while len(experiment_configs) < num_levels:
        # level 1
        entry = random.randint(100, 500)
        
        # second level
        # sample two dist randomly, set near and far wo biasing
        side = random.choice([-1, 1]) 
        dist_near = random.randint(int(screen_width/30), int(screen_width/4))
        dist_far = random.randint(int(screen_width/3.5), int(screen_width/2))
        
        cand_a_x = entry + (side * dist_near)
        cand_b_x = entry - (side * dist_far)
        
        # checking, just for fun
        if not (0 < cand_a_x < screen_width and 0 < cand_b_x < screen_width):
            continue

        # third level
        h3_goal_x = cand_b_x + random.randint(-50, 50)
        if not (0 < h3_goal_x < screen_width):
            continue

        # costs
        cost_g_a = greedy_func(entry, cand_a_x, h3_goal_x, c)
        cost_p_a = planning_func(entry, cand_a_x, h3_goal_x, c)
        
        cost_g_b = greedy_func(entry, cand_b_x, h3_goal_x, c)
        cost_p_b = planning_func(entry, cand_b_x, h3_goal_x, c)

        greedy_prefers_a = (cost_g_a * degree_of_conflict < cost_g_b)
        planner_prefers_b = (cost_p_b * degree_of_conflict < cost_p_a)
        
        if greedy_prefers_a and planner_prefers_b:
            trial = {
                "trial_id": len(experiment_configs) + 1,
                "levels": [
                    {"level": 1, "holes": [entry]},
                    {"level": 2, "holes": [cand_a_x, cand_b_x]},
                    {"level": 3, "holes": [h3_goal_x]}
                ],
                "metadata": {
                    "greedy_choice": cand_a_x,
                    "planner_choice": cand_b_x,
                    "switch_cost_param": c,
                    "cost_greedy_diff": cost_g_b - cost_g_a,
                    "cost_planner_diff": cost_p_a - cost_p_b
                }
            }
            experiment_configs.append(trial)
            
    return experiment_configs


# %%
levels = generate_levels(num_levels = 10000, screen_width= 1)

with open('trials_new.json', 'w') as f:
    json.dump(levels, f, indent=4)


# %% [markdown]
# # Generating New Trials w Oversampling Disagreement

# %%
#import random

def generate_levels(num_levels=50, 
                    greedy_func=calculate_greedy_cost, 
                    planning_func=calculate_planning_cost, 
                    c=0):
    
    target_conflict = num_levels // 2
    target_agreement = num_levels - target_conflict
    
    conflict_trials = []
    agreement_trials = []
    
    while len(conflict_trials) < target_conflict or len(agreement_trials) < target_agreement:
        
        # 1. Generate Entry and Goal
        # Restricted to [1, 10] so there is always at least one space available at 0 and 11
        entry = random.randint(1, 10)
        h3_goal_x = random.randint(1, 10)
        
        # Find the absolute left-most and right-most points of the entry/goal pair
        min_x = min(entry, h3_goal_x)
        max_x = max(entry, h3_goal_x)
        
        # 2. Place Candidates Strictly on the Outside
        # One candidate must be to the left of `min_x`, the other to the right of `max_x`
        cand_left = random.randint(0, min_x - 1)
        cand_right = random.randint(max_x + 1, 11)
        
        # Randomly assign left and right to A and B so A isn't always the left one
        candidates = [cand_left, cand_right]
        random.shuffle(candidates)
        cand_a_x, cand_b_x = candidates[0], candidates[1]
        
        # 3. Calculate Costs
        cost_g_a = greedy_func(entry, cand_a_x, h3_goal_x, c)
        cost_g_b = greedy_func(entry, cand_b_x, h3_goal_x, c)
        
        cost_p_a = planning_func(entry, cand_a_x, h3_goal_x, c)
        cost_p_b = planning_func(entry, cand_b_x, h3_goal_x, c)
        
        # Skip configurations where either strategy perceives a perfect tie
        if cost_g_a == cost_g_b or cost_p_a == cost_p_b:
            continue
            
        # Determine strict preferences
        greedy_prefers_a = cost_g_a < cost_g_b
        planner_prefers_a = cost_p_a < cost_p_b
        
        # If they prefer different holes, it's a conflict
        is_conflict = (greedy_prefers_a != planner_prefers_a)
        
        # 4. Build the trial object
        trial = {
            "levels": [
                {"level": 1, "holes": [entry]},
                {"level": 2, "holes": [cand_a_x, cand_b_x]},
                {"level": 3, "holes": [h3_goal_x]}
            ]
        }
        
        # 5. Route to the correct bucket if it isn't full yet
        if is_conflict and len(conflict_trials) < target_conflict:
            conflict_trials.append(trial)
        elif not is_conflict and len(agreement_trials) < target_agreement:
            agreement_trials.append(trial)
            
    # Combine the lists and shuffle so the conflict trials aren't all clustered
    all_trials = conflict_trials + agreement_trials
    random.shuffle(all_trials)
    
    # Assign sequential trial_ids after shuffling
    for idx, trial in enumerate(all_trials):
        trial["trial_id"] = idx + 1
        
    return all_trials


# %%
levels = generate_levels(num_levels = 10000)

with open('trials_new.json', 'w') as f:
    json.dump(levels, f, indent=4)

# %% [markdown]
# # Run the code from here down in order to re-generate the trials 4/29

# %%
trials = json.load(open('../configs/default_experiment-7-10.json'))
#trials

# %%
def format_experiment_blocks(filepath='../configs/default_experiment-7-10.json'):
    # Load the raw data
    with open(filepath, 'r') as f:
        trials = json.load(f)

    new_trials = []


    # 2. Define and insert instruction trials
    instruction_trials = [
        {
            "params": {
            "instructions": [
                "[PRACTICE] Complete the maze as as quickly and accurately as possible."
                ]
            },
            "levels": [[3],[5],[2, 10],[5]]
        },
        {
            'params': {
                'pre_instructions': [
                    "Sometimes, going to the closest hole will make you take longer to complete the level",
                    "On this level, take the most effecient possible path",
                    "If you take the less effecient paths, you'll be asked to repeat this block"
                ],
                'instructions': [
                    "Sometimes, going to the closest hole will make you take longer to complete the level",
                    "On this level, take the most effecient possible path",
                    "If you take the less effecient paths, you'll be asked to repeat this block"
                ]
            },
            'levels': [[5], [3, 10], [9]]
        }, 
        {
            'params': {
                'pre_instructions': [
                    "Sometimes, going to the closest hole will make you take longer to complete the level",
                    "On this level, take the most effecient possible path",
                    "If you take the less effecient paths, you'll be asked to repeat this block"
                ],
                'instructions': [
                    "Sometimes, going to the closest hole will make you take longer to complete the level",
                    "On this level, take the most effecient possible path",
                    "If you take the less effecient paths, you'll be asked to repeat this block"
                ]
            },
            'levels': [[6], [0, 10], [2]]
        }, 
        {
            'params': {
                'pre_instructions': [
                    "Sometimes, going to the closest hole will make you take longer to complete the level",
                    "On this level, take the most effecient possible path",
                    "If you take the less effecient paths, you'll be asked to repeat this block"
                ],
                'instructions': [
                    "Sometimes, going to the closest hole will make you take longer to complete the level",
                    "On this level, take the most effecient possible path",
                    "If you take the less effecient paths, you'll be asked to repeat this block"
                ]
            },
            'levels': [[4], [1, 11], [10]]
        }
    ]
    
    # Insert them sequentially after the practice trial
    new_trials.extend(instruction_trials)

    print(trials[0])

    # 3. Process the massive block
    if len(trials) > 0:
        massive_block = trials[0]
        all_levels = massive_block.get('levels', [])
        base_params = massive_block.get('params', {})
        
        # Clean the base params so pre_instructions don't accidentally copy into every block
        if 'pre_instructions' in base_params:
            del base_params['pre_instructions']

        # Math for slicing the chunks
        target_blocks = 40
        trials_per_block = 35
        levels_per_trial = 3
        chunk_size = trials_per_block * levels_per_trial  # 105 levels per block

        for block_idx in range(target_blocks):
            start_idx = block_idx * chunk_size
            end_idx = start_idx + chunk_size
            
            # Slice exactly 105 levels. 
            # (If the massive block runs out of levels before 40 blocks, it will stop naturally)
            chunk_levels = all_levels[start_idx:end_idx]
            
            # Ensure we only append complete blocks
            if len(chunk_levels) < chunk_size:
                break
                
            split_block = {
                'params': copy.deepcopy(base_params),
                'levels': chunk_levels
            }

            # Add the introductory text only to the very first generated block
            if block_idx == 0:
                split_block['params']['pre_instructions'] = [
                    "Now we can begin. There are 40 mazes in total.",
                    "Each maze will take around two minutes to complete.",
                    "Complete each maze as quickly as possible."
                ]
                
            new_trials.append(split_block)

    return new_trials

# %%
formatted_trials = format_experiment_blocks()

# %%
len(formatted_trials)

# %%
# change this to allow the patient to do more levels if they want

#new_trials1 = new_trials[0:15]
#new_trials2 = [new_trials[0]] + new_trials[15:29]
#new_trials2[1]['params'] = new_trials1[1]['params']
#del new_trials2[2]

formatted_trials = formatted_trials[0:44]

for i in range(4, len(formatted_trials)):
    formatted_trials[i]['params']['startCameraMode'] = 1

# %%
formatted_trials[0]

# %%
with open('../configs/short_trials_experiment-7-10.json', 'w') as f:
     json.dump(formatted_trials, f, indent=2)

# %% [markdown]
# # Adding E.params.startCameraMode

# %%
trials = json.load(open('../configs/short_trials_experiment.json'))

# %%
for i in range(4, len(trials)):
    if i % 2 == 1:
        trials[i]['params']['startCameraMode'] = 1

# %%
trials = trials[:36]

# %%
trials[35]

# %%
with open('../configs/short_trials_experiment.json', 'w') as f:
     json.dump(trials, f, indent=2)
