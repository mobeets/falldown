#%% load trials

import json
import numpy as np
import matplotlib.pyplot as plt
import random

#%% generate levels

def generate_levels(num_trials=100, screen_width=1, 
                    greedy_func=calculate_greedy_cost, 
                    planning_func=calculate_planning_cost, 
                    pct_agree=0.3, pct_disagree=0.4,
                    c=0, nsegments=12,
                    degree_of_conflict=1.5):
    
    trials = []
    to_segment = lambda x: int(x * nsegments)
    n_agreeing = 0
    n_disagreeing = 0
    n_null = 0
    target_agreeing = int(pct_agree*num_trials)
    target_disagreeing = int(pct_disagree*num_trials)
    target_null = num_trials - target_agreeing - target_disagreeing
    
    while len(trials) < num_trials:
        # --- LEVEL 1 ---
        # Constraint: Between 0.2 and 0.8
        h1_entry = random.uniform(0.2, 0.8)
        
        # --- LEVEL 2 ---
        # Constraint: Must be at least 0.1 away from h1
        # Left side:  0 to (h1 - 0.1)
        # Right side: (h1 + 0.1) to 1
        
        # Safety check: ensure bounds are valid (though 0.2-0.8 range ensures they usually are)
        max_left = h1_entry - 0.15
        min_right = h1_entry + 0.15
        
        # If the gap is too small or out of bounds, restart
        if max_left <= 0 or min_right >= 1: 
            continue

        h2_left_side = random.uniform(0, max_left)
        h2_right_side = random.uniform(min_right, 1)

        # --- LEVEL 3 ---
        # Constraint: Must be at least 0.1 away from BOTH h2 holes
        # Since h3 is between the two holes, it must be:
        # Greater than (h2_left + 0.1) AND Less than (h2_right - 0.1)
        
        min_h3 = h2_left_side + 0.1
        max_h3 = h2_right_side - 0.1
        
        # If the gap between level 2 holes is too tight (< 0.2), 
        # we cannot fit a hole in level 3. Restart.
        if min_h3 >= max_h3:
            continue

        h3_goal_x = random.uniform(min_h3, max_h3)
        
        # Final screen width check
        if not (0 < h3_goal_x < screen_width):
            continue

        h1_entry = to_segment(h1_entry)
        h2_left_side = to_segment(h2_left_side)
        h2_right_side = to_segment(h2_right_side)
        h3_goal_x = to_segment(h3_goal_x)

        # --- CALCULATE COSTS ---
        cost_g_left = abs(h1_entry - h2_left_side)
        cost_g_right = abs(h1_entry - h2_right_side)
        cost_p_left = cost_g_left + abs(h2_left_side - h3_goal_x)
        cost_p_right = cost_g_right + abs(h2_right_side - h3_goal_x)

        delta_greedy_cost = cost_g_right - cost_g_left
        delta_planning_cost = cost_p_right - cost_p_left
        greedy_planning_agreement = delta_greedy_cost * delta_planning_cost
        
        if delta_greedy_cost == 0 and delta_planning_cost == 0:
            # ignore cases where they're both zero
            continue

        # make sure we generate the desired number of agreeing and disagreeing trials
        if greedy_planning_agreement > 0:
            if n_agreeing >= target_agreeing:
                continue
            n_agreeing += 1
        elif greedy_planning_agreement < 0:
            if n_disagreeing >= target_disagreeing:
                continue
            n_disagreeing += 1
        else:
            if n_null >= target_null:
                continue
            n_null += 1

        trial = {
            "trial_id": len(trials),
            "levels": [
                {"level": 1, "holes": [h1_entry]},
                {"level": 2, "holes": [h2_left_side, h2_right_side]},
                {"level": 3, "holes": [h3_goal_x]}
            ],
            "metadata": {
                "left_greedy_cost": cost_g_left,
                "right_greedy_cost": cost_g_right,
                "left_planning_cost": cost_p_left,
                "right_planning_cost": cost_p_right,
                "delta_greedy_cost": delta_greedy_cost,
                "delta_planning_cost": delta_planning_cost,
                "greedy_planning_agreement": greedy_planning_agreement
            }
        }
        trials.append(trial)
    print(f'Generated {len(trials)} trials: {n_agreeing} agreeing, {n_disagreeing} disagreeing')
    return trials

def trials_to_levels(trials):
    levels = []
    for trial in trials:
        for level in trial['levels']:
            hole_indices = level['holes']
            levels.append(hole_indices)
    return levels

def downsample_trials(trials, mode='greedy', percentile=50, threshold=None):
    trial_ids = []
    dists = []
    for trial in trials:
        trial_ids.append(trial['trial_id'])
        if mode == 'greedy':
            dists.append((trial['metadata']['left_greedy_cost'], trial['metadata']['right_greedy_cost']))
        elif mode == 'planning':
            dists.append((trial['metadata']['left_planning_cost'], trial['metadata']['right_planning_cost']))
        elif mode == 'greedy_plan':
            dists.append((trial['metadata']['delta_greedy_cost'], trial['metadata']['delta_planning_cost']))

    # count unique occurrences of each distance pair
    from collections import Counter
    counts = Counter(dists)

    # find the counts at the specified percentile
    if threshold is None:
        threshold = np.percentile(list(counts.values()), percentile)
    
    # for any distance pair occuring above threshold, downsample corresponding trial ids to threshold count
    downsampled_ids = []
    for dist_pair, count in counts.items():
        ids_for_pair = [trial_ids[i] for i, d in enumerate(dists) if d == dist_pair]
        if count > threshold:
            downsampled_ids.extend(random.sample(ids_for_pair, int(threshold)))
        else:
            downsampled_ids.extend(ids_for_pair)
    
    # return downsampled trials
    downsampled_trials = [trial for trial in trials if trial['trial_id'] in downsampled_ids]
    return downsampled_trials

#%% function definitions

def load_experiment(fnm):
    data = json.load(open(fnm))
    return data

def calculate_distances(levels):
    """
    e.g., for first block, levels = data[0]['levels']
    """
    # calculate distances
    dists = []
    for i, holes in enumerate(levels):
        if i == 0 or i == len(levels)-1:
            continue
        if len(holes) != 2:
            continue
        if len(levels[i-1]) != 1 or len(levels[i+1]) != 1:
            continue
        h_prev = levels[i-1][0]
        h_next = levels[i+1][0]
        h_curr1, h_curr2 = holes
        assert h_curr1 <= h_curr2

        # 1-step distances
        d1_1 = abs(h_curr1 - h_prev)
        d1_2 = abs(h_curr2 - h_prev)

        # 2-step distances
        d1 = abs(h_curr1 - h_prev) + abs(h_curr1 - h_next)
        d2 = abs(h_curr2 - h_prev) + abs(h_curr2 - h_next)
        
        dists.append((d1_1, d1_2, d1, d2))
    dists = np.vstack(dists)
    return dists

def make_heatmaps(dists):
    names = {0: 'Greedy', 2: '2-Step', 3: 'Greedy vs. 2-step'}
    plt.figure(figsize=(9,6), dpi=300)
    d = 0
    for c in [0,2,3]:
        if c in [0,2]:
            xs = dists[:,c+0]
            ys = dists[:,c+1]
            xlbl = 'Dist. to Left'
            ylbl = 'Dist. to Right'
        else:
            xs = dists[:,1]-dists[:,0]
            ys = dists[:,3]-dists[:,2]
            xlbl = '$\Delta$ distance, Greedy'
            ylbl = '$\Delta$ distance, 2-Step'
        zs = np.ones_like(xs)

        xs_all = np.unique(xs)
        ys_all = np.unique(ys)
        xs_all = np.arange(xs_all.min(), xs_all.max()+1)
        ys_all = np.arange(ys_all.min(), ys_all.max()+1)
        Z = np.full((len(ys_all), len(xs_all)), np.nan)

        for i, xi in enumerate(xs_all):
            for j, yi in enumerate(ys_all):
                ix = (xs == xi) & (ys == yi)
                Z[j, i] = np.sum(zs[ix])

        plt.subplot(2,3,d+1)
        plt.imshow(
            Z,
            origin='lower',
            interpolation='nearest',
            cmap='Reds',
            extent=[xs_all.min(), xs_all.max(), ys_all.min(), ys_all.max()]
        )
        plt.xticks(xs_all, fontsize=6)
        plt.yticks(ys_all, fontsize=6)
        plt.clim([0, np.max(Z)])
        plt.colorbar()
        plt.xlabel(xlbl)
        plt.ylabel(ylbl)
        plt.title(names[c])

        # now make histogram
        plt.subplot(2,3,3+d+1)
        if c in [0,2]:
            diffs = ys - xs
            lbl = ylbl + ' - ' + xlbl
        else:
            diffs = ys*xs
            lbl = ylbl + ' * ' + xlbl
            d_count = sum(diffs < 0)
            a_count = sum(diffs > 0)
            pct_da = d_count / (d_count + a_count)
            pct_tot = (a_count + d_count) / len(diffs)
            print(f'Agree {a_count} vs disagreement {d_count}, ({pct_da:.1%} disagree vs agree, {pct_tot:.1%} of total agree or disagree)')

        plt.hist(diffs, bins=np.arange(-10,11)-0.5)
        plt.xlabel(lbl)
        plt.ylabel('Count')
        plt.title(names[c] + ' histogram')

        d += 1
    plt.tight_layout()

#%%

def prune_levels(levels):
    """
    skip one level if we have 1 hole two levels in a row
    """
    pruned = []
    for i, holes in enumerate(levels):
        # if i == 0:
        #     pruned.append(holes)
        #     continue
        # if len(levels[i-1]) == 1 and len(levels[i]) == 1:
        #     continue
        # pruned.append(holes)
        if i == len(levels)-1:
            pruned.append(holes)
            continue
        if len(levels[i]) == 1 and len(levels[i+1]) == 1:
            continue
        pruned.append(holes)
    return pruned

#%%

out_path = '../../app/configs/default_experiment1.json'
levels_per_block = 300

trials = generate_levels(num_trials=10000)
# trials = downsample_trials(trials, mode='greedy', percentile=80)
# trials = downsample_trials(trials, mode='planning', percentile=80)
trials = downsample_trials(trials, mode='greedy_plan', percentile=90)
levels = trials_to_levels(trials)
# levels = prune_levels(levels)

dists = calculate_distances(levels)
make_heatmaps(dists)

# split up levels (a list) into blocks of size levels_per_block
blocks = []
for i in range(0, len(levels), levels_per_block):
    block_levels = levels[i:i+levels_per_block]
    blocks.append({'params': {}, 'levels': block_levels})
json.dump(blocks, open(out_path, 'w'))
