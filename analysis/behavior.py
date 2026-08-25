#%%

import json
import numpy as np
import matplotlib.pyplot as plt

def load(fnm):
	return json.load(open(fnm))

def get_choices(data):
    choices = []
    for block in data['blocks']:
        trials = block['trials']
        for i, trial in enumerate(trials):
            if i == 0:
                continue
            if 'holes' in trial:
                hole_locs = sorted(trial['holes']['hole_locations'])
            elif 'events' in trial and len(trial['events']) > 0:
                hole_locs = sorted(trial['events'][0]['hole_locations'])
            else:
                print('no choice')
                continue
            if len(hole_locs) == 2:
                if 'holeUsed' in trial:
                    h_cur = trial['holeUsed']
                    h_prev = trials[i-1]['holeUsed']
                elif 'events' in trial and len(trial['events']) > 0 and 'events' in trials[i-1] and len(trials[i-1]['events']) > 0:
                    h_cur = trial['events'][0]['holeUsed']
                    h_prev = trials[i-1]['events'][0]['holeUsed']
                else:
                    continue
                dist_L = np.abs(hole_locs[0] - h_prev)
                dist_R = np.abs(hole_locs[1] - h_prev)
                choice = hole_locs.index(h_cur)
                if 'timePassedThru' in trial:
                    rt = trial['timePassedThru'] - trials[i-1]['timePassedThru']
                elif 'events' in trial and len(trial['events']) > 0 and len(trials[i-1]['events']) > 0:
                    rt = trial['events'][0]['time'] - trials[i-1]['events'][0]['time']
                else:
                    rt = np.nan
                choices.append((dist_L, dist_R, rt, choice))
        return np.vstack(choices)

def compare_greedy_vs_rollout(data, only_use_disagreements=False):
    choices = []
    for block in data['blocks']:
        trials = block['trials']
        for i, trial in enumerate(trials):
            if i == 0:
                continue
            if 'holes' in trial:
                hole_locs = sorted(trial['holes']['hole_locations'])
                prev_hole_locs = sorted(trials[i-1]['holes']['hole_locations'])
                next_hole_locs = sorted(trials[i+1]['holes']['hole_locations'])
                h_cur = trial['holeUsed']
                h_prev = trials[i-1]['holeUsed']
                h_next = trials[i+1]['holeUsed']
            elif 'events' in trial and len(trial['events']) > 0 and len(trials[i-1]['events']) > 0 and len(trials[i+1]['events']) > 0:
                hole_locs = sorted(trial['events'][0]['hole_locations'])
                prev_hole_locs = sorted(trials[i-1]['events'][0]['hole_locations'])
                next_hole_locs = sorted(trials[i+1]['events'][0]['hole_locations'])
                h_cur = trial['events'][0]['holeUsed']
                h_prev = trials[i-1]['events'][0]['holeUsed']
                h_next = trials[i+1]['events'][0]['holeUsed']
            else:
                hole_locs = []
                prev_hole_locs = []
                next_hole_locs = []
                h_cur = None
                h_prev = None
            if len(hole_locs) != 2 or len(prev_hole_locs) != 1 or len(next_hole_locs) != 1:
                continue

            dist_L1 = np.abs(hole_locs[0] - h_prev)
            dist_R1 = np.abs(hole_locs[1] - h_prev)
            dist_total_L2 = dist_L1 + np.abs(hole_locs[0] - h_next)
            dist_total_R2 = dist_R1 + np.abs(hole_locs[1] - h_next)
            if only_use_disagreements:
                # we ignore trials where both greedy and rollout would make the same choice
                if (dist_L1 < dist_R1 and dist_total_L2 < dist_total_R2) or (dist_R1 < dist_L1 and dist_total_R2 < dist_total_L2):
                    continue
            
            choice = hole_locs.index(h_cur)
            if 'timePassedThru' in trial:
                rt = trial['timePassedThru'] - trials[i-1]['timePassedThru']
            elif 'events' in trial and len(trial['events']) > 0 and len(trials[i-1]['events']) > 0:
                rt = trial['events'][0]['time'] - trials[i-1]['events'][0]['time']
            else:
                rt = np.nan
            choices.append((dist_L1, dist_R1, dist_total_L2, dist_total_R2, rt, choice))
    return np.vstack(choices)

def plot_psychometric_curve(X, y, fig=None, color='k', xlabel='Δ Distance to hole (L - R)', label='_'):
	xs = np.unique(X)
	ys = []
	ses = []
	for x in xs:
		ix = X == x
		ymu = np.nanmean(y[ix])
		ys.append(ymu)
		ses.append(np.nanstd(y[ix]) / np.sqrt(sum(ix)))

	if fig is None:
		plt.figure(figsize=(3,3), dpi=300)
	for x,y,se in zip(xs,ys,ses):
		plt.plot([x,x],[y-se,y+se],'-', color=color, alpha=0.3)
	plt.plot(xs, ys, '.-', color=color, label=label)
	plt.xlabel(xlabel)
	plt.ylabel('Prob. of choosing Right Hole')
	return fig

def plot_rt_vs_conflict(X, y, fig=None, color='purple', xlabel='Degree of Conflict', label='_'):
    xs = np.unique(X)
    ys = []
    ses = []
    
    for x in xs:
        ix = X == x
        ymu = np.nanmean(y[ix])
        ys.append(ymu)
        ses.append(np.nanstd(y[ix]) / np.sqrt(np.sum(ix)))

    if fig is None:
        fig = plt.figure(figsize=(3,3), dpi=300)
        
    for x, y_val, se in zip(xs, ys, ses):
        plt.plot([x, x], [y_val - se, y_val + se], '-', color=color, alpha=0.3)
        
    plt.plot(xs, ys, '.-', color=color, label=label)
    
    plt.xlabel(xlabel)
    plt.ylabel('Reaction Time (units?)')
    
    if label != '_':
        plt.legend()
        
    return fig

def plot_rt_residuals_vs_conflict(choices, fig=None):
    # 1. Extract columns from the compare_greedy_vs_rollout output
    # Columns: (dist_L1, dist_R1, dist_total_L2, dist_total_R2, rt, choice)
    dist_L1 = choices[:, 0]
    dist_R1 = choices[:, 1]
    dist_total_L2 = choices[:, 2]
    dist_total_R2 = choices[:, 3]
    rts = choices[:, 4]
    choice_made = choices[:, 5]

    # Filter out trials with NaN reaction times
    valid_mask = ~np.isnan(rts)
    dist_L1, dist_R1 = dist_L1[valid_mask], dist_R1[valid_mask]
    dist_total_L2, dist_total_R2 = dist_total_L2[valid_mask], dist_total_R2[valid_mask]
    rts = rts[valid_mask]
    choice_made = choice_made[valid_mask]

    # 2. Identify the total distance traveled over 2-steps based on the user's choice
    chosen_dist_2 = np.where(choice_made == 0, dist_total_L2, dist_total_R2)

    # 3. Regress the reaction time vs the chosen 2-step distance
    slope, intercept = np.polyfit(chosen_dist_2, rts, 1)
    predicted_rts = (slope * chosen_dist_2) + intercept

    # 4. Collect residuals (Actual RT - Predicted RT)
    residuals = rts - predicted_rts

    # 5. Define Conflict (1-step vs 2-step)
    # Calculated as the difference in magnitude between the 1-step delta and 2-step delta
    conflict_1step = np.abs(dist_L1 - dist_R1)
    conflict_2step = np.abs(dist_total_L2 - dist_total_R2)
    conflicts = np.abs(conflict_1step - conflict_2step)
    
    # Round slightly to prevent floating-point precision issues from creating too many bins
    conflicts = np.round(conflicts, decimals=5)
    unique_conflicts = np.unique(conflicts)

    if fig is None:
        fig = plt.figure(figsize=(8, 5), dpi=300)

    # Plot raw trial residuals in the background
    plt.scatter(conflicts, residuals, alpha=0.2, color='gray', label='Trial Residuals')

    # Calculate and plot mean residuals (with standard error) for each conflict level
    mean_residuals = []
    ses = []
    
    for c in unique_conflicts:
        ix = conflicts == c
        res_c = residuals[ix]
        
        mean_val = np.mean(res_c)
        se_val = np.std(res_c) / np.sqrt(len(res_c)) if len(res_c) > 0 else 0
        
        mean_residuals.append(mean_val)
        ses.append(se_val)
        
        # Error bar
        plt.plot([c, c], [mean_val - se_val, mean_val + se_val], '-', color='blue', alpha=0.7)

    # Line connecting the means
    plt.plot(unique_conflicts, mean_residuals, 'o-', color='blue', linewidth=2, markersize=6, label='Mean Residual')

    # Baseline at 0 (representing the exact regression prediction)
    plt.axhline(0, color='black', linestyle='--', linewidth=1.5, label='Expected RT (Baseline)')

    # Labels and Formatting
    plt.xlabel('Conflict Margin (|1-Step Diff - 2-Step Diff|)')
    plt.ylabel('Reaction Time Residual (ms)')
    plt.title('RT Residuals (Controlling for 2-Step Distance) vs. Conflict')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    return fig


def plot_rt_over_time_by_conflict(trial_nums, conflicts, rts, window_size=10, fig=None):
    sort_idx = np.argsort(trial_nums)
    c_sorted = conflicts[sort_idx]
    rt_sorted = rts[sort_idx]

    unique_conflicts = np.unique(c_sorted)

    if fig is None:
        fig = plt.figure(figsize=(8, 5), dpi=300)

    for c in unique_conflicts:
        ix = c_sorted == c
        rt_c = rt_sorted[ix]
        
        local_trials = np.arange(1, len(rt_c) + 1)
        
        if len(rt_c) >= window_size:
            moving_avg = np.convolve(rt_c, np.ones(window_size)/window_size, mode='valid')
            t_moving_avg = local_trials[window_size - 1:]
            
            plt.plot(t_moving_avg, moving_avg, linewidth=2.5, linestyle='-', label=f'Conflict: {c:.2f}')

    plt.xlabel('Trial Number')
    plt.ylabel('Reaction Time (units?)')
    plt.title(f'(Rxn time by conflict)')
    
    # Place legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    return fig

def plot_rt_residuals_over_time(choices, window_size=10, fig=None):
    # 1. Extract columns from the compare_greedy_vs_rollout output
    dist_L1 = choices[:, 0]
    dist_R1 = choices[:, 1]
    dist_total_L2 = choices[:, 2]
    dist_total_R2 = choices[:, 3]
    rts = choices[:, 4]
    choice_made = choices[:, 5]

    # Filter out trials with NaN reaction times
    valid_mask = ~np.isnan(rts)
    dist_L1, dist_R1 = dist_L1[valid_mask], dist_R1[valid_mask]
    dist_total_L2, dist_total_R2 = dist_total_L2[valid_mask], dist_total_R2[valid_mask]
    rts = rts[valid_mask]
    choice_made = choice_made[valid_mask]

    # 2. Regress RT vs the chosen 2-step distance to get residuals
    chosen_dist_2 = np.where(choice_made == 0, dist_total_L2, dist_total_R2)
    slope, intercept = np.polyfit(chosen_dist_2, rts, 1)
    predicted_rts = (slope * chosen_dist_2) + intercept
    residuals = rts - predicted_rts

    # 3. Define Conflict (1-step vs 2-step)
    conflict_1step = np.abs(dist_L1 - dist_R1)
    conflict_2step = np.abs(dist_total_L2 - dist_total_R2)
    conflicts = np.abs(conflict_1step - conflict_2step)
    
    # Round slightly to group similar conflict levels
    conflicts = np.round(conflicts, decimals=5)
    unique_conflicts = np.unique(conflicts)

    if fig is None:
        fig = plt.figure(figsize=(8, 5), dpi=300)

    # Add a baseline at 0 to show the "expected" reaction time
    plt.axhline(0, color='black', linestyle='--', linewidth=1.5, label='Expected RT (0 Residual)')

    # 4. Plot moving average for each degree of conflict independently
    for c in unique_conflicts:
        ix = conflicts == c
        res_c = residuals[ix]
        
        # Because we read the array top-to-bottom, the data is naturally chronological.
        # We just count the local encounters for this specific conflict.
        local_trials = np.arange(1, len(res_c) + 1)
        
        # Calculate and plot the moving average trendline
        if len(res_c) >= window_size:
            moving_avg = np.convolve(res_c, np.ones(window_size)/window_size, mode='valid')
            t_moving_avg = local_trials[window_size - 1:]
            
            plt.plot(t_moving_avg, moving_avg, linewidth=2.5, linestyle='-', label=f'Conflict: {c:.2f}')

    # Labels and Formatting
    plt.xlabel('Number of Encounters (Local Trial Count)')
    plt.ylabel('Reaction Time Residual (ms)')
    plt.title(f'RT Residuals by Conflict Exposure ({window_size}-Trial Moving Avg)')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    return fig

def plot_rt_vs_distance(choices, step=1):
    dist_L1 = choices[:, 0]
    dist_R1 = choices[:, 1]
    dist_total_L2 = choices[:, 2]
    dist_total_R2 = choices[:, 3]
    rt = choices[:, 4]
    choice_made = choices[:, 5]

    valid_mask = rt>0
    
    if step == 1:
        chosen_dist = np.where(choice_made == 0, dist_L1, dist_R1)
        xlabel = '1-Step Distance Traveled'
    elif step == 2:
        chosen_dist = np.where(choice_made == 0, dist_total_L2, dist_total_R2)
        xlabel = 'Total 2-Step Path Distance'
    else:
        raise ValueError("The 'step' parameter must be 1 or 2.")

    x_data = chosen_dist[valid_mask]
    y_data = rt[valid_mask]
    choices_valid = choice_made[valid_mask]

    fig = plt.figure(figsize=(7, 5), dpi=300)
    
    idx_L = choices_valid == 0
    idx_R = choices_valid == 1
    
    plt.scatter(x_data[idx_L], y_data[idx_L], alpha=0.4, label='Chose Left', color='blue')
    plt.scatter(x_data[idx_R], y_data[idx_R], alpha=0.4, label='Chose Right', color='orange')
    
    plt.xlabel(xlabel)
    plt.ylabel('Reaction Time (ms)')
    plt.title(f'Reaction Time vs. {xlabel}')
    plt.legend()
    plt.tight_layout()
    
    return fig

#%% load data

#fnm = '../logs/unknown-2026-01-30T20-34-28-374Z-jr71.json'
#fnm = '../logs/unknown-2026-02-10T20-42-37-419Z-99s9.json'
#fnm = '../analysis/2-11-trial1.json'
#fnm = '../logs/unknown-2026-02-12T19-44-31-094Z-suyb.json'
#fnm = '../logs/unknown-2026-02-12T19-51-47-046Z-10wy.json'
# fnm = '../logs/unknown-2026-02-12T20-45-28-597Z-ash0.json'
#fnm = '../logs//RAH-2026-02-15T15-42-43-790Z-oopr.json'
#fnm = '../logs/EMU/YFV-2026-02-20T20-13-23-929Z-iel9.json'
#fnm = 'real_trial1.json'
fnm = '../data/cloud_study/65D6694BE06947289BE4336BC1DE271A-019e9464-b9d3-798d-aa65-c87d82961db6-019e8386-74e7-7359-827b-6b4e4bc47db9-2026-06-04T21-03-48-346Z-fg8d.json'
data = load(fnm)

#%% visualize task

plt.figure(figsize=(1,6), dpi=300)
for block in data['blocks']:
    trials = block['trials']
    for i, trial in enumerate(trials[:100]):
        hole_locs = sorted(trial['hole_locations'])
        segments = np.ones(12)
        for hole in hole_locs:
            segments[hole] = 0
        for j,s in enumerate(segments):
            if s == 0:
                continue
            plt.plot([j-0.5, j+0.5], [i,i], 'k-')
        if 'events' in trial and len(trial['events']) > 0:
            hole_used = [event['holeUsed'] for event in trial['events'] if 'holeUsed' in event]
            if len(hole_used) == 0:
                continue
            plt.plot(hole_used[0], i, 'r.', markersize=2)
        else:
            plt.plot(np.arange(len(segments)), np.full(len(segments), i), 'b-', markersize=2, alpha=0.5)
    break
# flip y-axis
plt.gca().invert_yaxis()
plt.axis('off')

#%% plot psychometric curve (greedy)

# plot psychometric curve
choices = get_choices(data)
X = choices[:,0] - choices[:,1]
# X = choices[:,2] - choices[:,3]
y = choices[:,-1]
plot_psychometric_curve(X, y)

#%% plot 2D choice heatmap (L dist vs R dist, greedy)

# heatmap where choices[:,:2] are the coordinates and choices[:,2] is the value
plt.figure(figsize=(3,3), dpi=300)
# plt.tricontourf(choices[:,0], choices[:,1], choices[:,2], levels=10, cmap='viridis')

xs = choices[:,0]
ys = choices[:,1]
zs = choices[:,2]

xs_all = np.unique(xs)
ys_all = np.unique(ys)
Z = np.full((len(ys_all), len(xs_all)), np.nan)

for i, xi in enumerate(xs_all):
	for j, yi in enumerate(ys_all):
		ix = (xs == xi) & (ys == yi)
		z = np.nanmean(zs[ix])
		Z[j, i] = (z - 0.5) * 2  # scale to -1 to 1
    # ix = np.where(xs_all == xi)[0][0]
    # iy = np.where(ys_all == yi)[0][0]
    # Z[iy, ix] = np.nanmean(zs[(xs == xi) & (ys == yi)])

# Step 2: plot with no interpolation
plt.imshow(
    Z,
    origin='lower',
    interpolation='nearest',  # <- NO smoothing
	cmap='RdBu',
    extent=[xs_all.min(), xs_all.max(), ys_all.min(), ys_all.max()]
)
plt.clim([-1,1])

# plt.xticks([0, np.max(choices[:,0])])
# plt.yticks([0, np.max(choices[:,1])])
plt.colorbar()
plt.xlabel('Distance to Left Hole')
plt.ylabel('Distance to Right Hole')
plt.show()

#%% compare greedy vs rollout heatmaps

choices = compare_greedy_vs_rollout(data, only_use_disagreements=True)
fig = plt.figure(figsize=(3,3), dpi=300)

X = choices[:,0] - choices[:,1]
y = choices[:,-1]
plot_psychometric_curve(X, y, fig=fig, color='r')
X = choices[:,2] - choices[:,3]
y = choices[:,-1]
plot_psychometric_curve(X, y, fig=fig, color='b')

#%% plot psychometric curves for greedy vs rollout

choices = compare_greedy_vs_rollout(data, only_use_disagreements=True)
fig = plt.figure(figsize=(3,3), dpi=300)
clrs = ['g', 'r', 'b', 'm', 'k', 'c']
names = ['Greedy L', 'Greedy R', 'Rollout L', 'Rollout R', 'Greedy (L-R)', 'Rollout (L-R)']

y = choices[:,-1]

for d in range(len(names)):
	if d < 4:
		X = choices[:,d]
	elif d == 4:
		plt.legend(fontsize=6)
		fig = plt.figure(figsize=(3,3), dpi=300)
		X = choices[:,0] - choices[:,1]
	elif d == 5:
		X = choices[:,2] - choices[:,3]
	plot_psychometric_curve(X, y, fig=fig, color=clrs[d], xlabel='Distance to hole', label=names[d])

plt.legend(fontsize=6)

# %%
choices = compare_greedy_vs_rollout(data, only_use_disagreements=False)

dist_total_L2 = choices[:, 2]
dist_total_R2 = choices[:, 3]
rt = choices[:, 4]

X = np.abs(dist_total_L2 - dist_total_R2) 
y = rt

plot_rt_vs_conflict(X, y)
plt.show()
# %%
choices = compare_greedy_vs_rollout(data, only_use_disagreements=True)

dist_total_L2 = choices[:, 2]
dist_total_R2 = choices[:, 3]
dist_L1 = choices[:, 0]
dist_R1 = choices[:, 1]
rts = choices[:, 4]

conflicts = np.abs(dist_total_L2 - dist_total_R2) 
trial_nums = np.arange(len(rts))

plot_rt_over_time_by_conflict(trial_nums, conflicts, rts, window_size=20)
plt.show()

# %%
plot_rt_vs_distance(choices, step = 2)
plt.show()
plot_rt_vs_distance(choices, step = 1)
plt.show()
# %%
plot_rt_residuals_vs_conflict(choices)

# %%
plot_rt_residuals_over_time(choices, window_size= 10)
plt.show()
# %%

def plot_rt_min_residuals_vs_conflict(choices, fig=None):
    # 1. Extract columns from the compare_greedy_vs_rollout output
    dist_L1 = choices[:, 0]
    dist_R1 = choices[:, 1]
    dist_total_L2 = choices[:, 2]
    dist_total_R2 = choices[:, 3]
    rts = choices[:, 4]
    choice_made = choices[:, 5]

    # Filter out trials with NaN reaction times
    valid_mask = ~np.isnan(rts)
    dist_L1, dist_R1 = dist_L1[valid_mask], dist_R1[valid_mask]
    dist_total_L2, dist_total_R2 = dist_total_L2[valid_mask], dist_total_R2[valid_mask]
    rts = rts[valid_mask]
    choice_made = choice_made[valid_mask]

    # 2. Identify the total distance traveled over 2-steps
    chosen_dist_2 = np.where(choice_made == 0, dist_total_L2, dist_total_R2)

    # 3. Calculate the new "Residual" (Actual RT - Minimum RT for that distance)
    residuals = np.zeros_like(rts)
    unique_distances = np.unique(chosen_dist_2)
    
    for d in unique_distances:
        # Find all trials where the user traveled exactly this distance
        idx = chosen_dist_2 == d
        
        # Find their absolute fastest reaction time for this distance
        min_rt_for_d = np.min(rts[idx])
        
        # The residual is how much slower this specific trial was compared to their best
        residuals[idx] = rts[idx] - min_rt_for_d
    
    #plt.scatter(unique_distances)

    # 4. Define Conflict (1-step vs 2-step margin difference)
    conflict_1step = np.abs(dist_L1 - dist_R1)
    conflict_2step = np.abs(dist_total_L2 - dist_total_R2)
    conflicts = np.abs(conflict_1step - conflict_2step)
    
    conflicts = np.round(conflicts, decimals=5)
    unique_conflicts = np.unique(conflicts)

    if fig is None:
        fig = plt.figure(figsize=(8, 5), dpi=300)

    # Plot raw trial residuals in the background
    plt.scatter(conflicts, residuals, alpha=0.2, color='gray', label='Trial Residuals')

    # Calculate and plot mean residuals for each conflict level
    mean_residuals = []
    ses = []
    
    for c in unique_conflicts:
        ix = conflicts == c
        res_c = residuals[ix]
        
        mean_val = np.mean(res_c)
        se_val = np.std(res_c) / np.sqrt(len(res_c)) if len(res_c) > 0 else 0
        
        mean_residuals.append(mean_val)
        ses.append(se_val)
        
        # Error bar
        plt.plot([c, c], [mean_val - se_val, mean_val + se_val], '-', color='purple', alpha=0.7)

    # Line connecting the means
    plt.plot(unique_conflicts, mean_residuals, 'o-', color='purple', linewidth=2, markersize=6, label='Mean Delay')

    # Baseline at 0 (representing the absolute minimum time)
    plt.axhline(0, color='black', linestyle='--', linewidth=1.5, label='Theoretical Minimum (0 Delay)')

    # Labels and Formatting
    plt.xlabel('Conflict Margin (|1-Step Diff - 2-Step Diff|)')
    plt.ylabel('Delay above Minimum RT (ms)')
    plt.title('Cognitive Delay vs. Decision Conflict')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    return fig

# %%
plot_rt_min_residuals_vs_conflict(choices)

#%% load choice times

def get_chosen_hole(trial):
    if 'events' in trial and len(trial['events']) > 0:
        assert len(trial['events']) == 1, "Expected exactly one event per trial for RT analysis"
        event = trial['events'][0]
        return event.get('holeUsed', None)
    elif 'holeUsed' in trial:
        return trial['holeUsed']
    return None

def get_time_passed_thru(trial):
    if 'events' in trial and len(trial['events']) > 0:
        assert len(trial['events']) == 1, "Expected exactly one event per trial for RT analysis"
        event = trial['events'][0]
        return event.get('time', None)
    elif 'timePassedThru' in trial:
        return trial['timePassedThru']
    return None
if __name__ == '__main__':

    total_trial_index = 0
    results = []
    nfailed = 0
    ngood = 0
    for block in data['blocks']:
        trials = block['trials']
        if len(trials) < 10:
            continue

        has_decisions = any('hole_locations' in trial and len(trial['hole_locations']) == 2 for trial in trials)
        pct_cors = []
        for i, trial in enumerate(trials):
            total_trial_index += 1
            if i == 0 or trials[i]['block_index'] != trials[i-1]['block_index']:
                continue
            if i+1 == len(trials):
                continue
        
            is_decision = len(trial['hole_locations']) == 2 and len(trials[i-1]['hole_locations']) == 1
            is_fixed = len(trial['hole_locations']) == 1 and len(trials[i-1]['hole_locations']) == 1
            if not (is_decision or is_fixed):
                continue

            curHole = get_chosen_hole(trial)
            origHole = get_chosen_hole(trials[i-1])
            distTraveled = np.abs(curHole - origHole) if curHole is not None and origHole is not None else np.nan

            if has_decisions and is_decision:
                nonchosenHole = [h for h in trial['hole_locations'] if h != curHole][0]
                distToNonchosen = np.abs(nonchosenHole - origHole)
                if distToNonchosen > distTraveled:
                    pct_cors.append(1)
                else:
                    pct_cors.append(0)

            rt = get_time_passed_thru(trial) - get_time_passed_thru(trials[i-1])
            if rt < 0:
                # print(trials[i-1])
                # print('-----')
                # print(trials[i])
                # print('========')
                nfailed += 1
                continue
            else:
                ngood += 1
            results.append({'rel_trial_index': i, 'trial_index': total_trial_index, 'block_index': trial['block_index'], 'decision_block': has_decisions, 'rt': rt, 'origHole': origHole, 'curHole': curHole, 'distTraveled': distTraveled, 'is_decision': is_decision})
    
        if has_decisions:
            print(f"Block {block['block_index']} had {len(trials)} trials with decisions, pct correct: {np.mean(pct_cors):.2f}, mean RT: {np.mean([r['rt']/1000 for r in results if r['block_index'] == block['block_index']]):.2f} s")

    print(f"Extracted RTs for {ngood} valid trial pairs, skipped {nfailed} pairs with negative RTs.")

    #%%

    # for each distTraveled, plot the distribution of RTs (e.g. boxplot or violin plot)
    # separately for decision_blocks vs non-decision blocks

    plt.figure(figsize=(6,4), dpi=300)
    for decision_block in [False, True]:
    
        dists = []
        rts = []
        for res in results:
            if res['decision_block'] == decision_block and not np.isnan(res['distTraveled']) and not np.isnan(res['rt']):
                if res['rt'] > 5000:
                    continue
                dists.append(res['distTraveled'])
                rts.append(res['rt'])

        # plot violin plot of RTs for each distance traveled
        unique_dists = np.unique(dists)
        data_to_plot = [ [rts[i] for i in range(len(rts)) if dists[i] == ud] for ud in unique_dists ]
        plt.violinplot(data_to_plot, positions=unique_dists, showmeans=True)

        plt.xlabel('Distance Traveled')
        plt.ylabel('Reaction Time (ms)')
        plt.tight_layout()
    plt.show()

    #%%

    # for each (origHole, curHole) pair, compare RTs during decision blocks vs non-decision blocks (e.g. bar plot with error bars)

    rt_holes = {}
    for res in results:
        if not np.isnan(res['origHole']) and not np.isnan(res['curHole']) and not np.isnan(res['rt']):
            if res['rt'] > 5000:
                continue
            key = (res['origHole'], res['curHole'])
            if key not in rt_holes:
                rt_holes[key] = {'decision': [], 'non_decision': []}
            if res['decision_block']:
                rt_holes[key]['decision'].append(res['rt'])
            else:
                rt_holes[key]['non_decision'].append(res['rt'])

    plt.figure(figsize=(3,3), dpi=300)
    for key, rt_dict in rt_holes.items():
        decision_rts = rt_dict['decision']
        non_decision_rts = rt_dict['non_decision']
        if len(decision_rts) > 0 and len(non_decision_rts) > 0:
            means = [np.mean(non_decision_rts), np.mean(decision_rts)]
            ses = [np.std(non_decision_rts)/np.sqrt(len(non_decision_rts)), np.std(decision_rts)/np.sqrt(len(decision_rts))]
            # decision on x, non-decision on y
            plt.plot(means[0], means[1], '.', color='blue')

    ymx = 3000
    plt.axis('square')
    plt.plot([0, ymx], [0, ymx], 'k-', zorder=-1, alpha=0.5)
    plt.xlim([0, ymx])
    plt.ylim([0, ymx])
    plt.xlabel('Mean RT (Non-Decision Blocks)')
    plt.ylabel('Mean RT (Decision Blocks)')
    plt.tight_layout()
    plt.show()
