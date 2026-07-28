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
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

import statsmodels.api as sm
from sklearn.metrics import log_loss

import ssm
from ssm.util import find_permutation



# %%
# simply loads data
def load(fnm):
	return json.load(open(fnm))

def get_choices(data):
    """Pre-Processes Experiment Data

    Returns
        Right choice distance and left choice distance from the entry hole.
        The actual choice made.
        Reaction time.
    """
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
    """Pre-Processes Experiment Data

    Returns
        - Right choice distance and left choice distance from the entry hole.
        - 2-step distances from the beginning of the trial
        - The actual choice made
        - Reaction time: calculated the time passing through the choice level minus the time passing through the entry funnel
    """
    choices = []
    for block in data['blocks']:
        if len(block['trials']) == 4:
            continue
        
        trials = block['trials']
        for i, trial in enumerate(trials):
            if i == len(trials)-1:
                continue
            if 'holes' in trial:
                hole_locs = sorted(trial['holes']['hole_locations'])
                prev_hole_locs = sorted(trials[i-1]['holes']['hole_locations'])
                next_hole_locs = sorted(trials[i+1]['holes']['hole_locations'])
                h_cur = trial['holeUsed']
                h_prev = trials[i-1]['holeUsed']
                h_next = trials[i+1]['holeUsed']
            elif 'events' in trial and len(trial['events']) > 0 and len(trials[i-1]['events']) > 0 and len(trials[i+1]['events']) > 0:
                #print(i)
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


# %%
# data analysis block

def plot_psychometric_curve(X, y, fig=None, color='k', xlabel='Δ Distance to hole (L - R)', label='_'):
    """Plots a Psychometric curve

    Inputs:
        - X: difference between the 1-step left distance and 1-step right distance
        - y: the actual choice made

    Returns:
        - A psychometric curve
    """

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
    """Raw reaction time vs conflict (probably not helpful because it doesn't take into account the actual time it takes to move)
    """
    
    
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

def plot_rt_residuals_vs_conflict(choices, fig=None, rt_regression_type = 'linear'):
    """ Regress residuals against the distance and then plot that against the 'conflict'

    The regression is on the 1-step distance because that's what the reaction time is actually measured against

    Inputs:
        - Take the output of compare_greedy_vs_rollout
        - rt_regression_type determines whether we 'regress' the residuals on distance by taking the linear relationship or by subtracting the minimum amount of time it took to travel that distance

    Outputs:
        - Plots a graph where the y-axis is the residual of the regression, the x-axis is the 'conflict'2-11-trial1.json

    NOTE: I gotta figure out if I'm calculating conflict correctly or not
    """
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
    chosen_dist_1 = np.where(choice_made == 0, dist_L1, dist_R1)
    predicted_rts = np.zeros(chosen_dist_1.shape)

    # 3. Regress the reaction time vs the chosen 2-step distance
    if rt_regression_type == 'linear':
        slope, intercept = np.polyfit(chosen_dist_1, rts, 1)
        predicted_rts = (slope * chosen_dist_1) + intercept
    elif rt_regression_type == 'minimum':
        unique_distances = np.unique(chosen_dist_1)
        for d in unique_distances:
            idx = chosen_dist_1 == d

            min_rt_for_d = np.min(rts[idx])
            predicted_rts[idx] = min_rt_for_d

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

    plt.close(fig)
    
    return fig

def plot_rt_over_time_by_conflict(trial_nums, conflicts, rts, window_size=10, fig=None):
    """
    Similarly obsolete as the previous plot_rt_vs_conflict function 
    """
    
    
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

def plot_rt_residuals_over_time(choices, window_size=10, fig=None, rt_regression_type = 'linear'):
    """The 'residuals' of the reaction time plotted over time as a moving average

    Inputs:
        - Takes the output of compare_greedy_vs_rollout for 'choices'
        - The window size determines the moving average
        - rt_regression_type is 'linear' if we want to calculate the predicted residuals via linear regression, or 'minimum' if we want to compare against the minimum amount of time taken to move 

    Outputs:
        - Moving average plot of reaction times over expeceted reaction time, separated by the 'degree of conflict'2-11-trial1.json

    NOTE: Is it valid to plot moving averages like this?
    """
    
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
    chosen_dist_1 = np.where(choice_made == 0, dist_L1, dist_R1)
    slope, intercept = np.polyfit(chosen_dist_1, rts, 1)

    predicted_rts = np.zeros_like(rts)

    if rt_regression_type == 'linear':
        slope, intercept = np.polyfit(chosen_dist_1, rts, 1)
        predicted_rts = (slope * chosen_dist_1) + intercept
    elif rt_regression_type == 'minimum':
        unique_distances = np.unique(chosen_dist_1)
        for d in unique_distances:
            idx = chosen_dist_1 == d

            min_rt_for_d = np.min(rts[idx])
            predicted_rts[idx] = min_rt_for_d

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

    plt.close(fig)
    
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

def plot_conflict_vs_rt_heatmap(choices, rt_regression_type='linear'):
    dist_L1 = choices[:, 0]
    dist_R1 = choices[:, 1]
    dist_total_L2 = choices[:, 2]
    dist_total_R2 = choices[:, 3]
    rts = choices[:, 4]
    choice_made = choices[:, 5]

    conflict_1step = dist_L1 - dist_R1
    conflict_2step = dist_total_L2 - dist_total_R2

    chosen_dist_1 = np.where(choice_made == 0, dist_L1, dist_R1)
    
    predicted_rts = np.zeros_like(rts)

    if rt_regression_type == 'linear':
        slope, intercept = np.polyfit(chosen_dist_1, rts, 1)
        predicted_rts = (slope * chosen_dist_1) + intercept
    elif rt_regression_type == 'minimum':
        unique_distances = np.unique(chosen_dist_1)
        for d in unique_distances:
            idx = chosen_dist_1 == d
            min_rt_for_d = np.min(rts[idx])
            predicted_rts[idx] = min_rt_for_d

    residuals = rts - predicted_rts

    unique_1step_conflicts = np.sort(np.unique(conflict_1step))
    unique_2step_conflicts = np.sort(np.unique(conflict_2step))

    rt_heatmap = np.zeros((unique_1step_conflicts.size, unique_2step_conflicts.size))

    for r_idx, val_1step in enumerate(unique_1step_conflicts):
        for c_idx, val_2step in enumerate(unique_2step_conflicts):
            
            idx = (conflict_1step == val_1step) & (conflict_2step == val_2step)
            
            if np.sum(idx) > 0:
                rt_heatmap[r_idx, c_idx] = np.mean(residuals[idx])
            else:
                rt_heatmap[r_idx, c_idx] = np.nan

    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(rt_heatmap, origin='lower', cmap='viridis')
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Mean RT Residual", rotation=-90, va="bottom")

    ax.set_xticks(range(len(unique_2step_conflicts)))
    ax.set_xticklabels(np.round(unique_2step_conflicts, 2), rotation=45)
    ax.set_xlabel('2-Step Conflict (L2-R2)')

    ax.set_yticks(range(len(unique_1step_conflicts)))
    ax.set_yticklabels(np.round(unique_1step_conflicts, 2))
    ax.set_ylabel('1-Step Conflict (L1-R1)')

    ax.set_title("RT Residuals by 1-Step and 2-Step Conflict")
    
    fig.tight_layout()
    plt.show()
            


# %%
if __name__ == "__main__":
    data = load("YFW-2026-04-29T17-01-30-883Z-ikoe.json")
    data = load("YFX-2026-05-13T22-06-02-493Z-461m.json")

    choices = compare_greedy_vs_rollout(data, only_use_disagreements=False)

    #plot_rt_residuals_vs_conflict(choices)
    rt_over_time_minimum_30 = plot_rt_residuals_over_time(choices, window_size= 10, rt_regression_type= 'minimum')
    display(rt_over_time_minimum_30)

    rt_vs_conflict_minimum = plot_rt_residuals_vs_conflict(choices, rt_regression_type= 'minimum')
    display(rt_vs_conflict_minimum)

    choices = compare_greedy_vs_rollout(data, only_use_disagreements=False)
    plot_conflict_vs_rt_heatmap(choices)

# %%
if __name__ == "__main__":
    #data = load("Mani_NoFutureSight_Trial.json")
    #data['blocks'][1]['trials'][7]['events'][0]
    block = data['blocks'][1]
    trials = block['trials']
    #len(trials)
    trials[0]


# %%
def pre_proccess_data_from_choice_vs_no_choice(data):
    output = []

    for block_num, block in enumerate(data['blocks']):
        if block_num == 0:
            continue

        trials = block['trials']

        for i in range(len(trials) // 3):
            choice_trial_sequence = [
                trials[3*i]['hole_locations'],
                trials[3*i+1]['hole_locations'],
                trials[3*i+2]['hole_locations']
            ]
            
            chosen_path = [
                trials[3*i]['events'][0]['holeUsed'], 
                trials[3*i+1]['events'][0]['holeUsed'],
                trials[3*i+2]['events'][0]['holeUsed']
            ]
            
            observed_rt = trials[3*i+2]['events'][0]['time'] - trials[3*i]['events'][0]['time']
            
            # does the middle trial have two options
            is_choice = len(trials[3*i+1]['hole_locations']) == 2

            if is_choice:
                options = trials[3*i+1]['hole_locations']
                chosen_hole = trials[3*i+1]['events'][0]['holeUsed']
                
                unchosen_hole = options[0] if options[0] != chosen_hole else options[1]
                
                non_chosen_path = [
                    trials[3*i]['events'][0]['holeUsed'], 
                    unchosen_hole, 
                    trials[3*i+2]['events'][0]['holeUsed']
                ]
                
                chosen_1step_dist = abs(chosen_path[1] - chosen_path[0])
                unchosen_1step_dist = abs(non_chosen_path[1] - non_chosen_path[0])
                
                chosen_2step_dist = chosen_1step_dist + abs(chosen_path[2] - chosen_path[1])
                unchosen_2step_dist = unchosen_1step_dist + abs(non_chosen_path[2] - non_chosen_path[1])
                
            else:
                non_chosen_path = None
                chosen_1step_dist = None
                unchosen_1step_dist = None
                chosen_2step_dist = None
                unchosen_2step_dist = None
            
            output.append({
                'block_number': block_num,
                'trial_sequence_number': i,
                'hole_sequence': choice_trial_sequence,
                'chosen_path': chosen_path,
                'non_chosen_path': non_chosen_path,
                'observed_rt': observed_rt,
                'choice_trial': is_choice,
                'chosen_1step_dist': chosen_1step_dist,
                'unchosen_1step_dist': unchosen_1step_dist,
                'chosen_2step_dist': chosen_2step_dist,
                'unchosen_2step_dist': unchosen_2step_dist
            })
    return output


# %%
choice_vs_no_choice = pre_proccess_data_from_choice_vs_no_choice(data)
#choice_vs_no_choice[3]

choice_vs_no_choice_df = pd.DataFrame(choice_vs_no_choice)

choice_sequences = choice_vs_no_choice_df[choice_vs_no_choice_df['choice_trial'] == True]
#sequence1 = choice_sequences[1]

#no_choice_sequences = choice_vs_no_choice_df[choice_vs_no_choice_df['choice_trial'] == False]
#no_choice_sequences

# %% [markdown]
# # Reaction Times

# %%
def calculate_approx_time_planning(choice_vs_no_choice_df):
    choice_sequences = choice_vs_no_choice_df[choice_vs_no_choice_df['choice_trial'] == True]
    no_choice_sequences = choice_vs_no_choice_df[choice_vs_no_choice_df['choice_trial'] == False]

    planning_times = []

    for index, sequence in choice_sequences.iterrows():
        current_path = sequence['chosen_path']
        
        path_mask = no_choice_sequences['chosen_path'].apply(lambda x: x == current_path)
        relevant_no_choice = no_choice_sequences[path_mask]
        
        if not relevant_no_choice.empty:
            baseline_rt = relevant_no_choice['observed_rt'].mean()
            planning_times.append(sequence['observed_rt'] - baseline_rt)
        else:

            planning_times.append(np.nan)

    return planning_times

def compare_different_path_execution_times(choice_vs_no_choice_df):
    choice_sequences = choice_vs_no_choice_df[choice_vs_no_choice_df['choice_trial'] == True]
    no_choice_sequences = choice_vs_no_choice_df[choice_vs_no_choice_df['choice_trial'] == False]

    execution_time_diffs = []

    for index, sequence in choice_sequences.iterrows():
        current_path_chosen = sequence['chosen_path']
        current_path_not_chosen = sequence['non_chosen_path']
        
        path_mask1 = no_choice_sequences['chosen_path'].apply(lambda x: x == current_path_chosen)
        relevant_no_choice_taken = no_choice_sequences[path_mask1]
        path_mask2 = no_choice_sequences['chosen_path'].apply(lambda x: x == current_path_not_chosen)
        relevant_no_choice_not_taken = no_choice_sequences[path_mask2]
        
        if not (relevant_no_choice_taken.empty and relevant_no_choice_not_taken.empty):
            rt1 = relevant_no_choice_taken['observed_rt'].mean()
            rt2 = relevant_no_choice_not_taken['observed_rt'].mean()
            execution_time_diffs.append(-rt1+rt2)
        else:

            execution_time_diffs.append(np.nan)

    return execution_time_diffs


# %%
planning_times = calculate_approx_time_planning(choice_vs_no_choice_df)
planning_times
# plt.hist(planning_times)

clean_times = [time for time in planning_times if not np.isnan(time)]

fig, ax = plt.subplots(figsize=(8, 5))

ax.hist(clean_times, bins=40, range=(-2000, 2000), edgecolor='black')

ax.set_title('Distribution of Planning Times')
ax.set_xlabel('Planning Time vs Baseline (ms)')
ax.set_ylabel('Frequency (Number of Trials)')

plt.show()

# %%
execution_time_diffs = compare_different_path_execution_times(choice_vs_no_choice_df)
#planning_times
# plt.hist(planning_times)

clean_times = [time for time in execution_time_diffs if not np.isnan(time)]

fig, ax = plt.subplots(figsize=(8, 5))

ax.hist(clean_times, bins=40, range=(-2500, 2500), edgecolor='black')

ax.set_title('Distribution of Execution Time Differences')
ax.set_xlabel('(Not Chosen Path Time) - (Chosen Path Time)')
ax.set_ylabel('Frequency (Number of Trials)')

plt.show()

# %%
if __name__ == '__main__':
    choice_trials = choice_vs_no_choice_df[choice_vs_no_choice_df['choice_trial']]

    diff_1step = choice_trials['chosen_1step_dist'] - choice_trials['unchosen_1step_dist']
    diff_2step = choice_trials['chosen_2step_dist'] - choice_trials['unchosen_2step_dist']

    disagree_trials = choice_trials[(diff_1step * diff_2step) < 0]
    disagree_trials

    percent_greedy_choice = sum(disagree_trials['chosen_1step_dist'] <= disagree_trials['unchosen_1step_dist'])/len(disagree_trials)

    percent_planning_choice = sum(disagree_trials['chosen_2step_dist'] <= disagree_trials['unchosen_2step_dist'])/len(disagree_trials)

    print(percent_greedy_choice)
    print(percent_planning_choice)

    # %%
    choice_trials

    # %% [markdown]
    # # Predicting Left vs Right Choices

    # %%
    ### Whether Left or Right was Picked, and Left vs Right Distances

    chosen_middle = choice_vs_no_choice_df['chosen_path'].str[1]
    unchosen_middle = choice_vs_no_choice_df['non_chosen_path'].str[1]

    choice_vs_no_choice_df['chosen_left'] = (chosen_middle < unchosen_middle) & (choice_vs_no_choice_df['choice_trial'])

    #choice_vs_no_choice_df['chosen_left']


    choice_trials

    ### Direction

    prev_end_hole = choice_vs_no_choice_df['chosen_path'].shift(1).str[2]
    curr_start_hole = choice_vs_no_choice_df['chosen_path'].str[0]

    direction = np.sign(prev_end_hole - curr_start_hole)

    prev_seq_num = choice_vs_no_choice_df['trial_sequence_number'].shift(1)
    curr_seq_num = choice_vs_no_choice_df['trial_sequence_number']

    prev_block = choice_vs_no_choice_df['block_number'].shift(1)
    curr_block = choice_vs_no_choice_df['block_number']

    is_valid_sequence = (prev_seq_num + 1 == curr_seq_num) & (prev_block == curr_block)

    choice_vs_no_choice_df['incoming_direction'] = np.where(is_valid_sequence, direction, np.nan)



    ### Putting together the dataframe
    choice_trials = choice_vs_no_choice_df[choice_vs_no_choice_df['choice_trial']]
    choice_trials = choice_trials.dropna()


    # %%

    #logistic_regression_df = pd.DataFrame()

    X = pd.DataFrame({
        'L1': np.where(choice_trials['chosen_left'], choice_trials['chosen_1step_dist'], choice_trials['unchosen_1step_dist']),
        'R1': np.where(~choice_trials['chosen_left'], choice_trials['chosen_1step_dist'], choice_trials['unchosen_1step_dist']),
    
        'L2': np.where(choice_trials['chosen_left'], 
                       choice_trials['chosen_2step_dist'] - choice_trials['chosen_1step_dist'], 
                       choice_trials['unchosen_2step_dist'] - choice_trials['unchosen_1step_dist']),
        'R2': np.where(~choice_trials['chosen_left'], 
                       choice_trials['chosen_2step_dist'] - choice_trials['chosen_1step_dist'], 
                       choice_trials['unchosen_2step_dist'] - choice_trials['unchosen_1step_dist']),

        'Direction': choice_trials['incoming_direction']
    })
    y = choice_trials['chosen_left']

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.2)

    model = sklearn.linear_model.LogisticRegression().fit(X_train, y_train)
    #predictions = model.predict(X_test)

    print("======MODEL 1 RESULTS======\n\n")

    accuracy = model.score(X_test, y_test)
    print(f'Accuracy:', accuracy,'\n')

    print(f'Coeffecients:\n', pd.Series(model.coef_[0], index=X_train.columns),'\n')
    print(f'Model Intercept: ', model.intercept_)
    predictions = model.predict(X)
    print(f'Confusion Matrix: ', sklearn.metrics.confusion_matrix(predictions, y)/len(predictions), '\n')

    y_probs = model.predict_proba(X_test)

    loss = log_loss(y_test, y_probs)

    total_log_likelihood = -len(y_test) * loss

    print(f'Total Log Likelihood: ', total_log_likelihood)
    X2 = pd.DataFrame({'L1-R1': X['L1']-X['R1'],
                       'Direction': X['Direction']})

    X3 = pd.DataFrame({'L1+L2-R1-R2': X['L1']+X['L2']-X['R1']-X['R2'],
                       'Direction': X['Direction']})

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X2, y, test_size = 0.2)

    model = sklearn.linear_model.LogisticRegression(penalty=None).fit(X_train, y_train)
    #predictions = model.predict(X_test)

    print("\n\n======MODEL 2 RESULTS======\n\n")

    accuracy = model.score(X_test, y_test)
    print(f'Accuracy 2:', accuracy,'\n')

    print(f'Coeffecients 2:\n', pd.Series(model.coef_[0], index=X_train.columns),'\n')
    print(f'Model Intercept: ', model.intercept_)
    predictions = model.predict(X2)

    print(f'Confusion Matrix 2: ', sklearn.metrics.confusion_matrix(predictions, y)/len(predictions))

    y_probs = model.predict_proba(X_test)

    loss = log_loss(y_test, y_probs)

    total_log_likelihood = -len(y_test) * loss

    print(f'Total Log Likelihood: ', total_log_likelihood, '\n')

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X3, y, test_size = 0.2)

    model = sklearn.linear_model.LogisticRegression().fit(X_train, y_train)
    #predictions = model.predict(X_test)

    print("\n\n======MODEL 3 RESULTS======\n\n")

    accuracy = model.score(X_test, y_test)
    print(f'Accuracy 3:', accuracy,'\n')

    mle_coefficients = model.coef_
    mle_intercept = model.intercept_


    print(f'Coeffecients3:\n', pd.Series(model.coef_[0], index=X_train.columns),'\n', )
    print(f'Model Intercept: ', mle_intercept)
    predictions = model.predict(X3)
    print(f'Confusion Matrix3: ', sklearn.metrics.confusion_matrix(predictions, y)/len(predictions))

    y_probs = model.predict_proba(X_test)

    loss = log_loss(y_test, y_probs)

    total_log_likelihood = -len(y_test) * loss

    print(f'Total Log Likelihood: ', total_log_likelihood, '\n')

    # %% [markdown]
    # # Seeing if the Strategy is Switched

    # %%
    choice_trials = choice_vs_no_choice_df[choice_vs_no_choice_df['choice_trial']]
    sum(choice_trials['incoming_direction'].dropna())

    is_greedy_choice = choice_trials['chosen_1step_dist'] < choice_trials['unchosen_1step_dist']
    is_planned_choice = choice_trials['chosen_2step_dist'] < choice_trials['unchosen_2step_dist']

    strategy_switching_record = np.where(is_greedy_choice, 1, 0) - np.where(is_planned_choice, 1, 0)
    smoothed_switching_record = pd.DataFrame(strategy_switching_record).rolling(window=13).mean()

    plt.plot(range(len(choice_trials)), strategy_switching_record, linewidth = 1, color = 'orange')
    plt.plot(range(len(choice_trials)), smoothed_switching_record, linewidth = 3, color = 'red')
    plt.title('Strategy Time Series')
    plt.ylabel('Moving Average Strategy\n(Positive Indicates Using the Greedy Strategy)')
    plt.xlabel('Trial Number')
    plt.show()

    # %%
    sum(choice_trials['chosen_left'])/len(choice_trials)

    # %%
    ARMAmodel1 = auto_arima(smoothed_switching_record.dropna(), seasonal= False) 
    ARMAmodel2 = auto_arima(strategy_switching_record, seasonal= False) 

    #plt.plot(smoothed_switching_record, label='Actual')
    #plt.plot(results.predict(), label='Fitted')
    #plt.legend()
    #plt.show()

    print(f'Smoothed Input Model: ', ARMAmodel1.summary())
    print(f'Normal Model: ', ARMAmodel2.summary())

    # %% [markdown]
    # # Fitting a GLM HMM

    # %%
    data = load("YFX-2026-05-13T22-06-02-493Z-461m.json")

    choice_vs_no_choice = pre_proccess_data_from_choice_vs_no_choice(data)
    #choice_vs_no_choice[3]

    choice_vs_no_choice_df = pd.DataFrame(choice_vs_no_choice)
    chosen_middle = choice_vs_no_choice_df['chosen_path'].str[1]
    unchosen_middle = choice_vs_no_choice_df['non_chosen_path'].str[1]

    choice_vs_no_choice_df['chosen_left'] = (chosen_middle < unchosen_middle) & (choice_vs_no_choice_df['choice_trial'])

    #choice_vs_no_choice_df['chosen_left']

    ### Direction

    prev_end_hole = choice_vs_no_choice_df['chosen_path'].shift(1).str[2]
    curr_start_hole = choice_vs_no_choice_df['chosen_path'].str[0]

    direction = np.sign(prev_end_hole - curr_start_hole)

    prev_seq_num = choice_vs_no_choice_df['trial_sequence_number'].shift(1)
    curr_seq_num = choice_vs_no_choice_df['trial_sequence_number']

    prev_block = choice_vs_no_choice_df['block_number'].shift(1)
    curr_block = choice_vs_no_choice_df['block_number']

    is_valid_sequence = (prev_seq_num + 1 == curr_seq_num) & (prev_block == curr_block)

    choice_vs_no_choice_df['incoming_direction'] = np.where(is_valid_sequence, direction, np.nan)



    ### Putting together the dataframe
    choice_trials = choice_vs_no_choice_df[choice_vs_no_choice_df['choice_trial']]
    choice_trials = choice_trials.dropna()

    X = pd.DataFrame({
        'L1': np.where(choice_trials['chosen_left'], choice_trials['chosen_1step_dist'], choice_trials['unchosen_1step_dist']),
        'R1': np.where(~choice_trials['chosen_left'], choice_trials['chosen_1step_dist'], choice_trials['unchosen_1step_dist']),
    
        'L2': np.where(choice_trials['chosen_left'], 
                       choice_trials['chosen_2step_dist'] - choice_trials['chosen_1step_dist'], 
                       choice_trials['unchosen_2step_dist'] - choice_trials['unchosen_1step_dist']),
        'R2': np.where(~choice_trials['chosen_left'], 
                       choice_trials['chosen_2step_dist'] - choice_trials['chosen_1step_dist'], 
                       choice_trials['unchosen_2step_dist'] - choice_trials['unchosen_1step_dist']),

        'Direction': choice_trials['incoming_direction']
    })
    y = choice_trials['chosen_left']

    # %%
    choices = []
    for choice in y:
        choices.append([int(choice)])
    choices = [np.array(choices)]

    # %%
    num_states = 2
    obs_dim = 1
    num_categories = 2
    input_dim = 4

    full_inpts = np.ones((len(choice_trials), 4))

    full_inpts[:, 0] = -(X['L1'] - X['R1'])
    full_inpts[:, 1] = -(X['L2'] - X['R2'])
    full_inpts[:, 2] = X['Direction']


    # %%
    new_glmhmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs", 
                       observation_kwargs=dict(C=num_categories), transitions="standard")

    N_iters = 200
    fit_ll = new_glmhmm.fit(choices, inputs=full_inpts, method="em", num_iters=N_iters, tolerance=10**-4)

    # %%
    fig = plt.figure(figsize=(4, 3), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(fit_ll, label="EM")
    plt.legend(loc="lower right")
    plt.xlabel("EM Iteration")
    plt.xlim(0, len(fit_ll))
    plt.ylabel("Log Probability")
    plt.show()

    # %%
    #new_glmhmm.permute(find_permutation(true_latents[0], new_glmhmm.most_likely_states(true_choices[0], input=inpts_per_session[0])))

    fig = plt.figure(figsize=(4, 3), dpi=80, facecolor='w', edgecolor='k')
    cols = ['#ff7f00', '#4daf4a', '#377eb8']
    recovered_weights = new_glmhmm.observations.params
    for k in range(num_states):
        if k ==0:
            plt.plot(range(input_dim), recovered_weights[k][0], color=cols[k],
                         lw=1.5,  label = "recovered", linestyle = '--')
        else:
            plt.plot(range(input_dim), recovered_weights[k][0], color=cols[k],
                         lw=1.5,  label = '', linestyle = '--')
    plt.yticks(fontsize=10)
    plt.ylabel("GLM weight", fontsize=15)
    plt.xlabel("covariate", fontsize=15)
    plt.xticks([0, 1, 2, 3], ['L1-R1', 'L2-R2', 'direction', 'bias'], fontsize=12, rotation=45)
    plt.axhline(y=0, color="k", alpha=0.5, ls="--")
    plt.legend()
    plt.title("Weight recovery", fontsize=15)

    # %%
    recovered_trans_mat = np.exp(new_glmhmm.transitions.log_Ps)
    plt.imshow(recovered_trans_mat, vmin=-0.8, vmax=1, cmap='bone')
    for i in range(recovered_trans_mat.shape[0]):
        for j in range(recovered_trans_mat.shape[1]):
            text = plt.text(j, i, str(np.around(recovered_trans_mat[i, j], decimals=2)), ha="center", va="center",
                            color="k", fontsize=12)
    plt.xlim(-0.5, num_states - 0.5)
    plt.xticks(range(0, num_states), ('1', '2'), fontsize=10)
    plt.yticks(range(0, num_states), ('1', '2'), fontsize=10)
    plt.ylim(num_states - 0.5, -0.5)
    plt.title("recovered", fontsize = 15)
    plt.subplots_adjust(0, 0, 1, 1)
