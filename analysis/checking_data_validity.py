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


# %% [markdown]
# # Importing Functions and Data

# %%
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
            dist_L2 = dist_L1 + np.abs(hole_locs[0] - h_next)
            dist_R2 = dist_R1 + np.abs(hole_locs[1] - h_next)
            if only_use_disagreements:
                # we ignore trials where both greedy and rollout would make the same choice
                if (dist_L1 < dist_R1 and dist_L2 < dist_R2) or (dist_R1 < dist_L1 and dist_R2 < dist_L2):
                    continue
            
            choice = hole_locs.index(h_cur)
            if 'timePassedThru' in trial:
                rt = trial['timePassedThru'] - trials[i-1]['timePassedThru']
            elif 'events' in trial and len(trial['events']) > 0 and len(trials[i-1]['events']) > 0:
                rt = trial['events'][0]['time'] - trials[i-1]['events'][0]['time']
            else:
                rt = np.nan
            choices.append((dist_L1, dist_R1, dist_L2, dist_R2, rt, choice))
    return np.vstack(choices)

def get_switches_per_level(data):

    switches_per_level = []

    for block in data['blocks']:
        if 'user_inputs' not in block:
            continue
            
        input_times = np.array(block['user_inputs']['time'])
        input_vals = np.array(block['user_inputs']['input'])
        
        trials = block['trials']
        
        for i, trial in enumerate(trials):
            if i == 0:
                continue
                
            if 'events' in trial and len(trial['events']) > 0 and len(trials[i-1]['events']) > 0:
                t_start = trials[i-1]['events'][0]['time']
                t_end = trial['events'][0]['time']
            elif 'timePassedThru' in trial:
                t_start = trials[i-1]['timePassedThru']
                t_end = trial['timePassedThru']
            else:
                continue 
            mask = (input_times > t_start) & (input_times <= t_end)
            level_inputs = input_vals[mask]
            
            if len(level_inputs) > 0:
                changes = np.diff(level_inputs)
                
                num_switches = np.count_nonzero(changes)
            else:
                num_switches = 0
                
            switches_per_level.append(num_switches)
            
    return np.array(switches_per_level)


# %% [markdown]
# # Analysis Functions

# %%
def plot_rt_residuals_histogram(choices, rt_regression_type='linear', bins=50, fig=None):
   
    # 1. Extract columns from the compare_greedy_vs_rollout output
    dist_L1 = choices[:, 0]
    dist_R1 = choices[:, 1]
    rts = choices[:, 4]
    choice_made = choices[:, 5]

    # Filter out trials with NaN reaction times
    valid_mask = ~np.isnan(rts)
    dist_L1, dist_R1 = dist_L1[valid_mask], dist_R1[valid_mask]
    rts = rts[valid_mask]
    choice_made = choice_made[valid_mask]

    # 2. Regress RT vs the chosen 1-step distance to get residuals
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

    if fig is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    else:
        # If a figure is passed in, clear it and add two subplots
        fig.clf()
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

    lower_bound = np.percentile(residuals, 1)
    upper_bound = np.percentile(residuals, 99)
    clean_residuals = residuals[(residuals >= lower_bound) & (residuals <= upper_bound)]

    ax1.hist(residuals, bins=bins, color='lightcoral', edgecolor='black', alpha=0.8)
    ax1.set_xlabel('Reaction Time Residual (ms)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('With Outliers')
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, label=f'Expected RT (N={len(residuals)})')
    ax1.legend(loc='upper right')

    ax2.hist(clean_residuals, bins=bins, color='skyblue', edgecolor='black', alpha=0.8)
    ax2.set_xlabel('Reaction Time Residual (ms)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('No Outliers')
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label=f'Expected RT (N={len(clean_residuals)})')
    ax2.legend(loc='upper right')

    # Ensure the plots don't squish together
    fig.tight_layout()
    
    return fig

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

def plot_switch_distribution(data):
    """
    Plots a histogram showing how frequently users switched directions per level.
    """
    switches = get_switches_per_level(data)
    
    if len(switches) == 0:
        print("No valid user inputs found in the data.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    
    max_switches = np.max(switches)
    bins = np.arange(-0.5, max_switches + 1.5, 1)
    
    ax.hist(switches, bins=bins, color='coral', edgecolor='black', alpha=0.8)
    
    # Formatting
    ax.set_xticks(range(max_switches + 1))
    ax.set_title('Distribution of Direction Switches Per Level')
    ax.set_xlabel('Number of Switches (per level)')
    ax.set_ylabel('Frequency (Number of Levels)')
    
    plt.tight_layout()
    plt.show()

def plot_trials_per_block(data, fig=None):
    """
    Extracts the number of trials in each block and plots it as a line graph.
    """
    block_numbers = []
    trial_counts = []

    # 1. Extract the data
    for i, block in enumerate(data['blocks']):
        # If your blocks are 0-indexed but you want the graph to start at Block 1, use i + 1
        block_numbers.append(i + 1) 
        
        # Count the number of trials in this specific block
        if 'trials' in block:
            trial_counts.append(len(block['trials']))
        else:
            trial_counts.append(0)  # Safe fallback if a block is totally empty

    block_numbers = block_numbers[1:]
    trial_counts = trial_counts[1:]

    # 2. Create the Plot
    if fig is None:
        fig = plt.figure(figsize=(8, 5), dpi=300)

    # Plot the line with 'o' markers to clearly show each individual block
    plt.plot(block_numbers, trial_counts, marker='o', linestyle='-', color='indigo', linewidth=2, markersize=6)

    # 3. Formatting
    plt.title('Number of Trials per Block')
    plt.xlabel('Block Number')
    plt.ylabel('Total Trials')
    
    # Force the X-axis to only show whole numbers (since you can't have Block 1.5)
    plt.xticks(block_numbers)
    
    # Add a subtle grid to make it easier to read across
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    return fig


# %% [markdown]
# # Merging Shitty Data Functions

# %%
import os
import glob
import json
from collections import defaultdict


# %%
def merge_participant_files(folder_path, participant_id=None, file_glob=None):
    """
    If participant_id is given, matches files with that prefix (old behavior).
    If file_glob is given (e.g. 'analysis/cloud study data 2/*'), uses glob directly.
    Otherwise treats folder_path as a directory and groups files by first 30 chars.

    Prints merge statistics including:
      - total_repeats: number of times a (block, trial) pair was encountered
        that already existed in the merged state
      - cloned_repeats: repeats where the incoming trial has the same timestamp
        as the existing one — typically snapshots of already-finished blocks
        saved again without new play data
      - updated_repeats: repeats where the timestamp differs — usually blocks
        the participant was actively playing between saves, or data from
        a different session (WebSocket reconnect) with genuinely new timestamps

    Returns {participant_prefix: merged_data} dict.
    """
    # Resolve file list
    if file_glob is not None:
        file_list = glob.glob(file_glob)
    elif participant_id is not None:
        file_list = glob.glob(os.path.join(folder_path, f"{participant_id}*.json"))
    else:
        file_list = glob.glob(os.path.join(folder_path, "*.json"))

    if not file_list:
        print("No files found")
        return {}

    # Group by first 30 characters of basename
    groups = defaultdict(list)
    for fp in file_list:
        prefix = os.path.basename(fp)[:30]
        groups[prefix].append(fp)

    def get_trial_timestamp(trial):
        if 'timePassedThru' in trial:
            return ('timePassedThru', trial['timePassedThru'])
        if 'events' in trial and len(trial['events']) > 0:
            ev = trial['events'][0]
            for key in ('time', 'timeBallFirstTouched'):
                if key in ev:
                    return (key, ev[key])
        return None

    def timestamps_match(a, b):
        if a is None or b is None:
            return False
        return a[0] == b[0] and abs(a[1] - b[1]) < 1e-6

    all_merged = {}
    for prefix, fp_list in groups.items():
        merged_blocks = {}
        total_processed = 0
        total_repeats = 0
        cloned_repeats = 0
        updated_repeats = 0

        for file_path in fp_list:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

            if 'blocks' not in data:
                continue

            for block in data['blocks']:
                b_idx = block.get('block_index')
                if b_idx is None:
                    continue
                if b_idx not in merged_blocks:
                    merged_blocks[b_idx] = {'meta': {}, 'trials': {}}

                for k, v in block.items():
                    if k not in ('block_index', 'trials'):
                        merged_blocks[b_idx]['meta'][k] = v

                if 'trials' not in block:
                    continue
                for trial in block['trials']:
                    t_idx = trial.get('index')
                    if t_idx is not None:
                        total_processed += 1
                        if t_idx in merged_blocks[b_idx]['trials']:
                            total_repeats += 1
                            existing = merged_blocks[b_idx]['trials'][t_idx]
                            existing_ts = get_trial_timestamp(existing)
                            incoming_ts = get_trial_timestamp(trial)
                            if timestamps_match(existing_ts, incoming_ts):
                                cloned_repeats += 1
                            else:
                                updated_repeats += 1
                        merged_blocks[b_idx]['trials'][t_idx] = trial

        # Convert to sorted list form
        final_data = {'blocks': []}
        for b_idx in sorted(merged_blocks.keys()):
            block_dict = {'block_index': b_idx}
            block_dict.update(merged_blocks[b_idx]['meta'])
            block_dict['trials'] = [
                merged_blocks[b_idx]['trials'][t_idx]
                for t_idx in sorted(merged_blocks[b_idx]['trials'])
            ]
            final_data['blocks'].append(block_dict)

        unique_trials = sum(len(merged_blocks[b]['trials']) for b in merged_blocks)
        unique_blocks = len(merged_blocks)

        print(f"Participant {prefix}: {len(fp_list)} files -> {total_repeats} repeated trials, "
              f"{unique_trials} unique trials, {unique_blocks} unique blocks")
        if total_repeats > 0:
            print(f"  Of repeats: {cloned_repeats} cloned (same timestamp), "
                  f"{updated_repeats} updated (different timestamp)")

        all_merged[prefix] = final_data

    return all_merged


# %%
def save_merged_participants(merged_dict, output_dir='cloud study data'):
    """
    Save each participant's cleaned merged data as a JSON file in output_dir.

    The filename is {prefix}_cleaned.json so it doesn't collide with raw files.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved = []
    for prefix, data in merged_dict.items():
        out_path = os.path.join(output_dir, f'{prefix}_cleaned.json')
        with open(out_path, 'w') as f:
            json.dump(data, f)
        saved.append(out_path)
        print(f'Saved {out_path}')
    print(f'Done — {len(saved)} participant file(s) written to {output_dir}/')
    return saved


# %%
def get_trial_timestamp(trial):
    if 'timePassedThru' in trial:
        return ('timePassedThru', trial['timePassedThru'])
    if 'events' in trial and len(trial['events']) > 0:
        ev = trial['events'][0]
        for key in ('time', 'timeBallFirstTouched'):
            if key in ev:
                return (key, ev[key])
    return None

def timestamps_match(a, b):
    if a is None or b is None:
        return False
    return a[0] == b[0] and abs(a[1] - b[1]) < 1e-6

def report_repeated_trials(folder_path, participant_id=None, file_glob=None):
    """
    For each participant, prints the start time and end time of each raw file
    so you can see the active recording window per file and how they overlap.

    If participant_id is given, only that participant is analyzed (like
    merge_participant_files). If file_glob is given, uses that glob directly.
    Otherwise all .json files in folder_path are scanned.

    Does NOT merge — only inspects raw files.
    """
    if file_glob is not None:
        file_list = glob.glob(file_glob)
    elif participant_id is not None:
        file_list = glob.glob(os.path.join(folder_path, f"{participant_id}*.json"))
    else:
        file_list = glob.glob(os.path.join(folder_path, "*.json"))

    if not file_list:
        print("No files found")
        return

    groups = defaultdict(list)
    for fp in file_list:
        prefix = os.path.basename(fp)[:30]
        groups[prefix].append(fp)

    for prefix, fp_list in groups.items():
        print(f"\n{'='*72}")
        print(f"Participant: {prefix}  ({len(fp_list)} files)")
        print(f"{'='*72}")

        file_times = []
        for fi, fp in enumerate(fp_list):
            data = json.load(open(fp))
            timestamps = []
            for block in data.get('blocks', []):
                for trial in block.get('trials', []):
                    ts = get_trial_timestamp(trial)
                    if ts is not None:
                        timestamps.append(ts[1])
            if timestamps:
                file_times.append((fi, min(timestamps), max(timestamps)))
            else:
                file_times.append((fi, None, None))

        print(f"  {'File':>3}  {'Start (ms)':>14}  {'End (ms)':>14}  {'Duration (s)':>14}  {'Filename':<30}")
        print(f"  {'-'*3}  {'-'*14}  {'-'*14}  {'-'*14}  {'-'*30}")
        for fi, start, end in file_times:
            fname = os.path.basename(fp_list[fi])[:30]
            if start is not None:
                dur = (end - start) / 1000
                print(f"  {fi:>3}  {start:>14.2f}  {end:>14.2f}  {dur:>14.2f}  {fname}")
            else:
                print(f"  {fi:>3}  {'N/A':>14}  {'N/A':>14}  {'N/A':>14}  {fname}")

        # Gap analysis
        if len(file_times) > 1:
            print(f"\n  Gaps between consecutive files:")
            for i in range(len(file_times) - 1):
                _, _, end_cur = file_times[i]
                _, start_next, _ = file_times[i + 1]
                if end_cur is not None and start_next is not None:
                    gap = (start_next - end_cur) / 1000
                    gap_str = f"{gap:.1f}s"
                    if gap > 60:
                        gap_str = f"{gap/60:.1f}min ({gap_str})"
                    print(f"    File {i} -> {i+1}: gap of {gap_str}")


# %%
# Merge all JSON files from the cloud study folder into per-participant data
all_participant_data = merge_participant_files(
    'cloud study data 2',
    file_glob='cloud study data 2/*.json'
)

# %%
# Save cleaned versions to /cloud study data/
#save_merged_participants(all_participant_data, 'cloud study data')

# %%
# Report on repeated trials — shows timestamps side by side
report_repeated_trials(
    'cloud study data 2',
    'EA4EE5B954A749C8BEED8F06A43F582C'
)

# %% [markdown]
# # Output

# %%
# Pick a participant from the merged data to analyze
prefixes = list(all_participant_data.keys())
prefix = prefixes[4] if prefixes else None
data = all_participant_data[prefix] if prefix else {'blocks': []}
print(f"Analyzing participant: {prefix}")

# %%
#data['blocks'][0]['trials']

# %%
#for block in data['blocks']:
#    print(block['block_index'])

# %%
choices = compare_greedy_vs_rollout(data, only_use_disagreements=False)
histogram = plot_rt_residuals_histogram(choices, rt_regression_type='minimum')

# %%
plot_switch_distribution(data)

# %%
#choices = get_choices(data)
X = choices[:,0] - choices[:,1]
# X = choices[:,2] - choices[:,3]
y = choices[:,-1]
plot_psychometric_curve(X, y)


# %%
trials_per_block = plot_trials_per_block(data)

# %%
def plot_deaths_per_block(data, fig=None):
    """
    Plots the number of deaths (trials with no events) per block.
    """
    block_numbers = []
    death_counts = []

    for i, block in enumerate(data['blocks']):
        block_numbers.append(i + 1)

        if 'trials' in block:
            deaths = sum(1 for trial in block['trials']
                         if 'events' not in trial or len(trial.get('events', [])) == 0)
            death_counts.append(deaths)
        else:
            death_counts.append(0)

    block_numbers = block_numbers[1:]
    death_counts = death_counts[1:]

    if fig is None:
        fig = plt.figure(figsize=(8, 5), dpi=300)

    plt.bar(block_numbers, death_counts, color='tomato', edgecolor='black', alpha=0.8)

    plt.title('Number of Deaths per Block')
    plt.xlabel('Block Number')
    plt.ylabel('Deaths')

    plt.xticks(block_numbers)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()

    return fig


# %%
deaths_per_block = plot_deaths_per_block(data)

# %%