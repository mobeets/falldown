#%%

import os.path
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%

def load(fnm):
	return json.load(open(fnm))

def get_block_choices(trials, gameInfo=None):
    choices = []
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
        elif 'events' in trial and len(trial['events']) > 0 and len(trials) >= i+2 and len(trials[i-1]['events']) > 0 and len(trials[i+1]['events']) > 0:
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
            h_next = None
        if len(hole_locs) != 2 or len(prev_hole_locs) != 1 or len(next_hole_locs) != 1:
            continue

        if i > 2 and h_prev is not None:
            preceding_hole_used = trials[i-2]['holeUsed'] if 'holes' in trials[i-2] else trials[i-2]['events'][0]['holeUsed']
            direction_entered = -1 if preceding_hole_used < h_prev else 1
        else:
            direction_entered = 0

        dist_L1 = np.abs(hole_locs[0] - h_prev)
        dist_R1 = np.abs(hole_locs[1] - h_prev)
        dist_L2 = dist_L1 + np.abs(hole_locs[0] - h_next)
        dist_R2 = dist_R1 + np.abs(hole_locs[1] - h_next)
        delta_2 = h_next - h_prev
        # we ignore trials where both greedy and rollout would make the same choice
        if (dist_L1 < dist_R1 and dist_L2 < dist_R2) or (dist_R1 < dist_L1 and dist_R2 < dist_L2):
            is_disagreement = False
        else:
            is_disagreement = True
        
        choice = hole_locs.index(h_cur)
        isPlanningChoice = dist_L2 < dist_R2 if choice == 0 else dist_R2 < dist_L2
        rt = np.nan
        rt_full = np.nan
        if 'timePassedThru' in trial:
            rt = trial['timePassedThru'] - trials[i-1]['timePassedThru']
            if len(trials) >= i+2:
                rt_full = trials[i+1]['timePassedThru'] - trials[i-1]['timePassedThru']
        elif 'events' in trial and len(trial['events']) > 0 and len(trials[i-1]['events']) > 0:
            rt = trial['events'][0]['time'] - trials[i-1]['events'][0]['time']
            if len(trials) >= i+2 and len(trials[i+1]['events']) > 0:
                rt_full = trials[i+1]['events'][0]['time'] - trials[i-1]['events'][0]['time']

        if 'events' in trial and len(trial['events']) > 0 and gameInfo is not None:
            event = trial['events'][0]
            heightOnScreen = (event['ballY'] - event['cameraY']) / gameInfo['height']
            scrollSpeed = event['scrollSpeed'] if 'scrollSpeed' in event else np.nan
        else:
            heightOnScreen = np.nan
            scrollSpeed = trial['scrollSpeed'] if 'scrollSpeed' in trial else np.nan

        choices.append({
            'subject_index': None,
            'subject_id': None,
            'block_count': None,
            'block_index': trial['block_index'],
            'trial_index': trial['index'],
            'hole_locations': trial['hole_locations'],
            'dist_L1': dist_L1,
            'dist_R1': dist_R1,
            'dist_L2': dist_L2,
            'dist_R2': dist_R2,
            'delta_2': delta_2,
            'rt': rt,
            'rt_full': rt_full,
            'choice': choice,
            'scrollSpeed': scrollSpeed,
            'is_disagreement': is_disagreement,
            'direction_entered': direction_entered,
            'isPlanningChoice': isPlanningChoice,
            'heightOnScreen': heightOnScreen})
    return pd.DataFrame(choices)

def load_all_sessions(fnms, block_filter=None):
    if block_filter is None:
        # by default, only include blocks with drift
        block_filter = lambda block: block['block_config']['params']['startCameraMode'] == 1
    
    sessions = {}
    subj_id_to_index = {}
    for fnm in fnms:
        data = json.load(open(fnm))
        subj_id = data['subject_id']
        if subj_id not in sessions:
            subj_id_to_index[subj_id] = len(subj_id_to_index)
            sessions[subj_id] = {}
        c_choices = sessions[subj_id]
        for block in data['blocks']:
            if 'startCameraMode' not in block['block_config']['params'] or block_filter(block) is False:
                # ignore practice levels
                continue
            bid = block['block_count']
            df = get_block_choices(block['trials'], data['gameInfo'])
            df['subject_index'] = subj_id_to_index[subj_id]
            df['subject_id'] = subj_id
            df['block_count'] = bid
            df['is_drift_block'] = block['block_config']['params']['startCameraMode'] == 1
            if bid not in c_choices or len(df) > len(c_choices[bid]):
                sessions[subj_id][bid] = df
    
    # now concatenate all dataframes for each subject into a single df
    pd_sessions = {}
    for subj_id in sessions:
        c_sessions = [sessions[subj_id][bid] for bid in sorted(sessions[subj_id].keys())]
        if len(c_sessions) == 0:
            continue
        pd_sessions[subj_id] = pd.concat(c_sessions, ignore_index=True)
    # now concatenate all dfs for all subjects into a single dataframe
    df = pd.concat([pd_sessions[subj_id] for subj_id in sorted(pd_sessions.keys())], ignore_index=True)
    return df

#%%

if __name__ == '__main__':
    data_dir = os.path.expanduser('~/Downloads/Second 5 Participants')
    fnms = glob.glob(os.path.join(data_dir, '*.json'))
    df = load_all_sessions(fnms)
