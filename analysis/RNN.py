# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: pt_env
#     language: python
#     name: python3
# ---

# %% [markdown]
# # RNN & Feedforward Models
#
# Contains the `TinyDecisionRNN`, `FeedforwardDecisionNN` (distance and
# position variants), data prep, training, and evaluation functions.

# %%
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import confusion_matrix

# %%
def load(fnm):
    return json.load(open(fnm))


def load_cloud_study_data_2():
    """Load all participant JSON files from analysis/cloud study data/."""
    data_dir = Path("analysis/cloud study data")
    participants = {}
    for fpath in sorted(data_dir.glob("*.json")):
        data = load(str(fpath))
        sid = data.get("subject_id", fpath.stem)
        short_key = sid.split("-")[0] if "-" in sid else sid
        participants[short_key] = data
        print(f"  {short_key}: {len(data.get('blocks', []))} blocks")
    return participants


# %% [markdown]
# ## RNN — data prep, model, training, evaluation

# %%
def prepare_rnn_tensors(raw_data, batch_size=1, test_split=0.2):
    """Processes raw maze data into padded block-wise tensors for the RNN."""
    from exploratory_data_analysis import pre_proccess_data_from_choice_vs_no_choice

    processed_data = pre_proccess_data_from_choice_vs_no_choice(raw_data)

    if isinstance(processed_data, list):
        df_raw = pd.DataFrame(processed_data)
    else:
        df_raw = processed_data

    is_left = df_raw['chosen_left'].astype(bool)

    L1 = np.where(is_left, df_raw['chosen_1step_dist'], df_raw['unchosen_1step_dist'])
    R1 = np.where(~is_left, df_raw['chosen_1step_dist'], df_raw['unchosen_1step_dist'])

    chosen_2step_diff = df_raw['chosen_2step_dist'] - df_raw['chosen_1step_dist']
    unchosen_2step_diff = df_raw['unchosen_2step_dist'] - df_raw['unchosen_1step_dist']

    L2 = np.where(is_left, chosen_2step_diff, unchosen_2step_diff)
    R2 = np.where(~is_left, chosen_2step_diff, unchosen_2step_diff)

    X = pd.DataFrame({
        'L1-R1': L1 - R1,
        'L2-R2': L2 - R2,
        'block_drift': df_raw['block_drift'],
        'block_number': df_raw['block_number'],
        'chosen_left': df_raw['chosen_left'],
        'ball_y': df_raw['ball_y_at_top'],
        'cost': df_raw['observed_rt']
    })

    ball_y_mean = X['ball_y'].mean()
    ball_y_std = X['ball_y'].std()

    X['ball_y'] = (X['ball_y'] - ball_y_mean) / ball_y_std

    trials_per_block = X.groupby('block_number').size()
    large_blocks = trials_per_block[trials_per_block > 4].index

    if not large_blocks.empty:
        first_real_block = large_blocks.min()
        X = X[X['block_number'] >= first_real_block].copy()

    for col in ['L1-R1', 'L2-R2', 'cost']:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    X = X.dropna(subset=['L1-R1', 'L2-R2', 'cost', 'chosen_left', 'ball_y'])

    max_time = X['cost'].max()
    min_time = X['cost'].min()
    X['cost'] = (X['cost'] - min_time) / (max_time - min_time + 1e-6)

    valid_trials_per_block = X.groupby('block_number').size().sort_index()

    if test_split > 0.0:
        cumulative_trials = valid_trials_per_block.cumsum()
        total_trials = cumulative_trials.iloc[-1]

        train_threshold = total_trials * (1 - test_split)

        train_blocks = set(valid_trials_per_block[cumulative_trials <= train_threshold].index)
        test_blocks = set(valid_trials_per_block[cumulative_trials > train_threshold].index)

        if len(test_blocks) == 0 and len(valid_trials_per_block) > 1:
            test_blocks = {valid_trials_per_block.index[-1]}
            train_blocks = set(valid_trials_per_block.index[:-1])
    else:
        train_blocks = set(valid_trials_per_block.index)
        test_blocks = set()

    feature_cols = ['L1-R1', 'L2-R2', 'block_drift', 'ball_y', 'cost']

    def build_tensors(target_blocks):
        features_list = []
        targets_list = []

        for b in sorted(target_blocks):
            group = X[X['block_number'] == b].sort_index()
            features_list.append(torch.tensor(group[feature_cols].values, dtype=torch.float32))
            targets_list.append(torch.tensor(group['chosen_left'].values, dtype=torch.long))

        if not features_list:
            return None, None, None

        x_pad = pad_sequence(features_list, batch_first=True, padding_value=0.0)
        y_pad = pad_sequence(targets_list, batch_first=True, padding_value=-1)

        dataset = TensorDataset(x_pad, y_pad)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return x_pad, y_pad, loader

    X_train, y_train, train_loader = build_tensors(train_blocks)
    X_test, y_test, test_loader = build_tensors(test_blocks)

    print(f"-> Extracted {len(train_blocks)} Training Blocks, {len(test_blocks)} Testing Blocks.")

    if X_train is not None:
        print(f"-> Train X_padded shape: {X_train.shape} | Train y_padded shape: {y_train.shape}")
    if X_test is not None:
        print(f"-> Test X_padded shape:  {X_test.shape} | Test y_padded shape:  {y_test.shape}")

    return (X_train, y_train, train_loader), (X_test, y_test, test_loader)


# %%
class TinyDecisionRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_actions):
        super(TinyDecisionRNN, self).__init__()

        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          batch_first=True)

        self.readout = nn.Linear(in_features=hidden_size,
                                 out_features=num_actions)

    def forward(self, x, h_0=None):
        gru_out, h_n = self.gru(x, h_0)

        logits = self.readout(gru_out)

        probabilities = torch.softmax(logits, dim=-1)

        return probabilities, h_n


# %%
def evaluate_model_performance(model, data_loader):
    model.eval()

    total_log_likelihood = 0.0
    all_predictions = []
    all_actuals = []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            h_0 = torch.zeros(1, batch_x.size(0), model.hidden_size)
            probabilities, _ = model(batch_x, h_0)

            for run_idx in range(batch_x.size(0)):
                run_probs = probabilities[run_idx].view(-1, model.readout.out_features)
                run_actuals = batch_y[run_idx].view(-1)

                for step_idx in range(run_probs.size(0)):
                    actual_action = run_actuals[step_idx].item()

                    if actual_action == -1:
                        break

                    if actual_action not in [0, 1]:
                        continue

                    try:
                        chosen_prob = run_probs[step_idx][actual_action].item()

                        if np.isnan(chosen_prob) or chosen_prob == 0.0:
                            print(f"Something's wrong @ {run_idx}, Step {step_idx}!")
                        total_log_likelihood += np.log(max(chosen_prob, 1e-7))

                        predicted_action = torch.argmax(run_probs[step_idx]).item()

                        all_predictions.append(predicted_action)
                        all_actuals.append(actual_action)

                    except IndexError as e:
                        print(f"Index alignment error at step {step_idx}: {e}")
                        continue

    all_actuals = np.array(all_actuals)
    all_predictions = np.array(all_predictions)

    total_steps = len(all_actuals)
    accuracy = np.sum(all_predictions == all_actuals) / total_steps
    err_matrix = confusion_matrix(all_actuals, all_predictions, labels=[0, 1])

    return {
        "log_likelihood": total_log_likelihood / total_steps,
        "accuracy": accuracy,
        "error_matrix": err_matrix
    }


# %%
def train_RNN(model, train_loader, num_epochs=150, learning_rate=0.005, l1_lambda=1e-4):
    """
    Trains the TinyDecisionRNN model, applying L1 regularization to the
    recurrent weights and gradient clipping to prevent explosion.
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    model.train()

    epoch_losses = []

    for epoch in range(num_epochs):
        total_epoch_loss = 0.0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()

            h_0 = torch.zeros(1, batch_x.size(0), model.hidden_size)

            gru_out, _ = model.gru(batch_x, h_0)
            logits = model.readout(gru_out)

            logits = logits.view(-1, model.readout.out_features)
            batch_y = batch_y.view(-1)

            base_loss = criterion(logits, batch_y)

            l1_norm = sum(p.abs().sum() for name, p in model.named_parameters() if 'gru.weight' in name)
            loss = base_loss + (l1_lambda * l1_norm)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_epoch_loss += loss.item()

        avg_loss = total_epoch_loss / len(train_loader)
        epoch_losses.append(avg_loss)

        if (epoch + 1) % 25 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    print("\nTraining Complete!")
    return model, epoch_losses


# %%
def run_RNN_for_eval(participant_data, num_epochs=400):
    train_data, test_data = prepare_rnn_tensors(participant_data, batch_size=4, test_split=0.2)
    X_train, y_train, train_loader = train_data
    X_test, y_test, test_loader = test_data

    model_participant = TinyDecisionRNN(input_size=5, hidden_size=2, num_actions=2)
    model_participant, _ = train_RNN(model_participant, train_loader=train_loader, num_epochs=num_epochs)

    return evaluate_model_performance(model_participant, test_loader)


# %% [markdown]
# ## Feedforward NN (distance features)
#
# Same 5 features as the RNN (`L1-R1`, `L2-R2`, `block_drift`, `ball_y`,
# `cost`) but treats each trial independently — no sequential processing.

# %%
def prepare_ff_tensors(raw_data, test_split=0.2):
    """
    Processes raw maze data into individual trial features (no sequence padding)
    for the feedforward model. Returns train/test DataLoaders of single trials.
    """
    from exploratory_data_analysis import pre_proccess_data_from_choice_vs_no_choice

    processed_data = pre_proccess_data_from_choice_vs_no_choice(raw_data)

    if isinstance(processed_data, list):
        df_raw = pd.DataFrame(processed_data)
    else:
        df_raw = processed_data

    is_left = df_raw['chosen_left'].astype(bool)

    L1 = np.where(is_left, df_raw['chosen_1step_dist'], df_raw['unchosen_1step_dist'])
    R1 = np.where(~is_left, df_raw['chosen_1step_dist'], df_raw['unchosen_1step_dist'])

    chosen_2step_diff = df_raw['chosen_2step_dist'] - df_raw['chosen_1step_dist']
    unchosen_2step_diff = df_raw['unchosen_2step_dist'] - df_raw['unchosen_1step_dist']

    L2 = np.where(is_left, chosen_2step_diff, unchosen_2step_diff)
    R2 = np.where(~is_left, chosen_2step_diff, unchosen_2step_diff)

    X = pd.DataFrame({
        'L1-R1': L1 - R1,
        'L2-R2': L2 - R2,
        'block_drift': df_raw['block_drift'],
        'block_number': df_raw['block_number'],
        'chosen_left': df_raw['chosen_left'],
        'ball_y': df_raw['ball_y_at_top'],
        'cost': df_raw['observed_rt']
    })

    ball_y_mean = X['ball_y'].mean()
    ball_y_std = X['ball_y'].std()
    X['ball_y'] = (X['ball_y'] - ball_y_mean) / ball_y_std

    trials_per_block = X.groupby('block_number').size()
    large_blocks = trials_per_block[trials_per_block > 4].index

    if not large_blocks.empty:
        first_real_block = large_blocks.min()
        X = X[X['block_number'] >= first_real_block].copy()

    for col in ['L1-R1', 'L2-R2', 'cost']:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    X = X.dropna(subset=['L1-R1', 'L2-R2', 'cost', 'chosen_left', 'ball_y'])

    max_time = X['cost'].max()
    min_time = X['cost'].min()
    X['cost'] = (X['cost'] - min_time) / (max_time - min_time + 1e-6)

    valid_trials_per_block = X.groupby('block_number').size().sort_index()

    if test_split > 0.0:
        cumulative_trials = valid_trials_per_block.cumsum()
        total_trials = cumulative_trials.iloc[-1]
        train_threshold = total_trials * (1 - test_split)

        train_blocks = set(valid_trials_per_block[cumulative_trials <= train_threshold].index)
        test_blocks = set(valid_trials_per_block[cumulative_trials > train_threshold].index)

        if len(test_blocks) == 0 and len(valid_trials_per_block) > 1:
            test_blocks = {valid_trials_per_block.index[-1]}
            train_blocks = set(valid_trials_per_block.index[:-1])
    else:
        train_blocks = set(valid_trials_per_block.index)
        test_blocks = set()

    feature_cols = ['L1-R1', 'L2-R2', 'block_drift', 'ball_y', 'cost']

    def build_tensors(target_blocks):
        feats, tgts = [], []
        for b in sorted(target_blocks):
            group = X[X['block_number'] == b].sort_index()
            feats.append(torch.tensor(group[feature_cols].values, dtype=torch.float32))
            tgts.append(torch.tensor(group['chosen_left'].values, dtype=torch.long))

        if not feats:
            return None, None, None

        feats_cat = torch.cat(feats, dim=0)
        tgts_cat = torch.cat(tgts, dim=0)
        dataset = TensorDataset(feats_cat, tgts_cat)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        return feats_cat, tgts_cat, loader

    X_train, y_train, train_loader = build_tensors(train_blocks)
    X_test, y_test, test_loader = build_tensors(test_blocks)

    print(f"-> FF: {len(train_blocks)} Training Blocks, {len(test_blocks)} Testing Blocks.")
    if X_train is not None:
        print(f"-> FF Train: {X_train.shape[0]} trials | Test: {X_test.shape[0]} trials")

    return (X_train, y_train, train_loader), (X_test, y_test, test_loader)


# %%
class FeedforwardDecisionNN(nn.Module):
    def __init__(self, input_size, hidden_sizes=(16, 8), num_actions=2):
        super().__init__()
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.net(x)
        probabilities = torch.softmax(logits, dim=-1)
        return probabilities


# %%
def train_feedforward(model, train_loader, num_epochs=150, learning_rate=0.005):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            probs = model(batch_x)
            loss = criterion(torch.log(probs.clamp_min(1e-7)), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 25 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

    print("Training Complete!")
    return model


# %%
def evaluate_feedforward(model, test_loader):
    model.eval()
    total_log_likelihood = 0.0
    all_preds = []
    all_actuals = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            probs = model(batch_x)
            for i in range(batch_x.size(0)):
                actual = batch_y[i].item()
                p = probs[i][actual].item()
                if np.isnan(p) or p == 0.0:
                    print(f"Warning: bad probability {p}")
                total_log_likelihood += np.log(max(p, 1e-7))
                pred = torch.argmax(probs[i]).item()
                all_preds.append(pred)
                all_actuals.append(actual)

    all_actuals = np.array(all_actuals)
    all_preds = np.array(all_preds)
    total_steps = len(all_actuals)
    accuracy = np.sum(all_preds == all_actuals) / total_steps
    err_matrix = confusion_matrix(all_actuals, all_preds, labels=[0, 1])

    return {
        "log_likelihood": total_log_likelihood / total_steps,
        "accuracy": accuracy,
        "error_matrix": err_matrix
    }


# %%
def run_FF_for_eval(participant_data, num_epochs=400):
    train_data, test_data = prepare_ff_tensors(participant_data, test_split=0.2)
    _, _, train_loader = train_data
    _, _, test_loader = test_data

    model = FeedforwardDecisionNN(input_size=5, hidden_sizes=(16, 8), num_actions=2)
    model = train_feedforward(model, train_loader, num_epochs=num_epochs)

    return evaluate_feedforward(model, test_loader)


# %% [markdown]
# ## Feedforward NN (raw positions)
#
# Uses raw hole coordinates `[entry_hole, left_hole, right_hole, exit_hole]`
# instead of pre-computed distance differences. The model learns its own
# spatial relationships.

# %%
def prepare_ff_position_tensors(raw_data, test_split=0.2):
    """
    Processes raw maze data into individual trial features using raw hole
    positions instead of pre-computed greedy/planning distances.

    Features: [entry_hole, left_hole, right_hole, exit_hole]

    The model learns its own notion of spatial relationships rather than
    relying on the hand-crafted L1-R1 / L2-R2 distance features.
    """
    from exploratory_data_analysis import pre_proccess_data_from_choice_vs_no_choice

    processed_data = pre_proccess_data_from_choice_vs_no_choice(raw_data)

    if isinstance(processed_data, list):
        df_raw = pd.DataFrame(processed_data)
    else:
        df_raw = processed_data

    # Only decision trials have two middle holes (left/right) to choose
    # between; no-choice trials have a single middle hole and no prediction
    # target.
    df_raw = df_raw[df_raw['choice_trial']].reset_index(drop=True)

    # Determine which side was chosen (left=0, right=1)
    chosen_middle = df_raw['chosen_path'].str[1]
    unchosen_middle = df_raw['non_chosen_path'].str[1]
    chose_right = (chosen_middle > unchosen_middle).astype(bool)

    entry = df_raw['chosen_path'].str[0].values.astype(float)
    chosen_hole = chosen_middle.values.astype(float)
    unchosen_hole = unchosen_middle.values.astype(float)
    exit_hole = df_raw['chosen_path'].str[2].values.astype(float)

    # Always present holes in canonical order: [left, right] so the model
    # always sees the same hole ordering regardless of which was chosen.
    hole_sequences = df_raw['hole_sequence']
    left_hole = np.array([sorted(hs[1])[0] for hs in hole_sequences], dtype=float)
    right_hole = np.array([sorted(hs[1])[1] for hs in hole_sequences], dtype=float)

    X = pd.DataFrame({
        'entry': entry,
        'left_hole': left_hole,
        'right_hole': right_hole,
        'exit': exit_hole,
        'chose_right': chose_right,
        'block_number': df_raw['block_number'],
    })

    X = X.dropna()

    valid_trials_per_block = X.groupby('block_number').size().sort_index()

    if test_split > 0.0:
        cumulative_trials = valid_trials_per_block.cumsum()
        total_trials = cumulative_trials.iloc[-1]
        train_threshold = total_trials * (1 - test_split)

        train_blocks = set(valid_trials_per_block[cumulative_trials <= train_threshold].index)
        test_blocks = set(valid_trials_per_block[cumulative_trials > train_threshold].index)

        if len(test_blocks) == 0 and len(valid_trials_per_block) > 1:
            test_blocks = {valid_trials_per_block.index[-1]}
            train_blocks = set(valid_trials_per_block.index[:-1])
    else:
        train_blocks = set(valid_trials_per_block.index)
        test_blocks = set()

    feature_cols = ['entry', 'left_hole', 'right_hole', 'exit']

    def build_tensors(target_blocks):
        feats, tgts = [], []
        for b in sorted(target_blocks):
            group = X[X['block_number'] == b].sort_index()
            feats.append(torch.tensor(group[feature_cols].values, dtype=torch.float32))
            tgts.append(torch.tensor(group['chose_right'].values, dtype=torch.long))

        if not feats:
            return None, None, None

        feats_cat = torch.cat(feats, dim=0)
        tgts_cat = torch.cat(tgts, dim=0)
        dataset = TensorDataset(feats_cat, tgts_cat)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        return feats_cat, tgts_cat, loader

    X_train, y_train, train_loader = build_tensors(train_blocks)
    X_test, y_test, test_loader = build_tensors(test_blocks)

    print(f"-> FF-Position: {len(train_blocks)} Training Blocks, {len(test_blocks)} Testing Blocks.")
    if X_train is not None:
        print(f"-> FF-Position Train: {X_train.shape[0]} trials | Test: {X_test.shape[0]} trials")

    return (X_train, y_train, train_loader), (X_test, y_test, test_loader)


# %%
def run_FF_position_for_eval(participant_data, num_epochs=400):
    train_data, test_data = prepare_ff_position_tensors(participant_data, test_split=0.2)
    _, _, train_loader = train_data
    _, _, test_loader = test_data

    model = FeedforwardDecisionNN(input_size=4, hidden_sizes=(16, 8), num_actions=2)
    model = train_feedforward(model, train_loader, num_epochs=num_epochs)

    return evaluate_feedforward(model, test_loader)


# %% [markdown]
# ## Test runner — compare all three models

# %%
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    data_dir = Path("analysis/cloud study data")
    json_files = sorted(data_dir.glob("*.json"))
    if not json_files:
        print("No participant files found in analysis/cloud study data/")
    else:
        print("=" * 70)
        print("  FEEDFORWARD vs RNN COMPARISON")
        print("=" * 70)
        print(f"  Found {len(json_files)} participant file(s) in {data_dir}\n")

        rows = []
        for fpath in json_files:
            short = fpath.stem[:20]
            print(f"\n  --- {short} ---")
            data = load(str(fpath))

            for name, fn, n_epochs in [
                ("RNN (GRU)", run_RNN_for_eval, 150),
                ("FF (distances)", run_FF_for_eval, 150),
                ("FF (positions)", run_FF_position_for_eval, 150),
            ]:
                try:
                    m = fn(data, num_epochs=n_epochs)
                    if m:
                        rows.append({
                            "Participant": short,
                            "Model": name,
                            "Accuracy": m["accuracy"],
                            "LogLikelihood": m["log_likelihood"],
                        })
                        print(f"    {name:20s}  acc: {m['accuracy']*100:5.1f}%  ll: {m['log_likelihood']:.4f}")
                    else:
                        print(f"    {name:20s}  FAILED (no metrics)")
                except Exception as e:
                    print(f"    {name:20s}  FAILED ({e})")

        if rows:
            df = pd.DataFrame(rows)
            print("\n" + "=" * 70)
            print("  SUMMARY")
            print("=" * 70)
            pivot = df.pivot_table(index="Participant", columns="Model",
                                   values="Accuracy", aggfunc="first")
            print((pivot * 100).round(1).to_string())
            ranking = df.groupby("Model")["Accuracy"].mean().sort_values(ascending=False)
            print("\n  Mean Accuracy:")
            for rank, (model, acc) in enumerate(ranking.items(), 1):
                print(f"    {rank}. {model}: {acc*100:.1f}%")

#%%