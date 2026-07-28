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

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
import matplotlib.pyplot as plt
import seaborn as sns

# %%
class MazeDataset(Dataset):
    def __init__(self, trial_features, participant_ids, choices, rt_values=None,
                 time_bin_ids=None):
        self.features = torch.tensor(trial_features, dtype=torch.float32)
        self.p_ids = torch.tensor(participant_ids, dtype=torch.long)
        self.choices = torch.tensor(choices, dtype=torch.float32)
        self.rt_values = (torch.tensor(rt_values, dtype=torch.float32)
                          if rt_values is not None else None)
        self.time_bin_ids = (torch.tensor(time_bin_ids, dtype=torch.long)
                             if time_bin_ids is not None else None)

    def __len__(self):
        return len(self.choices)

    def __getitem__(self, idx):
        if self.time_bin_ids is not None:
            if self.rt_values is not None:
                return (self.features[idx], self.p_ids[idx],
                        self.time_bin_ids[idx], self.choices[idx],
                        self.rt_values[idx])
            return (self.features[idx], self.p_ids[idx],
                    self.time_bin_ids[idx], self.choices[idx])
        if self.rt_values is not None:
            return (self.features[idx], self.p_ids[idx],
                    self.choices[idx], self.rt_values[idx])
        return (self.features[idx], self.p_ids[idx], self.choices[idx])


# %%
def orthogonality_penalty(bases):
    bases_norm = F.normalize(bases, p=2, dim=0)
    correlation_matrix = torch.matmul(bases_norm.T, bases_norm)
    identity = torch.eye(correlation_matrix.size(0), device=bases.device)
    return torch.norm(correlation_matrix - identity, p='fro')


# %% [markdown]
# # StrategyDeepONet: Mixture-of-Strategies Model
#
# Replaces the single participant embedding with K separate strategy networks
# and a gate that selects the active strategy per trial. Draws on:
# - Ashwood et al 2022: GLM-HMM discrete strategy switching
# - Kirsch 2019: strategies as points in a computational-constraint space

# %%
class StrategyDeepONet(nn.Module):
    def __init__(self, num_participants, num_features=5, num_bases=4, num_strategies=3,
                 shared_bases=False):
        super().__init__()

        self.num_strategies = num_strategies
        self.num_bases = num_bases
        self.shared_bases = shared_bases

        if shared_bases:
            self.basis_net = nn.Sequential(
                nn.Linear(num_features, 16),
                nn.ReLU(),
                nn.Linear(16, 16),
                nn.ReLU(),
                nn.Linear(16, num_bases),
                nn.Tanh()
            )
        else:
            self.basis_nets = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(num_features, 16),
                    nn.ReLU(),
                    nn.Linear(16, 16),
                    nn.ReLU(),
                    nn.Linear(16, num_bases),
                    nn.Tanh()
                ) for _ in range(num_strategies)
            ])

        coeff_dim = num_strategies * num_bases
        self.participant_coeffs = nn.Embedding(num_participants, coeff_dim)
        nn.init.normal_(self.participant_coeffs.weight, mean=0.0, std=0.1)

        gate_input_dim = num_features + coeff_dim
        self.gate = nn.Sequential(
            nn.Linear(gate_input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_strategies)
        )

    def forward(self, trial_features, participant_ids):
        coeffs_flat = self.participant_coeffs(participant_ids)
        coeffs = coeffs_flat.view(-1, self.num_strategies, self.num_bases)

        gate_input = torch.cat([trial_features, coeffs_flat], dim=-1)
        strategy_weights = F.softmax(self.gate(gate_input), dim=-1)

        all_logits = []
        all_bases = []
        for k in range(self.num_strategies):
            if self.shared_bases:
                bases_k = self.basis_net(trial_features)
            else:
                bases_k = self.basis_nets[k](trial_features)
            logit_k = (bases_k * coeffs[:, k, :]).sum(dim=-1)
            all_logits.append(logit_k)
            all_bases.append(bases_k)

        stacked_logits = torch.stack(all_logits, dim=-1)
        final_logit = (stacked_logits * strategy_weights).sum(dim=-1)
        stacked_bases = torch.stack(all_bases, dim=1)

        return final_logit, stacked_bases, strategy_weights


# %% [markdown]
# # StrategyDeepONetMultiTask: Add Reaction-Time Prediction
#
# Adds a parallel set of participant coefficients that predict RT using the
# same basis functions. A basis that predicts choices but not RT captures
# deliberative planning; a basis that predicts both captures heuristic responding.
# Draws on:
# - Resulaj et al 2009: change-of-mind bounded diffusion
# - Keung et al 2020: divisive evidence accumulation

# %%
class StrategyDeepONetMultiTask(StrategyDeepONet):
    def __init__(self, num_participants, num_features=5, num_bases=4, num_strategies=3, shared_bases=False):
        super().__init__(num_participants, num_features, num_bases, num_strategies, shared_bases=shared_bases)

        coeff_dim = self.num_strategies * self.num_bases
        self.rt_coeffs = nn.Embedding(num_participants, coeff_dim)
        nn.init.normal_(self.rt_coeffs.weight, mean=0.0, std=0.1)

    def forward(self, trial_features, participant_ids):
        logit, bases, strategy_weights = super().forward(trial_features, participant_ids)

        rt_coeffs_flat = self.rt_coeffs(participant_ids)
        rt_coeffs = rt_coeffs_flat.view(-1, self.num_strategies, self.num_bases)

        # Weighted by the same strategy gate
        all_rt_preds = []
        for k in range(self.num_strategies):
            bases_k = bases[:, k, :]
            rt_pred_k = (bases_k * rt_coeffs[:, k, :]).sum(dim=-1)
            all_rt_preds.append(rt_pred_k)
        stacked_rt = torch.stack(all_rt_preds, dim=-1)
        rt_pred = (stacked_rt * strategy_weights).sum(dim=-1)

        return logit, bases, strategy_weights, rt_pred


# %% [markdown]
# # Time-Binned Strategy DeepONet
#
# Splits each participant's trials into T temporal bins and learns separate
# strategy embeddings per bin, producing a coefficient trajectory over time.
# Combines strategy gating with temporal dynamics.
# Draws on: Ashwood PsyTrack model (tracks GLM weights over time).

# %%
class TimeBinnedStrategyDeepONet(StrategyDeepONet):
    def __init__(self, num_participants, num_time_bins, num_features=5,
                 num_bases=4, num_strategies=3, shared_bases=False):
        super().__init__(num_participants, num_features, num_bases, num_strategies,
                         shared_bases=shared_bases)
        self.num_time_bins = num_time_bins

        coeff_dim = self.num_strategies * self.num_bases
        self.participant_coeffs = nn.Embedding(num_participants * num_time_bins, coeff_dim)
        nn.init.normal_(self.participant_coeffs.weight, mean=0.0, std=0.1)

        gate_input_dim = num_features + coeff_dim
        self.gate = nn.Sequential(
            nn.Linear(gate_input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_strategies)
        )

    def forward(self, trial_features, participant_ids, time_bin_ids):
        flat_ids = participant_ids * self.num_time_bins + time_bin_ids
        return super().forward(trial_features, flat_ids)


# %% [markdown]
# # Training Functions

# %%
def train_strategy_deeponet(model, dataloader, num_epochs=200, lr=0.001,
                            penalty_weight=0.5, entropy_weight=0.05):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    is_time_binned = isinstance(model, TimeBinnedStrategyDeepONet)

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_bce = 0.0
        total_orth = 0.0
        total_entropy = 0.0

        for batch in dataloader:
            if is_time_binned:
                features, p_ids, bin_ids, true_choices = batch
            else:
                features, p_ids, true_choices = batch
            optimizer.zero_grad()

            if is_time_binned:
                logits, bases, strategy_weights = model(features, p_ids, bin_ids)
            else:
                logits, bases, strategy_weights = model(features, p_ids)
            bce_loss = criterion(logits, true_choices)

            # Orthogonality penalty: with shared bases, penalize once
            orth_loss = 0.0
            if model.shared_bases:
                orth_loss = orthogonality_penalty(bases[:, 0, :])
            else:
                for k in range(model.num_strategies):
                    orth_loss += orthogonality_penalty(bases[:, k, :])
                orth_loss /= model.num_strategies

            # Entropy bonus: prevent the gate from collapsing to one strategy
            probs = torch.clamp(strategy_weights, min=1e-8)
            entropy = -(probs * torch.log(probs)).sum(dim=-1).mean()

            loss = bce_loss + penalty_weight * orth_loss - entropy_weight * entropy
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_bce += bce_loss.item()
            total_orth += orth_loss.item()
            total_entropy += entropy.item()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            n = len(dataloader)
            print(f"Epoch {epoch+1:03d}/{num_epochs} | "
                  f"Loss: {total_loss/n:.4f} | BCE: {total_bce/n:.4f} | "
                  f"Orth: {total_orth/n:.4f} | Ent: {total_entropy/n:.4f}")

    return model


# %%
def train_strategy_deeponet_multitask(model, dataloader, num_epochs=200, lr=0.001,
                                      penalty_weight=0.5, entropy_weight=0.05,
                                      rt_weight=0.3):
    choice_criterion = nn.BCEWithLogitsLoss()
    rt_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_bce = 0.0
        total_mse = 0.0
        total_orth = 0.0
        total_entropy = 0.0

        for batch in dataloader:
            features, p_ids, true_choices, true_rt = batch
            optimizer.zero_grad()

            logits, bases, strategy_weights, rt_pred = model(features, p_ids)
            bce_loss = choice_criterion(logits, true_choices)
            mse_loss = rt_criterion(rt_pred, true_rt)

            orth_loss = 0.0
            if model.shared_bases:
                orth_loss = orthogonality_penalty(bases[:, 0, :])
            else:
                for k in range(model.num_strategies):
                    orth_loss += orthogonality_penalty(bases[:, k, :])
                orth_loss /= model.num_strategies

            probs = torch.clamp(strategy_weights, min=1e-8)
            entropy = -(probs * torch.log(probs)).sum(dim=-1).mean()

            loss = (bce_loss + rt_weight * mse_loss
                    + penalty_weight * orth_loss - entropy_weight * entropy)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_bce += bce_loss.item()
            total_mse += mse_loss.item()
            total_orth += orth_loss.item()
            total_entropy += entropy.item()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            n = len(dataloader)
            print(f"Epoch {epoch+1:03d}/{num_epochs} | "
                  f"Loss: {total_loss/n:.4f} | BCE: {total_bce/n:.4f} | "
                  f"MSE: {total_mse/n:.4f} | Orth: {total_orth/n:.4f} | "
                  f"Ent: {total_entropy/n:.4f}")

    return model


# %%
def train_time_binned(model, dataloader, num_epochs=200, lr=0.001,
                      penalty_weight=0.5, entropy_weight=0.05):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_bce = 0.0
        total_orth = 0.0
        total_entropy = 0.0

        for batch in dataloader:
            features, p_ids, bin_ids, true_choices = batch
            optimizer.zero_grad()

            logits, bases, strategy_weights = model(features, p_ids, bin_ids)
            bce_loss = criterion(logits, true_choices)

            orth_loss = 0.0
            if model.shared_bases:
                orth_loss = orthogonality_penalty(bases[:, 0, :])
            else:
                for k in range(model.num_strategies):
                    orth_loss += orthogonality_penalty(bases[:, k, :])
                orth_loss /= model.num_strategies

            probs = torch.clamp(strategy_weights, min=1e-8)
            entropy = -(probs * torch.log(probs)).sum(dim=-1).mean()

            loss = bce_loss + penalty_weight * orth_loss - entropy_weight * entropy
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_bce += bce_loss.item()
            total_orth += orth_loss.item()
            total_entropy += entropy.item()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            n = len(dataloader)
            print(f"Epoch {epoch+1:03d}/{num_epochs} | "
                  f"Loss: {total_loss/n:.4f} | BCE: {total_bce/n:.4f} | "
                  f"Orth: {total_orth/n:.4f} | Ent: {total_entropy/n:.4f}")

    return model


# %% [markdown]
# # Evaluation

# %%
def evaluate_strategy_model(model, dataloader, with_rt=False):
    model.eval()
    total_ll = 0.0
    correct = 0
    total = 0
    all_strategy_weights = []
    is_time_binned = isinstance(model, TimeBinnedStrategyDeepONet)

    p_ll = {}
    p_correct = {}
    p_total = {}

    with torch.no_grad():
        for batch in dataloader:
            if is_time_binned:
                features, p_ids, bin_ids, true_choices = batch
                logits, bases, strategy_weights = model(features, p_ids, bin_ids)
            elif with_rt:
                features, p_ids, true_choices, _ = batch
                logits, bases, strategy_weights, _ = model(features, p_ids)
            else:
                features, p_ids, true_choices = batch
                logits, bases, strategy_weights = model(features, p_ids)

            bce_sum = F.binary_cross_entropy_with_logits(
                logits, true_choices, reduction='sum')
            total_ll += -bce_sum.item()

            trial_bce = F.binary_cross_entropy_with_logits(
                logits, true_choices, reduction='none')
            trial_ll = -trial_bce

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            is_correct = (preds == true_choices)
            correct += is_correct.sum().item()
            total += true_choices.size(0)
            all_strategy_weights.append(strategy_weights.cpu())

            for j in range(true_choices.size(0)):
                pid = p_ids[j].item()
                p_ll[pid] = p_ll.get(pid, 0.0) + trial_ll[j].item()
                p_correct[pid] = p_correct.get(pid, 0) + is_correct[j].item()
                p_total[pid] = p_total.get(pid, 0) + 1

    avg_ll = total_ll / total
    acc = correct / total
    all_weights = torch.cat(all_strategy_weights, dim=0)

    per_participant = {}
    for pid in sorted(p_ll.keys()):
        per_participant[pid] = {
            'accuracy': p_correct[pid] / p_total[pid],
            'log_likelihood': p_ll[pid] / p_total[pid],
        }

    print(f"--- Evaluation ---")
    print(f"Avg Log-Likelihood: {avg_ll:.4f}")
    print(f"Accuracy: {acc*100:.2f}% ({correct}/{total})")
    print(f"Strategy usage: {all_weights.mean(dim=0).tolist()}")

    return avg_ll, acc, all_weights, per_participant


# %% [markdown]
# # Data Pipeline (compatible with cognitivedeepOnet.py)

# %%
def pre_proccess_data_from_choice_vs_no_choice(data):
    output = []
    for block_num, block in enumerate(data['blocks']):
        if block_num == 0:
            continue
        try:
            block_drift = block['block_config']['params']['startCameraMode']
        except (KeyError, TypeError):
            block_drift = 0

        game_states = block.get('game_states', {})
        state_times = np.array(game_states.get('time', []))
        ball_y_coords = np.array(game_states.get('ball_y', []))
        camera_y_coords = np.array(game_states.get('camera_y', []))
        trials = block['trials']

        for i in range(len(trials) // 3):
            try:
                tier1_event_time = trials[3*i]['events'][0]['time']
                if len(state_times) > 0:
                    closest_idx = np.abs(state_times - tier1_event_time).argmin()
                    relative_ball_y = float(ball_y_coords[closest_idx]
                                            - camera_y_coords[closest_idx])
                else:
                    relative_ball_y = None

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
                observed_rt = (trials[3*i+2]['events'][0]['time']
                               - trials[3*i]['events'][0]['time'])
                is_choice = len(trials[3*i+1]['hole_locations']) == 2

                if is_choice:
                    options = trials[3*i+1]['hole_locations']
                    chosen_hole = trials[3*i+1]['events'][0]['holeUsed']
                    unchosen = options[0] if options[0] != chosen_hole else options[1]
                    non_chosen_path = [
                        trials[3*i]['events'][0]['holeUsed'],
                        unchosen,
                        trials[3*i+2]['events'][0]['holeUsed']
                    ]
                    chosen_1step = abs(chosen_path[1] - chosen_path[0])
                    unchosen_1step = abs(non_chosen_path[1] - non_chosen_path[0])
                    chosen_2step = chosen_1step + abs(chosen_path[2] - chosen_path[1])
                    unchosen_2step = (unchosen_1step
                                      + abs(non_chosen_path[2] - non_chosen_path[1]))
                else:
                    non_chosen_path = None
                    chosen_1step = unchosen_1step = None
                    chosen_2step = unchosen_2step = None

                output.append({
                    'block_number': block_num,
                    'trial_sequence_number': i,
                    'hole_sequence': choice_trial_sequence,
                    'chosen_path': chosen_path,
                    'non_chosen_path': non_chosen_path,
                    'observed_rt': observed_rt,
                    'choice_trial': is_choice,
                    'chosen_1step_dist': chosen_1step,
                    'unchosen_1step_dist': unchosen_1step,
                    'chosen_2step_dist': chosen_2step,
                    'unchosen_2step_dist': unchosen_2step,
                    'block_drift': block_drift,
                    'ball_y_at_top': relative_ball_y
                })
            except (KeyError, IndexError, TypeError):
                continue

    output = pd.DataFrame(output)
    Q1 = output['observed_rt'].quantile(0.25)
    Q3 = output['observed_rt'].quantile(0.75)
    IQR = Q3 - Q1
    output = output[(output['observed_rt'] >= Q1 - 2.5*IQR)
                    & (output['observed_rt'] <= Q3 + 2.5*IQR)]
    output = output.reset_index(drop=True)

    chosen_middle = output['chosen_path'].str[1]
    unchosen_middle = output['non_chosen_path'].str[1]
    output['chosen_left'] = (chosen_middle < unchosen_middle)

    prev_end = output['chosen_path'].shift(1).str[2]
    curr_start = output['chosen_path'].str[0]
    direction = np.sign(prev_end - curr_start)
    prev_seq = output['trial_sequence_number'].shift(1)
    curr_seq = output['trial_sequence_number']
    prev_block = output['block_number'].shift(1)
    curr_block = output['block_number']
    is_valid = (prev_seq + 1 == curr_seq) & (prev_block == curr_block)
    output['incoming_direction'] = np.where(is_valid, -direction, np.nan)
    output = output.dropna(subset=['incoming_direction']).reset_index(drop=True)
    return output


def build_deeponet_dataset(participant_data_dict):
    all_features, all_choices, all_p_ids, all_rt = [], [], [], []

    for p_idx, (_, raw_data) in enumerate(participant_data_dict.items()):
        processed = pre_proccess_data_from_choice_vs_no_choice(raw_data)
        processed = processed[processed['choice_trial'] == True]
        processed = processed.dropna(subset=['chosen_1step_dist', 'ball_y_at_top'])
        processed = processed.reset_index(drop=True)

        is_left = processed['chosen_left']
        L1 = np.where(is_left, processed['chosen_1step_dist'],
                      processed['unchosen_1step_dist'])
        R1 = np.where(~is_left, processed['chosen_1step_dist'],
                      processed['unchosen_1step_dist'])
        Total_L = np.where(is_left, processed['chosen_2step_dist'],
                           processed['unchosen_2step_dist'])
        Total_R = np.where(~is_left, processed['chosen_2step_dist'],
                           processed['unchosen_2step_dist'])
        incoming = processed['incoming_direction']

        X = pd.DataFrame({
            'L1_minus_R1': L1 - R1,
            'Total_L_minus_Total_R': Total_L - Total_R,
            'ball_y_at_top': processed['ball_y_at_top'],
            'incoming_pos': (incoming == 1).astype(float),
            'incoming_neg': (incoming == -1).astype(float)
        })
        features = X.values
        choices = (~processed['chosen_left']).astype(float).values
        p_ids_arr = np.full(len(processed), p_idx)
        rt_arr = processed['observed_rt'].values.astype(float)

        all_features.append(features)
        all_choices.append(choices)
        all_p_ids.append(p_ids_arr)
        all_rt.append(rt_arr)

    return (np.vstack(all_features), np.concatenate(all_p_ids),
            np.concatenate(all_choices), np.concatenate(all_rt),
            len(participant_data_dict))


# %% [markdown]
# # Visualization Functions

# %%
def plot_strategy_timeline(strategy_weights, participant_id=0, n_trials=200):
    """Plot how strategy engagement changes over trials for one participant."""
    weights = strategy_weights[strategy_weights.shape[0] >= n_trials
                               and strategy_weights.numpy()[:n_trials, :]]
    plt.figure(figsize=(14, 4), dpi=120)
    num_strategies = strategy_weights.shape[1]
    colors = plt.cm.tab10(np.linspace(0, 1, num_strategies))
    for k in range(num_strategies):
        plt.plot(strategy_weights[:n_trials, k].numpy(), color=colors[k],
                 alpha=0.7, linewidth=0.8, label=f"Strategy {k+1}")
    plt.title(f"Strategy Engagement Timeline (Participant {participant_id})")
    plt.xlabel("Trial")
    plt.ylabel("Strategy Weight (softmax)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_strategy_heatmap(strategy_weights, participant_id=0, n_trials=200):
    """Heatmap view of strategy switching patterns."""
    weights_slice = strategy_weights[:n_trials].numpy()
    plt.figure(figsize=(12, 4), dpi=120)
    ax = sns.heatmap(weights_slice.T, cmap="YlOrRd", cbar_kws={'label': 'Weight'},
                     xticklabels=20, yticklabels=[f"S{k+1}" for k in range(weights_slice.shape[1])])
    plt.title(f"Strategy Engagement Heatmap (Participant {participant_id})")
    plt.xlabel("Trial")
    plt.tight_layout()
    plt.show()


def plot_strategy_distribution(all_strategy_weights, participant_ids):
    """Distribution of strategy usage per participant."""
    from collections import defaultdict
    p_strategies = defaultdict(list)
    for w, pid in zip(all_strategy_weights, participant_ids):
        p_strategies[pid].append(w.numpy())

    p_means = {}
    pids_sorted = sorted(p_strategies.keys())
    for pid in pids_sorted[:15]:
        p_means[pid] = np.mean(p_strategies[pid], axis=0)

    num_strats = all_strategy_weights.shape[1]
    plt.figure(figsize=(10, 6), dpi=120)
    for pid in pids_sorted[:15]:
        plt.plot(range(num_strats), p_means[pid], 'o-', alpha=0.5,
                 label=f"P{pid}")
    plt.xticks(range(num_strats), [f"S{k+1}" for k in range(num_strats)])
    plt.title("Mean Strategy Weights per Participant")
    plt.xlabel("Strategy")
    plt.ylabel("Average Weight")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_coefficients_heatmap(model, participant_ids=None):
    coeffs = model.participant_coeffs.weight.detach().cpu().numpy()
    # Reshape: [num_participants, num_strategies, num_bases]
    coeffs_reshaped = coeffs.reshape(-1, model.num_strategies, model.num_bases)
    n_p = min(15, coeffs_reshaped.shape[0])

    fig, axes = plt.subplots(1, model.num_strategies, figsize=(5*model.num_strategies, 6), dpi=120)
    if model.num_strategies == 1:
        axes = [axes]
    for k in range(model.num_strategies):
        matrix = coeffs_reshaped[:n_p, k, :]
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                    ax=axes[k], cbar_kws={'label': 'Coefficient'})
        axes[k].set_title(f"Strategy {k+1} Coefficients")
        axes[k].set_xlabel("Basis Function")
        axes[k].set_ylabel("Participant ID")
    plt.tight_layout()
    plt.show()


# %% [markdown]
# # Training Runner

# %%
def run_model(model_type='gated', participant_data_paths=None, num_strategies=3,
              num_bases=4, num_epochs=200, num_time_bins=5):
    """
    Full training pipeline for StrategyDeepONet variants.

    Args:
        model_type: 'gated' | 'multitask' | 'timebinned'
        participant_data_paths: list of file paths to participant JSON files
        num_strategies: K strategies for the mixture model
        num_bases: D basis functions per strategy
        num_epochs: training epochs
        num_time_bins: T temporal bins (timebinned mode only)

    Returns:
        model, all_strategy_weights, metrics
    """
    if participant_data_paths is None:
        raise ValueError("Provide participant_data_paths (list of JSON file paths)")

    participants = {}
    for i, path in enumerate(participant_data_paths):
        participants[f"P{i}"] = json.load(open(path))

    features, p_ids, choices, rt_values, num_participants = build_deeponet_dataset(participants)

    X_train, X_test, id_train, id_test, y_train, y_test = train_test_split(
        features, p_ids, choices, test_size=0.2, random_state=42)

    X_cont = X_train[:, :3]
    X_disc = X_train[:, 3:]
    Xt_cont = X_test[:, :3]
    Xt_disc = X_test[:, 3:]

    scaler = StandardScaler()
    X_train_final = np.hstack((scaler.fit_transform(X_cont), X_disc))
    X_test_final = np.hstack((scaler.transform(Xt_cont), Xt_disc))

    if model_type == 'gated':
        model = StrategyDeepONet(num_participants, num_features=5,
                                 num_bases=num_bases, num_strategies=num_strategies,
                                 shared_bases= True)
        train_set = MazeDataset(X_train_final, id_train, y_train)
        test_set = MazeDataset(X_test_final, id_test, y_test)
        train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
        trained = train_strategy_deeponet(model, train_loader, num_epochs=num_epochs)
        _, acc, weights, per_participant = evaluate_strategy_model(trained, test_loader)

    elif model_type == 'multitask':
        model = StrategyDeepONetMultiTask(num_participants, num_features=5,
                                          num_bases=num_bases, num_strategies=num_strategies,
                                          shared_bases= True)
        rt_mean = rt_values.mean()
        rt_std = rt_values.std() + 1e-8
        rt_values_norm = (rt_values - rt_mean) / rt_std
        rt_train, rt_test = train_test_split(rt_values_norm, test_size=0.2, random_state=42)
        train_set = MazeDataset(X_train_final, id_train, y_train, rt_train)
        test_set = MazeDataset(X_test_final, id_test, y_test, rt_test)
        train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
        trained = train_strategy_deeponet_multitask(model, train_loader, num_epochs=num_epochs)
        _, acc, weights, per_participant = evaluate_strategy_model(trained, test_loader, with_rt=True)

    elif model_type == 'timebinned':
        model = TimeBinnedStrategyDeepONet(num_participants, num_time_bins,
                                           num_features=5, num_bases=num_bases,
                                           num_strategies=num_strategies,
                                           shared_bases= True)
        bin_ids_train = np.floor(np.linspace(0, num_time_bins - 0.001, len(id_train))).astype(int)
        bin_ids_test = np.floor(np.linspace(0, num_time_bins - 0.001, len(id_test))).astype(int)
        train_set = MazeDataset(X_train_final, id_train, y_train, time_bin_ids=bin_ids_train)
        test_set = MazeDataset(X_test_final, id_test, y_test, time_bin_ids=bin_ids_test)
        train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
        trained = train_strategy_deeponet(model, train_loader, num_epochs=num_epochs)
        _, acc, weights, per_participant = evaluate_strategy_model(trained, test_loader)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    print(f"\nModel: {model_type} | Strategies: {num_strategies} | "
          f"Bases: {num_bases} | Accuracy: {acc*100:.1f}%")
    return trained, weights, {'accuracy': acc, 'per_participant': per_participant}

# %% [markdown]
# # Execution: Train and Visualize All Variants
#
# This section loads the same participant data as cognitivedeepOnet.py,
# trains all three model variants, and generates diagnostic plots.
# Run individual cells or the whole block — each variant is independent.

if __name__ == '__main__':
    # %%
    import glob
    import os

    # --- Resolve paths from this file's location ---
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

    # --- Find participant JSON files ---
    DATA_DIR = os.path.join(_SCRIPT_DIR, "cloud study data")
    available_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))

    print(f"Found {len(available_files)} participant files in {DATA_DIR}")
    for f in available_files:
        print(f"  {os.path.basename(f)}")

    if len(available_files) == 0:
        print("ERROR: No participant JSON files found. Place files in 'cloud study data/' "
              "in the project root.")
    else:
        participant_paths = available_files


    # %% [markdown]
    # ## Variant 1: Gated Strategy Model
    #
    # Trains `StrategyDeepONet` with K=3 strategies. Outputs trial-level
    # `strategy_weights` for every participant — the primary diagnostic.

    # %%
    if len(available_files) >= 3:
        print("Training gated StrategyDeepONet (K=3 strategies)...")
        gated_model, gated_weights, gated_metrics = run_model(
            model_type='gated',
            participant_data_paths=participant_paths,
            num_strategies=3,
            num_bases=4,
            num_epochs=200
        )

        # --- Visualizations ---
        print("\n--- Strategy Timeline (participant 0, first 200 trials) ---")
        plot_strategy_timeline(gated_weights, participant_id=0, n_trials=200)

        print("\n--- Strategy Heatmap (participant 0) ---")
        plot_strategy_heatmap(gated_weights, participant_id=0, n_trials=200)

        print("\n--- Coefficient Heatmap ---")
        plot_coefficients_heatmap(gated_model)

        print(f"\nGated model complete. Accuracy: {gated_metrics['accuracy']*100:.1f}%")

        # Save model for later clustering
        save_dir = os.path.join(_SCRIPT_DIR)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "gated_strategy_model.pt")
        torch.save(gated_model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
    else:
        print("Need at least 3 participants for gated model. Skipping.")


    # %% [markdown]
    # ## Variant 2: Multi-Task Model (Choice + RT)
    #
    # Adds RT prediction head to decompose strategies into deliberative
    # (choice-predictive only) vs. heuristic (predicts both choice and RT).

    # %%
    if len(available_files) >= 3:
        print("Training multi-task StrategyDeepONet (K=3, RT head)...")
        mt_model, mt_weights, mt_metrics = run_model(
            model_type='multitask',
            participant_data_paths=participant_paths,
            num_strategies=3,
            num_bases=4,
            num_epochs=200
        )
        print(f"\nMulti-task model complete. Accuracy: {mt_metrics['accuracy']*100:.1f}%")

        print("\n--- Strategy Distribution Across Participants ---")
        # Build participant ID list from the test split to match weights
        features, p_ids, choices, rt_values, n_p = build_deeponet_dataset(
            {f"P{i}": json.load(open(p)) for i, p in enumerate(participant_paths)})
        _, _, id_test, _ = train_test_split(
            features, p_ids, test_size=0.2, random_state=42)
        plot_strategy_distribution(mt_weights[:len(id_test)], id_test)
    else:
        print("Need at least 3 participants. Skipping.")


    # %% [markdown]
    # ## Variant 3: Time-Binned Model
    #
    # Learns separate strategy embeddings per temporal bin to track
    # how strategies evolve over the course of the experiment.

    # %%
    if len(available_files) >= 4:
        print("Training time-binned StrategyDeepONet (T=5 bins, K=3 strategies)...")
        tb_model, tb_weights, tb_metrics = run_model(
            model_type='timebinned',
            participant_data_paths=participant_paths,
            num_strategies=3,
            num_bases=4,
            num_time_bins=5,
            num_epochs=200
        )
        print(f"\nTime-binned model complete. Accuracy: {tb_metrics['accuracy']*100:.1f}%")

        # Extract and plot coefficient trajectories over time bins
        coeffs_raw = tb_model.participant_coeffs.weight.detach().cpu().numpy()
        num_participants = len(participant_paths)
        num_time_bins = 5
        num_strategies = 3
        num_bases = 4

        reshaped = coeffs_raw.reshape(num_participants, num_time_bins, num_strategies, num_bases)
        # Show trajectory for participant 0, strategy 1
        fig, axes = plt.subplots(1, num_strategies, figsize=(5*num_strategies, 4), dpi=120)
        if num_strategies == 1:
            axes = [axes]
        for s in range(num_strategies):
            for b in range(num_bases):
                axes[s].plot(range(num_time_bins), reshaped[0, :, s, b],
                             'o-', label=f"Basis {b+1}", linewidth=2)
            axes[s].set_title(f"Strategy {s+1} — Participant 0")
            axes[s].set_xlabel("Time Bin")
            axes[s].set_ylabel("Coefficient Value")
            axes[s].legend(fontsize=8)
            axes[s].axhline(0, color='gray', linestyle='--', alpha=0.3)
        plt.suptitle("Coefficient Trajectory Over Time Bins", fontsize=13)
        plt.tight_layout()
        plt.show()
    else:
        print("Need at least 4 participants for time-binned model. Skipping.")


    # %% [markdown]
    # ## Compare Variants
    #
    # Summary table of accuracy across all trained variants.

    # %%
    print("\n" + "=" * 50)
    print("MODEL COMPARISON")
    print("=" * 50)

    results = {}
    if 'gated_model' in dir() and 'gated_metrics' in dir():
        results['Gated (K=3)'] = gated_metrics['accuracy']
    if 'mt_model' in dir() and 'mt_metrics' in dir():
        results['Multi-Task (K=3)'] = mt_metrics['accuracy']
    if 'tb_model' in dir() and 'tb_metrics' in dir():
        results['Time-Binned (T=5, K=3)'] = tb_metrics['accuracy']

    for name, acc in results.items():
        print(f"  {name}: {acc*100:.2f}%")

    if len(results) >= 2:
        best = max(results, key=results.get)
        print(f"\nBest variant: {best}")

    print("\nAll models saved. Run analysis/strategy_clustering.py next to cluster "
          "participants by strategy type.")

    # %%
