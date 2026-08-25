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


# %%
class SequenceDataset(Dataset):
    """Yields one ordered participant sequence per item.

    Each sequence is (features (T, M), participant_id, choices (T,),
    rt (T,) or None, time_bin_ids (T,) or None). Because sequences have
    variable length, use DataLoader(batch_size=None) so no collation occurs.
    """
    def __init__(self, sequences, with_rt=False, time_binned=False):
        self.sequences = sequences
        self.with_rt = with_rt
        self.time_binned = time_binned

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        features, p_id, choices, rt, bins = self.sequences[idx]
        item = [torch.as_tensor(features, dtype=torch.float32),
                torch.as_tensor(p_id, dtype=torch.long),
                torch.as_tensor(choices, dtype=torch.float32)]
        if self.with_rt:
            item.append(torch.as_tensor(rt, dtype=torch.float32))
        if self.time_binned:
            item.append(torch.as_tensor(bins, dtype=torch.long))
        return tuple(item)


# %% [markdown]
# # StrategyDeepONet: Mixture-of-Strategies Model
#
# Replaces the single participant embedding with K separate strategy networks
# and a gate that selects the active strategy per trial. Draws on:
# - Ashwood et al 2022: GLM-HMM discrete strategy switching
# - Kirsch 2019: strategies as points in a computational-constraint space

# %%
class StrategyDeepONet(nn.Module):
    """HMM-gated strategy DeepONet.

    Replaces the per-trial feature-driven softmax gate with an explicit
    Markov chain over the K strategy states, so strategy switches are
    persistent rather than i.i.d. per trial (GLM-HMM analogue).

    Per participant p:
      - initial state distribution   pi_p = softmax(initial_logits[p])      (K,)
      - transition matrix            A_p  = row-softmax(transition_logits[p]) (K, K)

    Emissions are unchanged from the original DeepONet: the choice
    probability under state k is sigmoid of the basis-coefficient dot product.
    The full sequence log-likelihood is computed with the log-space forward
    algorithm (fully differentiable in torch).

    forward() consumes one ordered participant sequence at a time:
      trial_features: (T, M)
      participant_ids: () scalar
      choices: (T,) 0/1
      coeff_ids (optional, time-binned use): (T,) embedding ids per trial
    Returns (seq_ll, logits, bases, strategy_weights):
      seq_ll: scalar sequence log-likelihood
      logits: (T,) marginal logit of P(y=1) from the filtered posterior
      bases: (T, K, D)
      strategy_weights: (T, K) filtered posterior P(z_t | y_{1:t})
    """
    def __init__(self, num_participants, num_features=5, num_bases=4, num_strategies=3,
                 shared_bases=False):
        super().__init__()

        self.num_participants = num_participants
        self.num_features = num_features
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

        # Markov chain over strategies (static, per-participant)
        self.transition_logits = nn.Parameter(
            torch.zeros(num_participants, num_strategies, num_strategies))
        self.initial_logits = nn.Parameter(
            torch.zeros(num_participants, num_strategies))

    def _bases(self, trial_features):
        """Return basis outputs as (T, K, D)."""
        if self.shared_bases:
            b = self.basis_net(trial_features)  # (T, D)
            return b.unsqueeze(1).expand(-1, self.num_strategies, -1)
        return torch.stack([net(trial_features) for net in self.basis_nets], dim=1)

    def forward(self, trial_features, participant_ids, choices, coeff_ids=None):
        T = trial_features.shape[0]
        pid = participant_ids
        if coeff_ids is None:
            coeff_ids = torch.full((T,), pid, dtype=torch.long, device=trial_features.device)

        coeffs = self.participant_coeffs(coeff_ids)                      # (T, K*D)
        coeffs = coeffs.view(T, self.num_strategies, self.num_bases)

        bases = self._bases(trial_features)                              # (T, K, D)
        state_logits = (bases * coeffs).sum(dim=-1)                      # (T, K)

        eps = 1e-8
        probs = torch.clamp(torch.sigmoid(state_logits), eps, 1 - eps)   # (T, K)

        y = choices.float().view(T, 1)
        emission_ll = (y * torch.log(probs)
                       + (1 - y) * torch.log(1 - probs))                 # (T, K)

        log_A = F.log_softmax(self.transition_logits[pid], dim=-1)       # (K, K)
        log_pi = F.log_softmax(self.initial_logits[pid], dim=-1)         # (K,)

        # Log-space forward algorithm (differentiable)
        alphas = torch.empty(T, self.num_strategies, device=trial_features.device)
        alphas[0] = log_pi + emission_ll[0]
        for t in range(1, T):
            alphas[t] = (torch.logsumexp(alphas[t - 1].unsqueeze(-1) + log_A, dim=0)
                         + emission_ll[t])
        seq_ll = torch.logsumexp(alphas[T - 1], dim=0)

        strategy_weights = F.softmax(alphas, dim=-1)                     # filtered posterior (T, K)
        marginal_p = torch.clamp((strategy_weights * probs).sum(dim=-1), eps, 1 - eps)
        logits = torch.log(marginal_p / (1 - marginal_p))                # (T,)

        return seq_ll, logits, bases, strategy_weights


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

    def forward(self, trial_features, participant_ids, choices, coeff_ids=None):
        seq_ll, logits, bases, strategy_weights = super().forward(
            trial_features, participant_ids, choices, coeff_ids=coeff_ids)

        rt_coeffs = self.rt_coeffs(participant_ids)                  # (K*D,)
        rt_coeffs = rt_coeffs.view(self.num_strategies, self.num_bases)
        rt_state = (bases * rt_coeffs).sum(dim=-1)                   # (T, K)
        rt_pred = (strategy_weights * rt_state).sum(dim=-1)          # (T,)

        return seq_ll, logits, bases, strategy_weights, rt_pred


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

        # Per (participant, time-bin) coefficient sets; transitions/initial
        # distribution remain per-participant (from the base class).
        coeff_dim = self.num_strategies * self.num_bases
        self.participant_coeffs = nn.Embedding(num_participants * num_time_bins, coeff_dim)
        nn.init.normal_(self.participant_coeffs.weight, mean=0.0, std=0.1)

    def forward(self, trial_features, participant_ids, time_bin_ids, choices):
        coeff_ids = participant_ids * self.num_time_bins + time_bin_ids
        return super().forward(trial_features, participant_ids, choices, coeff_ids=coeff_ids)


# %% [markdown]
# # Training Functions

# %%
def train_strategy_deeponet(model, dataloader, num_epochs=200, lr=0.001,
                            penalty_weight=0.5, entropy_weight=0.05):
    """Train an HMM-gated StrategyDeepONet by sequence negative log-likelihood.

    Loss = seq-NLL (per-trial normalized) + orthogonality penalty
           - entropy bonus on mean state occupancy (prevents single-state collapse).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    is_time_binned = isinstance(model, TimeBinnedStrategyDeepONet)

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_nll = 0.0
        total_orth = 0.0
        total_entropy = 0.0
        n = 0

        for batch in dataloader:
            optimizer.zero_grad()

            if is_time_binned:
                features, p_ids, true_choices, bin_ids = batch
                seq_ll, logits, bases, strategy_weights = model(
                    features, p_ids, bin_ids, true_choices)
            else:
                features, p_ids, true_choices = batch
                seq_ll, logits, bases, strategy_weights = model(
                    features, p_ids, true_choices)

            T = features.shape[0]
            nll = -seq_ll / T

            # Orthogonality penalty: with shared bases, penalize once
            orth_loss = 0.0
            if model.shared_bases:
                orth_loss = orthogonality_penalty(bases[:, 0, :])
            else:
                for k in range(model.num_strategies):
                    orth_loss += orthogonality_penalty(bases[:, k, :])
                orth_loss /= model.num_strategies

            # Entropy bonus: keep mean state occupancy spread out (no collapse)
            occ = strategy_weights.mean(dim=0)
            entropy = -(occ * torch.log(occ.clamp(min=1e-8))).sum()

            loss = nll + penalty_weight * orth_loss - entropy_weight * entropy
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_nll += nll.item()
            total_orth += orth_loss.item()
            total_entropy += entropy.item()
            n += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}/{num_epochs} | "
                  f"Loss: {total_loss/n:.4f} | NLL: {total_nll/n:.4f} | "
                  f"Orth: {total_orth/n:.4f} | Ent: {total_entropy/n:.4f}")

    return model


# %%
def train_strategy_deeponet_multitask(model, dataloader, num_epochs=200, lr=0.001,
                                      penalty_weight=0.5, entropy_weight=0.05,
                                      rt_weight=0.3):
    """Train the HMM-gated multi-task model (choice + RT) by seq-NLL + RT MSE."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_nll = 0.0
        total_mse = 0.0
        total_orth = 0.0
        total_entropy = 0.0
        n = 0

        for batch in dataloader:
            features, p_ids, true_choices, true_rt = batch
            optimizer.zero_grad()

            seq_ll, logits, bases, strategy_weights, rt_pred = model(
                features, p_ids, true_choices)

            T = features.shape[0]
            nll = -seq_ll / T
            mse_loss = F.mse_loss(rt_pred, true_rt)

            orth_loss = 0.0
            if model.shared_bases:
                orth_loss = orthogonality_penalty(bases[:, 0, :])
            else:
                for k in range(model.num_strategies):
                    orth_loss += orthogonality_penalty(bases[:, k, :])
                orth_loss /= model.num_strategies

            occ = strategy_weights.mean(dim=0)
            entropy = -(occ * torch.log(occ.clamp(min=1e-8))).sum()

            loss = (nll + rt_weight * mse_loss
                    + penalty_weight * orth_loss - entropy_weight * entropy)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_nll += nll.item()
            total_mse += mse_loss.item()
            total_orth += orth_loss.item()
            total_entropy += entropy.item()
            n += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}/{num_epochs} | "
                  f"Loss: {total_loss/n:.4f} | NLL: {total_nll/n:.4f} | "
                  f"MSE: {total_mse/n:.4f} | Orth: {total_orth/n:.4f} | "
                  f"Ent: {total_entropy/n:.4f}")

    return model


# %%
def train_time_binned(model, dataloader, num_epochs=200, lr=0.001,
                      penalty_weight=0.5, entropy_weight=0.05):
    return train_strategy_deeponet(model, dataloader, num_epochs=num_epochs, lr=lr,
                                   penalty_weight=penalty_weight,
                                   entropy_weight=entropy_weight)


# %% [markdown]
# # Evaluation

# %%
def evaluate_strategy_model(model, dataloader, with_rt=False):
    model.eval()
    is_time_binned = isinstance(model, TimeBinnedStrategyDeepONet)

    total_ll = 0.0
    correct = 0
    total = 0
    all_strategy_weights = []
    p_ll = {}
    p_correct = {}
    p_total = {}

    with torch.no_grad():
        for batch in dataloader:
            if is_time_binned:
                features, p_ids, true_choices, bin_ids = batch
                seq_ll, logits, bases, strategy_weights = model(
                    features, p_ids, bin_ids, true_choices)
            elif with_rt:
                features, p_ids, true_choices, _ = batch
                seq_ll, logits, bases, strategy_weights, _ = model(
                    features, p_ids, true_choices)
            else:
                features, p_ids, true_choices = batch
                seq_ll, logits, bases, strategy_weights = model(
                    features, p_ids, true_choices)

            trial_bce = F.binary_cross_entropy_with_logits(
                logits, true_choices, reduction='none')
            trial_ll = -trial_bce

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            is_correct = (preds == true_choices)
            T = features.shape[0]
            correct += is_correct.sum().item()
            total += T
            total_ll += trial_ll.sum().item()
            all_strategy_weights.append(strategy_weights.cpu())

            pid = int(p_ids)
            p_ll[pid] = p_ll.get(pid, 0.0) + trial_ll.sum().item()
            p_correct[pid] = p_correct.get(pid, 0) + is_correct.sum().item()
            p_total[pid] = p_total.get(pid, 0) + T

    avg_ll = total_ll / total if total else 0.0
    acc = correct / total if total else 0.0
    all_weights = (torch.cat(all_strategy_weights, dim=0)
                   if all_strategy_weights else torch.empty(0))

    per_participant = {}
    for pid in sorted(p_ll.keys()):
        per_participant[pid] = {
            'accuracy': p_correct[pid] / p_total[pid],
            'log_likelihood': p_ll[pid] / p_total[pid],
        }

    print(f"--- Evaluation ---")
    print(f"Avg Log-Likelihood: {avg_ll:.4f}")
    print(f"Accuracy: {acc*100:.2f}% ({correct}/{total})")
    if all_weights.numel() > 0:
        print(f"Strategy usage (mean posterior): {all_weights.mean(dim=0).tolist()}")

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


# %%
def _build_participant_trials(processed):
    """Extract the standard feature set + choices + RT from preprocessed rows,
    preserving trial order. Returns (features, choices, rt) or None if too few."""
    processed = processed[processed['choice_trial'] == True]
    processed = processed.dropna(subset=['chosen_1step_dist', 'ball_y_at_top'])
    processed = processed.reset_index(drop=True)
    if len(processed) < 4:
        return None

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
        'incoming_neg': (incoming == -1).astype(float),
    })
    features = X.values.astype(np.float32)
    choices = (~processed['chosen_left']).astype(np.float32).values
    rt = processed['observed_rt'].values.astype(np.float32)
    return features, choices, rt


def build_sequence_dataset(participant_data_dict, test_frac=0.2, num_time_bins=None):
    """Build ordered per-participant train/test sequences for the HMM-gated models.

    Each participant's choice trials are split temporally (first (1-test_frac)
    trials for train, the rest for test). Continuous features and RT are z-scored
    using statistics from the pooled training sequences only. Participants with
    too few trials are dropped and re-indexed 0..N-1.

    Returns (train_seqs, test_seqs, num_participants), where each sequence is
    (features (T, M), participant_id, choices (T,), rt (T,), time_bin_ids (T,) or None).
    """
    kept = []
    for _, raw_data in participant_data_dict.items():
        processed = pre_proccess_data_from_choice_vs_no_choice(raw_data)
        trials = _build_participant_trials(processed)
        if trials is not None:
            kept.append(trials)

    if not kept:
        return [], [], 0

    train_raw, test_raw = [], []
    for new_p, (features, choices, rt) in enumerate(kept):
        n = len(features)
        split = int(n * (1 - test_frac))
        split = max(min(split, n - 2), 2)

        tr_f, te_f = features[:split], features[split:]
        tr_c, te_c = choices[:split], choices[split:]
        tr_r, te_r = rt[:split], rt[split:]

        if num_time_bins:
            tr_b = np.minimum((np.arange(split) * num_time_bins) // split,
                              num_time_bins - 1)
            te_b = np.minimum((np.arange(n - split) * num_time_bins) // (n - split),
                              num_time_bins - 1)
        else:
            tr_b = te_b = None

        train_raw.append([tr_f, new_p, tr_c, tr_r, tr_b])
        test_raw.append([te_f, new_p, te_c, te_r, te_b])

    # z-score continuous features and RT on the pooled train part
    all_tr = np.vstack([s[0] for s in train_raw])
    mu = all_tr[:, :3].mean(axis=0)
    std = all_tr[:, :3].std(axis=0) + 1e-8
    all_rt = np.concatenate([s[3] for s in train_raw])
    rt_mu, rt_std = all_rt.mean(), all_rt.std() + 1e-8

    for s in train_raw:
        s[0] = s[0].copy()
        s[0][:, :3] = (s[0][:, :3] - mu) / std
        s[3] = (s[3] - rt_mu) / rt_std
    for s in test_raw:
        s[0] = s[0].copy()
        s[0][:, :3] = (s[0][:, :3] - mu) / std
        s[3] = (s[3] - rt_mu) / rt_std

    train_seqs = [tuple(s) for s in train_raw]
    test_seqs = [tuple(s) for s in test_raw]
    return train_seqs, test_seqs, len(kept)


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
    Full training pipeline for the HMM-gated StrategyDeepONet variants.

    Args:
        model_type: 'gated' | 'multitask' | 'timebinned'
        participant_data_paths: list of file paths to participant JSON files
        num_strategies: K strategies (latent HMM states)
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

    is_time_binned = (model_type == 'timebinned')
    train_seqs, test_seqs, num_participants = build_sequence_dataset(
        participants, test_frac=0.2,
        num_time_bins=(num_time_bins if is_time_binned else None))

    if num_participants == 0:
        raise ValueError("No participants with enough trials to build sequences.")

    if model_type == 'gated':
        model = StrategyDeepONet(num_participants, num_features=5,
                                 num_bases=num_bases, num_strategies=num_strategies,
                                 shared_bases=True)
        train_set = SequenceDataset(train_seqs, with_rt=False, time_binned=False)
        test_set = SequenceDataset(test_seqs, with_rt=False, time_binned=False)
        train_loader = DataLoader(train_set, batch_size=None, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=None, shuffle=False)
        trained = train_strategy_deeponet(model, train_loader, num_epochs=num_epochs)
        _, acc, weights, per_participant = evaluate_strategy_model(trained, test_loader)

    elif model_type == 'multitask':
        model = StrategyDeepONetMultiTask(num_participants, num_features=5,
                                          num_bases=num_bases, num_strategies=num_strategies,
                                          shared_bases=True)
        train_set = SequenceDataset(train_seqs, with_rt=True, time_binned=False)
        test_set = SequenceDataset(test_seqs, with_rt=True, time_binned=False)
        train_loader = DataLoader(train_set, batch_size=None, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=None, shuffle=False)
        trained = train_strategy_deeponet_multitask(model, train_loader, num_epochs=num_epochs)
        _, acc, weights, per_participant = evaluate_strategy_model(trained, test_loader, with_rt=True)

    elif model_type == 'timebinned':
        model = TimeBinnedStrategyDeepONet(num_participants, num_time_bins,
                                           num_features=5, num_bases=num_bases,
                                           num_strategies=num_strategies,
                                           shared_bases=True)
        train_set = SequenceDataset(train_seqs, with_rt=False, time_binned=True)
        test_set = SequenceDataset(test_seqs, with_rt=False, time_binned=True)
        train_loader = DataLoader(train_set, batch_size=None, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=None, shuffle=False)
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
    import glob
    import os

    # --- Resolve paths from this file's location ---
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

    # --- Find participant JSON files ---
    DATA_DIR = os.path.join(_SCRIPT_DIR, "..", "data", "cloud_study")
    available_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))

    print(f"Found {len(available_files)} participant files in {DATA_DIR}")
    for f in available_files:
        print(f"  {os.path.basename(f)}")

    if len(available_files) == 0:
        print("ERROR: No participant JSON files found. Place files in 'data/cloud_study/' "
              "in the project root.")
    else:
        participant_paths = available_files


    if len(available_files) >= 3:
        print("Training gated StrategyDeepONet (K=3 strategies)...")
        gated_model, gated_weights, gated_metrics = run_model(
            model_type='gated',
            participant_data_paths=participant_paths,
            num_strategies=3,
            num_bases=4,
            num_epochs=400
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


    if len(available_files) >= 3:
        print("Training multi-task StrategyDeepONet (K=3, RT head)...")
        mt_model, mt_weights, mt_metrics = run_model(
            model_type='multitask',
            participant_data_paths=participant_paths,
            num_strategies=3,
            num_bases=4,
            num_epochs=400
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


    if len(available_files) >= 4:
        print("Training time-binned StrategyDeepONet (T=5 bins, K=3 strategies)...")
        tb_model, tb_weights, tb_metrics = run_model(
            model_type='timebinned',
            participant_data_paths=participant_paths,
            num_strategies=3,
            num_bases=4,
            num_time_bins=5,
            num_epochs=400
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
