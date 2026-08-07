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
# # Model Comparison
#
# Run every model on every participant and compare.
#
# **Models compared:**
#   1. Logistic Regression
#   2. RNN (TinyDecisionRNN, GRU)
#   3. Feedforward NN (distance features)
#   4. Feedforward NN (raw position features)
#   5. GLM-HMM (2-state and 3-state, same inputs as logistic regression)
#   6. CognitiveDeepONet (shared basis)
#   7. StrategyDeepONet — HMM-gated (Markov transitions over strategies)
#   8. StrategyDeepONet — multi-task (choice + RT)
#   9. StrategyDeepONet — time-binned
#   10. Custom cognitive model (planning + greedy mixture)
#
# **Output:** accuracy pivot table + mean ranking + box plot.

# %%

PARTICIPANT_FILES = {
        "P1": "cloud study data/65D6694BE06947289BE4336BC1DE271A-019e9464-b9d3-798d-aa65-c87d82961db6-019e8386-74e7-7359-827b-6b4e4bc47db9-2026-06-04T21-03-48-346Z-fg8d.json",
        "P2": "cloud study data/88AD64F00C6B43489770A02E7A1AE2C2-019e8fd9-16e9-7876-8e3b-d51a48df0526-019e8386-74e7-7359-827b-6b4e4bc47db9-2026-06-03T23-37-31-300Z-4ecm.json",
        "P3": "cloud study data/6462D588260B4356936047A04A336EBE-019e9464-f99c-77c5-bf47-327c7a7cf4f1-019e8386-74e7-7359-827b-6b4e4bc47db9-2026-06-04T21-41-26-943Z-c5do.json",
        "P4": "cloud study data/46331EBA4F494FAD901E83106523FF12-019e9464-9d12-7cc3-8cba-8f0dd00eeb20-019e8386-74e7-7359-827b-6b4e4bc47db9-2026-06-04T20-48-33-792Z-sop6.json",
        "P5": "cloud study data/BB4D2ACD4DAB45F5BAB68A472EB2E06C-019e9464-9a85-718c-9964-ec6755cdcd1c-019e8386-74e7-7359-827b-6b4e4bc47db9-2026-06-04T20-48-17-611Z-i0am.json",
        "P6": "cloud study data/C47CEEC22AD9448E9F87D0577BA7FC80-019e946e-abeb-723a-8d4d-50881fc0551f-019e8386-74e7-7359-827b-6b4e4bc47db9-2026-06-04T20-59-12-508Z-e1tl.json",
        "P7": "cloud study data/CEFD2FE92E6847B2B27FF0175811CE81-019e9464-988c-7240-bf66-336f77c05049-019e8386-74e7-7359-827b-6b4e4bc47db9-2026-06-04T20-50-03-371Z-34zm.json",
        "P8": "cloud study data/EC07396CE23248F2855499612FEB8ACA-019e9464-92a5-7d10-b713-7022c5b049fc-019e8386-74e7-7359-827b-6b4e4bc47db9-2026-06-04T20-48-16-501Z-olib.json",
        "P9": "cloud study data/FD2A6686546A4D689BE4A684CD264636-019e946a-96b4-78df-ac42-63e6e82c3209-019e8386-74e7-7359-827b-6b4e4bc47db9-2026-06-04T20-54-42-499Z-j7h3.json",

        "P10": "cloud study data/32FC87F1C127480BA90BCC97640655_cleaned.json",
        "P11": "cloud study data/96CA2FB7709946BB8EB38CAB5B713E_cleaned.json",
        "P12": "cloud study data/B0525260D0F8488D8D4695DD76FF64_cleaned.json",
        "P13": "cloud study data/C8C4C97C01AA45CA9064DA1A7635A4_cleaned.json",
        "P14": "cloud study data/EA4EE5B954A749C8BEED8F06A43F58_cleaned.json"
   }

import json
import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-blocking backend — prevents plt.show() from hanging on import

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from RNN import (
    run_RNN_for_eval,
    run_FF_for_eval,
    run_FF_position_for_eval,
)

from exploratory_data_analysis import (
    get_participants_data,
    pre_proccess_data_from_choice_vs_no_choice,
    evaluate_logistic_baseline,
    run_participant_fits,
)

import ssm

from cognitivedeepOnet import (
    CognitiveDeepONet,
    MazeDataset as CogMazeDataset,
    build_deeponet_dataset as cog_build_dataset,
    train_deeponet,
    evaluate_deeponet,
)

from strategy_deeponet import (
    MazeDataset as StratMazeDataset,
    build_deeponet_dataset as strat_build_dataset,
    train_strategy_deeponet,
    train_strategy_deeponet_multitask,
    train_time_binned,
    evaluate_strategy_model,
    run_model,
    StrategyDeepONet,
    StrategyDeepONetMultiTask,
    TimeBinnedStrategyDeepONet,
)

# Silence sklearn spurious warnings
warnings.filterwarnings("ignore", category=UserWarning)


# %% [markdown]
# ## Per-participant models

# %%
def run_glmhmm_for_participant(raw_data, num_states=2, sticky=False,
                               num_iters=200, test_split=0.2, kappa=100.0):
    """
    Fit a K-state GLM-HMM on a single participant's data using the SAME
    features, target, standardization, and chronological block split as the
    logistic-regression baseline (evaluate_logistic_baseline).

    Features per trial (built identically to the logistic baseline):
        L1-R1                 (z-scored, fit on train blocks)
        L1+L2-R1-R2           (z-scored, fit on train blocks)
        Incoming Direction
        Block Drift x Incoming Direction interaction (added if drift varies)
        + constant bias column (mirrors sklearn's intercept)

    Target: chosen_left (1 = left). Trained/tested on the same train/test
    blocks as the logistic baseline, so accuracy / log-likelihood are directly
    comparable.

    Returns a dict with accuracy, log-likelihood, and, for reporting, the
    per-state GLM weights, transition matrix, state occupancy, and switching
    rate. Returns None if there are too few trials.
    """
    processed = pre_proccess_data_from_choice_vs_no_choice(raw_data)
    if isinstance(processed, list):
        processed = pd.DataFrame(processed)
    processed = processed[processed['choice_trial'] == True].dropna().reset_index(drop=True)

    if len(processed) < 20:
        return None

    is_left = processed['chosen_left'].astype(bool)
    L1 = np.where(is_left, processed['chosen_1step_dist'], processed['unchosen_1step_dist'])
    R1 = np.where(~is_left, processed['chosen_1step_dist'], processed['unchosen_1step_dist'])
    L2 = np.where(is_left,
                  processed['chosen_2step_dist'] - processed['chosen_1step_dist'],
                  processed['unchosen_2step_dist'] - processed['unchosen_1step_dist'])
    R2 = np.where(~is_left,
                  processed['chosen_2step_dist'] - processed['chosen_1step_dist'],
                  processed['unchosen_2step_dist'] - processed['unchosen_1step_dist'])

    # ---- identical feature construction as the logistic baseline ----
    X = pd.DataFrame({
        'diff_1step': L1 - R1,
        'diff_planning': L1 + L2 - R2 - R1,
        'Block Drift': processed['block_drift'],
        'block_number': processed['block_number'],
        'chosen_left': processed['chosen_left'].astype(int),
        'Incoming Direction': processed['incoming_direction'],
    }).copy()

    # ---- same chronological block split as run_logistic_regression_baseline ----
    valid_trials_per_block = X.groupby('block_number').size().sort_index()
    cumulative_trials = valid_trials_per_block.cumsum()
    total_trials = cumulative_trials.values[-1] if len(cumulative_trials) else 0
    train_threshold = total_trials * (1 - test_split)
    train_blocks = valid_trials_per_block[cumulative_trials <= train_threshold].index
    test_blocks = valid_trials_per_block[cumulative_trials > train_threshold].index
    if len(test_blocks) == 0 and len(valid_trials_per_block) > 1:
        test_blocks = [valid_trials_per_block.index[-1]]
        train_blocks = valid_trials_per_block.index[:-1]

    train_df = X[X['block_number'].isin(train_blocks)].copy()
    test_df = X[X['block_number'].isin(test_blocks)].copy()

    if len(train_df) < 10 or len(test_df) < 5:
        return None

    # ---- same z-scoring (fit on train blocks only) ----
    mu_1, sig_1 = train_df['diff_1step'].mean(), train_df['diff_1step'].std()
    mu_2, sig_2 = train_df['diff_planning'].mean(), train_df['diff_planning'].std()
    sig_1 = sig_1 if sig_1 != 0 else 1e-6
    sig_2 = sig_2 if sig_2 != 0 else 1e-6

    train_df['L1-R1'] = (train_df['diff_1step'] - mu_1) / sig_1
    test_df['L1-R1'] = (test_df['diff_1step'] - mu_1) / sig_1
    train_df['L1+L2-R1-R2'] = (train_df['diff_planning'] - mu_2) / sig_2
    test_df['L1+L2-R1-R2'] = (test_df['diff_planning'] - mu_2) / sig_2

    has_drift_variance = train_df['Block Drift'].nunique() > 1
    if has_drift_variance:
        for df in [train_df, test_df]:
            df['Block Drift + Incoming Direction Interaction'] = (
                df['Incoming Direction'] * df['Block Drift'])
        feature_cols = ['L1-R1', 'L1+L2-R1-R2', 'Incoming Direction',
                        'Block Drift + Incoming Direction Interaction']
    else:
        feature_cols = ['L1-R1', 'L1+L2-R1-R2', 'Incoming Direction']

    def to_inputs(df):
        cols = df[feature_cols].values.astype(float)
        return np.column_stack([cols, np.ones(len(df))])  # bias column = intercept

    train_inpts = to_inputs(train_df)
    test_inpts = to_inputs(test_df)
    y_train = train_df['chosen_left'].astype(int).values
    y_test = test_df['chosen_left'].astype(int).values

    # ssm's input_driven_obs (D=1) expects observation arrays shaped (T, 1)
    choices_train = [np.array(y_train, dtype=int).reshape(-1, 1)]
    choices_test = [np.array(y_test, dtype=int).reshape(-1, 1)]
    input_dim = train_inpts.shape[1]

    try:
        if sticky:
            transitions = "sticky"
            transition_kwargs = dict(alpha=2.0, kappa=kappa)
        else:
            transitions = "standard"
            transition_kwargs = {}
        glmhmm = ssm.HMM(num_states, 1, input_dim,
                         observations="input_driven_obs",
                         observation_kwargs=dict(C=2),
                         transitions=transitions,
                         transition_kwargs=transition_kwargs)
        glmhmm.fit(choices_train, inputs=train_inpts, method="em",
                   num_iters=num_iters, tolerance=10**-4)

        log_probs = glmhmm.log_likelihood(choices_test, inputs=test_inpts)
        avg_ll = float(log_probs) / len(choices_test[0])

        posterior = glmhmm.filter(choices_test[0], input=test_inpts)  # (T, K)

        # ssm's input_driven_obs stores C-1 weight sets per state (the last
        # category is the softmax baseline with logit 0).
        W = glmhmm.observations.params  # (K, C-1, M)
        C = 2
        preds = []
        for t in range(len(choices_test[0])):
            class_logits = np.zeros((posterior.shape[1], C))
            class_logits[:, :C - 1] = W @ test_inpts[t]  # (K, C-1)
            class_logits -= class_logits.max(axis=-1, keepdims=True)
            class_probs = np.exp(class_logits)
            class_probs /= class_probs.sum(axis=-1, keepdims=True)
            marginal = posterior[t] @ class_probs  # (C,)
            preds.append(int(np.argmax(marginal)))

        accuracy = np.mean(preds == y_test)

        # Reporting: per-state decision weights (positive -> left; the baseline
        # class-1 weights are 0 by construction), transition matrix, occupancy,
        # switching rate
        state_weights = glmhmm.observations.params[:, 0, :]
        transition_matrix = np.exp(glmhmm.transitions.log_Ps)
        occupancy = posterior.mean(axis=0)
        switching_rate = 1.0 - np.trace(transition_matrix) / num_states

        return {
            "log_likelihood": avg_ll,
            "accuracy": accuracy,
            "num_states": num_states,
            "sticky": sticky,
            "state_weights": state_weights,
            "transition_matrix": transition_matrix,
            "occupancy": occupancy,
            "switching_rate": switching_rate,
            "features": feature_cols,
        }
    except Exception as e:
        warnings.warn(f"GLM-HMM failed: {e}")
        return None


def print_glmhmm_summary(m):
    """Compact per-participant GLM-HMM reporting (weights, transitions, occupancy)."""
    if m is None:
        return
    tm = m['transition_matrix']
    print(f"      transition matrix:\n{tm.round(2)}")
    for s in range(m['num_states']):
        w = m['state_weights'][s]
        top = int(np.argmax(np.abs(w[:len(m['features'])])))
        print(f"      state {s+1} weights: {np.round(w, 2)}  "
              f"(strongest: {m['features'][top]})")
    print(f"      occupancy: {np.round(m['occupancy'], 2)}  "
          f"switching rate: {m['switching_rate']:.2f}")


def run_custom_cognitive_model(participant_data_dict):
    """Run run_participant_fits and return mean test accuracy + per-participant DataFrame."""
    results_df = run_participant_fits(participant_data_dict)
    if results_df.empty:
        return None, results_df
    mean_test_acc = results_df['Test_Accuracy'].mean()
    return mean_test_acc, results_df


# %% [markdown]
# ## Multi-participant DeepONet models

# %%
def run_cognitivedeeponet_all(participant_data_dict, num_epochs=200):
    """Train CognitiveDeepONet on all participants, return per-participant accuracy + LL."""
    features, p_ids, choices, num_p = cog_build_dataset(participant_data_dict)

    X_train, X_test, id_train, id_test, y_train, y_test = train_test_split(
        features, p_ids, choices, test_size=0.2, random_state=42)

    X_cont = X_train[:, :3]
    X_disc = X_train[:, 3:]
    Xt_cont = X_test[:, :3]
    Xt_disc = X_test[:, 3:]

    scaler = StandardScaler()
    X_train_final = np.hstack((scaler.fit_transform(X_cont), X_disc))
    X_test_final = np.hstack((scaler.transform(Xt_cont), Xt_disc))

    model = CognitiveDeepONet(num_participants=num_p, num_features=5, num_bases=4)
    train_set = CogMazeDataset(X_train_final, id_train, y_train)
    test_set = CogMazeDataset(X_test_final, id_test, y_test)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    train_deeponet(model, train_loader, num_epochs=num_epochs, lr=0.0015, penalty_weight=0.5)
    _, test_acc, per_participant = evaluate_deeponet(model, test_loader)
    return test_acc, per_participant


def run_strategy_variant_all(participant_data_paths, model_type, num_epochs=200):
    """Train a StrategyDeepONet variant on all participants, return per-participant accuracy + LL."""
    _, _, metrics = run_model(
        model_type=model_type,
        participant_data_paths=participant_data_paths,
        num_strategies=3,
        num_bases=4,
        num_epochs=num_epochs,
        num_time_bins=5,
    )
    return metrics['accuracy'], metrics['per_participant']


# %% [markdown]
# ## Main comparison — run all 10 models

# %%
def compare_all_models(data_dir="cloud study data",
                       participant_ids=None,
                       num_epochs_rnn=400,
                       num_epochs_ff=400,
                       num_epochs_deeponet=200,
                       plot=True):
    """
    Run every model on every participant and return a comparison DataFrame.

    Parameters
    ----------
    data_dir : str
        Directory containing participant JSON files for DeepONet file-path lookup.
    participant_ids : list of str, optional
        Specific participant IDs to use (e.g. ["P1","P3"]). If None, loads all.
    num_epochs_rnn : int
        Training epochs for RNN.
    num_epochs_ff : int
        Training epochs for both feedforward variants.
    num_epochs_deeponet : int
        Training epochs for all DeepONet variants.
    plot : bool
        Whether to produce the box-plot figure.

    Returns
    -------
    pd.DataFrame with columns: Participant, Model, Accuracy, LogLikelihood.
    """
    print("=" * 70)
    print("  MODEL COMPARISON — All Models, All Participants")
    print("=" * 70)

    # ── Load participants ──────────────────────────────────────────────
    # 1. Known participants (hardcoded in exploratory_data_analysis.py)
    known = get_participants_data()
    known_ids = set()
    for d in known:
        sid = d.get("subject_id", "")
        if sid:
            known_ids.add(sid)

    # 2. Scan cloud study data/ for any files not in the hardcoded set
    data_dir_abs = Path(__file__).resolve().parent / "cloud study data"
    extra_data = []
    if data_dir_abs.is_dir():
        for fpath in sorted(data_dir_abs.glob("*.json")):
            with open(fpath) as f:
                d = json.load(f)
            sid = d.get("subject_id", "")
            if sid and sid not in known_ids:
                known.append(d)
                extra_data.append(d)
                known_ids.add(sid)
        if extra_data:
            print(f"  Loaded {len(extra_data)} additional participant(s) from {data_dir_abs}")

    participants_data = known
    labels = [f"P{i + 1}" for i in range(len(participants_data))]

    if participant_ids is not None:
        # Filter to requested IDs (match by numeric index P1, P2, …)
        idx_map = {f"P{i + 1}": i for i in range(len(labels))}
        selected = [idx_map[pid] for pid in participant_ids if pid in idx_map]
        participants_data = [participants_data[i] for i in selected]
        labels = [participant_ids[i] for i in range(len(participant_ids)) if participant_ids[i] in idx_map]

    participant_data_dict = {
        f"Participant_{i + 1}": data for i, data in enumerate(participants_data)
    }

    # Resolve file paths for StrategyDeepONet (requires paths, not dicts)
    participant_paths = []
    for pid in participant_data_dict.keys():
        pdata = participant_data_dict[pid]
        subject_id = pdata.get("subject_id", "")
        if not subject_id:
            tmp_name = f"graphify-out/_tmp_{pid}.json"
            Path(tmp_name).parent.mkdir(parents=True, exist_ok=True)
            Path(tmp_name).write_text(json.dumps(pdata))
            participant_paths.append(tmp_name)
            continue
        found = False
        for candidate in sorted(Path(data_dir).glob("*.json")):
            with open(candidate) as f:
                if json.load(f).get("subject_id", "") == subject_id:
                    participant_paths.append(str(candidate))
                    found = True
                    break
        if not found:
            tmp_name = f"graphify-out/_tmp_{pid}.json"
            Path(tmp_name).parent.mkdir(parents=True, exist_ok=True)
            Path(tmp_name).write_text(json.dumps(pdata))
            participant_paths.append(tmp_name)

    results = []

    # ── 1. Logistic Regression ─────────────────────────────────────────
    print("\n  --- Logistic Regression ---")
    for i, pdata in enumerate(participants_data):
        try:
            m = evaluate_logistic_baseline(pdata)
            if m:
                results.append({
                    "Participant": labels[i], "Model": "Logistic Regression",
                    "Accuracy": m["accuracy"], "LogLikelihood": m["log_likelihood"],
                })
                print(f"    {labels[i]}: {m['accuracy'] * 100:.1f}%")
        except Exception as e:
            print(f"    {labels[i]}: FAILED ({e})")

    # ── 2. RNN ──────────────────────────────────────────────────────────
    print("\n  --- RNN (GRU) ---")
    for i, pdata in enumerate(participants_data):
        try:
            m = run_RNN_for_eval(pdata, num_epochs=num_epochs_rnn)
            if m:
                results.append({
                    "Participant": labels[i], "Model": "RNN (GRU)",
                    "Accuracy": m["accuracy"], "LogLikelihood": m["log_likelihood"],
                })
                print(f"    {labels[i]}: {m['accuracy'] * 100:.1f}%")
        except Exception as e:
            print(f"    {labels[i]}: FAILED ({e})")

    # ── 3. Feedforward (distances) ──────────────────────────────────────
    print("\n  --- Feedforward NN (distances) ---")
    for i, pdata in enumerate(participants_data):
        try:
            m = run_FF_for_eval(pdata, num_epochs=num_epochs_ff)
            if m:
                results.append({
                    "Participant": labels[i], "Model": "FF (distances)",
                    "Accuracy": m["accuracy"], "LogLikelihood": m["log_likelihood"],
                })
                print(f"    {labels[i]}: {m['accuracy'] * 100:.1f}%")
        except Exception as e:
            print(f"    {labels[i]}: FAILED ({e})")

    # ── 4. Feedforward (positions) ──────────────────────────────────────
    print("\n  --- Feedforward NN (positions) ---")
    for i, pdata in enumerate(participants_data):
        try:
            m = run_FF_position_for_eval(pdata, num_epochs=num_epochs_ff)
            if m:
                results.append({
                    "Participant": labels[i], "Model": "FF (positions)",
                    "Accuracy": m["accuracy"], "LogLikelihood": m["log_likelihood"],
                })
                print(f"    {labels[i]}: {m['accuracy'] * 100:.1f}%")
        except Exception as e:
            print(f"    {labels[i]}: FAILED ({e})")

    # ── 5. GLM-HMM ─────────────────────────────────────────────────────
    for n_states in (2, 3):
        model_name = f"GLM-HMM ({n_states}-state)"
        print(f"\n  --- {model_name} ---")
        for i, pdata in enumerate(participants_data):
            try:
                m = run_glmhmm_for_participant(pdata, num_states=n_states)
                if m:
                    results.append({
                        "Participant": labels[i], "Model": model_name,
                        "Accuracy": m["accuracy"], "LogLikelihood": m["log_likelihood"],
                    })
                    print(f"    {labels[i]}: {m['accuracy'] * 100:.1f}%  "
                          f"(LL {m['log_likelihood']:.3f}, switching {m['switching_rate']:.2f})")
                    if n_states == 2:
                        print_glmhmm_summary(m)
                else:
                    print(f"    {labels[i]}: SKIPPED (too few trials)")
            except Exception as e:
                print(f"    {labels[i]}: FAILED ({e})")

    # ── 6. CognitiveDeepONet ───────────────────────────────────────────
    print("\n  --- CognitiveDeepONet (all participants) ---")
    try:
        cog_acc, cog_per_p = run_cognitivedeeponet_all(participant_data_dict,
                                                       num_epochs=num_epochs_deeponet)
        for p_idx, p_metrics in cog_per_p.items():
            if p_idx < len(labels):
                results.append({
                    "Participant": labels[p_idx], "Model": "CognitiveDeepONet",
                    "Accuracy": p_metrics['accuracy'],
                    "LogLikelihood": p_metrics['log_likelihood'],
                })
                print(f"    {labels[p_idx]}: {p_metrics['accuracy'] * 100:.1f}%")
        print(f"    Overall test accuracy: {cog_acc * 100:.1f}%")
    except Exception as e:
        print(f"    FAILED ({e})")

    # ── 7. StrategyDeepONet — gated ─────────────────────────────────────
    print("\n  --- StrategyDeepONet (gated, all participants) ---")
    try:
        gated_acc, gated_per_p = run_strategy_variant_all(participant_paths, "gated",
                                                          num_epochs=num_epochs_deeponet)
        for p_idx, p_metrics in gated_per_p.items():
            if p_idx < len(labels):
                results.append({
                    "Participant": labels[p_idx], "Model": "Strategy-Gated",
                    "Accuracy": p_metrics['accuracy'],
                    "LogLikelihood": p_metrics['log_likelihood'],
                })
                print(f"    {labels[p_idx]}: {p_metrics['accuracy'] * 100:.1f}%")
        print(f"    Overall test accuracy: {gated_acc * 100:.1f}%")
    except Exception as e:
        print(f"    FAILED ({e})")

    # ── 8. StrategyDeepONet — multi-task ────────────────────────────────
    print("\n  --- StrategyDeepONet (multi-task, all participants) ---")
    try:
        mt_acc, mt_per_p = run_strategy_variant_all(participant_paths, "multitask",
                                                     num_epochs=num_epochs_deeponet)
        for p_idx, p_metrics in mt_per_p.items():
            if p_idx < len(labels):
                results.append({
                    "Participant": labels[p_idx], "Model": "Strategy-MultiTask",
                    "Accuracy": p_metrics['accuracy'],
                    "LogLikelihood": p_metrics['log_likelihood'],
                })
                print(f"    {labels[p_idx]}: {p_metrics['accuracy'] * 100:.1f}%")
        print(f"    Overall test accuracy: {mt_acc * 100:.1f}%")
    except Exception as e:
        print(f"    FAILED ({e})")

    # ── 9. StrategyDeepONet — time-binned ──────────────────────────────
    print("\n  --- StrategyDeepONet (time-binned, all participants) ---")
    try:
        tb_acc, tb_per_p = run_strategy_variant_all(participant_paths, "timebinned",
                                                     num_epochs=num_epochs_deeponet)
        for p_idx, p_metrics in tb_per_p.items():
            if p_idx < len(labels):
                results.append({
                    "Participant": labels[p_idx], "Model": "Strategy-TimeBinned",
                    "Accuracy": p_metrics['accuracy'],
                    "LogLikelihood": p_metrics['log_likelihood'],
                })
                print(f"    {labels[p_idx]}: {p_metrics['accuracy'] * 100:.1f}%")
        print(f"    Overall test accuracy: {tb_acc * 100:.1f}%")
    except Exception as e:
        print(f"    FAILED ({e})")

    # ── 10. Custom cognitive model ──────────────────────────────────────
    print("\n  --- Custom Cognitive Model ---")
    try:
        cog_acc, cog_df = run_custom_cognitive_model(participant_data_dict)
        if cog_df is not None and not cog_df.empty:
            for _, row in cog_df.iterrows():
                p_idx = int(row["Participant"].split("_")[1]) - 1
                if p_idx < len(labels):
                    results.append({
                        "Participant": labels[p_idx], "Model": "Custom Cognitive",
                        "Accuracy": row["Test_Accuracy"],
                        "LogLikelihood": row["Test_LL"],
                    })
            print(f"    Mean test accuracy: {cog_acc * 100:.1f}%")
    except Exception as e:
        print(f"    FAILED ({e})")

    # ── Build results DataFrame ────────────────────────────────────────
    df = pd.DataFrame(results)
    if not df.empty and plot:
        _plot_comparison(df)
        _plot_comparison_ll(df)
        _plot_participant_heatmap(df, "Accuracy")
        _plot_participant_heatmap(df, "LogLikelihood")

    return df


def _plot_comparison(df):
    """Box plot of per-model test accuracy across participants."""
    ranking = df.groupby("Model")["Accuracy"].mean().sort_values(ascending=False)
    models_ordered = ranking.index.tolist()

    fig, ax = plt.subplots(figsize=(14, 6), dpi=120)
    x = np.arange(len(models_ordered))

    bp = ax.boxplot(
        [df[df["Model"] == m]["Accuracy"].dropna().values for m in models_ordered],
        positions=x, widths=0.5, patch_artist=True,
    )

    colors = plt.cm.Set2(np.linspace(0, 1, len(models_ordered)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    rng = np.random.default_rng(42)
    for mi, m in enumerate(models_ordered):
        sub = df[df["Model"] == m]["Accuracy"].dropna()
        jitter = rng.uniform(-0.1, 0.1, len(sub))
        ax.scatter(np.full(len(sub), mi) + jitter, sub.values,
                   color="black", alpha=0.5, s=20, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(models_ordered, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Test Accuracy")
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4, label="Chance")
    ax.legend(fontsize=8)
    ax.set_title("Model Comparison — Accuracy", fontsize=13)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("model_comparison_accuracy.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\n  Plot saved: model_comparison_accuracy.png")


def _plot_comparison_ll(df):
    """Box plot of per-model test log-likelihood across participants."""
    ranking = df.groupby("Model")["LogLikelihood"].mean().sort_values(ascending=False)
    models_ordered = ranking.index.tolist()

    fig, ax = plt.subplots(figsize=(14, 6), dpi=120)
    x = np.arange(len(models_ordered))

    bp = ax.boxplot(
        [df[df["Model"] == m]["LogLikelihood"].dropna().values for m in models_ordered],
        positions=x, widths=0.5, patch_artist=True,
    )

    colors = plt.cm.Set2(np.linspace(0, 1, len(models_ordered)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    rng = np.random.default_rng(42)
    for mi, m in enumerate(models_ordered):
        sub = df[df["Model"] == m]["LogLikelihood"].dropna()
        jitter = rng.uniform(-0.1, 0.1, len(sub))
        ax.scatter(np.full(len(sub), mi) + jitter, sub.values,
                   color="black", alpha=0.5, s=20, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(models_ordered, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Test Log-Likelihood (per trial)")
    ax.set_title("Model Comparison — Log-Likelihood", fontsize=13)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("model_comparison_ll.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Plot saved: model_comparison_ll.png")


def _plot_participant_heatmap(df, metric="Accuracy"):
    """Participant x Model heatmap showing per-participant scores across all models."""
    pivot = df.pivot_table(
        index="Participant", columns="Model",
        values=metric, aggfunc="first",
    )
    if pivot.empty:
        return

    row_means = pivot.mean(axis=1).sort_values(ascending=False)
    col_means = pivot.mean(axis=0).sort_values(ascending=False)
    pivot = pivot.loc[row_means.index, col_means.index]

    fig_width = max(14, len(pivot.columns) * 1.2)
    fig_height = max(8, len(pivot.index) * 0.55)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=120)

    fmt = ".3f" if metric == "LogLikelihood" else ".1%"
    annot_matrix = pivot.copy()
    if metric == "Accuracy":
        annot_matrix = pivot * 100
    else:
        annot_matrix = pivot

    sns.heatmap(
        pivot, annot=annot_matrix.values, fmt=fmt,
        cmap="YlOrRd", center=None, cbar_kws={"label": metric},
        linewidths=0.5, linecolor="white", ax=ax,
        annot_kws={"fontsize": 8},
    )

    ax.set_title(f"Participant × Model — {metric}", fontsize=13, pad=12)
    ax.set_xlabel("Model")
    ax.set_ylabel("Participant")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    plt.tight_layout()
    fname = f"model_comparison_heatmap_{metric.lower()}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Plot saved: {fname}")


# %% [markdown]
# ## Run and display results

# %%


results_df = compare_all_models()

if not results_df.empty:
    # ── Accuracy pivot table ───────────────────────────────────────────
    pivot_acc = results_df.pivot_table(
        index="Participant", columns="Model",
        values="Accuracy", aggfunc="first",
    )
    print("\n--- Accuracy (%) ---")
    print((pivot_acc * 100).round(1).to_string())

    # ── Log-likelihood pivot table ─────────────────────────────────────
    pivot_ll = results_df.pivot_table(
        index="Participant", columns="Model",
        values="LogLikelihood", aggfunc="first",
    )
    print("\n--- Log-Likelihood ---")
    print(pivot_ll.round(4).to_string())

    # ── Mean accuracy ranking ──────────────────────────────────────────
    ranking = results_df.groupby("Model")["Accuracy"].mean().sort_values(ascending=False)
    print("\n--- Mean Accuracy Ranking ---")
    for rank, (model, acc) in enumerate(ranking.items(), 1):
        print(f"  {rank}. {model}: {acc * 100:.1f}%")

#%%