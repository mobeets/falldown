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
# # DeepONet Transfer-Scaling Study
#
# Does the shared DeepONet basis trained on more participants transfer better to a
# **completely new** participant? For a held-out participant we freeze the basis and
# only fit that participant's coefficients (embedding) on their own trials, then
# evaluate on their held-out trials.
#
# Two model types are supported through the same loop:
#
# - `cognitive`: `CognitiveDeepONet`. The held-out fit is a **logistic regression on
#   the frozen basis features** — exact MLE of the embedding since the model is linear
#   in the coefficients.
# - `strategy`: HMM-gated `StrategyDeepONet` (shared basis). With the basis frozen the
#   model reduces to a **GLM-HMM whose emission inputs are the basis features**; the
#   held-out fit runs `ssm` EM on the new participant's trials (K strategies, Markov
#   transitions).
#
# A **few-shot** dimension varies how many of the held-out participant's train trials
# are used to fit the embedding.
#
# Draws on: Lu et al 2021 (DeepONet transfer of trunk/basis functions), Ashwood et al
# 2022 (GLM-HMM), Kirsch 2019 (per-participant strategy coefficients).

# %%
import io
import os
import sys
import glob
import json
import warnings
import contextlib
from math import comb

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
import ssm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from cognitivedeepOnet import CognitiveDeepONet, MazeDataset as CogMazeDataset, train_deeponet
from strategy_deeponet import (
    pre_proccess_data_from_choice_vs_no_choice,
    _build_participant_trials,
    StrategyDeepONet,
    SequenceDataset,
    train_strategy_deeponet,
)

warnings.filterwarnings("ignore", category=UserWarning)


# %% [markdown]
# ## Data preparation

# %%
def load_participants(data_dir):
    """Load every participant JSON, preprocess, and return per-participant records.

    Each record is (name, raw_data, trials) where trials = (features (n, 5),
    choices (n,), rt (n,)) in trial order.
    """
    parts = []
    for f in sorted(glob.glob(os.path.join(data_dir, "*.json"))):
        d = json.load(open(f))
        processed = pre_proccess_data_from_choice_vs_no_choice(d)
        trials = _build_participant_trials(processed)
        if trials is not None:
            parts.append((os.path.basename(f), d, trials))
    return parts


def temporal_split(features, choices, test_frac=0.2):
    """Per-participant chronological split: first (1-test_frac) trials for train."""
    n = len(features)
    split = int(n * (1 - test_frac))
    split = max(min(split, n - 2), 2)
    return (features[:split], choices[:split]), (features[split:], choices[split:])


def fit_pool_scaler(pool_train_features):
    """z-score statistics for the 3 continuous features, fit on pooled pool train."""
    X = np.vstack(pool_train_features)
    mu = X[:, :3].mean(axis=0)
    std = X[:, :3].std(axis=0) + 1e-8
    return mu, std


def scale_features(features, mu, std):
    f = features.copy()
    f[:, :3] = (f[:, :3] - mu) / std
    return f


# %% [markdown]
# ## Pool basis training

# %%
def train_pool_basis(model_type, pool_train, num_participants, num_epochs, seed,
                     num_strategies=3, lr=0.0015):
    """Train the shared basis on the pool's (already scaled) train trials.

    pool_train: list of (features (n,5), choices (n,)) in participant order (ids
    assigned by list position). Returns the trained model with a frozen basis_net.
    """
    torch.manual_seed(seed)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        if model_type == "cognitive":
            X = np.vstack([t[0] for t in pool_train])
            y = np.concatenate([t[1] for t in pool_train])
            ids = np.concatenate([np.full(len(t[0]), i) for i, t in enumerate(pool_train)])
            ds = CogMazeDataset(X, ids, y)
            dl = DataLoader(ds, batch_size=64, shuffle=True)
            model = CognitiveDeepONet(num_participants=num_participants,
                                      num_features=5, num_bases=4)
            train_deeponet(model, dl, num_epochs=num_epochs, lr=lr, penalty_weight=0.5)
        elif model_type == "strategy":
            seqs = [(t[0], i, t[1], None, None) for i, t in enumerate(pool_train)]
            ds = SequenceDataset(seqs, with_rt=False, time_binned=False)
            dl = DataLoader(ds, batch_size=None, shuffle=True)
            model = StrategyDeepONet(num_participants=num_participants, num_features=5,
                                     num_bases=4, num_strategies=num_strategies,
                                     shared_bases=True)
            train_strategy_deeponet(model, dl, num_epochs=num_epochs, lr=0.001,
                                    penalty_weight=0.5)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    # Freeze the basis (the only part that transfers)
    for p in model.basis_net.parameters():
        p.requires_grad_(False)
    model.eval()
    return model


def basis_features(model, X):
    """Frozen-basis outputs for a (n, 5) feature matrix -> (n, D)."""
    with torch.no_grad():
        return model.basis_net(torch.tensor(X, dtype=torch.float32)).numpy()


# %% [markdown]
# ## Held-out participant fitting (frozen basis)

# %%
def fit_heldout_logistic(model, train_feats, train_choices, test_feats, test_choices):
    """num_states = 1: logistic regression on the frozen basis features.

    This is the exact MLE of the participant's DeepONet embedding (the model is
    linear in the coefficients, and the DeepONet logit has no bias term, so we fit
    without an intercept).
    """
    B_train = basis_features(model, train_feats)
    B_test = basis_features(model, test_feats)
    clf = LogisticRegression(penalty="l2", C=10.0, fit_intercept=False,
                             solver="lbfgs", max_iter=1000)
    clf.fit(B_train, train_choices)
    p = clf.predict_proba(B_test)[:, 1]
    acc = np.mean((p >= 0.5).astype(int) == test_choices)
    eps = 1e-12
    ll = np.mean(test_choices * np.log(p + eps) + (1 - test_choices) * np.log(1 - p + eps))
    return acc, ll


def fit_heldout_glmhmm(model, train_feats, train_choices, test_feats, test_choices,
                       num_states=3, num_iters=200):
    """num_states = K: GLM-HMM on the frozen basis features.

    With the basis frozen, the HMM-gated StrategyDeepONet reduces to a GLM-HMM whose
    emission inputs are the basis outputs. Fit the new participant's per-state
    weights + Markov transitions with ssm EM; predict via the filtered posterior.
    """
    B_train = basis_features(model, train_feats)
    B_test = basis_features(model, test_feats)

    # bias column mirrors the GLM intercept (basis carries no constant feature)
    train_inpts = np.column_stack([B_train, np.ones(len(B_train))])
    test_inpts = np.column_stack([B_test, np.ones(len(B_test))])
    y_train = train_choices.astype(int).reshape(-1, 1)
    y_test = test_choices.astype(int).reshape(-1, 1)

    glmhmm = ssm.HMM(num_states, 1, train_inpts.shape[1],
                     observations="input_driven_obs",
                     observation_kwargs=dict(C=2),
                     transitions="standard")
    with contextlib.redirect_stderr(io.StringIO()):  # silence tqdm progress bars
        glmhmm.fit([y_train], inputs=train_inpts, method="em",
                   num_iters=num_iters, tolerance=10**-4)

    log_probs = glmhmm.log_likelihood([y_test], inputs=test_inpts)
    ll = float(log_probs) / len(y_test)

    posterior = glmhmm.filter(y_test, input=test_inpts)  # (T, K)
    W = glmhmm.observations.params  # (K, C-1=1, M)
    preds = []
    for t in range(len(y_test)):
        class_logits = np.zeros((num_states, 2))
        class_logits[:, :1] = W @ test_inpts[t]  # class 1 is the softmax baseline
        class_logits -= class_logits.max(axis=-1, keepdims=True)
        class_probs = np.exp(class_logits)
        class_probs /= class_probs.sum(axis=-1, keepdims=True)
        marginal = posterior[t] @ class_probs
        preds.append(int(np.argmax(marginal)))
    acc = np.mean(preds == y_test[:, 0])

    return acc, ll


def fit_heldout(model, model_type, num_states, train_feats, train_choices,
                test_feats, test_choices, num_iters=200):
    if model_type == "cognitive" or num_states == 1:
        return fit_heldout_logistic(model, train_feats, train_choices,
                                    test_feats, test_choices)
    return fit_heldout_glmhmm(model, train_feats, train_choices, test_feats,
                              test_choices, num_states=num_states, num_iters=num_iters)


# %% [markdown]
# ## Reference baselines (no transfer)

# %%
def logistic_baseline_accuracy(raw_data):
    """Within-participant logistic regression on raw features (no transfer)."""
    from exploratory_data_analysis import evaluate_logistic_baseline
    m = evaluate_logistic_baseline(raw_data)
    return m["accuracy"] if m else np.nan


def scratch_deeponet_accuracy(features, choices, num_epochs=100, seed=0):
    """CognitiveDeepONet trained on the held-out participant alone (no transfer)."""
    mu, std = fit_pool_scaler([features])
    X = scale_features(features, mu, std)
    torch.manual_seed(seed)
    ds = CogMazeDataset(X, np.zeros(len(choices)), choices)
    dl = DataLoader(ds, batch_size=64, shuffle=True)
    model = CognitiveDeepONet(num_participants=1, num_features=5, num_bases=4)
    with contextlib.redirect_stdout(io.StringIO()):
        train_deeponet(model, dl, num_epochs=num_epochs, lr=0.0015, penalty_weight=0.5)
    with torch.no_grad():
        logits, _ = model(torch.tensor(X, dtype=torch.float32),
                          torch.zeros(len(choices), dtype=torch.long))
    p = torch.sigmoid(logits).numpy()
    return np.mean((p >= 0.5).astype(int) == choices)


def _build_exploratory_feature_frame(raw_data):
    """Return the exploratory feature frame used by `evaluate_logistic_baseline`.

    Mirrors the frame construction in exploratory_data_analysis.py
    (evaluate_logistic_baseline): diff_1step, diff_planning, Block Drift,
    Incoming Direction. The optional drift interaction is added at fit time
    when drift varies. The frame preserves the trial order produced by
    `pre_proccess_data_from_choice_vs_no_choice`.
    """
    processed = pre_proccess_data_from_choice_vs_no_choice(raw_data)
    df_raw = pd.DataFrame(processed) if isinstance(processed, list) else processed

    is_left = df_raw['chosen_left'].astype(bool)
    L1 = np.where(is_left, df_raw['chosen_1step_dist'], df_raw['unchosen_1step_dist'])
    R1 = np.where(~is_left, df_raw['chosen_1step_dist'], df_raw['unchosen_1step_dist'])

    chosen_2step_diff = df_raw['chosen_2step_dist'] - df_raw['chosen_1step_dist']
    unchosen_2step_diff = df_raw['unchosen_2step_dist'] - df_raw['unchosen_1step_dist']
    L2 = np.where(is_left, chosen_2step_diff, unchosen_2step_diff)
    R2 = np.where(~is_left, chosen_2step_diff, unchosen_2step_diff)

    X = pd.DataFrame({
        'diff_1step': L1 - R1,
        'diff_planning': L1 + L2 - R2 - R1,
        'Block Drift': df_raw['block_drift'],
        'block_number': df_raw['block_number'],
        'chosen_left': df_raw['chosen_left'].astype(int),
        'Incoming Direction': df_raw['incoming_direction']
    })
    X = X.dropna(subset=['diff_1step', 'diff_planning', 'Block Drift',
                         'Incoming Direction', 'chosen_left']).copy()
    return X


def matched_logistic_accuracy(raw_data, test_frac=0.2, fit_fracs=(1.0, 0.5, 0.25)):
    """Within-participant logistic baseline trained on the SAME amount of data
    the DeepONet embedding fit receives at each fit_frac.

    Uses the exploratory feature set (same as the full-data `logistic (no
    transfer)` reference) and the same temporal train/test split as the scaling
    study, so the test set is identical to the DeepONet's. For each fit_frac it
    fits a plain logistic regression on the first `n_fit = int(n_train * ff)`
    train rows and evaluates accuracy on the shared test set.

    Returns {fit_frac: accuracy} (NaN if a fit is not possible).
    """
    X = _build_exploratory_feature_frame(raw_data)
    if X.empty:
        return {ff: np.nan for ff in fit_fracs}

    # chronological split by trial order (same convention as temporal_split)
    n = len(X)
    split = int(n * (1 - test_frac))
    split = max(min(split, n - 2), 2)
    X_train = X.iloc[:split].reset_index(drop=True)
    X_test = X.iloc[split:].reset_index(drop=True)
    if len(X_test) == 0:
        return {ff: np.nan for ff in fit_fracs}

    out = {}
    for ff in fit_fracs:
        n_fit = max(2, int(len(X_train) * ff))
        n_fit = min(n_fit, len(X_train))
        tr = X_train.iloc[:n_fit]

        # standardize the two distance features on this train subset only
        mu_1, sig_1 = tr['diff_1step'].mean(), tr['diff_1step'].std()
        mu_2, sig_2 = tr['diff_planning'].mean(), tr['diff_planning'].std()
        sig_1 = sig_1 if sig_1 != 0 else 1e-6
        sig_2 = sig_2 if sig_2 != 0 else 1e-6

        tr = tr.copy()
        tr['L1-R1'] = (tr['diff_1step'] - mu_1) / sig_1
        tr['L1+L2-R1-R2'] = (tr['diff_planning'] - mu_2) / sig_2
        te = X_test.copy()
        te['L1-R1'] = (te['diff_1step'] - mu_1) / sig_1
        te['L1+L2-R1-R2'] = (te['diff_planning'] - mu_2) / sig_2

        has_drift_variance = tr['Block Drift'].nunique() > 1
        if has_drift_variance:
            tr['Block Drift + Incoming Direction Interaction'] = (
                tr['Incoming Direction'] * tr['Block Drift'])
            te['Block Drift + Incoming Direction Interaction'] = (
                te['Incoming Direction'] * te['Block Drift'])
            features = ['L1-R1', 'L1+L2-R1-R2', 'Incoming Direction',
                        'Block Drift + Incoming Direction Interaction']
        else:
            features = ['L1-R1', 'L1+L2-R1-R2', 'Incoming Direction']

        if len(tr['chosen_left'].unique()) < 2:
            out[ff] = np.nan
            continue

        clf = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
        clf.fit(tr[features], tr['chosen_left'])
        y_pred = clf.predict(te[features])
        out[ff] = np.mean(y_pred == te['chosen_left'])

    return out


# %% [markdown]
# ## Main scaling study

# %%
def run_scaling_study(model_type="cognitive", data_dir="cloud study data",
                      held_out_ids=None, pool_sizes=None,
                      subsets_per_size=3, seeds=(0, 1),
                      num_epochs=100, fit_fracs=(1.0, 0.5, 0.25),
                      num_states=1, num_strategies=3, test_frac=0.2,
                      plot=True, out_prefix="scaling", scratch_epochs=100):
    """Run the leave-out / pool-size scaling study.

    Returns (results, summary_df, pool_sizes, held_out_names).
    results[(h_name, N, fit_frac)] = list of held-out test accuracies across
    subsets x seeds.
    """
    print("=" * 70)
    print(f"  SCALING STUDY — model={model_type}, held-out fit num_states={num_states}")
    print("=" * 70)

    parts = load_participants(data_dir)
    if not parts:
        raise FileNotFoundError(f"No participant JSON files in {data_dir}")

    # per-participant temporal split (consistent across every cohort)
    records = []
    for name, raw, (feats, choices, _rt) in parts:
        (tr_f, tr_c), (te_f, te_c) = temporal_split(feats, choices, test_frac=test_frac)
        records.append({"name": name, "raw": raw, "train": (tr_f, tr_c), "test": (te_f, te_c)})

    # --- held-out set (default: the 3 participants with the most trials) ---
    if held_out_ids is None:
        held_out_ids = [r["name"] for r in
                        sorted(records, key=lambda r: -len(r["train"][0]))[:3]]
    by_name = {r["name"]: r for r in records}
    held_out = [by_name[n] for n in held_out_ids]
    pool = [r for r in records if r["name"] not in held_out_ids]
    held_out_names = [r["name"] for r in held_out]
    print(f"  Held-out participants ({len(held_out)}):")
    for r in held_out:
        print(f"    {r['name'][:30]:32s} train={len(r['train'][0])} test={len(r['test'][0])}")
    print(f"  Pool participants: {len(pool)}")

    if pool_sizes is None:
        pool_sizes = sorted({1, 2, 4, 6, 8, len(pool)})
    pool_sizes = sorted({min(N, len(pool)) for N in pool_sizes})

    # --- no-transfer baselines for the held-out participants ---
    baselines = {}
    for h in held_out:
        logistic = logistic_baseline_accuracy(h["raw"])
        scratch = (scratch_deeponet_accuracy(*h["train"], num_epochs=scratch_epochs)
                   if model_type == "cognitive" else np.nan)
        matched = matched_logistic_accuracy(h["raw"], test_frac=test_frac,
                                            fit_fracs=fit_fracs)
        baselines[h["name"]] = {"logistic": logistic, "scratch": scratch,
                                "matched": matched}
        print(f"  Baseline {h['name'][:30]}: logistic={logistic:.3f} "
              f"scratch_deeponet={scratch:.3f}")
        print(f"    matched logistic per fit_frac: " +
              ", ".join(f"{ff}={matched.get(ff, float('nan')):.3f}"
                        for ff in fit_fracs))

    # --- main loop ---
    results = {}
    n_total = len(pool_sizes) * subsets_per_size * len(seeds)
    done = 0
    for N in pool_sizes:
        n_subsets = min(subsets_per_size,
                        comb(len(pool), N) if N <= len(pool) else 1)
        for rep in range(n_subsets):
            for seed in seeds:
                done += 1
                rng = np.random.default_rng(1000 * N + 7 * rep + 13 * seed)
                subset_idx = rng.choice(len(pool), size=N, replace=False)
                subset = [pool[i] for i in subset_idx]

                # scale with pool-train statistics (the pool basis was trained on these)
                mu, std = fit_pool_scaler([s["train"][0] for s in subset])
                pool_train = [(scale_features(s["train"][0], mu, std), s["train"][1])
                              for s in subset]

                model = train_pool_basis(model_type, pool_train, num_participants=N,
                                         num_epochs=num_epochs, seed=seed,
                                         num_strategies=num_strategies)

                for h in held_out:
                    h_train_f, h_train_c = h["train"]
                    h_test_f, h_test_c = h["test"]
                    h_train_f = scale_features(h_train_f, mu, std)
                    h_test_f = scale_features(h_test_f, mu, std)
                    n_fit_max = len(h_train_f)
                    for ff in fit_fracs:
                        n_fit = max(2, int(n_fit_max * ff))
                        acc, _ll = fit_heldout(
                            model, model_type, num_states,
                            h_train_f[:n_fit], h_train_c[:n_fit],
                            h_test_f, h_test_c)
                        results.setdefault((h["name"], N, ff), []).append(acc)

                if done % 10 == 0 or done == n_total:
                    print(f"  [{done}/{n_total}] N={N} rep={rep + 1}/{n_subsets} seed={seed}")

    # --- aggregate summary ---
    rows = []
    for h_name in held_out_names:
        for N in pool_sizes:
            for ff in fit_fracs:
                accs = results.get((h_name, N, ff), [])
                rows.append({"participant": h_name, "N": N, "fit_frac": ff,
                             "acc_mean": np.mean(accs) if accs else np.nan,
                             "acc_se": (np.std(accs) / np.sqrt(len(accs))) if accs else np.nan,
                             "n_repeats": len(accs),
                             "logistic_matched": baselines.get(h_name, {}).get("matched", {}).get(ff, np.nan)})
    summary = pd.DataFrame(rows)

    if plot:
        plot_scaling(results, fit_fracs, pool_sizes, held_out_names, baselines,
                     model_type=model_type, out_prefix=out_prefix)

    return results, summary, pool_sizes, held_out_names, baselines


def plot_scaling(results, fit_fracs, pool_sizes, held_out_names, baselines,
                 model_type, out_prefix="scaling"):
    ncol = len(fit_fracs)
    fig, axes = plt.subplots(1, ncol, figsize=(5.2 * ncol, 4.5), dpi=150,
                             sharey=True)
    if ncol == 1:
        axes = [axes]

    for ax, ff in zip(axes, fit_fracs):
        agg = {}
        for N in pool_sizes:
            agg[N] = [np.mean(results[(h, N, ff)]) for h in held_out_names]

        for h in held_out_names:
            xs = list(pool_sizes)
            ys = [np.mean(results[(h, N, ff)]) for N in pool_sizes]
            ax.plot(xs, ys, "o-", alpha=0.35, lw=1.2, ms=3)

        m = [np.mean(agg[N]) for N in pool_sizes]
        se = [np.std(agg[N]) / np.sqrt(len(agg[N])) for N in pool_sizes]
        ax.errorbar(pool_sizes, m, yerr=se, marker="o", color="k", lw=2,
                    capsize=3, label="mean \u00b1 SE")

        ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.7, label="chance")
        log_vals = [baselines[h]["logistic"] for h in held_out_names
                    if not np.isnan(baselines[h]["logistic"])]
        if log_vals:
            ax.axhline(np.mean(log_vals), color="tomato", ls=":", lw=1.5,
                       label=f"logistic (no transfer)={np.mean(log_vals):.2f}")
        matched_vals = [baselines[h]["matched"].get(ff, np.nan)
                        for h in held_out_names]
        matched_vals = [v for v in matched_vals if not np.isnan(v)]
        if matched_vals:
            ax.axhline(np.mean(matched_vals), color="darkorange", ls="--", lw=1.5,
                       label=f"logistic (matched n_fit)={np.mean(matched_vals):.2f}")
        scr_vals = [baselines[h]["scratch"] for h in held_out_names
                    if not np.isnan(baselines[h]["scratch"])]
        if scr_vals:
            ax.axhline(np.mean(scr_vals), color="steelblue", ls=":", lw=1.5,
                       label=f"scratch deeponet={np.mean(scr_vals):.2f}")

        ax.set_title(f"fit_frac = {ff} (held-out train data)")
        ax.set_xlabel("N participants (basis pool)")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Held-out test accuracy")
    fig.suptitle(f"Basis transfer scaling — {model_type}", fontsize=13)
    fig.tight_layout()
    fname = f"{out_prefix}_accuracy_{model_type}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n  Plot saved: {fname}")

    # print the summary table
    print("\n--- Mean held-out accuracy by N and fit_frac ---")
    for ff in fit_fracs:
        sub = [np.mean([np.mean(results[(h, N, ff)]) for h in held_out_names])
               for N in pool_sizes]
        print(f"  fit_frac={ff}: " + ", ".join(f"N{N}={v:.3f}" for N, v in zip(pool_sizes, sub)))
        mvals = [baselines[h]["matched"].get(ff, np.nan) for h in held_out_names]
        mvals = [v for v in mvals if not np.isnan(v)]
        if mvals:
            print(f"    matched logistic at fit_frac={ff}: {np.mean(mvals):.3f}")


# %% [markdown]
# ## Run

# %%
if __name__ == "__main__":
    # CognitiveDeepONet — full study
    results, summary, pool_sizes, held_out_names, baselines = run_scaling_study(
        model_type="cognitive",
        data_dir=os.path.join(_SCRIPT_DIR, "cloud study data"),
        subsets_per_size=3,
        seeds=(0, 1),
        num_epochs=100,
        fit_fracs=(1.0, 0.25, 0.1),
    )
    summary.to_csv("scaling_cognitive_summary.csv", index=False)

    # HMM-gated StrategyDeepONet — same loop, held-out fit is a GLM-HMM on basis features
    results_s, summary_s, _, _, _ = run_scaling_study(
         model_type="strategy",
         data_dir=os.path.join(_SCRIPT_DIR, "cloud study data"),
         num_states=3,
         subsets_per_size=2,
         seeds=(0,),
         num_epochs=100,
         fit_fracs=(1.0, 0.25, 0.1),
     )
    summary_s.to_csv("scaling_strategy_summary.csv", index=False)
#%%
