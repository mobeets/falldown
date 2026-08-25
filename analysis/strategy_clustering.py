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
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# %% [markdown]
# # Strategy Clustering
#
# Post-hoc analysis that takes a trained StrategyDeepONet (or the original
# CognitiveDeepONet) and discovers participant strategy types via:
# 1. PCA on participant coefficient vectors
# 2. k-means clustering on the reduced space
# 3. Behavioral characterization of each cluster
# 4. Statistical validation
#
# Draws on:
# - Jach et al 2024: individual differences have low-dimensional structure (PCA)
# - Ashwood et al 2022: discrete strategy types detected via GLM-HMM
# - Kirsch 2019: strategies as points in constraint space

# %%
def extract_strategy_features(model, participant_data_dict,
                               preprocess_fn, build_dataset_fn,
                               feature_names=None):
    """
    Extract coefficient vectors and trial-level strategy weights for each
    participant from a trained StrategyDeepONet or CognitiveDeepONet.

    Works with both model types:
    - CognitiveDeepONet: yields num_bases coefficients per participant
    - StrategyDeepONet: yields num_strategies × num_bases coefficients per
      participant, plus trial-level gate weights

    Returns:
        coeffs_df: DataFrame with participant index + coefficients
        strategy_weights_df: per-trial strategy weights (None for basic model)
        feature_names: list of feature names used
    """
    coeffs_raw = model.participant_coeffs.weight.detach().cpu().numpy()
    num_participants = coeffs_raw.shape[0]
    coeff_dim = coeffs_raw.shape[1]

    has_strategies = hasattr(model, 'num_strategies')
    num_strategies = model.num_strategies if has_strategies else 1
    num_bases = coeff_dim // num_strategies

    if feature_names is None:
        feature_names = [f"Feature_{i+1}" for i in range(num_bases)]

    rows = []
    for p in range(num_participants):
        coeffs_matrix = coeffs_raw[p].reshape(num_strategies, num_bases)
        row = {'participant_id': p}
        for s in range(num_strategies):
            for b in range(num_bases):
                row[f'S{s+1}_B{b+1}'] = coeffs_matrix[s, b]
        rows.append(row)

    coeffs_df = pd.DataFrame(rows)

    return coeffs_df, num_strategies, num_bases, feature_names


# %%
def run_pca(coeffs_df, n_components=None, plot=True):
    """
    PCA decomposition of participant strategy coefficients.
    Tests Jach et al's claim that individual differences in decision-making
    have a low-dimensional structure.
    """
    feature_cols = [c for c in coeffs_df.columns if c != 'participant_id']
    X = coeffs_df[feature_cols].values

    if X.shape[0] < 3:
        print("Need at least 3 participants for PCA.")
        return None, None, X

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    max_components = min(X.shape[0] - 1, X.shape[1])
    if n_components is None:
        n_components = min(max_components, 5)

    pca = PCA(n_components=min(n_components, max_components))
    X_pca = pca.fit_transform(X_scaled)

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=120)

        axes[0].plot(range(1, len(pca.explained_variance_ratio_) + 1),
                     np.cumsum(pca.explained_variance_ratio_), 'o-', linewidth=2,
                     color='steelblue', markersize=8)
        axes[0].axhline(y=0.9, linestyle='--', color='gray', alpha=0.5,
                        label='90% variance')
        axes[0].set_xlabel('Number of Components')
        axes[0].set_ylabel('Cumulative Explained Variance')
        axes[0].set_title('PCA: Dimensionality of Strategy Space')
        axes[0].legend()

        colors = plt.cm.tab10(np.linspace(0, 1, len(feature_cols)))
        loadings = pca.components_[:2].T
        for i, name in enumerate(feature_cols):
            axes[1].arrow(0, 0, loadings[i, 0]*3, loadings[i, 1]*3,
                          color=colors[i], alpha=0.7, width=0.02,
                          head_width=0.08)
            axes[1].text(loadings[i, 0]*3.2, loadings[i, 1]*3.2, name,
                         fontsize=7, ha='center', va='center')
        axes[1].set_xlim(-4, 4)
        axes[1].set_ylim(-4, 4)
        axes[1].axhline(0, color='gray', linestyle='--', alpha=0.3)
        axes[1].axvline(0, color='gray', linestyle='--', alpha=0.3)
        axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        axes[1].set_title('PCA Loadings (PC1 vs PC2)')
        plt.tight_layout()
        plt.show()

        print(f"Top components explain: {np.cumsum(pca.explained_variance_ratio_)[:3]}")
        if pca.explained_variance_ratio_[0] < 0.3:
            print("Note: low first-PC variance suggests heterogeneous strategies.")
        elif pca.explained_variance_ratio_[0] > 0.7:
            print("Note: high first-PC variance suggests a dominant strategy axis.")

    return pca, X_pca, feature_cols


# %%
def cluster_participants(X_pca, max_k=None, plot=True):
    """
    k-means clustering in PCA space. Uses silhouette score to select optimal k.
    Tests Ashwood's claim that participants fall into discrete strategy types.
    """
    n = X_pca.shape[0]
    if n < 4:
        print(f"Need at least 4 participants for clustering (have {n}).")
        return None, None, None

    if max_k is None:
        max_k = min(8, n - 1)

    k_values = range(2, max_k + 1)
    silhouette_scores = []
    inertias = []

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_pca)
        silhouette_scores.append(silhouette_score(X_pca, labels))
        inertias.append(km.inertia_)

    best_k = k_values[np.argmax(silhouette_scores)]
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_pca)

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=120)

        axes[0].plot(list(k_values), silhouette_scores, 'o-', linewidth=2,
                     color='darkorange', markersize=8)
        axes[0].axvline(x=best_k, linestyle='--', color='red', alpha=0.7,
                        label=f'Best k={best_k}')
        axes[0].set_xlabel('Number of Clusters (k)')
        axes[0].set_ylabel('Silhouette Score')
        axes[0].set_title('Silhouette Analysis for Optimal k')
        axes[0].legend()

        colors = plt.cm.Set2(np.linspace(0, 1, best_k))
        for c in range(best_k):
            mask = cluster_labels == c
            axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1], s=120,
                            color=colors[c], edgecolors='black', linewidth=0.5,
                            label=f'Cluster {c+1} ({mask.sum()})')
        axes[1].set_xlabel('PC1')
        axes[1].set_ylabel('PC2')
        axes[1].set_title(f'Participant Strategy Clusters (k={best_k})')
        axes[1].legend()
        plt.tight_layout()
        plt.show()

    print(f"Optimal k: {best_k} (silhouette: {max(silhouette_scores):.3f})")
    for c in range(best_k):
        count = (cluster_labels == c).sum()
        print(f"  Cluster {c+1}: {count} participants")

    return kmeans, cluster_labels, best_k


# %%
def characterize_clusters(model, coeffs_df, cluster_labels, num_strategies,
                           num_bases, feature_names=None):
    """
    For each cluster, compute the mean strategy profile and describe
    what makes that cluster's participants distinct.
    """
    if cluster_labels is None:
        print("No cluster labels to characterize.")
        return None

    feature_cols = [c for c in coeffs_df.columns if c != 'participant_id']
    coeffs_df = coeffs_df.copy()
    coeffs_df['cluster'] = cluster_labels

    profiles = {}
    for c in sorted(np.unique(cluster_labels)):
        members = coeffs_df[coeffs_df['cluster'] == c][feature_cols].mean()
        profiles[c] = members

        print(f"\n--- Cluster {c+1} ({len(coeffs_df[coeffs_df['cluster']==c])} participants) ---")

        if num_strategies > 1:
            matrix = members.values.reshape(num_strategies, num_bases)
            for s in range(num_strategies):
                dominant_basis = np.argmax(np.abs(matrix[s]))
                sign = "positive" if matrix[s, dominant_basis] > 0 else "negative"
                print(f"  Strategy {s+1}: strongest on Basis {dominant_basis+1} "
                      f"({sign}, weight={matrix[s, dominant_basis]:.3f})")
        else:
            sorted_idx = np.argsort(-np.abs(members.values))
            top_3 = sorted_idx[:3]
            for rank, idx in enumerate(top_3):
                label = feature_names[idx] if feature_names else f"Basis_{idx+1}"
                print(f"  Top {rank+1}: {label} (weight={members.values[idx]:.3f})")

    return profiles


# %%
def plot_cluster_heatmap(coeffs_df, cluster_labels, num_strategies):
    """
    Heatmap of all participants' coefficients, sorted by cluster assignment.
    """
    feature_cols = [c for c in coeffs_df.columns if c != 'participant_id']
    df_sorted = coeffs_df.copy()
    df_sorted['cluster'] = cluster_labels
    df_sorted = df_sorted.sort_values('cluster')
    labels = df_sorted['cluster'].values
    matrix = df_sorted[feature_cols].values

    plt.figure(figsize=(14, max(6, matrix.shape[0] * 0.4)), dpi=120)
    ax = sns.heatmap(matrix, cmap="coolwarm", center=0, annot=False,
                     cbar_kws={'label': 'Coefficient Value'},
                     yticklabels=False)
    ax.set_xlabel("Strategy × Basis Feature")
    ax.set_ylabel("Participant (sorted by cluster)")
    ax.set_title("Participant Strategy Profiles by Cluster")

    prev = -1
    for i, c in enumerate(labels):
        if c != prev:
            ax.axhline(y=i, color='black', linewidth=1.5)
            ax.text(-1.5, i + (labels == c).sum() / 2,
                    f"C{c+1}", fontsize=9, ha='right', va='center')
            prev = c

    plt.tight_layout()
    plt.show()


# %%
def plot_behavioural_comparison(cluster_labels, participant_data_dict,
                                 preprocess_fn):
    """
    Compare behavioral metrics across clusters: RT, accuracy proxy,
    strategy purity (from gate output), and consistency.
    """
    if cluster_labels is None:
        return None

    metrics = []
    for p_idx, (p_name, raw_data) in enumerate(participant_data_dict.items()):
        try:
            processed = preprocess_fn(raw_data)
            choice_trials = processed[processed['choice_trial'] == True]
            if len(choice_trials) < 5:
                continue

            rt_mean = choice_trials['observed_rt'].mean()
            rt_std = choice_trials['observed_rt'].std()
            n_trials = len(choice_trials)
            cluster = cluster_labels[p_idx] if p_idx < len(cluster_labels) else -1

            metrics.append({
                'participant': p_idx,
                'cluster': cluster,
                'mean_rt': rt_mean,
                'rt_variability': rt_std,
                'n_trials': n_trials,
            })
        except Exception:
            continue

    df = pd.DataFrame(metrics)
    if len(df) < 2:
        return df

    clusters_present = sorted(df['cluster'].unique())
    n_clusters = len(clusters_present)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=120)

    for ax, col, title in [(axes[0], 'mean_rt', 'Mean RT by Strategy Cluster'),
                            (axes[1], 'rt_variability', 'RT Variability by Strategy Cluster')]:
        cluster_data = [df[df['cluster'] == c][col].values
                        for c in clusters_present]
        bp = ax.boxplot(cluster_data, labels=[f"C{c+1}" for c in clusters_present],
                         patch_artist=True)
        for patch, c in zip(bp['boxes'], clusters_present):
            patch.set_facecolor(plt.cm.Set2(c / max(1, n_clusters - 1)))
        ax.set_title(title)
        ax.set_xlabel('Cluster')

    plt.tight_layout()
    plt.show()

    return df


# %%
def full_clustering_pipeline(model, participant_data_dict,
                              preprocess_fn, build_dataset_fn,
                              feature_names=None):
    """
    End-to-end strategy clustering pipeline:
    1. Extract coefficients from trained model
    2. PCA decomposition
    3. k-means clustering
    4. Characterize clusters
    5. Behavioral comparison

    Args:
        model: trained StrategyDeepONet or CognitiveDeepONet
        participant_data_dict: {participant_name: raw_json_data}
        preprocess_fn: function to preprocess raw data (from strategy_deeponet)
        build_dataset_fn: function to build features (from strategy_deeponet)
        feature_names: names of basis features for interpretation

    Returns:
        results dict with keys: pca, X_pca, kmeans, cluster_labels,
        profiles, behavioural_df
    """
    print("=" * 60)
    print("STRATEGY CLUSTERING PIPELINE")
    print("=" * 60)

    # Step 1: Extract
    coeffs_df, num_strategies, num_bases, fnames = extract_strategy_features(
        model, participant_data_dict, preprocess_fn, build_dataset_fn,
        feature_names=feature_names)

    n_participants = len(coeffs_df)
    if n_participants < 4:
        print(f"Only {n_participants} participants — skipping clustering "
              "(need >= 4). Run PCA only.")
        pca, X_pca, cols = run_pca(coeffs_df, plot=True)
        return {'pca': pca, 'X_pca': X_pca, 'kmeans': None,
                'cluster_labels': None, 'profiles': None,
                'behavioural_df': None, 'coeffs_df': coeffs_df}

    # Step 2: PCA
    pca, X_pca, cols = run_pca(coeffs_df, plot=True)

    # Step 3: Cluster
    kmeans, cluster_labels, best_k = cluster_participants(X_pca, plot=True)

    # Step 4: Characterize
    profiles = characterize_clusters(model, coeffs_df, cluster_labels,
                                      num_strategies, num_bases, feature_names)

    # Step 5: Heatmap
    plot_cluster_heatmap(coeffs_df, cluster_labels, num_strategies)

    # Step 6: Behavioral comparison
    behaviour_df = plot_behavioural_comparison(
        cluster_labels, participant_data_dict, preprocess_fn)

    print("\n" + "=" * 60)
    print(f"Pipeline complete. {best_k} strategy clusters found "
          f"across {n_participants} participants.")
    if best_k <= 2:
        print("Note: only 2 clusters found. Try increasing num_strategies "
              "or num_bases in the model, or collect more participants "
              "to detect finer-grained clusters.")
    print("=" * 60)

    return {
        'pca': pca,
        'X_pca': X_pca,
        'kmeans': kmeans,
        'cluster_labels': cluster_labels,
        'profiles': profiles,
        'behavioural_df': behaviour_df,
        'coeffs_df': coeffs_df,
    }


# %% [markdown]
# # Execution: Full Clustering Pipeline
#
# Requires a trained model from `strategy_deeponet.py`. To run this file
# independently, either:
#   1. Run `strategy_deeponet.py` first to train + save a model, then load it here
#   2. Import and call `full_clustering_pipeline()` from a notebook/script that
#      already has a trained model in memory
#
# The cells below demonstrate option 1 (loading a saved model).

# %%
import json
import glob
import os
import torch
import numpy as np
import sys

# --- Fix import path: ensure analysis/ is on sys.path ---
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# --- Resolve project root from this file's location ---
PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

# --- Find participant files (resolve relative to project root) ---
DATA_DIR = os.path.join(_SCRIPT_DIR, "..", "data", "cloud_study")
available_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))

print(f"[diagnostic] Project root: {PROJECT_ROOT}")
print(f"[diagnostic] Data dir:    {DATA_DIR}")
print(f"[diagnostic] Files found: {len(available_files)}")
for f in available_files:
    print(f"  {os.path.basename(f)}")

if len(available_files) < 4:
    print("\n*** STOPPED: Need at least 4 participants for clustering. "
          f"Only {len(available_files)} found. ***")
    print("You can still import functions from this module for use in a notebook.")
    available_files = []
else:
    print(f"  ✓ {len(available_files)} participants — enough for clustering.\n")


# %% [markdown]
# ## Load Trained Model
#
# Load the gated StrategyDeepONet saved by `strategy_deeponet.py`.
# If you have a model in memory from a notebook, skip this cell and pass it
# directly to `full_clustering_pipeline()`.

# %%
MODEL_PATH = os.path.join(_SCRIPT_DIR, "gated_strategy_model.pt")

def load_saved_model(model_path, num_participants, num_strategies=3, num_bases=4):
    from strategy_deeponet import StrategyDeepONet

    model = StrategyDeepONet(
        num_participants=num_participants,
        num_features=5,
        num_bases=num_bases,
        num_strategies=num_strategies,
        shared_bases=True
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    print(f"  ✓ Loaded model from {model_path}")
    return model


trained_model = None

if not available_files:
    print("[diagnostic] Skipping model load — no participant files found.")
elif not os.path.exists(MODEL_PATH):
    print(f"\n*** STOPPED: Saved model not found at: ***")
    print(f"  {MODEL_PATH}")
    print("\nThis file expects a trained StrategyDeepONet saved by the execution")
    print("cells in analysis/strategy_deeponet.py. To fix:")
    print("  1. Run the execution section of analysis/strategy_deeponet.py first")
    print("  2. Or pass a trained model directly to full_clustering_pipeline()")
else:
    num_participants = len(available_files)
    trained_model = load_saved_model(MODEL_PATH, num_participants)


if __name__ == '__main__':

    participant_data_dict = {}

    if not available_files:
        print("[diagnostic] Skipping data load — no participant files.")
    elif trained_model is None:
        print("[diagnostic] Skipping data load — no trained model.")
    else:
        for i, path in enumerate(available_files):
            participant_data_dict[f"Participant_{i+1}"] = json.load(open(path, encoding='utf-8'))
        print(f"  ✓ Loaded {len(participant_data_dict)} participants for clustering.\n")

    results = None

    if not participant_data_dict:
        print("[diagnostic] Skipping pipeline — no participant data loaded.")
    elif trained_model is None:
        print("[diagnostic] Skipping pipeline — no trained model loaded.")
    else:
        from strategy_deeponet import (
            pre_proccess_data_from_choice_vs_no_choice,
            build_deeponet_dataset
        )

        FEATURE_NAMES = [
            "L1_minus_R1",
            "Total_L_minus_Total_R",
            "ball_y_at_top",
            "incoming_pos",
            "incoming_neg"
        ]

        print("Running full clustering pipeline...\n")
        results = full_clustering_pipeline(
            trained_model,
            participant_data_dict,
            preprocess_fn=pre_proccess_data_from_choice_vs_no_choice,
            build_dataset_fn=build_deeponet_dataset,
            feature_names=FEATURE_NAMES
        )

    if results is None:
        print("\n" + "=" * 60)
        print("DIAGNOSTIC: Why didn't clustering run?")
        print("=" * 60)
        print(f"  available_files:       {len(available_files)} files "
              f"({'✓' if available_files else '✗ — need >= 4'})")
        print(f"  trained_model:         {'✓ loaded' if trained_model is not None else '✗ — model file not found or not loaded'}")
        print(f"  participant_data_dict: {'✓ ' + str(len(participant_data_dict)) + ' participants' if participant_data_dict else '✗ — empty'}")
        print()
        if not available_files:
            print("  → Fix: ensure 'data/cloud_study/' exists in the project root")
            print(f"          (looking at: {DATA_DIR})")
        elif trained_model is None:
            print("  → Fix: run the execution cells in analysis/strategy_deeponet.py first")
            print(f"          (expected model at: {MODEL_PATH})")
        print("\n  Or pass a trained model directly:")
        print("    from strategy_clustering import full_clustering_pipeline")
        print("    results = full_clustering_pipeline(model, data_dict, ...)")
        print("=" * 60)

    elif results.get('cluster_labels') is not None:
        cluster_labels = results['cluster_labels']
        profiles = results['profiles']
        pca = results['pca']
        behaviour = results.get('behavioural_df')

        print("\n" + "=" * 50)
        print("CLUSTER ASSIGNMENTS")
        print("=" * 50)

        n_clusters = len(set(cluster_labels))
        for c in sorted(set(cluster_labels)):
            members = [i for i, lbl in enumerate(cluster_labels) if lbl == c]
            print(f"\nCluster {c+1} ({len(members)} participants):")
            for m in members:
                p_name = list(participant_data_dict.keys())[m]
                print(f"  Participant {m+1}: {p_name[:60]}...")

        print("\n" + "-" * 50)
        print("PER-CLUSTER BEHAVIORAL STATISTICS")
        print("-" * 50)

        if behaviour is not None and len(behaviour) > 0:
            for c in sorted(behaviour['cluster'].unique()):
                subset = behaviour[behaviour['cluster'] == c]
                print(f"\nCluster {c+1}:")
                print(f"  Mean RT: {subset['mean_rt'].mean():.0f} ms "
                      f"(±{subset['mean_rt'].std():.0f})")
                print(f"  Mean RT Variability: {subset['rt_variability'].mean():.0f} ms")
                print(f"  Avg Trials: {subset['n_trials'].mean():.0f}")

        print("\n" + "-" * 50)
        print("PCA VARIANCE EXPLAINED")
        print("-" * 50)
        if pca is not None:
            cumsum = np.cumsum(pca.explained_variance_ratio_)
            for i, (var, cum) in enumerate(zip(pca.explained_variance_ratio_, cumsum)):
                print(f"  PC{i+1}: {var*100:.1f}% (cumulative: {cum*100:.1f}%)")
            if cumsum[0] < 0.3:
                print("  → Low first-PC variance: strategies are heterogeneous "
                      "(supports multiple strategy types).")
            elif cumsum[0] > 0.6:
                print("  → High first-PC variance: a single dominant strategy axis "
                      "explains most variation.")

        print("\nPipeline complete. Use results['coeffs_df'] for custom analysis.")

    else:
        # results is not None but cluster_labels is None — PCA-only
        print(f"\n  PCA-only mode: {len(results['coeffs_df'])} participants. "
              "Need >= 4 participants for k-means clustering.")
        print("  PCA components available in results['pca'] and results['X_pca'].")


# %%