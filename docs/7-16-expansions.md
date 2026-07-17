# DeepONet Strategy Expansions — 2026-07-16

Four expansions to `cognitivedeepOnet.py` that capture and interpret individual
variation in decision-making strategies. Each is grounded in a specific paper in
the corpus. **No existing code was overwritten** — all additions are new files.

---

## Files Added

| File | Purpose |
|---|---|
| `analysis/strategy_deeponet.py` | Gated strategy model (all 3 architecture variants + training + viz) |
| `analysis/strategy_clustering.py` | Post-hoc PCA + k-means strategy typing pipeline |
| `docs/7-16-expansions.md` | This document |

---

## 1. Strategy Gate Network (`StrategyDeepONet`)

**Paper grounding:** Ashwood et al 2022 (GLM-HMM discrete strategy switching); Kirsch 2019 (strategies as points in a computational-constraint space).

**What it does:** Replaces the single participant embedding in the original `CognitiveDeepONet` with *K separate strategy networks*, each with its own basis trunk and participant coefficients. A **gate network** takes trial features + participant context and outputs a softmax over K strategies, allowing *trial-level* strategy switching.

### Forward Pass

Let $(\mathbf{x}_i \in \mathbb{R}^M)$ be the trial features for trial $(i)$, and $(p(i) \in \{1, \dots, N\})$ the participant who made it.

For each strategy $(k \in \{1, \dots, K\})$:

$$
\mathbf{b}_k(\mathbf{x}_i) = \tanh\big(\text{MLP}_k(\mathbf{x}_i)\big)
\in [-1, 1]^D
$$

where $MLP_k$ is a 3-layer network $(M \to 16 \to 16 \to D)$ with ReLU hidden activations and Tanh output, and $(D)$ is the number of basis functions.

Each participant $(p)$ has a learned coefficient vector per strategy:

$$
\mathbf{c}_{p,k} \in \mathbb{R}^D
\quad
\mathbf{c}_p = \big[\mathbf{c}_{p,1}, \dots, \mathbf{c}_{p,K}\big]
\in \mathbb{R}^{K \cdot D}
$$

The **logit** per strategy for trial $(i)$ is the dot product:

$$
\ell_{ik} = \mathbf{b}_k(\mathbf{x}_i) \cdot \mathbf{c}_{p(i),k}
$$

The **gate** computes a softmax mixture weight per strategy, conditioned on both the trial features and the participant's full coefficient profile:

$$
\mathbf{g}_i = \text{Softmax}\Big(
\text{MLP}_{\text{gate}}\big([\mathbf{x}_i, \mathbf{c}_{p(i)}]\big)
\Big)
\in [0, 1]^K
$$

where $(\text{MLP}_{\text{gate}}: \mathbb{R}^{M + K \cdot D} \to 16 \to K)$.

This is the key structural difference from the original model: instead of one fixed strategy per participant, the gate can switch per trial.

The final choice logit is the mixture-of-strategies combination:

$$
\boxed{\hat{y}_i = \sum_{k=1}^{K} g_{ik} \cdot \ell_{ik}}
$$

### Loss Function

Three terms:

**Binary cross-entropy** on the choice prediction:

$$
\mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum_{i=1}^{N}
\Big[
y_i \log \sigma(\hat{y}_i)
+ (1 - y_i) \log (1 - \sigma(\hat{y}_i))
\Big]
$$

**Orthogonality penalty** — encourages the \(D\) basis functions within each strategy to be linearly independent, so they discover distinct cognitive features rather than redundant solutions:

$$
\mathcal{L}_{\text{orth}}^{(k)} =
\Big\|
\underbrace{\tilde{\mathbf{B}}_k^\top \tilde{\mathbf{B}}_k}_{\text{correlation}}
- \mathbf{I}_D
\Big\|_F
\qquad
\tilde{\mathbf{B}}_k = \text{column-normalize}(\mathbf{B}_k)
$$

$$
\mathcal{L}_{\text{orth}} = \frac{1}{K} \sum_{k=1}^{K} \mathcal{L}_{\text{orth}}^{(k)}
$$

**Entropy bonus** — prevents the gate from collapsing to a single strategy (one-hot $(\mathbf{g}_i)$ for all trials), which would defeat the purpose of the mixture:

$$
\mathcal{H} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} g_{ik} \log g_{ik}
$$

**Total loss:**

$$
\boxed{
\mathcal{L} = \mathcal{L}_{\text{BCE}}
+ \lambda_{\text{orth}} \cdot \mathcal{L}_{\text{orth}}
- \lambda_{\text{ent}} \cdot \mathcal{H}
}
$$

With defaults $(\lambda_{\text{orth}} = 0.5)$, $(\lambda_{\text{ent}} = 0.05)$.

### Parameter Count

Per strategy: $(M \cdot 16 + 16 \cdot 16 + 16 \cdot D = 5 \cdot 16 + 256 + 4D = 336)$ weight parameters in the basis network plus $(D)$ final biases.

Participant embeddings: $(N \cdot (K \cdot D))$ parameters — for 9 participants, $(K=3), (D=4): (9 \cdot 12 = 108)$ parameters.

Gate network: $((M + K \cdot D) \cdot 16 + 16 \cdot K = (5 + 12) \cdot 16 + 48 = 320)$

Total ≈ 3 × 336 + 108 + 320 = **1,436 parameters**.

### Interpretation

The key diagnostic output is $(\mathbf{g}_i)$ — the gate's softmax probability vector per trial. Plotting $(\mathbf{g}_i)$ over the course of a session reveals strategy switching (e.g., a participant who starts using Strategy 1 for the first 50 trials, then switches to Strategy 2). These are the behavioral equivalent of Ashwood's HMM latent states, but learned end-to-end without a separate EM algorithm.

### Caveat
This DRASTICALLY increases the number of basis functions that we have, it might be better to vastly reduce the number of basis functions

---

## 2. Multi-Task Model with RT Prediction (`StrategyDeepONetMultiTask`)

**Paper grounding:** Resulaj et al 2009 (change-of-mind bounded diffusion uses RT to identify decision boundaries); Keung et al 2020 (divisive evidence accumulation links RT to evidence weighting).

### Forward Pass

Inherits the full gated architecture above. Adds a second set of participant coefficients for the RT prediction head:

$$
\mathbf{c}_{p,k}^{\text{RT}} \in \mathbb{R}^D
\quad
\mathbf{c}_p^{\text{RT}} \in \mathbb{R}^{K \cdot D}
$$

which are combined with the **same basis functions** $(\mathbf{b}_k)$ and the **same gate weights** $(\mathbf{g}_i)$ to produce an RT prediction:

$$
r_{ik}^{\text{pred}} = \mathbf{b}_k(\mathbf{x}_i) \cdot \mathbf{c}_{p(i),k}^{\text{RT}}
$$

$$
\boxed{\hat{r}_i = \sum_{k=1}^{K} g_{ik} \cdot r_{ik}^{\text{pred}}}
$$

Using the same gate here is the modeling assumption: the *strategy choice* determines which set of RT coefficients is active, just as it determines which choice coefficients are active.

### Loss Function

Adds an MSE term for RT:

$$
\mathcal{L}_{\text{RT}} = \frac{1}{N} \sum_{i=1}^{N} (\hat{r}_i - r_i)^2
$$

$$
\boxed{
\mathcal{L} = \mathcal{L}_{\text{BCE}}
+ \lambda_{\text{RT}} \cdot \mathcal{L}_{\text{RT}}
+ \lambda_{\text{orth}} \cdot \mathcal{L}_{\text{orth}}
- \lambda_{\text{ent}} \cdot \mathcal{H}
}
$$

### What the RT head reveals

Because both heads share the same basis functions $(\mathbf{b}_k)$, we can ask for each strategy $(k)$: *does this basis predict both choice and RT, or only one?*

- A basis with large $(\|\mathbf{c}_{p,k}\|)$ but small $(\|\mathbf{c}_{p,k}^{\text{RT}}\|)$ captures **deliberative planning** — it drives choice but doesn't correlate with response time (the participant takes however long they need).
- A basis with large $(\|\mathbf{c}_{p,k}\|)$ and large $(\|\mathbf{c}_{p,k}^{\text{RT}}\|)$ captures **heuristic / impulsive responding** — the same feature that drives choice also predicts faster (or slower) RT.

This decomposes participant strategies along Kirsch's heuristic-vs-deliberative axis.

---

## 3. Time-Binned Coefficients (`TimeBinnedStrategyDeepONet`)

**Paper grounding:** Ashwood et al 2022 PsyTrack model (tracks dynamic GLM weights over time); Ji-An et al 2025 (strategy discovery over temporal sequences).

### Forward Pass

Inherits the full gated architecture but replaces the per-participant embedding with a per-participant-per-time-bin embedding.

Let $(T)$ be the number of temporal bins. Each participant's trials are split into \(T\) equal-sized blocks by trial index. The embedding lookup becomes:

$$
\text{idx} = p \cdot T + t(i)
\qquad
t(i) = \left\lfloor \frac{\text{trial index of } i}{\text{trials per bin}} \right\rfloor
$$

$$
\mathbf{c}_{p,t(i),k} = \text{Embedding}(\text{idx})_{k}
$$

The gate now conditions on the time-specific coefficients:

$$
\mathbf{g}_i = \text{Softmax}\Big(
\text{MLP}_{\text{gate}}\big([\mathbf{x}_i, \mathbf{c}_{p(i),t(i)}]\big)
\Big)
$$

All other machinery (basis networks, dot product logits, loss function) remains identical.

### What the trajectories reveal

For each participant \(p\) and strategy \(k\), we now have a sequence of
coefficient vectors over time:

$$
\mathbf{c}_{p,1,k}, \mathbf{c}_{p,2,k}, \dots, \mathbf{c}_{p,T,k}
$$

Plotting these reveals:

- **Strategy adaptation:** does a participant shift from "planner" (large weight on $(\text{Total}_L - \text{Total}_R)$) to "reactor" (large weight on $(\text{ball\_y\_at\_top})$) over blocks?
- **Learning effects:** a monotonic drift in coefficient values across time suggests the participant refined their strategy with experience.
- **Sudden switches:** a sharp discontinuity between bin $(t)$ and $(t+1)$ resembles the HMM state transitions in Ashwood's PsyTrack model.

---

## 4. Post-Hoc Strategy Clustering (`strategy_clustering.py`)

**Paper grounding:** Jach et al 2024 (individual differences have low-dimensional PCA structure across curiosity and information-demand traits); Ashwood et al 2022 (discrete strategy types exist).

This pipeline takes a trained model's participant coefficients and discovers discrete strategy types using PCA + k-means. It does not require retraining the model — it's a post-hoc analysis of the learned representations.

### Step 1 — Extract Coefficients

From either `CognitiveDeepONet` or `StrategyDeepONet`, extract the participant embedding matrix:

$$
\mathbf{C} \in \mathbb{R}^{N \times K \cdot D}
$$

where row $(p)$ contains the concatenated strategy coefficients for participant $(p)$.

For the gated model, each participant is represented by $(K \cdot D = 12)$ values (3 strategies × 4 bases). This is the participant's **strategy fingerprint**.

### Step 2 — PCA Decomposition

Center $(\mathbf{C})$ and compute the covariance:

$$
\tilde{\mathbf{C}} = \mathbf{C} - \mathbf{1}_N \bar{\mathbf{c}}^\top
\qquad
\mathbf{S} = \frac{1}{N - 1} \tilde{\mathbf{C}}^\top \tilde{\mathbf{C}}
$$

Solve the eigendecomposition:

$$
\mathbf{S} \mathbf{v}_j = \lambda_j \mathbf{v}_j
\qquad
\lambda_1 \geq \lambda_2 \geq \dots \geq \lambda_{K \cdot D}
$$

Project participants onto the top principal components:

$$
\mathbf{Z} = \tilde{\mathbf{C}} \mathbf{V}_r
\in \mathbb{R}^{N \times r}
\quad
\mathbf{V}_r = [\mathbf{v}_1, \dots, \mathbf{v}_r]
$$

The variance explained by PC \(j\) is $(\lambda_j / \sum \lambda_j)$.

This tests Jach's hypothesis that individual differences in decision-making have a **low-dimensional** structure. If $(\lambda_1)$ explains > 70% of variance, a single strategy axis dominates; if it explains < 30%, strategies are genuinely multi-dimensional.

### Step 3 — k-Means Clustering

Run k-means on the PCA-reduced coordinates $(\mathbf{Z})$:

$$
\min_{\boldsymbol{\mu}_1, \dots, \boldsymbol{\mu}_C, \{z_i\}}
\sum_{i=1}^{N} \sum_{c=1}^{C} \mathbf{1}[z_i = c]
\| \mathbf{Z}_{i,:} - \boldsymbol{\mu}_c \|^2
$$

where $(C)$ is chosen by maximizing the **silhouette score**:

$$
s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}
$$

with $(a(i))$ = mean distance from participant $(i)$ to others in the same cluster, and $(b(i))$ = mean distance to the nearest other cluster. The optimal $(C)$ is the one with the highest mean silhouette across all participants.

### Step 4 — Cluster Characterization

For each cluster \(c\), compute the mean participant strategy profile:

$$
\bar{\mathbf{c}}_c = \frac{1}{| \text{Cluster } c |}
\sum_{p \in \text{Cluster } c} \mathbf{C}_{p,:}
$$

Reshape back to \((K, D)\) to see which strategy × basis weights define the cluster. This yields interpretations like:

- **Cluster 1:** "Immediate planners" — high weight on strategy 1's $(\text{L1\_minus\_R1})$ basis, low on everything else
- **Cluster 2:** "Look-ahead" — high weight on strategy 1's $(\text{Total\_L\_minus\_Total\_R})$ basis
- **Cluster 3:** "Direction followers" — high weight on $(\text{incoming\_pos})$ and $(\text{incoming\_neg})$

### Step 5 — Behavioral Validation

For each cluster, compute behavioral metrics:

$$
\mu_{\text{RT},c} = \frac{1}{|\text{Cluster } c|}
\sum_{p \in \text{Cluster } c} \frac{1}{N_p} \sum_{i=1}^{N_p} \text{RT}_{pi}
$$

$$
\sigma_{\text{RT},c} = \frac{1}{|\text{Cluster } c|}
\sum_{p \in \text{Cluster } c} \text{std}(\text{RT}_{p})
$$

Boxplots of RT and RT variability per cluster test whether discovered
strategy types map onto observable behavioral differences.

---

## Integration with Existing Code

The two new files are **fully independent** — they do not modify `cognitivedeepOnet.py`
or any other existing file. They reimplement `MazeDataset`, `orthogonality_penalty`,
and `pre_proccess_data_from_choice_vs_no_choice` locally so that changes to the
original don't break them, and so that the expansions can diverge as needed.

The `run_model()` function in `strategy_deeponet.py` is a convenience wrapper
that handles the full training pipeline (load → preprocess → split → scale →
train → evaluate) for all three model variants.

## Running Order

```python
# 1. Train a gated strategy model
from analysis.strategy_deeponet import run_model
model, weights, metrics = run_model(
    model_type='gated',
    participant_data_paths=["cloud study data/p1.json", ...],
    num_strategies=3
)

# 2. Cluster participants by strategy
from analysis.strategy_clustering import full_clustering_pipeline
from analysis.strategy_deeponet import (
    pre_proccess_data_from_choice_vs_no_choice,
    build_deeponet_dataset
)
import json

participants = {}
for i, path in enumerate(participant_paths):
    participants[f"P{i}"] = json.load(open(path))

results = full_clustering_pipeline(
    model, participants,
    preprocess_fn=pre_proccess_data_from_choice_vs_no_choice,
    build_dataset_fn=build_deeponet_dataset
)
```

## Caveats

- **Minimum participants:** Strategy gating needs at least ~5 participants to learn distinct strategies; clustering needs at least 4 participants. With the current 9 participants, k-means typically finds 2-3 clusters.
- **Gate collapse:** Set $(\lambda_{\text{ent}} \geq 0.05)$ to prevent the gate from assigning all trials to one strategy. Tune upward if collapse persists.
- **Time binned variant:** `TimeBinnedStrategyDeepONet` requires the dataset to return `(features, p_ids, bin_ids, choices)`. The current `MazeDataset` implementation handles this via the `time_bin_ids` parameter.
- **Overfitting risk:** With only 9 participants and 3 strategies × 4 bases, the model has \(\sim 1,400\) parameters. A small dataset may overfit. Monitor train/test accuracy gap; if >10%, reduce \(K\) or increase
  $(\lambda_{\text{ent}})$.
