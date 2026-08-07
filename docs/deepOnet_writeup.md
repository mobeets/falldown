# DeepONet Mathematical Formulations

Mathematical writeup of the original DeepONet operator-learning architecture and the
successive variations implemented in this project (`CognitiveDeepONet` in
`analysis/cognitivedeepOnet.py`, and the gated / multi-task / time-binned variants in
`analysis/strategy_deeponet.py`).

---

## Notation

| Symbol | Meaning |
|---|---|
| $(N)$ | number of participants |
| $(p(i) \in \{1, \dots, N\})$ | participant index of trial $(i)$ |
| $(M)$ | number of trial features (5) |
| $(D)$ | number of basis functions (4) |
| $(K)$ | number of strategies (3) |
| $(T)$ | number of time bins (5) |
| $(H)$ | hidden width of the MLPs (16) |
| $(\mathbf{x}_i \in \mathbb{R}^M)$ | feature vector of trial $(i)$ |
| $(y_i \in \{0, 1\})$ | choice label of trial $(i)$ (1 = chose right) |
| $(N_t)$ | number of trials in the (training) set |
| $(\sigma(\cdot))$ | logistic sigmoid |
| $(\|\cdot\|_F)$ | Frobenius norm |

---

## 1. The Original DeepONet (Lu et al., 2021)

### The Operator Learning Problem

We want to learn an **operator** $G: \mathcal{U} \to \mathcal{V}$ that maps an input
function $u$ to an output function $G(u)$, where $\mathcal{U}$ and $\mathcal{V}$ are
spaces of functions on compact domains. From observed input-output function pairs
$\{(u_j,\, G(u_j))\}$ the model must generalize to unseen input functions.

### Universal Approximation Theorem for Operators

DeepONet is grounded in the approximation theorem of Chen & Chen (1995). For any
continuous operator $G$ and any tolerance $\epsilon > 0$, there exist an integer
$(p)$, continuous functionals $(g_1, \dots, g_p)$ and continuous functions
$(\phi_1, \dots, \phi_p)$ such that

$$
\sup_{u \in U} \; \sup_{y \in K_2}
\Big| G(u)(y) - \sum_{k=1}^{p} g_k\big(u(x_1), \dots, u(x_m)\big) \, \phi_k(y) \Big|
< \epsilon,
$$

where each functional $g_k$ depends on $u$ **only through its values at $m$ fixed
sensor points** $(x_1, \dots, x_m)$.

### Architecture: Branch and Trunk

DeepONet parametrizes the two factors of the theorem with two neural networks:

- **Branch net** encodes the input function into coefficients:

$$
[b_1, \dots, b_p] = \mathcal{B}\big([u(x_1), \dots, u(x_m)]\big) \in \mathbb{R}^p
$$

- **Trunk net** encodes the query location $(y)$ into basis functions:

$$
[t_1, \dots, t_p] = \mathcal{T}(y) \in \mathbb{R}^p
$$

The output is the inner product of branch and trunk outputs (plus a bias $(b_0)$):

$$
\boxed{
G(u)(y) \approx \sum_{k=1}^{p} b_k \cdot t_k(y) + b_0
}
$$

### Interpretation

The branch net plays the role of the functionals $g_k$ (the "coefficients" of the
decomposition), and the trunk net plays the role of the functions $\phi_k$ (the
"basis"). The dot product generalizes a modal / Fourier-type expansion, and the theorem
guarantees that a sufficiently expressive parametrization can approximate the operator
to arbitrary accuracy.

---

## 2. CognitiveDeepONet — Project Adaptation

The project reinterprets the operator-learning view: **each participant defines a
cognitive operator** mapping trial context to a choice, and that operator is written as
a weighted sum of shared, participant-independent basis functions.

| Original DeepONet | CognitiveDeepONet |
|---|---|
| input function $u$ | participant $p$ (via its coefficient vector) |
| sensor values $[u(x_1), \dots, u(x_m)]$ | trial feature vector $\mathbf{x}_i$ |
| query location $y$ | trial feature vector $\mathbf{x}_i$ |
| branch net → coefficients $b_k$ | participant embedding $\mathbf{c}_p$ |
| trunk net → bases $t_k(y)$ | basis network $\mathbf{b}(\mathbf{x}_i)$ |
| operator $G(u)(y)$ | choice logit $\hat{y}_i$ |

### Trial Features

Each trial $(i)$ is summarized by $M = 5$ features. The three continuous features are
standardized (zero mean, unit variance, fit on the training split only); the two
direction indicators are binary and left unstandardized:

$$
\mathbf{x}_i = \big[
\text{L1} - \text{R1},\;
\text{Total}_L - \text{Total}_R,\;
\text{ball\_y\_at\_top},\;
\text{incoming\_pos},\;
\text{incoming\_neg}
\big] \in \mathbb{R}^5
$$

with

- $(\text{L1}, \text{R1})$ — 1-step distances of the left/right hole from the entry hole
- $(\text{Total}_L, \text{Total}_R)$ — cumulative 2-step distances (look-ahead cost)
- $(\text{ball\_y\_at\_top})$ — ball's screen-relative height when the sequence starts
- $(\text{incoming\_pos}, \text{incoming\_neg})$ — one-hot encoding of the direction the ball is coming from

### Basis (Trunk) Network

A 3-layer MLP $(M \to H \to H \to D)$ with ReLU hidden activations and a **Tanh output**
layer that forces all $D$ bases onto the same $[-1, 1]$ scale:

$$
\mathbf{b}(\mathbf{x}_i) = \tanh\big(\text{MLP}(\mathbf{x}_i)\big) \in [-1, 1]^D
$$

### Participant Coefficients (Branch)

A learned embedding matrix $\mathbf{C} \in \mathbb{R}^{N \times D}$; row $(\mathbf{c}_p)$
holds participant $(p)$'s coefficients. Initialized around 0
($\mathcal{N}(0, 0.1^2)$) to keep early predictions near chance:

$$
\mathbf{c}_p \in \mathbb{R}^D
$$

### Forward Pass

The choice logit is the inner product of the participant's coefficients with the basis
evaluated at the trial's features:

$$
\boxed{
\hat{y}_i = \mathbf{b}(\mathbf{x}_i)^\top \mathbf{c}_{p(i)}
}
$$

### Loss Function

**Binary cross-entropy** on the choice logits:

$$
\mathcal{L}_{\text{BCE}} = -\frac{1}{N_t} \sum_{i=1}^{N_t}
\Big[
y_i \log \sigma(\hat{y}_i) + (1 - y_i) \log \big(1 - \sigma(\hat{y}_i)\big)
\Big]
$$

**Orthogonality penalty** — over a batch of $(n)$ trials, stack the basis outputs into
$\mathbf{B} \in \mathbb{R}^{n \times D}$, normalize each column, and penalize how far the
resulting correlation matrix deviates from the identity (this is data-dependent: it
measures orthogonality of the bases *on the current batch*):

$$
\tilde{\mathbf{B}} = \text{column-normalize}(\mathbf{B})
\qquad
\mathcal{L}_{\text{orth}} = \Big\| \tilde{\mathbf{B}}^\top \tilde{\mathbf{B}} - \mathbf{I}_D \Big\|_F
$$

**Total loss:**

$$
\boxed{
\mathcal{L} = \mathcal{L}_{\text{BCE}} + \lambda_{\text{orth}} \cdot \mathcal{L}_{\text{orth}}
}
$$

with default $(\lambda_{\text{orth}} = 0.5)$.

---

## 3. Variation 1 — StrategyDeepONet (Gated Mixture of Strategies)

Replaces the single per-participant coefficient vector with $K$ separate strategies
plus a **gate network** that selects the active strategy per trial. This is the
behavioral analogue of Ashwood et al. (2022) GLM-HMM discrete strategy switching,
learned end-to-end.

> **Update:** the current implementation replaces the feature-driven gate below with an
> explicit Markov chain over strategies — see **Section 6 (HMM-StrategyDeepONet)**.
> The MLP gate here is retained in this document as the historical formulation.

### Per-Strategy Basis Networks

In the general form each strategy $(k \in \{1, \dots, K\})$ has its own basis network:

$$
\mathbf{b}_k(\mathbf{x}_i) = \tanh\big(\text{MLP}_k(\mathbf{x}_i)\big) \in [-1, 1]^D
$$

A `shared_bases` mode collapses all strategies onto one shared basis network,
$(\mathbf{b}_k(\mathbf{x}_i) = \mathbf{b}(\mathbf{x}_i))$ for all $k$ — this is what the
`run_model` pipeline uses by default.

### Per-Strategy Logits

Each participant now has a coefficient vector **per strategy**, concatenated into a
single profile:

$$
\mathbf{c}_{p,k} \in \mathbb{R}^D
\qquad
\mathbf{c}_p = \big[\mathbf{c}_{p,1}, \dots, \mathbf{c}_{p,K}\big] \in \mathbb{R}^{K \cdot D}
$$

The logit under strategy $(k)$ for trial $(i)$ is:

$$
\ell_{ik} = \mathbf{b}_k(\mathbf{x}_i)^\top \mathbf{c}_{p(i),k}
$$

### Gate Network

The gate outputs a softmax mixture weight per strategy, conditioned on both the trial
features and the participant's full strategy profile:

$$
\mathbf{g}_i = \text{Softmax}\Big(
\text{MLP}_{\text{gate}}\big([\mathbf{x}_i, \mathbf{c}_{p(i)}]\big)
\Big) \in [0, 1]^K
\qquad
\text{MLP}_{\text{gate}}: \mathbb{R}^{M + K \cdot D} \to H \to K
$$

### Mixture Output

The final choice logit is the gate-weighted mixture of per-strategy logits:

$$
\boxed{
\hat{y}_i = \sum_{k=1}^{K} g_{ik} \cdot \ell_{ik}
}
$$

### Loss Function

**Binary cross-entropy** as above, plus two regularizers:

**Orthogonality penalty** — averaged over strategies:

$$
\mathcal{L}_{\text{orth}} = \frac{1}{K} \sum_{k=1}^{K}
\Big\| \tilde{\mathbf{B}}_k^\top \tilde{\mathbf{B}}_k - \mathbf{I}_D \Big\|_F
\qquad
\tilde{\mathbf{B}}_k = \text{column-normalize}(\mathbf{B}_k)
$$

**Entropy bonus** — prevents the gate from collapsing to a single strategy
(one-hot $(\mathbf{g}_i)$ for all trials):

$$
\mathcal{H} = -\frac{1}{N_t} \sum_{i=1}^{N_t} \sum_{k=1}^{K} g_{ik} \log g_{ik}
$$

**Total loss:**

$$
\boxed{
\mathcal{L} = \mathcal{L}_{\text{BCE}}
+ \lambda_{\text{orth}} \cdot \mathcal{L}_{\text{orth}}
- \lambda_{\text{ent}} \cdot \mathcal{H}
}
$$

with defaults $(\lambda_{\text{orth}} = 0.5)$, $(\lambda_{\text{ent}} = 0.05)$.

The key diagnostic is the gate vector $(\mathbf{g}_i)$ itself — plotting it over trials
reveals trial-level strategy switching.

---

## 4. Variation 2 — StrategyDeepONetMultiTask (Choice + RT)

Adds a second regression head that predicts reaction time using the **same basis
functions** and the **same gate weights**. The modeling assumption: the active strategy
determines which set of RT coefficients are used, just as it determines which choice
coefficients are used.

### RT Head

A second set of participant coefficients is learned:

$$
\mathbf{c}_{p,k}^{\text{RT}} \in \mathbb{R}^D
\qquad
\mathbf{c}_p^{\text{RT}} \in \mathbb{R}^{K \cdot D}
$$

Per-strategy RT predictions and the gate-weighted mixture:

$$
r_{ik}^{\text{pred}} = \mathbf{b}_k(\mathbf{x}_i)^\top \mathbf{c}_{p(i),k}^{\text{RT}}
\qquad
\boxed{
\hat{r}_i = \sum_{k=1}^{K} g_{ik} \cdot r_{ik}^{\text{pred}}
}
$$

RT targets are standardized before training: $(r_i^{\text{norm}} = (r_i - \mu_r)/\sigma_r)$.

### Loss Function

Adds an MSE term for the RT prediction:

$$
\mathcal{L}_{\text{RT}} = \frac{1}{N_t} \sum_{i=1}^{N_t} \big(\hat{r}_i - r_i^{\text{norm}}\big)^2
$$

$$
\boxed{
\mathcal{L} = \mathcal{L}_{\text{BCE}}
+ \lambda_{\text{RT}} \cdot \mathcal{L}_{\text{RT}}
+ \lambda_{\text{orth}} \cdot \mathcal{L}_{\text{orth}}
- \lambda_{\text{ent}} \cdot \mathcal{H}
}
$$

with default $(\lambda_{\text{RT}} = 0.3)$.

### What the shared basis reveals

Because both heads share $(\mathbf{b}_k)$, comparing coefficient magnitudes separates
strategy types:

- large $(\|\mathbf{c}_{p,k}\|)$ but small $(\|\mathbf{c}_{p,k}^{\text{RT}}\|)$ —
  **deliberative planning**: the feature drives choice but not response time.
- large $(\|\mathbf{c}_{p,k}\|)$ and large $(\|\mathbf{c}_{p,k}^{\text{RT}}\|)$ —
  **heuristic / impulsive responding**: the same feature also predicts RT.

---

## 5. Variation 3 — TimeBinnedStrategyDeepONet

Splits each participant's trials into $T$ temporal bins and learns a separate strategy
profile per bin, producing a coefficient trajectory over time (analogue of Ashwood's
PsyTrack dynamic weights).

### Time-Binned Embeddings

The embedding lookup key becomes the participant-time pair:

$$
\text{idx} = p \cdot T + t(i)
\qquad
t(i) \in \{0, 1, \dots, T - 1\}
$$

$$
\mathbf{c}_{p, t(i), k} = \text{Embedding}(\text{idx})_k \in \mathbb{R}^D
$$

The embedding table has $N \cdot T$ rows (one profile per participant per bin).

### Forward Pass

The time-binned variant inherits the Markov-chain architecture of Section 6: strategy
transitions are a per-participant Markov chain, but the emission coefficients are
selected per trial from the participant-bin embedding:

$$
\mathbf{g}_i = \text{Softmax}\big(\alpha_{i}\big)
\qquad
\mathbf{c}_{p, t(i), k} = \text{Embedding}(p \cdot T + t(i))_k
$$

where $(\alpha_i)$ is the forward-algorithm state at trial $(i)$ (see Section 6). All
other machinery (basis networks, dot-product logits, sequence NLL loss) is shared.

### What the trajectories reveal

For each participant and strategy, a sequence of coefficient vectors over time:

$$
\mathbf{c}_{p,1,k}, \mathbf{c}_{p,2,k}, \dots, \mathbf{c}_{p,T,k}
$$

Plotting these reveals strategy adaptation (drift), learning effects (monotonic change),
and sudden strategy switches (sharp discontinuities between bins).

### Implementation note

Bins are assigned per participant from each trial's rank within that participant:
$(t(i) = \lfloor i \cdot T / N_p \rfloor)$ (equal-sized bins by trial index, clamped to
$T - 1$).

---

## 6. Variation 4 — HMM-StrategyDeepONet (Markov-Gated Strategies)

The current `StrategyDeepONet` replaces the per-trial feature-driven gate with an
explicit **Markov chain over strategy states**. Strategies are now latent states that
persist across trials and switch only occasionally — the DeepONet analogue of the
GLM-HMM (Ashwood et al., 2022), but with the per-strategy emissions replaced by
learned basis-function networks.

### Latent State Sequence

Let $(z_i \in \{1, \dots, K\})$ be the hidden strategy state on trial $(i)$. For each
participant $(p)$, the states evolve as a first-order Markov chain with **static**
(per-participant) transition probabilities:

$$
\pi_{p,k} = P(z_1 = k)
\qquad
A_p[i, j] = P(z_{i+1} = j \mid z_i = i)
$$

Both are obtained by softmax over learned logits:

$$
\boldsymbol{\pi}_p = \text{Softmax}\big(\boldsymbol{\ell}_p^{\text{init}}\big) \in [0, 1]^K
\qquad
A_p = \text{row-softmax}\big(\mathbf{L}_p^{\text{trans}}\big) \in [0, 1]^{K \times K}
$$

where $(\mathbf{L}_p^{\text{trans}} \in \mathbb{R}^{K \times K})$ is a per-participant
parameter. An optional **stickiness** can be imposed by inflating the diagonal
(self-transition) concentration, so that switches between strategies are rare:
$A_p[i,i] = (\alpha + \kappa)/(K\alpha + \kappa)$ for a Dirichlet prior with
concentration $(\alpha + \kappa \,\mathbf{1}[i = j])$.

### Emissions (unchanged from the gated model)

Under state $(k)$, the choice probability is the sigmoid of the basis-coefficient dot
product:

$$
\ell_{ik} = \mathbf{b}_k(\mathbf{x}_i)^\top \mathbf{c}_{p(i), k}
\qquad
P(y_i = 1 \mid z_i = k, \mathbf{x}_i) = \sigma(\ell_{ik})
$$

with per-trial emission log-likelihood:

$$
e_{ik} = y_i \log \sigma(\ell_{ik}) + (1 - y_i) \log\big(1 - \sigma(\ell_{ik})\big)
$$

### Forward Algorithm (log-space)

The sequence log-likelihood is computed with the forward algorithm, which is fully
differentiable in torch. Define the filtered state $(\alpha_{t, k} = \log P(z_t = k, y_{1:t}))$:

$$
\alpha_{1, k} = \log \pi_{p, k} + e_{1k}
$$

$$
\alpha_{t, j} = \log\!\!\sum_{k=1}^{K} \exp\Big(\alpha_{t-1, k} + \log A_p[k, j]\Big) + e_{tj}
$$

The per-sequence likelihood is:

$$
\boxed{
\log P(y_{1:T}) = \log\!\!\sum_{k=1}^{K} \exp\big(\alpha_{T, k}\big)
}
$$

### Predictions from the filtered posterior

The filtered posterior state probabilities and the marginal choice probability are:

$$
\gamma_{t, k} = P(z_t = k \mid y_{1:t}) = \frac{\exp(\alpha_{t, k})}{\sum_{j} \exp(\alpha_{t, j})}
\qquad
\boxed{
P(y_t = 1) = \sum_{k=1}^{K} \gamma_{t, k} \, \sigma(\ell_{tk})
}
$$

### Loss Function

Training minimizes the negative sequence log-likelihood (per-trial normalized) plus
the orthogonality penalty; the gate-entropy term is replaced by a bonus on the entropy
of the mean state occupancy (to avoid collapsing all trials into a single state):

$$
\boxed{
\mathcal{L} =
-\frac{1}{\sum_p T_p} \sum_{p} \log P\big(y_{p, 1:T}\big)
+ \lambda_{\text{orth}} \cdot \mathcal{L}_{\text{orth}}
- \lambda_{\text{ent}} \cdot \mathcal{H}(\bar{\gamma})
}
$$

with $(\bar{\gamma} = \frac{1}{\sum_p T_p}\sum_{p,t} \gamma_{p,t})$ and
$(\mathcal{H}(\bar{\gamma}) = -\sum_k \bar{\gamma}_k \log \bar{\gamma}_k)$.

### What the transition matrix reveals

The per-participant transition matrix $(A_p)$ is the key diagnostic: its diagonal
measures strategy persistence, off-diagonal entries measure switching, and the
filtered posterior $(\gamma_{t})$ gives a per-trial strategy timeline directly
comparable to the gate weights of the original gated model and to GLM-HMM state
posteriors.

---

## 7. Parameter Counts

Definitions: $(M = 5)$, $(H = 16)$, $(D = 4)$, $(K = 3)$, $(N = 9)$, $(T = 5)$.

Basis network (one): $MH + H^2 + HD + 2H + D$ weights $+$ biases $= 436$.

The MLP gate has been replaced by the Markov-chain parameters: per-participant
transition logits $N \cdot K^2$ and initial logits $N \cdot K$.

| Model | Formula | Defaults ($N = 9$) |
|---|---|---|
| CognitiveDeepONet | $MH + H^2 + HD + 2H + D + ND$ | $436 + 36 = 472$ |
| StrategyDeepONet — gated (MLP gate) | $K(MH + H^2 + HD + 2H + D) + (M + KD)H + H + KH + K + NKD$ | $1308 + 339 + 108 = 1755$ |
| HMM-StrategyDeepONet (shared bases) | $(MH + H^2 + HD + 2H + D) + NKD + NK^2 + NK$ | $436 + 108 + 81 + 27 = 652$ |
| HMM-StrategyDeepONet (per-strategy bases) | $K(MH + H^2 + HD + 2H + D) + NKD + NK^2 + NK$ | $1308 + 108 + 81 + 27 = 1524$ |
| HMM-StrategyDeepONetMultiTask (shared) | shared-basis HMM $+ NKD$ | $652 + 108 = 760$ |
| HMM-TimeBinnedStrategyDeepONet (shared) | shared-basis HMM $+ N \cdot T \cdot KD$ | $436 + 540 + 81 + 27 = 1084$ |

Note: `run_model` uses `shared_bases=True`, so the deployed variants are the shared-basis
rows.

---

## Caveats

- **State collapse (HMM variants):** set $(\lambda_{\text{ent}} \geq 0.05)$ to keep the
  mean state occupancy spread across strategies. With the MLP gate (historical), the
  entropy term instead prevents one-hot $(\mathbf{g}_i)$ collapse.
- **Sequence requirement:** the HMM-gated models are trained on ordered per-participant
  sequences, so the data pipeline uses a temporal train/test split per participant
  rather than a shuffled i.i.d. split.
- **Overfitting:** with a handful of participants and $K \cdot D$ coefficients per
  participant, these models can overfit quickly. Monitor the train/test accuracy gap.
- **Orthogonality is batch-dependent:** $(\mathcal{L}_{\text{orth}})$ measures
  basis orthogonality on the current batch, not globally — small batches will perturb it.
- **Tanh bounding:** the Tanh output caps every basis at $[-1, 1]$, which keeps the
  per-participant coefficients directly interpretable as "cognitive weights" but limits
  the effective output scale of each basis.
- **Label symmetry (predictions):** the marginal prediction $(\sum_k \gamma_{t,k} \sigma(\ell_{tk}))$
  weights each strategy's emission by its filtered posterior, which already conditions
  on the observed choices — interpret accuracy accordingly.
