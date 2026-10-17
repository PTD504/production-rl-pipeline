# Algorithm Selection: Choosing the Right RL Engine

## Introduction

In **[01_mdp_modeling.md](01_mdp_modeling.md)**, we reverse-engineered the core MDP specifications of the Push Block environment:
- **Action Space:** Discrete with 7 actions (`Discrete(7)`).
- **Observation Space:** 210-dimensional continuous vector (`Box(210,)`), pre-normalized in $[0.0, 1.0]$.
- **Reward Signal:** Sparse terminal reward ($+5.0$) upon goal completion with an existential step penalty ($-0.001$).
- **Parallel Infrastructure:** 32 synchronized arenas in a single Unity instance, delivering $\approx 6,000+$ FPS.

As engineers, our algorithm selection is grounded strictly in **algorithmic first-principles, mathematical properties, and systemic architectural fit** for the task at hand. We evaluate each algorithmic candidate based on its core optimization mechanics, memory dynamics, and exploration capabilities.

Let's walk through our elimination process from first principles.

---

## Step 1: Eliminating Continuous-First Algorithms

The first decision filter is straightforward: our action space is strictly **discrete** ($\mathcal{A} \in \{0, 1, \dots, 6\}$).

Algorithms designed around continuous control are fundamentally ill-suited:
- **DDPG (Deep Deterministic Policy Gradient)** & **TD3 (Twin Delayed DDPG)**: Depend directly on computing the Jacobian $\nabla_a Q(s, a)$ with respect to continuous action vectors. Applying them to discrete actions requires continuous relaxations (e.g., Gumbel-Softmax) that introduce high gradient variance and unnecessary training instability.
- **SAC (Soft Actor-Critic)**: While a discrete variant exists, SAC was natively derived for continuous maximum-entropy control. In discrete spaces, evaluating full softmax expectations across all action heads negates many of SAC's original analytical advantages over standard discrete policy gradient methods.

**Remaining Candidates:**
1. **A2C** (Advantage Actor-Critic)
2. **DQN Family** (Deep Q-Networks: Double DQN, Dueling DQN, Rainbow)
3. **PPO** (Proximal Policy Optimization)

---

## Step 2: The Actor-Critic Elimination — Why PPO Over A2C?

Within the on-policy policy gradient family, both A2C and PPO support **entropy regularization** ($\beta \mathcal{H}(\pi(\cdot | s))$) to stimulate exploration in sparse environments. However, their optimization dynamics differ fundamentally:

### 1. Sample Reuse: Multi-Epoch Minibatches vs. "One-and-Done"
- **A2C** is strictly 1-step on-policy: it collects a rollout across workers, computes a single gradient step, and immediately discards the data.
- **PPO** introduces a clipped surrogate objective:
  $$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left(r_t(\theta)\hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right) \right]$$
  The pessimistic clipping term ($\epsilon \approx 0.2$) bounds the likelihood ratio $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$, ensuring the new policy does not drift catastrophically far from the data-generating policy. This allows us to run **4 to 10 epochs of mini-batch stochastic gradient descent** on the exact same rollout buffer, squeezing orders of magnitude more learning signal out of every transition.

### 2. Safeguarding Against Policy Collapse (Trust Region)
- In A2C, an unluckily large gradient step (especially upon encountering the rare $+5.0$ goal signal) can push the policy into a destructive parameter subspace from which policy gradients cannot recover.
- PPO's clipping mechanism acts as an analytical soft trust region, providing reliable, monotonic or near-monotonic policy improvement without the second-order computational overhead of TRPO.

### 3. Vectorized Rollout Compatibility
- With 32 parallel arenas generating 6,000+ FPS, collecting 128 steps per arena yields a rollout buffer of **4,096 transitions** (`32 * 128`).
- PPO digests this entire batch in parallel on the GPU across multiple mini-batch passes. A2C, by contrast, relies on high-frequency, tiny gradient updates that cause severe CPU-GPU synchronization bottlenecks.

**Verdict:** PPO strictly dominates A2C in sample efficiency, optimization stability, and hardware utilization.

---

## Step 3: Value-Based (DQN Family) vs. Policy Gradient (PPO)

This brings us to the core architectural comparison: **Off-Policy Value-Based (DQN / Double / Dueling DQN)** versus **On-Policy Actor-Critic (PPO)**.

Even considering modern advancements in the DQN family (Double Q-learning, Dueling streams, and target networks), deep value-based methods face **fundamental, immutable bottlenecks** when applied to our specific environment:

### 1. The Replay Buffer Tax: Memory & I/O Overhead

Let's look at the raw memory demand side-by-side right from the start:

| Memory Metric | DQN Family (Replay Buffer) | PPO (Rollout Buffer) | Architectural Impact |
| :--- | :--- | :--- | :--- |
| **Active RAM Required** | **$\approx 1,750\text{ MB}$ ($1.75\text{ GB}$)** | **$\approx 3.4\text{ MB}$** | **$>500\times$ Memory Difference!** |
| **Memory Lifecycle** | Persistent, large circular buffer | Ephemeral, overwritten each rollout | Zero permanent memory bloat |
| **Hardware Cache Fit** | Exceeds CPU cache (forces DRAM access) | Fits entirely into L3 Cache / GPU VRAM | Ultra-fast contiguous execution |
| **Memory Access Pattern** | Non-contiguous random indexing | Contiguous sequential mini-batches | Zero cache thrashing |

When considered in isolation, $\approx 1.75\text{ GB}$ of RAM might sound tolerable on a modern desktop. However, placing it **directly next to PPO's lean $3.4\text{ MB}$ exposes a staggering $>500\times$ difference**:

- **The Math Behind DQN's $1.75\text{ GB}$:** Off-policy Q-learning strictly requires an Experience Replay Buffer to break temporal correlations and stabilize Bellman targets. For continuous 210-dimensional observations, storing a standard buffer of $10^6$ transitions requires:
  $$\text{DQN Memory} \approx 10^6 \times [210 \times 4\text{B (current obs)} + 210 \times 4\text{B (next obs)} + 4\text{B (action)} + 4\text{B (reward)} + 1\text{B (done)}] \approx \mathbf{1.75\text{ GB}}$$
- **The Math Behind PPO's $3.4\text{ MB}$:** PPO only stores the immediate rollout horizon ($32 \text{ arenas} \times 128 \text{ steps} = 4,096$ transitions):
  $$\text{PPO Memory} \approx 4,096 \times [210 \times 4\text{B (obs)} + \text{actions, values, logprobs, rewards}] \approx \mathbf{3.4\text{ MB}}$$
- **The I/O Bottleneck:** When 32 parallel arenas stream data at **6,000+ transitions/sec**, DQN's CPU process must continuously write high-dimensional float arrays across a massive 1.75 GB circular array while simultaneously performing random-access sampling for GPU batches. This introduces continuous cache thrashing and memory bus contention. PPO's 3.4 MB buffer, by contrast, is completely contiguous, updated linearly, and leaves the memory bus completely unburdened.

### 2. Exploration Dynamics: $\epsilon$-Greedy vs. Parameterized Entropy
- **DQN's Point-Estimate Limitation:** The DQN family learns a deterministic greedy policy $a^* = \arg\max_a Q(s, a)$. Exploration is typically injected externally via $\epsilon$-greedy heuristics (or parameter noise).
- **The Push Block Challenge:** Accidental goal completion has an astronomically low probability (our random policy benchmark logged **0 goals in 16,000 steps**). Under $\epsilon$-greedy, the agent takes isotropic, memoryless random walks that rarely maneuver behind the block and push it in a straight line. If $\epsilon$ decays before discovering the sparse goal, Q-values collapse to predicting only the $-0.001$ step penalty, and the policy freezes into a trivial local minimum (e.g., spinning or hugging walls).
- **PPO's Smooth Stochastic Landscape:** PPO explicitly optimizes a parameterized stochastic policy $\pi_\theta(a|s)$ via a `Categorical` distribution. Coupled with an **Entropy Regularization loss** ($\mathcal{H}(\pi)$), the agent maintains a continuous repulsive force against premature determinism across all 7 actions. It explores structured, smooth behavioral trajectories rather than jittery point-estimate noise.

### 3. The "Deadly Triad" and Optimization Stability
Richard Sutton’s **Deadly Triad** states that reinforcement learning algorithms can experience severe instability, bias explosion, or divergence when combining:
1. **Function Approximation** (Deep Neural Networks)
2. **Bootstrapping** (Bellman optimality updates: $r + \gamma \max_{a'} Q(s', a')$)
3. **Off-Policy Training** (Learning about greedy policy $\pi$ from historical buffer transitions $\mu$)

While Double Q-learning and target networks alleviate overestimation bias, deep Q-learning remains sensitive to distribution shift between the replay buffer and the current greedy policy—especially in high-dimensional continuous state spaces with sparse rewards.

**PPO fundamentally avoids the Deadly Triad:**
- It operates strictly **On-Policy** (data distribution matches the target policy).
- It bounds policy divergence using the **Clipped Surrogate Objective**.
- Value estimation utilizes **Generalized Advantage Estimation (GAE-$\lambda$)**, creating an optimal bias-variance trade-off without unstable bootstrapping loops.

### 4. Vectorized Architecture Alignment
- The Unity Push Block binary is architecturally structured as **32 synchronized arenas in a single engine instance**.
- PPO's execution model is a natural 1:1 match: all 32 agents step synchronously, accumulate a trajectory segment ($32 \times 128$), compute GAE advantages, run batched mini-batch updates on GPU, and proceed.
- Off-policy Q-learning on 32 synchronized envs requires awkward tuning of the update ratio (`gradient_steps` per `train_freq`). Running 32 gradient updates every step throttles simulation FPS, while updating too infrequently causes replay data staleness.

---

## Architectural Comparison Matrix

| Algorithmic Property | DQN Family (Double / Dueling) | PPO (Proximal Policy Optimization) | Push Block Production Impact |
| :--- | :--- | :--- | :--- |
| **Memory Footprint** | Massive Replay Buffer ($>1,750\text{ MB}$) | Ephemeral Rollout Buffer ($\approx 3.4\text{ MB}$) | **PPO Wins:** $>500\times$ less memory, zero cache thrashing. |
| **Exploration Mechanism** | $\epsilon$-greedy / Point-estimate noise | Parameterized `Categorical` + Entropy Bonus | **PPO Wins:** Crucial for escaping sparse reward traps. |
| **Data Throughput** | Random-access sampling bottleneck | Linear contiguous GPU mini-batches | **PPO Wins:** Maximizes the 6,000+ FPS engine stream. |
| **Deadly Triad Risk** | High (Bootstrapping + Function Approx + Off-Policy) | None (Strictly On-Policy with Clipped Surrogate) | **PPO Wins:** Guaranteed training stability & monotonic gains. |
| **Sample Reusability** | High (Transitions re-sampled many times) | Moderate (4–10 epochs on current rollout) | **DQN Advantage** (theoretical, but offset by exploration). |
| **Vectorized Env Fit** | Mismatched (synchronous 32-env buffer injection) | Native (1:1 lockstep batch rollouts) | **PPO Wins:** Seamless parallelization without synchronization lag. |

---

## Default Baseline Setup

In a disciplined reinforcement learning pipeline, we never jump blindly into extensive hyperparameter tuning right away. The standard engineering practice is to first establish an **uncalibrated default baseline**, verify that data flow and evaluation metrics are healthy, and only then proceed to systematic hyperparameter calibration and ablations.

Our baseline config is set up in **[configs/experiments/default_push_block_ppo.yaml](../configs/experiments/default_push_block_ppo.yaml)**. Rather than overthinking things early on, we stick to intuitive, well-tested defaults. We pull short rollout segments across all 32 arenas to keep updates moving fast, use surrogate clipping to stop the policy from overreacting when it finds a rare goal, and turn on a modest entropy bonus to keep the agent curiously trying new moves instead of freezing up from step penalties. Paired with GAE for smooth delayed-credit assignment and a few mini-batch passes per rollout to get the most out of our data, this gives us a clean, dependable baseline.

Once we spin up this initial run and watch how the policy behaves, we'll have real telemetry in hand to guide our next tuning iterations!