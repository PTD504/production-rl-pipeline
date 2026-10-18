# Production Reinforcement Learning Pipeline

A modular, production-ready Reinforcement Learning (RL) pipeline engineered to solve practical decision-making and control tasks.

---

## Objective

In real-world applications, RL projects often fail not because of algorithm choice, but due to brittle pipelines, dependency conflicts, unstandardized environment interfaces, and deployment gaps. 

This repository provides an end-to-end, reproducible RL engineering framework designed to bridge the gap between academic RL research and production deployment.

---

## Environments

### 1. Unity ML-Agents: Push Block
- **Domain:** Discrete navigation & physical manipulation.
- **Task:** The agent must navigate an arena, locate a movable block, and push it into a designated green target zone.
- **Action Space:** `Discrete(7)` (forward, backward, rotate left, rotate right, strafe left, strafe right, no-op).
- **Observation Space:** 210-dimensional continuous vector (stacked Raycast perception detecting walls, blocks, and goals).
- **Rewards:** Sparse positive reward (`+5.0`) on goal completion, small per-step negative penalty (`-0.001`) to encourage speed.
- **Multi-Arena Scaling:** Wrapped via `UnityToSB3VecEnv`, utilizing **32 parallel arenas** running inside a single Unity engine process to maximize data collection throughput without multi-window overhead.

---

## Getting Started & Testing

### Installation

Clone the repository and install dependencies with the Unity extra:

```bash
# Sync core dependencies + Unity ML-Agents environment extras
uv sync --extra unity
```

### Visual Testing (GUI Mode)

To verify the Push Block environment and observe the 32 parallel arenas live in 3D:

```bash
uv run --extra unity python scripts/test_push_block_env.py --gui
```

- Opens the standalone Unity game window on your desktop screen.
- Runs at real-time simulation speed (`--time-scale 1.0`).
- Dispatches random actions across all 32 arenas to test physics, raycasts, and reward signals.

### Headless Testing

For quick non-visual sanity checking in terminal/CI:

```bash
uv run python scripts/test_push_block_env.py --steps 100
```

---

## Phase 0: Environment Profiling & Baseline Benchmarking

Before training reinforcement learning policies, the pipeline requires **Automated MDP Profiling** and **Random Policy Benchmarking** to establish quantitative environment specifications and lower-bound reference points.

### 1. Automated MDP Profiler (`inspect_env.py`)

Inspects raw environment specs directly from the executable binary without assumptions. Measures true observation bounds, action dimensions, step rewards, horizon limits, and sampling throughput (FPS):

```bash
# Run MDP inspection across 500 steps
uv run python scripts/inspect_env.py --config configs/envs/push_block.yaml --steps 500
```

- **Metrics extracted:**
  - $\mathcal{S}$ (Observation Space): shape, dtype, dynamic range, NaN/Inf anomaly check.
  - $\mathcal{A}$ (Action Space): discrete action branches or continuous bounds.
  - Throughput: actual transition throughput (typically $>2,000$ FPS across 32 arenas).
  - Rewards: step penalties vs. terminal sparse signals.

### 2. Random Baseline Benchmarker (`run_baseline.py`)

Runs a stochastic random policy over a target number of episodes to establish the empirical **Lower Bound** (Return, Episode Length, Success Rate):

```bash
# Evaluate Random Policy over 100 episodes
uv run python scripts/run_baseline.py --config configs/envs/push_block.yaml --episodes 100

# Optional: save benchmark metrics to JSON for evaluation comparisons
uv run python scripts/run_baseline.py --config configs/envs/push_block.yaml --episodes 100 --output benchmarks/random_baseline.json
```

- **Output Metrics:**
  - `Mean Episode Return ± Std`: baseline score before learning.
  - `Mean Episode Length`: empirical horizon under random walk.
  - `Success Rate`: accidental goal completion rate.

---

## Phase 1: MDP Modeling & Formulation

Before choosing an algorithm or writing training scripts, we mathematically model the Push Block environment as a Markov Decision Process (MDP) tuple $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$.

- **Observation Space ($\mathcal{S}$):** Continuous 210-dimensional vector ($\mathbb{R}^{210}$), pre-normalized in $[0.0, 1.0]$. Composed of 14 Raycast sensors $\times$ 5 attributes (3 detectable object tags + distance + hit indicator) $\times$ 3 stacked history frames to convey velocity without requiring recurrent networks (RNN/LSTM).
- **Action Space ($\mathcal{A}$):** $\text{Discrete}(7)$ actuator commands (Forward, Backward, Strafe Left, Strafe Right, Rotate Clockwise, Rotate Counter-Clockwise, and No-op).
- **Reward Function ($\mathcal{R}$):** Sparse terminal success reward of $+5.0$ upon pushing the block into the green goal zone, with an existential time-penalty of $-0.001$ per step to encourage minimum-length trajectories.
- **Exploration Bottleneck:** Random walk benchmarking logged **0 goals in 16,000 steps**, demonstrating that accidental block navigation has near-zero probability. Exploration strategy is the primary learning bottleneck.

Detailed mathematical breakdown and empirical analysis: **[docs/01_mdp_modeling.md](docs/01_mdp_modeling.md)**.

---

## Phase 2: Algorithm Selection & Decision Matrix

Algorithm selection is grounded in mathematical properties, exploration dynamics, and architectural hardware fit:

1. **Continuous Algorithms (DDPG, TD3, continuous SAC) are eliminated:** The environment strictly requires discrete actions $\text{Discrete}(7)$.
2. **PPO strictly dominates A2C:** Multi-epoch mini-batch updates on the rollout buffer yield dramatically higher sample efficiency; surrogate clipping acts as a soft trust region preventing policy collapse; and batched rollouts map 1:1 to our high-throughput 32-arena setup.
3. **The Value-Based Reality Check (Why NOT DQN):**
   - **Replay Buffer Tax:** DQN requires a massive persistent buffer ($>1,750\text{ MB}$) with random-access sampling, introducing high cache misses and memory contention on a 6,000+ FPS stream. PPO uses an ephemeral rollout buffer of only $\approx 3.4\text{ MB}$ ($>500\times$ less memory) that fits entirely in CPU L3 / GPU VRAM.
   - **Exploration in Sparse Rewards:** DQN's $\epsilon$-greedy exploration takes memoryless isotropic random walks that fail in Push Block and freeze into local minima when $\epsilon$ decays. PPO optimizes a smooth `Categorical` distribution with an explicit **Entropy Regularization bonus** ($\mathcal{H}(\pi)$) to maintain persistent multi-modal exploration.
   - **The Deadly Triad:** DQN couples function approximation, bootstrapping, and off-policy data. PPO operates strictly on-policy with clipped surrogate objectives and GAE ($\lambda = 0.95$), guaranteeing monotonic learning stability.

Detailed decision matrix and trade-off analysis: **[docs/02_algorithm_selection.md](docs/02_algorithm_selection.md)**.

---

## Phase 3: Production Training Pipeline (`train.py`)

A modular, config-driven training pipeline supporting dual-environment evaluation, automatic checkpointing, and **Weights & Biases (W&B)** multi-seed run grouping.

### 1. Configuration & Default Baseline

All experiment hyperparameters live inside YAML configurations. The baseline experiment is located at **[configs/experiments/default_push_block_ppo.yaml](configs/experiments/default_push_block_ppo.yaml)**.

- **Rollout Batching:** 128 steps per arena across 32 arenas ($4,096$ transitions per update).
- **Optimization:** 4 mini-batch epochs per rollout with batch size 128 on GPU.
- **Exploration Engine:** Entropy bonus coefficient `ent_coef: 0.01` to conquer the sparse reward landscape.
- **Trust Region:** Surrogate clipping `clip_range: 0.2`.
- **Credit Assignment:** Discount factor $\gamma = 0.99$ with GAE $\lambda = 0.95$.

### 2. Running Training

#### Standard Run with Weights & Biases Logging:
```bash
uv run python scripts/train.py --config configs/experiments/default_push_block_ppo.yaml
```

#### Multi-Seed Automated Sweep (Mean ± Std in W&B):
Run multiple seeds sequentially; W&B automatically groups them by `experiment_name` to display mean curves with shaded standard deviation:
```bash
for seed in 42 100 2024; do
    uv run python scripts/train.py --config configs/experiments/default_push_block_ppo.yaml --seed $seed
done
```

#### Local / Offline Run (Without WandB Cloud):
```bash
# Disable WandB entirely (local TensorBoard only)
uv run python scripts/train.py --config configs/experiments/default_push_block_ppo.yaml --no-wandb

# Or run WandB in offline mode for later syncing
uv run python scripts/train.py --config configs/experiments/default_push_block_ppo.yaml --wandb-offline
```

#### Fast CI / Sanity Check Run:
Run a quick test of the training and evaluation loops (e.g., 256 steps without waiting for full convergence):
```bash
uv run python scripts/train.py --total-timesteps 256 --no-wandb --port 5035 --eval-port 5045
```

### 3. Training Artifacts Structure

Every run creates an isolated, self-contained output directory under `experiments/outputs/<experiment_name>/seed_<seed>/`:
```text
experiments/outputs/default_push_block_ppo/seed_42/
├── effective_config.yaml      # Exact configuration snapshot used for this run
├── best_model/                # Checkpoint with highest evaluation return
│   └── best_model.zip
├── checkpoints/               # Periodic model snapshots (every 50,000 steps)
│   └── model_*.zip
├── eval_logs/                 # Evaluation returns and episode lengths (eval_logs.npz)
├── tensorboard/               # Local TensorBoard event logs
├── final_model.zip            # Policy saved at the end of training
└── final_vec_normalize.pkl    # Final observation normalization statistics
```

---

## Policy Evaluation (`evaluate_policy.py` & `visualize.py`)

### 1. Fast Multi-Seed Headless Benchmark (`evaluate_policy.py`)

Statistically benchmark generalization across held-out seeds at maximum simulation speed (>2,500 FPS):

```bash
uv run python scripts/evaluate_policy.py --model experiments/outputs/default_push_block_ppo/seed_42/best_model/best_model.zip --num-seeds 10
```

### 2. 3D Real-Time Visual Inspection (`visualize.py`)

Observe the policy controlling agents live in desktop 3D graphics (supports both PyTorch `.zip` and standalone `.onnx` models):

```bash
# Evaluate using PyTorch checkpoint
uv run python scripts/visualize.py --model experiments/outputs/default_push_block_ppo/seed_42/best_model/best_model.zip

# Or evaluate directly using production ONNX Runtime (zero PyTorch dependency)
uv run python scripts/visualize.py --onnx models/push_block_policy.onnx
```

---

## Phase 4: Empirical Results & Verification

We evaluated the baseline PPO configuration across **3 independent seeds** (42, 100, 2024) for 500,000 steps each, tracking all runs in Weights & Biases under the group `default_push_block_ppo`.

### Key Results Summary

- **Evaluation Return:** Achieved **4.722 ± 0.216** mean reward across all seeds, with individual checkpoints reaching **4.971** (near the theoretical ceiling of 5.0).
- **Time to Goal:** Mean episode length dropped from 1,000 steps to **101.06 ± 47.36** steps (a ~90% reduction in completion time).
- **Critic Fidelity:** `explained_variance` averaged **0.656 ± 0.141** (reaching up to 0.753), providing high-accuracy baseline predictions and minimal policy gradient noise.
- **Trust Region Stability:** `approx_kl` maintained at **0.0074 ± 0.0008** and `clip_fraction` at **6.91% ± 0.90%**, confirming optimal policy updates without divergence.
- **Held-Out Generalization (10 Unseen Seeds):** Tested across 10 novel seeds (320 episodes); achieved **100.0%** success rate, **4.9812 ± 0.0018** return, and **30.05** steps average duration in 2.67 seconds.

### Engineering Verdict: Early Acceptance

The baseline PPO configuration achieved a 100% success rate without hyperparameter tuning. Because performance reached the physical ceiling of the environment and variance across seeds remained tightly bounded, hyperparameter tuning was intentionally bypassed to optimize compute resources.

Full charts, detailed analysis, and step-by-step metric breakdown: **[docs/03_baseline_experiments.md](docs/03_baseline_experiments.md)**.
Complementary guides: **[docs/03b_reward_shaping.md](docs/03b_reward_shaping.md)** and **[docs/04_hyperparameter_tuning.md](docs/04_hyperparameter_tuning.md)**.

---

## Phase 5: Production Deployment & ONNX Export (`export_onnx.py`)

To deploy trained policies into real-world applications (Unity game engine, C# runtime, embedded robotics, or C++ microservices) without Python/PyTorch dependencies, the pipeline packages models into the **ONNX** format.

### 1. Export Command & Verification

```bash
uv run python scripts/export_onnx.py \
  --model experiments/outputs/default_push_block_ppo/seed_42/best_model/best_model.zip \
  --output models/push_block_policy.onnx
```

### 2. Key Packaging Highlights

- **Fused Normalization:** Running mean ($\mu$) and variance ($\sigma^2$) are baked directly into the ONNX graph as constant tensor buffers. Downstream clients pass raw sensor raycasts $[0.0, 1.0]$ without writing custom normalization logic.
- **Critic Pruning & Self-Contained Binary:** Discards the Value network and optimizer states, packaging weights and graph into a single **84.57 KB** file (>99.99% smaller than Python/PyTorch runtime).
- **Mathematical Parity:** Verified across PyTorch and ONNX Runtime with a max logit discrepancy of **$1.70 \times 10^{-6}$** and **100.00%** action match agreement.
- **Sub-Millisecond Latency:** Evaluated at **0.008 ms** (8 microseconds) mean latency on a single CPU core (~125,000 inferences/sec), consuming <0.05% of a 60 FPS frame budget.

Detailed deployment architecture and integration examples: **[docs/05_model_export_onnx.md](docs/05_model_export_onnx.md)**.



