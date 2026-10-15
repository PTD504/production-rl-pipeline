# Production Reinforcement Learning Pipeline

A modular, production-ready Reinforcement Learning (RL) pipeline engineered to solve practical decision-making and control tasks.

---

## 🎯 Objective

In real-world applications, RL projects often fail not because of algorithm choice, but due to brittle pipelines, dependency conflicts, unstandardized environment interfaces, and deployment gaps. 

This repository provides an end-to-end, reproducible RL engineering framework designed to bridge the gap between academic RL research and production deployment.

---

## 🎮 Environments

### 1. Unity ML-Agents: Push Block
- **Domain:** Discrete navigation & physical manipulation.
- **Task:** The agent must navigate an arena, locate a movable block, and push it into a designated green target zone.
- **Action Space:** `Discrete(7)` (forward, backward, rotate left, rotate right, strafe left, strafe right, no-op).
- **Observation Space:** 210-dimensional continuous vector (stacked Raycast perception detecting walls, blocks, and goals).
- **Rewards:** Sparse positive reward (`+5.0`) on goal completion, small per-step negative penalty (`-0.001`) to encourage speed.
- **Multi-Arena Scaling:** Wrapped via `UnityToSB3VecEnv`, utilizing **32 parallel arenas** running inside a single Unity engine process to maximize data collection throughput without multi-window overhead.

---

## 🚀 Getting Started & Testing

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

## 🔬 Phase 0: Environment Profiling & Baseline Benchmarking

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
