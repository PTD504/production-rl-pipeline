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
uv run --extra unity python scripts/test_push_block_env.py --steps 100
```
