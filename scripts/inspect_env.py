"""Automated MDP Profiler (inspect_env.py).

Extracts and profiles empirical MDP properties (S, A, R, T) from the actual environment binary.
Validates environment wrappers and provides quantitative data for design documentation.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import yaml

from src.envs.registry import build_vec_env
from src.utils.seed import set_seed


def inspect_environment(config_path: str, n_steps: int = 500, port: int = None):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    set_seed(seed)

    # Disable normalization to profile the raw MDP properties
    cfg["env"]["norm_obs"] = False
    cfg["env"]["norm_reward"] = False

    # Allow custom port to avoid port collisions
    if port is not None:
        cfg["env"]["base_port"] = port
    elif "base_port" not in cfg["env"]:
        cfg["env"]["base_port"] = 5020

    env_name = cfg.get("experiment_name", cfg.get("env_name", cfg.get("env", {}).get("name", "PushBlock")))
    print("=" * 70)
    print(f"MDP AUDIT REPORT: {env_name}")
    print("=" * 70)

    # Instantiate the vectorized environment
    env = build_vec_env(cfg, is_eval=False)

    try:
        # 1. Inspect Observation Space (S)
        obs_space = env.observation_space
        # 2. Inspect Action Space (A)
        act_space = env.action_space

        print("• ENVIRONMENT SPECIFICATION:")
        print(f"  - Backend               : {cfg['env']['backend']}")
        print(f"  - Parallel Arenas (N)   : {env.num_envs}")
        print(f"  - Observation Space (S) : {obs_space}")
        print(f"    * Shape               : {obs_space.shape}")
        print(f"    * Dtype               : {obs_space.dtype}")
        print(f"  - Action Space (A)      : {act_space}")
        if hasattr(act_space, "n"):
            print(f"    * Type                : Discrete({act_space.n})")
        elif hasattr(act_space, "shape"):
            print(f"    * Type                : Continuous(shape={act_space.shape}, low={act_space.low}, high={act_space.high})")

        print("\n• SAMPLING TRAJECTORIES VIA RANDOM POLICY...")
        obs = env.reset()
        obs_min = float(np.min(obs))
        obs_max = float(np.max(obs))
        has_nan = bool(np.isnan(obs).any())
        has_inf = bool(np.isinf(obs).any())

        rewards_collected = []
        ep_returns = []
        ep_lengths = []
        current_returns = np.zeros(env.num_envs, dtype=np.float32)
        current_lengths = np.zeros(env.num_envs, dtype=np.int32)
        goals_reached = 0

        start_time = time.perf_counter()

        for step in range(n_steps):
            if hasattr(act_space, "sample"):
                actions = np.array([act_space.sample() for _ in range(env.num_envs)])
            else:
                actions = np.random.randint(0, act_space.n, size=env.num_envs)

            next_obs, rewards, dones, infos = env.step(actions)

            obs_min = min(obs_min, float(np.min(next_obs)))
            obs_max = max(obs_max, float(np.max(next_obs)))
            if not has_nan and np.isnan(next_obs).any():
                has_nan = True
            if not has_inf and np.isinf(next_obs).any():
                has_inf = True

            rewards_collected.extend(rewards.tolist())
            current_returns += rewards
            current_lengths += 1

            # Detect positive sparse goal rewards (typically +1.0 or +5.0 in PushBlock)
            goals_reached += int(np.sum(rewards > 0.5))

            for i, done in enumerate(dones):
                if done:
                    ep_returns.append(float(current_returns[i]))
                    ep_lengths.append(int(current_lengths[i]))
                    current_returns[i] = 0.0
                    current_lengths[i] = 0

        elapsed = time.perf_counter() - start_time
        total_transitions = n_steps * env.num_envs
        fps = total_transitions / elapsed if elapsed > 0 else 0.0

        rewards_arr = np.array(rewards_collected, dtype=np.float32)
        unique_rewards = np.unique(np.round(rewards_arr, 5))

        print("-" * 70)
        print(f"EMPIRICAL TELEMETRY ({total_transitions:,} transitions across {env.num_envs} arenas):")
        print(f"  - Profiling Wall Time   : {elapsed:.2f} s")
        print(f"  - Sampling Throughput   : {fps:,.1f} transitions/sec (FPS)")
        print(f"  - Observed State Range  : [{obs_min:.4f}, {obs_max:.4f}]")
        print(f"    * State Integrity     : NaN={has_nan}, Inf={has_inf} (Healthy: {not has_nan and not has_inf})")
        print(f"  - Step Reward Range     : [{np.min(rewards_arr):.5f}, {np.max(rewards_arr):.5f}]")
        print(f"    * Unique Reward Values: {unique_rewards[:10]}{'...' if len(unique_rewards) > 10 else ''}")
        print(f"  - Goals Reached (Raw)   : {goals_reached}")

        if ep_returns:
            print(f"  - Terminated Episodes   : {len(ep_returns)}")
            print(f"  - Episode Return (Mean) : {np.mean(ep_returns):.3f} ± {np.std(ep_returns):.3f}")
            print(f"  - Episode Return (Range): [{np.min(ep_returns):.3f}, {np.max(ep_returns):.3f}]")
            print(f"  - Episode Length (Mean) : {np.mean(ep_lengths):.1f} ± {np.std(ep_lengths):.1f} steps")
            print(f"  - Episode Length (Range): [{np.min(ep_lengths)}, {np.max(ep_lengths)}] steps")
        else:
            print(f"  - Terminated Episodes   : 0 (Increase --steps to observe full episode terminations)")

        print("=" * 70)
        print("MDP Inspection completed successfully.")
        print("=" * 70)

    finally:
        print("Closing environment...")
        env.close()
        print("Environment closed cleanly.")


def main():
    parser = argparse.ArgumentParser(description="Empirically profile MDP properties of an RL environment.")
    parser.add_argument("--config", type=str, default="configs/envs/push_block.yaml", help="Path to config YAML file.")
    parser.add_argument("--steps", type=int, default=500, help="Number of steps per arena to sample.")
    parser.add_argument("--port", type=int, default=5020, help="Base port for environment communication.")
    args = parser.parse_args()

    inspect_environment(args.config, args.steps, args.port)


if __name__ == "__main__":
    main()
