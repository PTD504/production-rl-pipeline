"""Random Baseline Benchmarker (run_baseline.py).

Evaluates a Random Policy across a target number of episodes to establish
an empirical lower-bound benchmark for Return, Episode Length, and Success Rate.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import yaml

from src.envs.registry import build_vec_env
from src.utils.seed import set_seed


def run_baseline(
    config_path: str,
    target_episodes: int = 100,
    port: int = 5025,
    output_path: str = None,
) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    set_seed(seed)

    # Disable observation and reward normalization to evaluate true environment returns
    cfg["env"]["norm_obs"] = False
    cfg["env"]["norm_reward"] = False

    if port is not None:
        cfg["env"]["base_port"] = port
    elif "base_port" not in cfg["env"]:
        cfg["env"]["base_port"] = 5025

    env_name = cfg.get("experiment_name", cfg.get("env_name", cfg.get("env", {}).get("name", "PushBlock")))
    print("=" * 70)
    print(f"RANDOM BASELINE BENCHMARK: {env_name}")
    print(f"Target Episodes: {target_episodes} | Seed: {seed} | Arenas: {cfg['env'].get('backend')}")
    print("=" * 70)

    # Use is_eval=False or is_eval=True depending on evaluation mode.
    # Note: For PushBlock, is_eval=False runs with time_scale=20.0 (high speed),
    # which is ideal for collecting 100 episodes fast.
    env = build_vec_env(cfg, is_eval=False)

    episode_returns = []
    episode_lengths = []
    episode_successes = []

    current_returns = np.zeros(env.num_envs, dtype=np.float32)
    current_lengths = np.zeros(env.num_envs, dtype=np.int32)
    current_success = np.zeros(env.num_envs, dtype=bool)

    start_time = time.perf_counter()
    total_steps = 0

    try:
        obs = env.reset()
        act_space = env.action_space

        print(f"Arenas: {env.num_envs} parallel instances. Collecting episodes...")

        while len(episode_returns) < target_episodes:
            if hasattr(act_space, "sample"):
                actions = np.array([act_space.sample() for _ in range(env.num_envs)])
            else:
                actions = np.random.randint(0, act_space.n, size=env.num_envs)

            obs, rewards, dones, infos = env.step(actions)
            total_steps += env.num_envs

            current_returns += rewards
            current_lengths += 1

            # In PushBlock, reaching goal yields a large positive reward (> 0.5)
            # Time step penalty is typically negative.
            goal_hit = rewards > 0.5
            current_success = np.logical_or(current_success, goal_hit)

            for i, done in enumerate(dones):
                if done and len(episode_returns) < target_episodes:
                    ep_ret = float(current_returns[i])
                    ep_len = int(current_lengths[i])
                    ep_suc = bool(current_success[i] or ep_ret > 0.0)

                    episode_returns.append(ep_ret)
                    episode_lengths.append(ep_len)
                    episode_successes.append(ep_suc)

                    current_returns[i] = 0.0
                    current_lengths[i] = 0
                    current_success[i] = False

                    if len(episode_returns) % 10 == 0 or len(episode_returns) == target_episodes:
                        print(
                            f"  Progress: [{len(episode_returns):3d}/{target_episodes}] episodes | "
                            f"Current Mean Return: {np.mean(episode_returns):.3f} | "
                            f"Success Rate: {np.mean(episode_successes) * 100:.1f}%"
                        )

        elapsed = time.perf_counter() - start_time
        fps = total_steps / elapsed if elapsed > 0 else 0.0

        ret_mean = float(np.mean(episode_returns))
        ret_std = float(np.std(episode_returns))
        ret_min = float(np.min(episode_returns))
        ret_max = float(np.max(episode_returns))

        len_mean = float(np.mean(episode_lengths))
        len_std = float(np.std(episode_lengths))
        len_min = int(np.min(episode_lengths))
        len_max = int(np.max(episode_lengths))

        success_rate = float(np.mean(episode_successes))
        success_count = int(np.sum(episode_successes))

        results = {
            "environment": env_name,
            "policy": "Random Policy",
            "episodes": len(episode_returns),
            "seed": seed,
            "mean_return": round(ret_mean, 4),
            "std_return": round(ret_std, 4),
            "min_return": round(ret_min, 4),
            "max_return": round(ret_max, 4),
            "mean_length": round(len_mean, 2),
            "std_length": round(len_std, 2),
            "min_length": len_min,
            "max_length": len_max,
            "success_rate": round(success_rate, 4),
            "success_count": success_count,
            "total_transitions": total_steps,
            "wall_time_sec": round(elapsed, 2),
            "throughput_fps": round(fps, 1),
        }

        print("\n" + "=" * 70)
        print("RANDOM POLICY BASELINE REPORT (LOWER BOUND)")
        print("=" * 70)
        print(f"• Total Evaluated Episodes : {len(episode_returns)}")
        print(f"• Wall Clock Time          : {elapsed:.2f} s ({fps:,.1f} FPS)")
        print(f"• Mean Episode Return      : {ret_mean:.3f} ± {ret_std:.3f} [Min: {ret_min:.3f}, Max: {ret_max:.3f}]")
        print(f"• Mean Episode Length      : {len_mean:.1f} ± {len_std:.1f} steps [Min: {len_min}, Max: {len_max}]")
        print(f"• Success Rate             : {success_rate * 100:.2f}% ({success_count}/{len(episode_returns)} episodes)")
        print("=" * 70)

        if output_path:
            out_file = Path(output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"Benchmark results saved to: {output_path}")

        return results

    finally:
        print("Closing environment...")
        env.close()
        print("Environment closed cleanly.")


def main():
    parser = argparse.ArgumentParser(description="Run random baseline benchmark on an environment.")
    parser.add_argument("--config", type=str, default="configs/envs/push_block.yaml", help="Path to config YAML file.")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to evaluate.")
    parser.add_argument("--port", type=int, default=5025, help="Base port for environment communication.")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save JSON results.")
    args = parser.parse_args()

    run_baseline(
        config_path=args.config,
        target_episodes=args.episodes,
        port=args.port,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
