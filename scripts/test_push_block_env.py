"""Quick script to test and visually inspect the Push Block Unity environment."""

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import numpy as np
from src.envs.registry import build_vec_env


def main():
    parser = argparse.ArgumentParser(description="Test and visualize Push Block environment.")
    parser.add_argument("--config", type=str, default="configs/envs/push_block.yaml", help="Path to config file.")
    parser.add_argument("--steps", type=int, default=300, help="Number of steps to run.")
    parser.add_argument("--gui", action="store_true", help="Launch with graphical window (no_graphics=False).")
    parser.add_argument("--time-scale", type=float, default=1.0, help="Simulation speed multiplier (1.0 for real-time visual inspection).")
    args = parser.parse_args()

    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {
            "seed": 42,
            "env": {
                "backend": "unity",
                "binary_path": "binaries/Startup/Startup.x86_64",
                "scene_name": "Assets/ML-Agents/Examples/PushBlock/Scenes/PushBlock.unity",
                "time_scale": 1.0,
                "no_graphics": True,
                "base_port": 5015,
                "norm_obs": True,
                "norm_reward": False,
            },
        }

    # Override config for visual testing
    if args.gui:
        cfg["env"]["no_graphics"] = False
    cfg["env"]["time_scale"] = args.time_scale
    cfg["env"]["base_port"] = 5015  # Dedicated port for testing to prevent collision

    print("=" * 60)
    print(f"Launching Push Block (GUI: {args.gui}, Time Scale: {args.time_scale}x)...")
    print("=" * 60)

    venv = build_vec_env(cfg, is_eval=True)

    try:
        obs = venv.reset()
        print(f"Environment connected successfully!")
        print(f" - Parallel Arenas: {venv.num_envs}")
        print(f" - Observation Shape: {obs.shape}")
        print(f" - Action Space: {venv.action_space}")
        print(f"Running {args.steps} random action steps...")

        total_rewards = np.zeros(venv.num_envs)
        goal_reached_count = 0

        for step in range(args.steps):
            # Sample random discrete actions for each arena [0..6]
            actions = np.random.randint(0, venv.action_space.n, size=(venv.num_envs,))
            obs, rewards, dones, infos = venv.step(actions)

            total_rewards += rewards
            # A reward > 1 indicates the agent pushed the block to the goal!
            goal_reached_count += np.sum(rewards > 1.0)

            if (step + 1) % 50 == 0 or step == args.steps - 1:
                print(f"Step [{step + 1:3d}/{args.steps}] | Goals reached: {goal_reached_count} | Mean accumulated reward: {np.mean(total_rewards):.3f}")

            # Small sleep when running with GUI so the user can easily observe movement
            if args.gui and args.time_scale <= 1.0:
                time.sleep(0.02)

        print("=" * 60)
        print("Test finished successfully!")
        print(f"Total goals reached across 32 arenas: {goal_reached_count}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        if "Communicator has exited" in str(e):
            print("\nUnity window closed by user.")
        else:
            raise e
    finally:
        print("Closing environment...")
        try:
            venv.close()
        except Exception:
            pass
        print("Closed cleanly.")


if __name__ == "__main__":
    main()
