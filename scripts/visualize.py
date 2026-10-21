"""Visual Evaluation Script (visualize.py).

Renders and evaluates a trained RL policy in real-time 3D GUI mode.
Loads the trained model weights and observation normalization statistics,
then steps the policy across all 32 arenas at 1.0x human-observable speed.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import yaml
from stable_baselines3.common.vec_env import VecNormalize

from src.algorithms.factory import ALGORITHM_REGISTRY
from src.envs.unity_wrapper import make_unity_vec_env
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize trained RL policy in Unity GUI")
    parser.add_argument(
        "--model",
        type=str,
        default="experiments/outputs/default_push_block_ppo/seed_42/best_model/best_model.zip",
        help="Path to trained model .zip file.",
    )
    parser.add_argument(
        "--onnx",
        type=str,
        default=None,
        help="Path to exported ONNX model (.onnx). If provided, runs directly via ONNX Runtime without PyTorch.",
    )
    parser.add_argument(
        "--vec-normalize",
        type=str,
        default=None,
        help="Path to final_vec_normalize.pkl (default: auto-detected from model directory).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/default_push_block_ppo.yaml",
        help="Path to environment config YAML.",
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="Simulation speed multiplier (1.0 for realistic real-time viewing).",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run headless without opening Unity graphical window.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5055,
        help="Dedicated TCP port for visualization to prevent collision.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Maximum timesteps to visualize.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions stochastically from policy distribution instead of taking deterministic mean.",
    )
    return parser.parse_args()


def find_vec_normalize_path(model_path: Path) -> Optional[Path]:
    """Search for vec_normalize.pkl in parent directories of the model."""
    candidates = [
        model_path.parent / "final_vec_normalize.pkl",
        model_path.parent.parent / "final_vec_normalize.pkl",
        model_path.parent / "vec_normalize.pkl",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def main():
    args = parse_args()

    is_onnx = bool(args.onnx or (args.model and args.model.endswith(".onnx")))
    onnx_path = Path(args.onnx or args.model).resolve() if is_onnx else None

    if not is_onnx:
        model_path = Path(args.model).resolve()
        if not model_path.exists():
            # Fallback check for final_model.zip
            fallback = model_path.parent.parent / "final_model.zip"
            if fallback.exists():
                model_path = fallback
            else:
                raise FileNotFoundError(f"Model file not found at: {model_path}")

        # Auto-detect VecNormalize stats
        vec_norm_path = Path(args.vec_normalize).resolve() if args.vec_normalize else find_vec_normalize_path(model_path)
    else:
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model file not found at: {onnx_path}")
        model_path = onnx_path
        vec_norm_path = None

    # Load environment config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    set_seed(seed)

    no_graphics = args.no_gui
    time_scale = args.time_scale

    exp_name = cfg.get("experiment_name", cfg.get("env", {}).get("name", "RL Agent"))
    print("=" * 70)
    print(f"VISUAL POLICY EVALUATION: {exp_name.upper()}")
    print("=" * 70)
    print(f"  - Engine           : {'ONNX Runtime (Standalone Model)' if is_onnx else 'Stable-Baselines3 (PyTorch)'}")
    print(f"  - Model Path       : {model_path}")
    print(f"  - Normalization    : {'Embedded in ONNX Graph (Fused)' if is_onnx else (vec_norm_path.name if vec_norm_path else 'None (Raw observations)')}")
    print(f"  - GUI Window       : {'Enabled (Desktop Screen)' if not no_graphics else 'Headless'}")
    print(f"  - Time Scale       : {time_scale}x (Real-time physical speed)")
    print(f"  - Dedicated Port   : {args.port}")
    print(f"  - Max Steps        : {args.steps}")
    print("=" * 70)

    raw_env = None
    env = None

    try:
        print("\n[INFO] Initializing Unity Environment...")
        raw_env = make_unity_vec_env(
            binary_path=cfg["env"]["binary_path"],
            scene_name=cfg["env"].get("scene_name", "Assets/ML-Agents/Examples/PushBlock/Scenes/PushBlock.unity"),
            time_scale=time_scale,
            no_graphics=no_graphics,
            base_port=args.port,
            seed=seed,
        )

        session = None
        model = None

        if is_onnx:
            import onnxruntime as ort

            print(f"[INFO] Initializing ONNX Runtime Session from: {onnx_path.name}")
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
            input_name = session.get_inputs()[0].name
            # Standalone ONNX model already has fused observation normalization, so it operates on raw observations directly
            env = raw_env
            print("[INFO] ONNX model loaded (fused normalization). Starting live demonstration...")
        else:
            if vec_norm_path and vec_norm_path.exists():
                print(f"[INFO] Loading observation normalization statistics from: {vec_norm_path.name}")
                env = VecNormalize.load(str(vec_norm_path), raw_env)
                env.training = False
                env.norm_reward = False
            else:
                print("[WARN] No VecNormalize file found; using raw observations.")
                env = raw_env

            algo_name = cfg.get("algorithm", {}).get("name")
            if not algo_name:
                if "sac" in str(model_path).lower():
                    algo_name = "SAC"
                elif "td3" in str(model_path).lower():
                    algo_name = "TD3"
                else:
                    algo_name = "PPO"
            algo_name = algo_name.upper()
            algo_cls = ALGORITHM_REGISTRY.get(algo_name)
            if algo_cls is None:
                raise ValueError(f"Algorithm '{algo_name}' not supported for visualization. Available: {list(ALGORITHM_REGISTRY.keys())}")

            print(f"[INFO] Loading trained {algo_name} policy from: {model_path.name}")
            model = algo_cls.load(str(model_path), env=env)
            print(f"[INFO] {algo_name} Policy loaded successfully. Starting live demonstration...")

        print("-" * 70)

        obs = env.reset()
        goals_reached = 0
        cumulative_rewards = np.zeros(env.num_envs, dtype=np.float32)
        episodes_completed = 0

        for step in range(args.steps):
            # Predict action from trained policy
            if is_onnx:
                outputs = session.run(None, {input_name: obs.astype(np.float32)})
                actions = outputs[0]
            else:
                actions, _ = model.predict(obs, deterministic=not args.stochastic)

            obs, rewards, dones, infos = env.step(actions)

            cumulative_rewards += rewards
            is_push_block = "pushblock" in cfg.get("env", {}).get("scene_name", "").lower()

            # Detect goal triggers for goal-reaching environments like PushBlock
            if is_push_block:
                step_goals = int(np.sum(rewards > 1.0))
                if step_goals > 0:
                    goals_reached += step_goals
                    print(f"[GOAL!] Step {step + 1:4d} | Arena scored a goal! Total Goals: {goals_reached}")

            for i, done in enumerate(dones):
                if done:
                    episodes_completed += 1
                    ep_return = float(cumulative_rewards[i])
                    cumulative_rewards[i] = 0.0
                    if is_push_block:
                        print(f"[EPISODE COMPLETED] Episode {episodes_completed:3d} (Arena {i:2d}) | Return: {ep_return:+.3f} | Total Goals: {goals_reached}")
                    else:
                        print(f"[EPISODE COMPLETED] Episode {episodes_completed:3d} (Arena {i:2d}) | Return: {ep_return:+.3f} | Step: {step + 1}")

            # Sleep slightly if real-time GUI mode to ensure silky smooth 50-60 FPS visual rendering
            if not no_graphics and time_scale <= 1.0:
                time.sleep(0.02)

        print("-" * 70)
        print("[INFO] Live demonstration completed.")
        if is_push_block:
            print(f"  - Total Goals Scored  : {goals_reached}")
        print(f"  - Episodes Completed  : {episodes_completed}")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n[INFO] Visualization interrupted by user.")
    except Exception as e:
        if "Communicator has exited" in str(e):
            print("\n[INFO] Unity window closed by user.")
        else:
            print(f"\n[ERROR] Visualization failed: {e}")
            raise e
    finally:
        print("\n[CLEANUP] Closing environment...")
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        elif raw_env is not None:
            try:
                raw_env.close()
            except Exception:
                pass
        print("[CLEANUP] Unity closed cleanly.")


if __name__ == "__main__":
    main()
