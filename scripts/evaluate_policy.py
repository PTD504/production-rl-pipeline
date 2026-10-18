"""Policy Evaluation Script (evaluate_policy.py).

Performs out-of-distribution evaluation of a trained RL policy across multiple
unseen random seeds. Evaluates task success rate, return distributions, and
efficiency (episode length) in headless, fast-forward mode.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from src.envs.unity_wrapper import make_unity_vec_env
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained RL policy across unseen seeds")
    parser.add_argument(
        "--model",
        type=str,
        default="experiments/outputs/default_push_block_ppo/seed_42/best_model/best_model.zip",
        help="Path to trained model .zip file.",
    )
    parser.add_argument(
        "--vec-normalize",
        type=str,
        default=None,
        help="Path to final_vec_normalize.pkl (default: auto-detected).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/default_push_block_ppo.yaml",
        help="Path to experiment config YAML.",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=10,
        help="Number of unseen evaluation seeds (default: 10).",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=1001,
        help="Starting seed for sequential test seeds (default: 1001).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Explicit list of evaluation seeds (overrides --num-seeds and --seed-start).",
    )
    parser.add_argument(
        "--episodes-per-seed",
        type=int,
        default=32,
        help="Number of evaluation episodes to collect per seed (default: 32).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5065,
        help="Dedicated TCP base port for evaluation (default: 5065).",
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        default=20.0,
        help="Simulation speed multiplier for fast evaluation (default: 20.0).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/held_out_evaluation_seed_42.json",
        help="Output JSON path to save evaluation results.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic policy predictions (default: True).",
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


def run_evaluation_for_seed(
    model: PPO,
    env: VecNormalize,
    seed: int,
    target_episodes: int,
    deterministic: bool = True,
) -> Dict[str, Any]:
    """Evaluate a policy on a specific seed until target_episodes are completed."""
    # Seed the vectorized environment
    env.seed(seed)
    obs = env.reset()

    num_envs = env.num_envs
    episode_returns: List[float] = []
    episode_lengths: List[int] = []
    episode_successes: List[bool] = []

    current_returns = np.zeros(num_envs, dtype=np.float32)
    current_lengths = np.zeros(num_envs, dtype=np.int32)

    step_count = 0
    max_steps = target_episodes * 500  # Generous safety cutoff

    while len(episode_returns) < target_episodes and step_count < max_steps:
        actions, _ = model.predict(obs, deterministic=deterministic)
        obs, rewards, dones, infos = env.step(actions)
        step_count += 1

        current_returns += rewards
        current_lengths += 1

        for i, done in enumerate(dones):
            if done and len(episode_returns) < target_episodes:
                ret = float(current_returns[i])
                length = int(current_lengths[i])
                # In Push Block, goal gives +5.0, so a return > 1.0 indicates task success
                success = bool(ret > 1.0)

                episode_returns.append(ret)
                episode_lengths.append(length)
                episode_successes.append(success)

                current_returns[i] = 0.0
                current_lengths[i] = 0

    returns_arr = np.array(episode_returns, dtype=np.float32)
    lengths_arr = np.array(episode_lengths, dtype=np.int32)
    success_rate = float(np.mean(episode_successes)) if episode_successes else 0.0

    return {
        "seed": seed,
        "episodes_evaluated": len(episode_returns),
        "mean_reward": float(np.mean(returns_arr)),
        "std_reward": float(np.std(returns_arr)),
        "min_reward": float(np.min(returns_arr)),
        "max_reward": float(np.max(returns_arr)),
        "mean_ep_length": float(np.mean(lengths_arr)),
        "std_ep_length": float(np.std(lengths_arr)),
        "success_rate": success_rate,
        "returns": [round(float(r), 4) for r in episode_returns],
        "lengths": [int(l) for l in episode_lengths],
    }


def main():
    args = parse_args()
    model_path = Path(args.model).resolve()
    if not model_path.exists():
        fallback = model_path.parent.parent / "final_model.zip"
        if fallback.exists():
            model_path = fallback
        else:
            raise FileNotFoundError(f"Model file not found at: {model_path}")

    vec_norm_path = Path(args.vec_normalize).resolve() if args.vec_normalize else find_vec_normalize_path(model_path)

    # Determine list of evaluation seeds
    if args.seeds is not None:
        test_seeds = args.seeds
    else:
        test_seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))

    # Load environment configuration
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("HELD-OUT MULTI-SEED POLICY EVALUATION")
    print("=" * 75)
    print(f"  - Model Path          : {model_path}")
    print(f"  - Normalization Stats : {vec_norm_path.name if vec_norm_path else 'None'}")
    print(f"  - Evaluation Seeds    : {len(test_seeds)} seeds ({test_seeds})")
    print(f"  - Episodes per Seed   : {args.episodes_per_seed}")
    print(f"  - Total Episodes      : {len(test_seeds) * args.episodes_per_seed}")
    print(f"  - Dedicated Port      : {args.port}")
    print(f"  - Simulation Speed    : {args.time_scale}x (Headless)")
    print(f"  - Output JSON Report  : {output_path}")
    print("=" * 75)

    raw_env = None
    env = None

    try:
        print("\n[INFO] Initializing Headless Unity Environment...")
        raw_env = make_unity_vec_env(
            binary_path=cfg["env"]["binary_path"],
            scene_name=cfg["env"].get("scene_name", "Assets/ML-Agents/Examples/PushBlock/Scenes/PushBlock.unity"),
            time_scale=args.time_scale,
            no_graphics=True,
            base_port=args.port,
            seed=test_seeds[0],
        )

        if vec_norm_path and vec_norm_path.exists():
            print(f"[INFO] Loading normalization statistics: {vec_norm_path.name}")
            env = VecNormalize.load(str(vec_norm_path), raw_env)
            env.training = False
            env.norm_reward = False
        else:
            print("[WARN] No VecNormalize statistics found; evaluating raw observations.")
            env = raw_env

        print(f"[INFO] Loading PPO policy weights: {model_path.name}")
        model = PPO.load(str(model_path), env=env)
        print("[INFO] Policy loaded successfully. Starting evaluation across seeds...")
        print("-" * 75)

        start_time = time.perf_counter()
        seed_results: List[Dict[str, Any]] = []

        print(f"{'Seed':<8} | {'Episodes':<8} | {'Mean Reward':<16} | {'Mean Ep Length':<16} | {'Success Rate':<12}")
        print("-" * 75)

        for seed in test_seeds:
            seed_start_t = time.perf_counter()
            res = run_evaluation_for_seed(
                model=model,
                env=env,
                seed=seed,
                target_episodes=args.episodes_per_seed,
                deterministic=args.deterministic,
            )
            seed_elapsed = time.perf_counter() - seed_start_t
            seed_results.append(res)

            print(
                f"{res['seed']:<8} | "
                f"{res['episodes_evaluated']:<8} | "
                f"{res['mean_reward']:+.3f} ± {res['std_reward']:.3f}     | "
                f"{res['mean_ep_length']:>6.1f} ± {res['std_ep_length']:<5.1f}    | "
                f"{res['success_rate'] * 100:>6.1f}% ({seed_elapsed:.1f}s)"
            )

        total_elapsed = time.perf_counter() - start_time
        total_episodes = sum(r["episodes_evaluated"] for r in seed_results)

        # Compute Aggregated Statistics across all test seeds
        all_mean_rewards = [r["mean_reward"] for r in seed_results]
        all_mean_lengths = [r["mean_ep_length"] for r in seed_results]
        all_success_rates = [r["success_rate"] for r in seed_results]

        agg_summary = {
            "total_seeds": len(test_seeds),
            "total_episodes": total_episodes,
            "total_elapsed_sec": round(total_elapsed, 2),
            "episodes_per_sec": round(total_episodes / total_elapsed, 1) if total_elapsed > 0 else 0.0,
            "mean_reward": float(np.mean(all_mean_rewards)),
            "std_reward": float(np.std(all_mean_rewards)),
            "min_seed_reward": float(np.min(all_mean_rewards)),
            "max_seed_reward": float(np.max(all_mean_rewards)),
            "mean_ep_length": float(np.mean(all_mean_lengths)),
            "std_ep_length": float(np.std(all_mean_lengths)),
            "overall_success_rate": float(np.mean(all_success_rates)),
        }

        print("=" * 75)
        print("AGGREGATED EVALUATION SUMMARY ACROSS UNSEEN SEEDS")
        print("=" * 75)
        print(f"  - Total Seeds Evaluated  : {agg_summary['total_seeds']}")
        print(f"  - Total Episodes Tested  : {agg_summary['total_episodes']}")
        print(f"  - Total Evaluation Time  : {agg_summary['total_elapsed_sec']} seconds ({agg_summary['episodes_per_sec']} eps/sec)")
        print(f"  - Mean Return across Seeds: {agg_summary['mean_reward']:+.4f} ± {agg_summary['std_reward']:.4f}")
        print(f"  - Mean Episode Length    : {agg_summary['mean_ep_length']:.2f} ± {agg_summary['std_ep_length']:.2f} steps")
        print(f"  - Overall Success Rate   : {agg_summary['overall_success_rate'] * 100:.2f}%")
        print("=" * 75)

        # Save results to JSON
        final_payload = {
            "metadata": {
                "model_path": str(model_path),
                "vec_normalize_path": str(vec_norm_path) if vec_norm_path else None,
                "config_path": str(Path(args.config).resolve()),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "summary": agg_summary,
            "seed_details": seed_results,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_payload, f, indent=2)

        print(f"[INFO] Full evaluation report saved to: {output_path}")

    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")
        raise e
    finally:
        print("\n[CLEANUP] Closing Unity evaluation environment...")
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
        print("[CLEANUP] Environment closed cleanly.")


if __name__ == "__main__":
    main()
