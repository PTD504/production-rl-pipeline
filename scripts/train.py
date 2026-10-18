"""Production RL Training Pipeline (train.py).

Config-driven, multi-seed training script supporting Stable-Baselines3 algorithms,
parallel Unity/Gym environments, automated evaluation, checkpointing, and Weights & Biases tracking.
"""

import argparse
import copy
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
import torch
import yaml
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecNormalize

from src.algorithms.factory import build_algorithm

from src.envs.registry import build_vec_env
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Production RL Training Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/default_push_block_ppo.yaml",
        help="Path to experiment configuration YAML file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed for multi-seed sweeps (default: seed from config).",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Override total training timesteps (e.g. for quick sanity checking).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Base port for training environment communication (default: from config).",
    )
    parser.add_argument(
        "--eval-port",
        type=int,
        default=None,
        help="Base port for evaluation environment (default: train_port + 20).",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Disable evaluation environment and periodic evaluation callbacks.",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging entirely.",
    )
    parser.add_argument(
        "--wandb-offline",
        action="store_true",
        help="Run Weights & Biases in offline mode.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Override WandB project name.",
    )
    parser.add_argument(
        "--wandb-group",
        type=str,
        default=None,
        help="Override WandB group name (default: experiment_name for multi-seed aggregation).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to train policy on ('auto', 'cuda', 'cpu').",
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    resolved_path = Path(config_path).resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Config file not found at: {resolved_path}")
    with open(resolved_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # 1. Apply CLI Overrides
    if args.seed is not None:
        cfg["seed"] = args.seed
    seed = cfg.get("seed", 42)

    if args.total_timesteps is not None:
        cfg["train"]["total_timesteps"] = args.total_timesteps
    total_timesteps = cfg["train"].get("total_timesteps", 500000)

    train_port = args.port or cfg.get("env", {}).get("base_port", 5005)
    cfg["env"]["base_port"] = train_port
    eval_port = args.eval_port or (train_port + 20)

    project_name = args.wandb_project or cfg.get("project_name", "production-rl-pipeline")
    experiment_name = cfg.get("experiment_name", "default_push_block_ppo")
    wandb_group = args.wandb_group or experiment_name
    run_name = f"{experiment_name}_seed_{seed}"

    # 2. Setup Output Directory
    output_dir = PROJECT_ROOT / "experiments" / "outputs" / experiment_name / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save effective config snapshot for reproducibility
    with open(output_dir / "effective_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)

    print("=" * 70)
    print("PRODUCTION RL PIPELINE: TRAINING INITIALIZATION")
    print("=" * 70)
    print(f"  - Project Name     : {project_name}")
    print(f"  - Experiment Name  : {experiment_name}")
    print(f"  - Random Seed      : {seed}")
    print(f"  - Target Timesteps : {total_timesteps:,}")
    print(f"  - Device           : {args.device}")
    print(f"  - Training Port    : {train_port}")
    print(f"  - Eval Port        : {eval_port if not args.no_eval else 'Disabled'}")
    print(f"  - Output Directory : {output_dir}")
    print("=" * 70)

    # 3. Seed Everything
    set_seed(seed)

    # 4. Initialize Weights & Biases (if enabled)
    wandb_run = None
    use_wandb = not args.no_wandb and cfg.get("train", {}).get("wandb", True)
    if use_wandb:
        try:
            import wandb

            if args.wandb_offline:
                os.environ["WANDB_MODE"] = "offline"

            wandb_run = wandb.init(
                project=project_name,
                group=wandb_group,
                name=run_name,
                config=cfg,
                sync_tensorboard=True,
                job_type="train",
                tags=["baseline" if "default" in experiment_name else "tuned", cfg["algorithm"]["name"].lower()],
            )
            # Define global_step as the default x-axis for all logged metrics
            wandb.define_metric("global_step")
            wandb.define_metric("*", step_metric="global_step")
            print(f"[INFO] Weights & Biases initialized: Group='{wandb_group}' | Name='{run_name}' (X-axis: global_step)")
        except Exception as e:
            print(f"[WARN] Failed to initialize Weights & Biases ({e}). Continuing with local logging only.")
            wandb_run = None

    train_env = None
    eval_env = None

    try:
        # 5. Build Training Environment
        print("\n[1/4] Initializing Training Environment...")
        train_env = build_vec_env(cfg, is_eval=False)
        num_arenas = train_env.num_envs
        print(f"  [INFO] Training Environment Ready ({num_arenas} parallel arenas, obs={train_env.observation_space.shape}).")

        # 6. Build Optional Evaluation Environment
        eval_freq = cfg["train"].get("eval_freq", 25000)
        enable_eval = (not args.no_eval) and (eval_freq > 0)

        if enable_eval:
            print("\n[2/4] Initializing Dedicated Evaluation Environment...")
            eval_cfg = copy.deepcopy(cfg)
            eval_cfg["env"]["base_port"] = eval_port
            eval_cfg["seed"] = seed + 1000  # Separate seed for objective validation
            eval_cfg["env"]["norm_obs"] = cfg["env"].get("norm_obs", True)
            eval_cfg["env"]["norm_reward"] = False  # Track true, unscaled returns
            eval_env = build_vec_env(eval_cfg, is_eval=True)
            print(f"  [INFO] Evaluation Environment Ready on port {eval_port} (unnormalized rewards).")
        else:
            print("\n[2/4] Dedicated Evaluation Environment disabled.")

        # 7. Initialize Algorithm via Factory
        print("\n[3/4] Constructing Algorithm Policy & Optimizer...")
        tb_log_dir = str(output_dir / "tensorboard")
        model = build_algorithm(
            cfg=cfg,
            env=train_env,
            tb_log_dir=tb_log_dir,
            device=args.device,
            seed=seed,
            verbose=1,
        )

        # 8. Setup Callbacks
        print("\n[4/4] Configuring Callbacks (Checkpoints, Evaluation, WandB)...")
        callbacks_list = []

        # Periodic Checkpoint Callback
        save_freq = cfg["train"].get("save_freq", 50000)
        save_freq_steps = max(1, save_freq // num_arenas)
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_cb = CheckpointCallback(
            save_freq=save_freq_steps,
            save_path=str(checkpoint_dir),
            name_prefix="model",
            save_vecnormalize=True,
            verbose=1,
        )
        callbacks_list.append(checkpoint_cb)
        print(f"  [INFO] CheckpointCallback: every {save_freq:,} steps ({save_freq_steps} env cycles).")

        # Periodic Evaluation Callback (if enabled)
        if enable_eval and eval_env is not None:
            eval_freq_steps = max(1, eval_freq // num_arenas)
            eval_episodes = cfg["train"].get("eval_episodes", 16)
            eval_logs_dir = output_dir / "eval_logs"
            eval_logs_dir.mkdir(parents=True, exist_ok=True)
            best_model_dir = output_dir / "best_model"
            best_model_dir.mkdir(parents=True, exist_ok=True)

            eval_cb = EvalCallback(
                eval_env=eval_env,
                callback_on_new_best=None,
                best_model_save_path=str(best_model_dir),
                log_path=str(eval_logs_dir),
                eval_freq=eval_freq_steps,
                n_eval_episodes=eval_episodes,
                deterministic=True,
                render=False,
                verbose=1,
            )
            callbacks_list.append(eval_cb)
            print(f"  [INFO] EvalCallback: every {eval_freq:,} steps ({eval_freq_steps} env cycles) across {eval_episodes} episodes.")

        # WandB Metrics Callback
        if wandb_run is not None:
            try:
                from wandb.integration.sb3 import WandbCallback

                wandb_cb = WandbCallback(
                    verbose=2,
                )
                callbacks_list.append(wandb_cb)
                print("  [INFO] WandbCallback attached.")
            except Exception as e:
                print(f"  [WARN] WandbCallback skipped: {e}")

        callbacks = CallbackList(callbacks_list)

        # 9. Execute Training Loop
        print("\n" + "=" * 70)
        print(f"STARTING TRAINING LOOP ({total_timesteps:,} TIMESTEPS)")
        print("=" * 70)
        start_time = time.perf_counter()

        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=cfg["train"].get("log_interval", 10),
            reset_num_timesteps=True,
        )

        elapsed = time.perf_counter() - start_time
        fps = total_timesteps / elapsed if elapsed > 0 else 0.0

        print("\n" + "=" * 70)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print(f"  - Total Elapsed Time : {elapsed:.2f} s ({elapsed / 60:.2f} min)")
        print(f"  - Overall Throughput : {fps:,.1f} steps/sec (FPS)")
        print("=" * 70)

        # Save Final Policy & VecNormalize statistics
        final_model_path = output_dir / "final_model"
        model.save(str(final_model_path))
        print(f"[INFO] Saved final model to: {final_model_path}.zip")

        if isinstance(train_env, VecNormalize):
            vec_norm_path = output_dir / "final_vec_normalize.pkl"
            train_env.save(str(vec_norm_path))
            print(f"[INFO] Saved normalization statistics to: {vec_norm_path}")

    except KeyboardInterrupt:
        print("\n[WARN] Training interrupted by user (KeyboardInterrupt).")
        if "model" in locals() and model is not None:
            emergency_path = output_dir / "interrupted_model"
            model.save(str(emergency_path))
            print(f"[INFO] Emergency checkpoint saved to: {emergency_path}.zip")
    except Exception as e:
        print(f"\n[ERROR] Training failed with exception: {e}")
        raise e
    finally:
        print("\n[CLEANUP] Executing resource cleanup...")
        if train_env is not None:
            try:
                train_env.close()
                print("  [INFO] Training environment closed cleanly.")
            except Exception as e:
                print(f"  [WARN] Error closing train_env: {e}")

        if eval_env is not None:
            try:
                eval_env.close()
                print("  [INFO] Evaluation environment closed cleanly.")
            except Exception as e:
                print(f"  [WARN] Error closing eval_env: {e}")

        if wandb_run is not None:
            try:
                wandb_run.finish()
                print("  [INFO] Weights & Biases run closed.")
            except Exception:
                pass
        print("[CLEANUP] Done.")


if __name__ == "__main__":
    main()
