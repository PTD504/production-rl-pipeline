"""Hyperparameter Optimization Pipeline for Continuous Control (tune.py).

Uses Optuna with Tree-structured Parzen Estimator (TPE) and Median Pruning
to find optimal hyperparameters for Soft Actor-Critic (SAC) on Unity ML-Agents Crawler.
"""

import argparse
import copy
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
import yaml
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize, sync_envs_normalization

from src.algorithms.factory import build_algorithm
from src.envs.registry import build_vec_env
from src.utils.seed import set_seed


class OptunaPruningEvalCallback(BaseCallback):
    """Callback for periodic evaluation and trial pruning via Optuna."""

    def __init__(
        self,
        eval_env,
        trial: optuna.Trial,
        eval_freq_steps: int,
        n_eval_episodes: int = 10,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.trial = trial
        self.eval_freq_steps = eval_freq_steps
        self.n_eval_episodes = n_eval_episodes
        self.eval_step = 0
        self.best_mean_reward = -float("inf")
        self.last_mean_reward = -float("inf")
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq_steps == 0:
            # Sync observation normalization statistics
            if isinstance(self.training_env, VecNormalize) and isinstance(self.eval_env, VecNormalize):
                sync_envs_normalization(self.training_env, self.eval_env)

            # Evaluate policy
            rewards, lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True,
                return_episode_rewards=True,
            )
            mean_reward = float(np.mean(rewards))
            mean_length = float(np.mean(lengths))
            self.last_mean_reward = mean_reward

            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward

            self.eval_step += 1
            print(
                f"    [TRIAL {self.trial.number} | EVAL {self.eval_step}] "
                f"Step {self.num_timesteps:,}: Return = {mean_reward:.2f} | Length = {mean_length:.1f}"
            )

            # Report intermediate evaluation metric to Optuna
            self.trial.report(mean_reward, step=self.eval_step)

            # Check if current trial should be pruned
            if self.trial.should_prune():
                print(
                    f"    [TRIAL {self.trial.number}] PRUNED by Optuna at step {self.num_timesteps:,} "
                    f"(Return: {mean_reward:.2f})"
                )
                self.is_pruned = True
                return False  # Halts SB3 training loop early

        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization via Optuna")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/default_crawler_sac.yaml",
        help="Path to base experiment configuration YAML file.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=10,
        help="Total number of Optuna trials to run.",
    )
    parser.add_argument(
        "--timesteps-per-trial",
        type=int,
        default=120000,
        help="Timesteps budget for each trial (e.g. 120,000 steps).",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=30000,
        help="Evaluation interval in environment steps.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes per evaluation checkpoint.",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="crawler_sac_optuna_study",
        help="Name of the Optuna study.",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///experiments/optuna/crawler_sac.db",
        help="Database URL for Optuna study persistence.",
    )
    parser.add_argument(
        "--base-port",
        type=int,
        default=5050,
        help="Base port for Unity environments.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampler and environments.",
    )
    parser.add_argument(
        "--output-config",
        type=str,
        default="configs/experiments/tuned_crawler_sac.yaml",
        help="Path to write the resulting best-hyperparameter experiment configuration.",
    )
    return parser.parse_args()


def sample_hyperparameters(trial: optuna.Trial) -> Dict[str, Any]:
    """Defines the search space for SAC continuous control on Crawler."""
    # 1. Learning Rate
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 7e-4, log=True)

    # 2. Mini-batch Size (Addressing gradient variance)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])

    # 3. Policy & Critic Network Architecture
    net_arch_name = trial.suggest_categorical("net_arch", ["256x256", "400x300", "512x512"])
    net_arch_map = {
        "256x256": [256, 256],
        "400x300": [400, 300],
        "512x512": [512, 512],
    }
    net_arch = net_arch_map[net_arch_name]

    # 4. Target Network Update Rate (tau)
    tau = trial.suggest_categorical("tau", [0.005, 0.01, 0.02])

    # 5. Discount Factor (gamma)
    gamma = trial.suggest_categorical("gamma", [0.98, 0.99, 0.995])

    # 6. Target Entropy Multiplier (relative to default -dim(A) = -20.0)
    target_entropy_mult = trial.suggest_categorical("target_entropy_mult", [0.5, 1.0, 1.5])
    target_entropy = -20.0 * target_entropy_mult

    return {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "policy_kwargs": {"net_arch": net_arch},
        "net_arch_name": net_arch_name,
        "tau": tau,
        "gamma": gamma,
        "target_entropy": target_entropy,
        "target_entropy_mult": target_entropy_mult,
    }


def objective(
    trial: optuna.Trial,
    base_cfg: Dict[str, Any],
    args: argparse.Namespace,
    optuna_dir: Path,
) -> float:
    """Optuna objective function for evaluating a single hyperparameter trial."""
    sampled_params = sample_hyperparameters(trial)

    # Build trial configuration
    trial_cfg = copy.deepcopy(base_cfg)
    trial_cfg["seed"] = args.seed + trial.number

    # Apply sampled hyperparameters to algorithm block
    trial_cfg["algorithm"]["learning_rate"] = sampled_params["learning_rate"]
    trial_cfg["algorithm"]["batch_size"] = sampled_params["batch_size"]
    trial_cfg["algorithm"]["policy_kwargs"] = sampled_params["policy_kwargs"]
    trial_cfg["algorithm"]["tau"] = sampled_params["tau"]
    trial_cfg["algorithm"]["gamma"] = sampled_params["gamma"]
    trial_cfg["algorithm"]["target_entropy"] = sampled_params["target_entropy"]

    # Assign distinct non-conflicting TCP ports for this trial
    train_port = args.base_port + (trial.number * 4)
    eval_port = train_port + 2
    trial_cfg["env"]["base_port"] = train_port

    print("\n" + "=" * 70)
    print(f"OPTUNA TRIAL {trial.number}/{args.n_trials} INITIALIZING")
    print("=" * 70)
    print(f"  - Learning Rate      : {sampled_params['learning_rate']:.6f}")
    print(f"  - Batch Size         : {sampled_params['batch_size']}")
    print(f"  - Network Arch       : {sampled_params['net_arch_name']}")
    print(f"  - Tau (Target Update): {sampled_params['tau']}")
    print(f"  - Gamma (Discount)   : {sampled_params['gamma']}")
    print(f"  - Target Entropy     : {sampled_params['target_entropy']} (mult={sampled_params['target_entropy_mult']})")
    print(f"  - Ports (Train/Eval) : {train_port} / {eval_port}")
    print("=" * 70)

    train_env = None
    eval_env = None
    trial_tb_dir = optuna_dir / "tensorboard" / f"trial_{trial.number}"
    trial_tb_dir.mkdir(parents=True, exist_ok=True)

    try:
        set_seed(trial_cfg["seed"])

        # 1. Build Training Environment
        train_env = build_vec_env(trial_cfg, is_eval=False)
        num_arenas = train_env.num_envs

        # 2. Build Dedicated Evaluation Environment
        eval_cfg = copy.deepcopy(trial_cfg)
        eval_cfg["env"]["base_port"] = eval_port
        eval_cfg["seed"] = trial_cfg["seed"] + 1000
        eval_cfg["env"]["norm_obs"] = trial_cfg["env"].get("norm_obs", True)
        eval_cfg["env"]["norm_reward"] = False  # Track true unscaled returns
        eval_env = build_vec_env(eval_cfg, is_eval=True)

        # 3. Instantiate Algorithm via Factory
        model = build_algorithm(
            cfg=trial_cfg,
            env=train_env,
            tb_log_dir=str(trial_tb_dir),
            device="auto",
            seed=trial_cfg["seed"],
            verbose=0,
        )

        # 4. Setup Pruning Callback
        eval_freq_steps = max(1, args.eval_freq // num_arenas)
        pruning_callback = OptunaPruningEvalCallback(
            eval_env=eval_env,
            trial=trial,
            eval_freq_steps=eval_freq_steps,
            n_eval_episodes=args.eval_episodes,
            verbose=0,
        )

        # 5. Train for trial budget
        model.learn(
            total_timesteps=args.timesteps_per_trial,
            callback=pruning_callback,
            reset_num_timesteps=True,
        )

        if pruning_callback.is_pruned:
            raise optuna.TrialPruned()

        final_score = pruning_callback.best_mean_reward
        print(f"  [TRIAL {trial.number} COMPLETE] Best Evaluation Return: {final_score:.2f}")
        return final_score

    finally:
        # Guarantee environment teardown and socket release
        if train_env is not None:
            train_env.close()
        if eval_env is not None:
            eval_env.close()
        time.sleep(1.0)  # Brief pause for OS socket release


def main():
    args = parse_args()

    # Load base configuration
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Base config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    # Setup Optuna directory
    optuna_dir = PROJECT_ROOT / "experiments" / "optuna"
    optuna_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("HYPERPARAMETER OPTIMIZATION (OPTUNA): CRAWLER SAC")
    print("=" * 70)
    print(f"  - Base Config       : {config_path.name}")
    print(f"  - Total Trials      : {args.n_trials}")
    print(f"  - Budget per Trial  : {args.timesteps_per_trial:,} steps")
    print(f"  - Eval Frequency    : {args.eval_freq:,} steps")
    print(f"  - Study Name        : {args.study_name}")
    print(f"  - Storage Backend   : {args.storage}")
    print(f"  - Output Config     : {args.output_config}")
    print("=" * 70)

    # Initialize TPE Sampler and Median Pruner
    sampler = TPESampler(seed=args.seed)
    pruner = MedianPruner(n_startup_trials=2, n_warmup_steps=1)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(trial, base_cfg, args, optuna_dir),
        n_trials=args.n_trials,
        gc_after_trial=True,
    )

    # Process and display best results
    print("\n" + "=" * 70)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION SUMMARY")
    print("=" * 70)
    print(f"  - Completed Trials  : {len(study.trials)}")
    print(f"  - Best Trial Number : {study.best_trial.number}")
    print(f"  - Best Score        : {study.best_value:.3f}")
    print("  - Best Hyperparameters:")
    for param_name, param_val in study.best_params.items():
        print(f"      * {param_name}: {param_val}")
    print("=" * 70)

    # Export Study Summary to CSV
    df_trials = study.trials_dataframe()
    csv_path = optuna_dir / f"{args.study_name}_results.csv"
    df_trials.to_csv(csv_path, index=False)
    print(f"[INFO] Detailed trial history exported to: {csv_path}")

    # Generate Tuned Experiment Configuration YAML
    best_params = study.best_params
    net_arch_map = {
        "256x256": [256, 256],
        "400x300": [400, 300],
        "512x512": [512, 512],
    }

    tuned_cfg = copy.deepcopy(base_cfg)
    tuned_cfg["experiment_name"] = "tuned_crawler_sac"
    tuned_cfg["algorithm"]["learning_rate"] = float(best_params["learning_rate"])
    tuned_cfg["algorithm"]["batch_size"] = int(best_params["batch_size"])
    tuned_cfg["algorithm"]["tau"] = float(best_params["tau"])
    tuned_cfg["algorithm"]["gamma"] = float(best_params["gamma"])
    tuned_cfg["algorithm"]["policy_kwargs"] = {
        "net_arch": net_arch_map[best_params["net_arch"]]
    }
    tuned_cfg["algorithm"]["target_entropy"] = float(-20.0 * best_params["target_entropy_mult"])

    # Set recommended scaled training timesteps (1M steps) for tuned model
    tuned_cfg["train"]["total_timesteps"] = 1000000

    output_config_path = Path(args.output_config).resolve()
    output_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(tuned_cfg, f, default_flow_style=False, sort_keys=False)

    print(f"[INFO] Ready-to-train tuned experiment config generated: {output_config_path}")


if __name__ == "__main__":
    main()
