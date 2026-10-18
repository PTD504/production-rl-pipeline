"""Algorithm Factory and Registry.

Enables extensible, config-driven algorithm instantiation (PPO, SAC, TD3)
without hardcoded if/else branching in training pipelines.
"""

import inspect
from typing import Any, Dict, Type

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv

# Central registry mapping algorithm names to their SB3 implementation classes
ALGORITHM_REGISTRY: Dict[str, Type[BaseAlgorithm]] = {
    "PPO": PPO,
    "SAC": SAC,
    "TD3": TD3,
}


def build_algorithm(
    cfg: Dict[str, Any],
    env: VecEnv,
    tb_log_dir: str,
    device: str = "auto",
    seed: int = 42,
    verbose: int = 1,
) -> BaseAlgorithm:
    """Builds and initializes an RL algorithm model from configuration.

    Uses signature inspection to automatically extract and validate hyperparameter
    kwargs matching the target algorithm's constructor, adhering to the Open-Closed Principle.

    Args:
        cfg: Configuration dictionary containing the 'algorithm' block.
        env: Vectorized training environment (VecEnv).
        tb_log_dir: Path for TensorBoard telemetry logging.
        device: Device to place tensors on ('auto', 'cuda', 'cpu').
        seed: Random seed for policy and action space.
        verbose: Verbosity level for Stable-Baselines3 (default: 1).

    Returns:
        Instantiated BaseAlgorithm model ready for training.
    """
    algo_cfg = cfg.get("algorithm", {})
    algo_name = algo_cfg.get("name", "").upper()

    if algo_name not in ALGORITHM_REGISTRY:
        available = list(ALGORITHM_REGISTRY.keys())
        raise ValueError(
            f"Algorithm '{algo_name}' is not registered in ALGORITHM_REGISTRY. "
            f"Available algorithms: {available}"
        )

    algo_cls = ALGORITHM_REGISTRY[algo_name]
    policy_name = algo_cfg.get("policy", "MlpPolicy")

    # Inspect constructor signature to dynamically extract valid arguments
    sig = inspect.signature(algo_cls.__init__)
    valid_params = set(sig.parameters.keys())

    # Filter kwargs from config, ignoring metadata fields
    ignored_keys = {"name", "policy"}
    algo_kwargs = {}
    for key, value in algo_cfg.items():
        if key in valid_params and key not in ignored_keys:
            # Handle float conversions for scientific notation if parsed as string
            if isinstance(value, str):
                try:
                    value = float(value)
                except ValueError:
                    pass  # Keep string values like 'auto' for ent_coef
            algo_kwargs[key] = value

    print(f"  [FACTORY] Initializing {algo_name} ({policy_name})")
    print(f"  [FACTORY] Injected Hyperparameters: {algo_kwargs}")

    # Common parameters shared across BaseAlgorithm implementations
    common_kwargs = {
        "policy": policy_name,
        "env": env,
        "tensorboard_log": tb_log_dir,
        "device": device,
        "seed": seed,
        "verbose": verbose,
    }

    # Merge common kwargs with algorithm-specific kwargs
    merged_kwargs = {**common_kwargs, **algo_kwargs}

    return algo_cls(**merged_kwargs)
