"""Environment registry using Factory Pattern with Lazy Imports."""

from typing import Any, Dict
from stable_baselines3.common.vec_env import VecNormalize, VecEnv


def build_vec_env(cfg: Dict[str, Any], is_eval: bool = False) -> VecNormalize:
    """Builds and wraps an SB3 VecEnv based on configuration.

    Uses Lazy Import so dependencies for one backend (e.g. unity) are not
    imported when running another backend (e.g. gymnasium).
    """
    backend = cfg["env"]["backend"]

    if backend == "unity":
        from src.envs.unity_wrapper import make_unity_vec_env

        time_scale = 1.0 if is_eval else cfg["env"].get("time_scale", 20.0)
        no_graphics = not is_eval if "no_graphics" not in cfg["env"] else cfg["env"]["no_graphics"]
        venv: VecEnv = make_unity_vec_env(
            binary_path=cfg["env"]["binary_path"],
            scene_name=cfg["env"].get("scene_name", "Assets/ML-Agents/Examples/PushBlock/Scenes/PushBlock.unity"),
            time_scale=time_scale,
            no_graphics=no_graphics,
            base_port=cfg["env"].get("base_port", 5005),
            seed=cfg["seed"],
        )
    elif backend == "gymnasium":
        from src.envs.gym_wrapper import make_gym_vec_env

        n_envs = 1 if is_eval else cfg["env"].get("n_envs", 8)
        venv = make_gym_vec_env(
            env_id=cfg["env"]["env_id"],
            n_envs=n_envs,
            seed=cfg["seed"],
        )
    else:
        raise ValueError(f"Unsupported environment backend: '{backend}'")

    # Wrap in VecNormalize for standardizing observation and reward streams
    norm_obs = cfg["env"].get("norm_obs", True)
    norm_reward = cfg["env"].get("norm_reward", True) if not is_eval else False

    venv = VecNormalize(
        venv,
        training=not is_eval,
        norm_obs=norm_obs,
        norm_reward=norm_reward,
        clip_obs=cfg["env"].get("clip_obs", 10.0),
        clip_reward=cfg["env"].get("clip_reward", 10.0),
    )
    return venv
