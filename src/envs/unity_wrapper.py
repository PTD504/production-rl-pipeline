"""Unity ML-Agents wrapper for Stable-Baselines3 VecEnv.

This module maps multi-agent scenes in a single Unity executable (e.g. PushBlock
with 32 parallel arenas) into a standard Stable-Baselines3 VecEnv.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel


class UnityToSB3VecEnv(VecEnv):
    """Wraps a Unity ML-Agents environment into an SB3-compatible VecEnv.

    Maps all parallel agent arenas inside a single Unity executable to parallel
    environments (num_envs = number of arena agents).
    """

    def __init__(
        self,
        file_name: str,
        scene_name: str = "Assets/ML-Agents/Examples/PushBlock/Scenes/PushBlock.unity",
        no_graphics: bool = True,
        time_scale: float = 20.0,
        base_port: int = 5005,
        seed: int = 42,
        additional_args: Optional[List[str]] = None,
    ):
        self.file_name = file_name
        self.scene_name = scene_name
        self.time_scale = time_scale
        self.engine_channel = EngineConfigurationChannel()
        self.engine_channel.set_configuration_parameters(time_scale=time_scale)

        args = additional_args or []
        if scene_name and "--mlagents-scene-name" not in args:
            args = ["--mlagents-scene-name", scene_name] + args

        self.env = UnityEnvironment(
            file_name=file_name,
            additional_args=args,
            no_graphics=no_graphics,
            side_channels=[self.engine_channel],
            base_port=base_port,
            seed=seed,
        )

        self.env.reset()
        self.behavior_name = list(self.env.behavior_specs.keys())[0]
        self.spec = self.env.behavior_specs[self.behavior_name]

        # Determine observation space
        total_obs_dim = sum(obs.shape[0] for obs in self.spec.observation_specs)
        observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_obs_dim,),
            dtype=np.float32,
        )

        # Determine action space
        if self.spec.action_spec.is_discrete():
            branches = self.spec.action_spec.discrete_branches
            if len(branches) == 1:
                action_space = spaces.Discrete(branches[0])
            else:
                action_space = spaces.MultiDiscrete(branches)
        elif self.spec.action_spec.is_continuous():
            action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.spec.action_spec.continuous_size,),
                dtype=np.float32,
            )
        else:
            raise ValueError(f"Unsupported action spec: {self.spec.action_spec}")

        # Get initial steps to discover all agents in the arenas
        decision_steps, _ = self.env.get_steps(self.behavior_name)
        self._agent_ids = list(decision_steps.agent_id)
        self._agent_id_to_index = {agent_id: idx for idx, agent_id in enumerate(self._agent_ids)}
        num_envs = len(self._agent_ids)

        super().__init__(num_envs=num_envs, observation_space=observation_space, action_space=action_space)

        self._actions: Optional[np.ndarray] = None
        self._last_obs = np.zeros((self.num_envs, total_obs_dim), dtype=np.float32)

    def _concat_obs(self, obs_list: Sequence[np.ndarray]) -> np.ndarray:
        """Concatenate multiple sensor observations (e.g. raycast + history) into 1D."""
        if len(obs_list) == 1:
            return obs_list[0].astype(np.float32)
        return np.concatenate([obs.astype(np.float32) for obs in obs_list], axis=-1)

    def reset(self) -> VecEnvObs:
        self.env.reset()
        decision_steps, _ = self.env.get_steps(self.behavior_name)

        obs = np.zeros((self.num_envs, self.observation_space.shape[0]), dtype=np.float32)
        # Concatenate observations across sensory streams for each agent
        full_obs = self._concat_obs(decision_steps.obs)

        for agent_id, agent_idx in self._agent_id_to_index.items():
            if agent_id in decision_steps:
                step_idx = decision_steps.agent_id_to_index[agent_id]
                obs[agent_idx] = full_obs[step_idx]

        self._last_obs = obs
        return np.copy(self._last_obs)

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = np.asarray(actions)

    def step_wait(self) -> VecEnvStepReturn:
        assert self._actions is not None, "step_async must be called before step_wait"

        # Format action tuple for Unity
        if self.spec.action_spec.is_discrete():
            discrete_actions = self._actions.reshape(self.num_envs, -1)
            action_tuple = ActionTuple(discrete=discrete_actions)
        else:
            continuous_actions = self._actions.reshape(self.num_envs, -1).astype(np.float32)
            action_tuple = ActionTuple(continuous=continuous_actions)

        self.env.set_actions(self.behavior_name, action_tuple)
        self.env.step()

        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)

        obs = np.copy(self._last_obs)
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos: List[Dict[str, Any]] = [{} for _ in range(self.num_envs)]

        # Process terminal steps first (episodes that completed or timed out)
        if len(terminal_steps) > 0:
            term_full_obs = self._concat_obs(terminal_steps.obs)
            for agent_id in terminal_steps:
                if agent_id in self._agent_id_to_index:
                    idx = self._agent_id_to_index[agent_id]
                    step_idx = terminal_steps.agent_id_to_index[agent_id]
                    terminal_agent_obs = term_full_obs[step_idx]

                    rewards[idx] = terminal_steps.reward[step_idx]
                    dones[idx] = True
                    infos[idx]["terminal_observation"] = terminal_agent_obs
                    infos[idx]["TimeLimit.truncated"] = terminal_steps.interrupted[step_idx]

        # Process decision steps (active agents and automatically reset agents)
        if len(decision_steps) > 0:
            dec_full_obs = self._concat_obs(decision_steps.obs)
            for agent_id in decision_steps:
                if agent_id in self._agent_id_to_index:
                    idx = self._agent_id_to_index[agent_id]
                    step_idx = decision_steps.agent_id_to_index[agent_id]
                    agent_obs = dec_full_obs[step_idx]

                    obs[idx] = agent_obs
                    # If the agent was not terminal, its reward comes from decision_steps
                    if not dones[idx]:
                        rewards[idx] = decision_steps.reward[step_idx]

        self._last_obs = obs
        return np.copy(obs), rewards, dones, infos

    def close(self) -> None:
        self.env.close()

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        target_indices = self._get_indices(indices)
        return [getattr(self, attr_name) for _ in target_indices]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        setattr(self, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        target_indices = self._get_indices(indices)
        method = getattr(self.env, method_name)
        result = method(*method_args, **method_kwargs)
        return [result for _ in target_indices]

    def env_is_wrapped(self, wrapper_class: Any, indices: VecEnvIndices = None) -> List[bool]:
        target_indices = self._get_indices(indices)
        return [False for _ in target_indices]


def make_unity_vec_env(
    binary_path: str,
    scene_name: str = "Assets/ML-Agents/Examples/PushBlock/Scenes/PushBlock.unity",
    time_scale: float = 20.0,
    no_graphics: bool = True,
    base_port: int = 5005,
    seed: int = 42,
) -> UnityToSB3VecEnv:
    """Factory helper to build a UnityToSB3VecEnv instance."""
    return UnityToSB3VecEnv(
        file_name=binary_path,
        scene_name=scene_name,
        time_scale=time_scale,
        no_graphics=no_graphics,
        base_port=base_port,
        seed=seed,
    )
