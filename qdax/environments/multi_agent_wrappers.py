from typing import Any, Dict

import jax
import jax.numpy as jnp
from brax import jumpy as jp
from brax.envs import Env, State, Wrapper

_agent_action_mapping = {
    "walker2d": {
        0: (slice(0, 3),),
        1: (slice(3, 6),),
    },
    "ant": {
        0: (slice(0, 2),),
        1: (slice(2, 4),),
        2: (slice(4, 6),),
        3: (slice(6, 8),),
    },
}

_agent_obs_mapping = {
    "walker2d": {
        0: jnp.array([0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13]),
        1: jnp.array([0, 1, 5, 6, 7, 8, 9, 10, 14, 15, 16]),
    },
    "ant": {
        0: jnp.array([0, 1, 2, 3, 4, 5, 6, 13, 14, 15, 16, 17, 18, 19, 20]),
        1: jnp.array([0, 1, 2, 3, 4, 7, 8, 13, 14, 15, 16, 17, 18, 21, 22]),
        2: jnp.array([0, 1, 2, 3, 4, 9, 10, 13, 14, 15, 16, 17, 18, 23, 24]),
        3: jnp.array([0, 1, 2, 3, 4, 11, 12, 13, 14, 15, 16, 17, 18, 25, 26]),
    },
}


class MultiAgentBraxWrapper(Wrapper):
    def __init__(self, env: Env, env_name: str, **kwargs: Any):
        self.env = env
        self.env_name = env_name
        self._kwargs = kwargs
        self.agent_action_mapping = _agent_action_mapping[env_name]
        self.agent_obs_mapping = _agent_obs_mapping[env_name]

    def step(self, state: State, agent_actions: jp.ndarray) -> State:
        global_action = self.map_agents_to_global_action(agent_actions)
        return self.env.step(state, global_action)

    def obs(self, state: State) -> Dict[int, jp.ndarray]:
        return self.map_global_obs_to_agents(state.obs)

    def get_obs_sizes(self) -> Dict[int, int]:
        return {k: v.size for k, v in self.agent_obs_mapping.items()}

    def map_agents_to_global_action(self, agent_actions: jp.ndarray) -> jp.ndarray:
        global_action = jnp.zeros(self.env.action_size)
        for agent_idx, action_indices in self.agent_action_mapping.items():
            # TODO: Remove assumption that all agents have the same action
            # space size and that each agent's actions are contiguous in the
            # global action space
            start, stop, _ = action_indices[0].indices(self.env.action_size)
            global_action = jax.lax.dynamic_update_slice(
                global_action, agent_actions[agent_idx], (start,)
            )
        return global_action

    def map_global_obs_to_agents(self, global_obs: jp.ndarray) -> Dict[int, jp.ndarray]:
        agent_obs = {}
        for agent_idx, obs_indices in self.agent_obs_mapping.items():
            agent_obs[agent_idx] = global_obs[obs_indices]
        return agent_obs

    def reset(self, key: jp.ndarray) -> State:
        return self.env.reset(key)
