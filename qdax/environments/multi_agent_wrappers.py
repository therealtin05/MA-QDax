from typing import Any, Dict, List, Tuple, Union

import jax.numpy as jnp
from brax import jumpy as jp
from brax.envs import Env, State, Wrapper

_agent_action_mapping = {
    "walker2d": {
        0: jnp.array([0, 1, 2]),
        1: jnp.array([3, 4, 5]),
    },
    "ant": {
        0: jnp.array([0, 1]),
        1: jnp.array([2, 3]),
        2: jnp.array([4, 5]),
        3: jnp.array([6, 7]),
    },
    "humanoid": {
        0: jnp.array([0, 1, 2, 11, 12, 13, 14, 15, 16]),
        1: jnp.array([3, 4, 5, 6, 7, 8, 9, 10]),
    },
}


def listerize(ranges: List[Union[int, Tuple[int, int]]]) -> List[int]:
    return [
        i
        for r in ranges
        for i in (range(r[0], r[1] + 1) if isinstance(r, tuple) else [r])
    ]


ranges: Dict[str, Dict[int, List[Union[int, Tuple[int, int]]]]] = {
    "walker2d": {
        0: [0, (2, 5), (8, 9), (11, 13)],
        1: [0, 2, (5, 9), (14, 16)],
    },
    "ant": {
        0: [(0, 5), 6, 7, 9, 11, (13, 18), 19, 20],
        1: [(0, 5), 7, 8, 9, 11, (13, 18), 21, 22],
        2: [(0, 5), 7, 9, 10, 11, (13, 18), 23, 24],
        3: [(0, 5), 7, 9, 11, 12, (13, 18), 25, 26],
    },
    "humanoid": {
        0: [
            (0, 10),
            (12, 14),
            (16, 30),
            (39, 44),
            (55, 94),
            (115, 124),
            (145, 184),
            (191, 214),
            (227, 232),
            (245, 277),
            (286, 291),
            (298, 321),
            (334, 339),
            (352, 375),
        ],
        1: [
            (0, 15),
            (22, 27),
            (31, 38),
            (85, 144),
            (209, 244),
            (269, 274),
            (278, 285),
            (316, 351),
        ],
    },
}

_agent_obs_mapping = {
    k: {k_: jnp.array(listerize(v_)) for k_, v_ in v.items()} for k, v in ranges.items()
}


class MultiAgentBraxWrapper(Wrapper):
    def __init__(self, env: Env, env_name: str, parameter_sharing: bool, **kwargs: Any):
        self.env = env
        self.env_name = env_name
        self.agent_action_mapping = _agent_action_mapping[env_name]
        self.agent_obs_mapping = _agent_obs_mapping[env_name]
        self.parameter_sharing = parameter_sharing
        self._kwargs = kwargs

    def step(self, state: State, agent_actions: Dict[int, jp.ndarray]) -> State:
        global_action = self.map_agents_to_global_action(agent_actions)
        return self.env.step(state, global_action)

    def obs(self, state: State) -> Dict[int, jp.ndarray]:
        return self.map_global_obs_to_agents(state.obs)

    def get_obs_sizes(self) -> Dict[int, int]:
        return {k: v.size for k, v in self.agent_obs_mapping.items()}

    def get_action_sizes(self) -> Dict[int, int]:
        return {k: v.size for k, v in self.agent_action_mapping.items()}

    def map_agents_to_global_action(
        self, agent_actions: Dict[int, jp.ndarray]
    ) -> jp.ndarray:
        global_action = jnp.zeros(self.env.action_size)
        for agent_idx, action_indices in self.agent_action_mapping.items():
            if self.parameter_sharing:
                global_action = global_action.at[action_indices].add(
                    agent_actions[agent_idx][action_indices]
                )
            else:
                global_action = global_action.at[action_indices].set(
                    agent_actions[agent_idx]
                )
        return global_action

    def map_global_obs_to_agents(self, global_obs: jp.ndarray) -> Dict[int, jp.ndarray]:
        agent_obs = {}
        for agent_idx, obs_indices in self.agent_obs_mapping.items():
            if self.parameter_sharing:
                # Zero vector except for the agent's own observations
                agent_obs[agent_idx] = (
                    jnp.zeros(global_obs.shape)
                    .at[obs_indices]
                    .set(global_obs[obs_indices])
                )
            else:
                # Just agent's own observations
                agent_obs[agent_idx] = global_obs[obs_indices]
        return agent_obs

    def reset(self, key: jp.ndarray) -> State:
        return self.env.reset(key)
