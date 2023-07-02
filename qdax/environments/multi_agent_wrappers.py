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
    "halfcheetah": {
        0: jnp.array([0]),
        1: jnp.array([1]),
        2: jnp.array([2]),
        3: jnp.array([3]),
        4: jnp.array([4]),
        5: jnp.array([5]),
    },
    "hopper": {
        0: jnp.array([0]),
        1: jnp.array([1]),
        2: jnp.array([2]),
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
    "halfcheetah": {
        0: [(1, 2), 3, 4, 6, (9, 11), 12],
        1: [(1, 2), 3, 4, 5, (9, 11), 13],
        2: [(1, 2), 4, 5, (9, 11), 14],
        3: [(1, 2), 3, 6, 7, (9, 11), 15],
        4: [(1, 2), 6, 7, 8, (9, 11), 16],
        5: [(1, 2), 7, 8, (9, 11), 17],
    },
    "hopper": {
        0: [(0, 1), 2, 3, (5, 7), 8],
        1: [(0, 1), 2, 3, 4, (5, 7), 9],
        2: [(0, 1), 3, 4, (5, 7), 10],
    },
}

_agent_obs_mapping = {
    k: {k_: jnp.array(listerize(v_)) for k_, v_ in v.items()} for k, v in ranges.items()
}


class MultiAgentBraxWrapper(Wrapper):
    def __init__(
        self,
        env: Env,
        env_name: str,
        parameter_sharing: bool,
        emitter_type: str,
        homogenisation_method: str,
        **kwargs: Any,
    ):
        self.env = env
        self.env_name = env_name
        self.agent_action_mapping = _agent_action_mapping[env_name]
        self.agent_obs_mapping = _agent_obs_mapping[env_name]
        self.parameter_sharing = parameter_sharing
        self.emitter_type = emitter_type
        self.homogenisation_method = homogenisation_method
        self._kwargs = kwargs

    def step(self, state: State, agent_actions: Dict[int, jp.ndarray]) -> State:
        global_action = self.map_agents_to_global_action(agent_actions)
        return self.env.step(state, global_action)

    def obs(self, state: State) -> Dict[int, jp.ndarray]:
        return self.map_global_obs_to_agents(state.obs)

    def get_obs_sizes(self) -> Dict[int, int]:
        if self.emitter_type == "shared_pool" or self.parameter_sharing:
            if self.homogenisation_method == "max":
                return {
                    k: 1 + max([v.size for v in self.agent_obs_mapping.values()])
                    for k in self.agent_obs_mapping.keys()
                }
            else:
                return {
                    k: self.env.observation_size for k in self.agent_obs_mapping.keys()
                }
        return {k: v.size for k, v in self.agent_obs_mapping.items()}

    def get_action_sizes(self) -> Dict[int, int]:
        if self.emitter_type == "shared_pool" or self.parameter_sharing:
            if self.homogenisation_method == "max":
                return {
                    k: max([v.size for v in self.agent_action_mapping.values()])
                    for k in self.agent_action_mapping.keys()
                }
            else:
                return {
                    k: self.env.action_size for k in self.agent_action_mapping.keys()
                }
        return {k: v.size for k, v in self.agent_action_mapping.items()}

    def map_agents_to_global_action(
        self, agent_actions: Dict[int, jp.ndarray]
    ) -> jp.ndarray:
        global_action = jnp.zeros(self.env.action_size)
        for agent_idx, action_indices in self.agent_action_mapping.items():
            if self.parameter_sharing or self.emitter_type == "shared_pool":
                if self.homogenisation_method == "max":
                    global_action = global_action.at[action_indices].set(
                        agent_actions[agent_idx][: action_indices.size]
                    )
                else:
                    global_action = global_action.at[action_indices].set(
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
            if self.parameter_sharing or self.emitter_type == "shared_pool":
                if self.homogenisation_method == "max":
                    # Vector with the agent idx as the first element and then the
                    # agent's own observations (zero padded to the size of the largest
                    # agent observation vector + 1)
                    agent_obs[agent_idx] = (
                        jnp.zeros(
                            1 + max([v.size for v in self.agent_obs_mapping.values()])
                        )
                        .at[0]
                        .set(agent_idx)
                        .at[1 : 1 + obs_indices.size]
                        .set(global_obs[obs_indices])
                    )
                else:
                    # Zero vector except for the agent's own observations
                    # (size of the global observation vector)
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
