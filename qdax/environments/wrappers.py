import copy
from typing import Dict

import brax.v1.envs
import flax.struct
import jax
from brax.v1 import jumpy as jp
from brax.v1.envs import Env, Wrapper

from qdax.environments.locomotion_wrappers import QDSystem


class CompletedEvalMetrics(flax.struct.PyTreeNode):
    current_episode_metrics: Dict[str, jp.ndarray]
    completed_episodes_metrics: Dict[str, jp.ndarray]
    completed_episodes: jp.ndarray
    completed_episodes_steps: jp.ndarray


class CompletedEvalWrapper(brax.v1.envs.env.Wrapper):
    """Brax env with eval metrics for completed episodes."""

    STATE_INFO_KEY = "completed_eval_metrics"

    def reset(self, rng: jp.ndarray) -> brax.v1.envs.env.State:
        reset_state = self.env.reset(rng)
        reset_state.metrics["reward"] = reset_state.reward
        eval_metrics = CompletedEvalMetrics(
            current_episode_metrics=jax.tree_util.tree_map(
                jp.zeros_like, reset_state.metrics
            ),
            completed_episodes_metrics=jax.tree_util.tree_map(
                lambda x: jp.zeros_like(jp.sum(x)), reset_state.metrics
            ),
            completed_episodes=jp.zeros(()),
            completed_episodes_steps=jp.zeros(()),
        )
        reset_state.info[self.STATE_INFO_KEY] = eval_metrics
        return reset_state

    def step(
        self, state: brax.v1.envs.env.State, action: jp.ndarray
    ) -> brax.v1.envs.env.State:
        state_metrics = state.info[self.STATE_INFO_KEY]
        if not isinstance(state_metrics, CompletedEvalMetrics):
            raise ValueError(f"Incorrect type for state_metrics: {type(state_metrics)}")
        del state.info[self.STATE_INFO_KEY]
        nstate = self.env.step(state, action)
        nstate.metrics["reward"] = nstate.reward
        # steps stores the highest step reached when done = True, and then
        # the next steps becomes action_repeat
        completed_episodes_steps = state_metrics.completed_episodes_steps + jp.sum(
            nstate.info["steps"] * nstate.done
        )
        current_episode_metrics = jax.tree_util.tree_map(
            lambda a, b: a + b, state_metrics.current_episode_metrics, nstate.metrics
        )
        completed_episodes = state_metrics.completed_episodes + jp.sum(nstate.done)
        completed_episodes_metrics = jax.tree_util.tree_map(
            lambda a, b: a + jp.sum(b * nstate.done),
            state_metrics.completed_episodes_metrics,
            current_episode_metrics,
        )
        current_episode_metrics = jax.tree_util.tree_map(
            lambda a, b: a * (1 - nstate.done) + b * nstate.done,
            current_episode_metrics,
            nstate.metrics,
        )

        eval_metrics = CompletedEvalMetrics(
            current_episode_metrics=current_episode_metrics,
            completed_episodes_metrics=completed_episodes_metrics,
            completed_episodes=completed_episodes,
            completed_episodes_steps=completed_episodes_steps,
        )
        nstate.info[self.STATE_INFO_KEY] = eval_metrics
        return nstate


class GravityWrapper(Wrapper):
    def __init__(self, env: Env, gravity_multiplier: float) -> None:
        super().__init__(env)
        self._gravity_multiplier = gravity_multiplier
        config = copy.copy(self.env.sys.config)
        config.gravity.z *= gravity_multiplier
        self._config = config

        self.unwrapped.sys = QDSystem(self._config)


class ActuatorStrengthWrapper(Wrapper):
    def __init__(
        self, env: Env, actuator_name: str, strength_multiplier: float
    ) -> None:
        super().__init__(env)
        self._actuator_name = actuator_name
        self._strength_multiplier = strength_multiplier

        config = copy.copy(self.env.sys.config)

        actuators = config.actuators
        for actuator in actuators:
            if actuator.name == actuator_name:
                actuator.strength *= strength_multiplier

        self._config = config
        self.unwrapped.sys = QDSystem(self._config)
