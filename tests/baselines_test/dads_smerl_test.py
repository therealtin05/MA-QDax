"""Testing script for the algorithm DADS"""
from functools import partial
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import pytest
from brax.v1.envs import State as EnvState

from qdax import environments
from qdax.baselines.dads import DadsTrainingState
from qdax.baselines.dads_smerl import DADSSMERL, DadsSmerlConfig
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.buffers.trajectory_buffer import TrajectoryBuffer
from qdax.core.neuroevolution.sac_td3_utils import do_iteration_fn, warmstart_buffer


def test_dads_smerl() -> None:
    """Launches and monitors the training of the agent."""

    env_name = "pointmaze"
    seed = 0
    env_batch_size = 100
    num_steps = 10000
    warmup_steps = 0
    buffer_size = 10000

    # SAC config
    batch_size = 256
    episode_length = 100
    tau = 0.005
    grad_updates_per_step = 0.25
    normalize_observations = False
    hidden_layer_sizes = (256, 256)
    alpha_init = 1.0
    fix_alpha = False
    discount = 0.97
    reward_scaling = 1.0
    learning_rate = 3e-4
    # DADS config
    num_skills = 5
    dynamics_update_freq = 1
    normalize_target = True
    descriptor_full_state = False

    # SMERL specific
    diversity_reward_scale = 2.0
    smerl_target = -200
    smerl_margin = 40

    # Initialize environments
    env_batch_size = env_batch_size
    assert (
        env_batch_size % num_skills == 0
    ), "Parameter env_batch_size should be a multiple of num_skills"
    num_env_per_skill = env_batch_size // num_skills

    env = environments.create(
        env_name=env_name,
        batch_size=env_batch_size,
        episode_length=episode_length,
        auto_reset=True,
    )

    eval_env = environments.create(
        env_name=env_name,
        batch_size=env_batch_size,
        episode_length=episode_length,
        auto_reset=True,
        eval_metrics=True,
    )

    key = jax.random.PRNGKey(seed)
    env_state = jax.jit(env.reset)(rng=key)
    eval_env_first_state = jax.jit(eval_env.reset)(rng=key)

    # Initialize buffer
    dummy_transition = QDTransition.init_dummy(
        observation_dim=env.observation_size + num_skills,
        action_dim=env.action_size,
        descriptor_dim=env.behavior_descriptor_length,
    )
    replay_buffer = TrajectoryBuffer.init(
        buffer_size=buffer_size,
        transition=dummy_transition,
        env_batch_size=env_batch_size,
        episode_length=episode_length,
    )

    if descriptor_full_state:
        descriptor_size = env.observation_size
    else:
        descriptor_size = env.behavior_descriptor_length

    dads_smerl_config = DadsSmerlConfig(
        # SAC config
        batch_size=batch_size,
        episode_length=episode_length,
        tau=tau,
        normalize_observations=normalize_observations,
        learning_rate=learning_rate,
        alpha_init=alpha_init,
        discount=discount,
        reward_scaling=reward_scaling,
        hidden_layer_sizes=hidden_layer_sizes,
        fix_alpha=fix_alpha,
        # DADS config
        num_skills=num_skills,
        descriptor_full_state=descriptor_full_state,
        omit_input_dynamics_dim=env.behavior_descriptor_length,
        dynamics_update_freq=dynamics_update_freq,
        normalize_target=normalize_target,
        # SMERL config
        diversity_reward_scale=diversity_reward_scale,
        smerl_margin=smerl_margin,
        smerl_target=smerl_target,
    )
    dads_smerl = DADSSMERL(
        config=dads_smerl_config,
        action_size=env.action_size,
        descriptor_size=env.state_descriptor_length,
    )
    training_state = dads_smerl.init(
        key,
        action_size=env.action_size,
        observation_size=env.observation_size,
        descriptor_size=descriptor_size,
    )

    skills = jnp.concatenate(
        [jnp.eye(num_skills)] * num_env_per_skill,
        axis=0,
    )

    # Make play_step* functions scannable by passing static args beforehand
    play_eval_step = partial(
        dads_smerl.play_step_fn,
        deterministic=True,
        env=eval_env,
        skills=skills,
        evaluation=True,
    )

    play_step = partial(
        dads_smerl.play_step_fn,
        env=env,
        deterministic=False,
        skills=skills,
    )

    eval_policy = partial(
        dads_smerl.eval_policy_fn,
        play_step_fn=play_eval_step,
        eval_env_first_state=eval_env_first_state,
        env_batch_size=env_batch_size,
    )

    # warmstart the buffer
    replay_buffer, env_state, training_state = warmstart_buffer(
        replay_buffer=replay_buffer,
        training_state=training_state,
        env_state=env_state,
        num_warmstart_steps=warmup_steps,
        env_batch_size=env_batch_size,
        play_step_fn=play_step,
    )

    total_num_iterations = num_steps // env_batch_size

    do_iteration = partial(
        do_iteration_fn,
        env_batch_size=env_batch_size,
        grad_updates_per_step=grad_updates_per_step,
        play_step_fn=play_step,
        update_fn=dads_smerl.update,
    )

    @jax.jit
    def _scan_do_iteration(
        carry: Tuple[DadsTrainingState, EnvState, TrajectoryBuffer],
        unused_arg: Any,
    ) -> Tuple[Tuple[DadsTrainingState, EnvState, TrajectoryBuffer], Any]:
        (
            training_state,
            env_state,
            replay_buffer,
            metrics,
        ) = do_iteration(*carry)
        return (training_state, env_state, replay_buffer), metrics

    # Training part
    (training_state, env_state, replay_buffer), (metrics) = jax.lax.scan(
        _scan_do_iteration,
        (training_state, env_state, replay_buffer),
        (),
        length=total_num_iterations,
    )

    # Evaluation part
    # Policy evaluation
    true_return, true_returns, diversity_returns, state_desc = eval_policy(
        training_state=training_state
    )

    print("True return : ", true_return)
    pytest.assume(true_return is not None)


if __name__ == "__main__":
    test_dads_smerl()
