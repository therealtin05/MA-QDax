from typing import Callable, Tuple, List, Dict

import jax
import jax.numpy as jnp

from qdax.core.neuroevolution.buffers.buffer import Transition
from qdax.types import Action, Observation, Params, RNGKey

from functools import partial

def make_maddpg_loss_fn(
    policy_fns_apply: Callable[[int, Params, Observation], Action],
    critic_fn: Callable[[Params, Observation, Action], jnp.ndarray],
    unflatten_obs_fn: Callable[
        [Observation],
        dict[int, jnp.ndarray]
    ],
    reward_scaling: float,
    discount: float,
) -> Tuple[
    Callable[[List[Params], Params, Transition], jnp.ndarray],
    Callable[[Params, List[Params], Params, Transition, RNGKey], jnp.ndarray],
]:
    """Creates the loss used in SAC.

    Args:
        policy_fn: the apply function of the policy
        critic_fn: the apply function of the critic
        parametric_action_distributions: the distribution over actions
        reward_scaling: a multiplicative factor to the reward
        discount: the discount factor
        action_size: the size of the environment's action space

    Returns:
        the loss of the policy
        the loss of the critic
    """

    _policy_loss_fn = partial(
        maddpg_policy_loss_fn,
        policy_fns_apply=policy_fns_apply,
        critic_fn=critic_fn,
        unflatten_obs_fn=unflatten_obs_fn
    )

    _critic_loss_fn = partial(
        maddpg_critic_loss_fn,
        policy_fns_apply=policy_fns_apply,
        critic_fn=critic_fn,
        unflatten_obs_fn=unflatten_obs_fn,
        reward_scaling=reward_scaling,
        discount=discount,
    )

    return _policy_loss_fn, _critic_loss_fn


def maddpg_policy_loss_fn(
    policy_params: List[Params],
    critic_params: Params,
    transitions: Transition,

    policy_fns_apply: Callable[[int, Params, Observation], jnp.ndarray],  # Changed type
    critic_fn: Callable[[Params, Observation, Action], jnp.ndarray],
    unflatten_obs_fn: Callable[
        [Observation],
        dict[int, jnp.ndarray]
    ],
    unflatten_actions_fn: Callable[
        [Observation],
        dict[int, jnp.ndarray]
    ],
) -> Tuple[List[jnp.ndarray], List[Params]]:
    """Policy loss function for MADDPG agent."""

    unflatten_obs = unflatten_obs_fn(transitions.obs)
    num_agents = len(policy_params)
    
    def single_agent_policy_loss(agent_params: Params, agent_idx: int) -> jnp.ndarray:
        """Compute policy loss for a single agent"""
        
        # Get all current actions using current policy parameters
        agent_actions = []
        for i in range(num_agents):
            if i == agent_idx:
                # Use the agent_params being optimized for this agent
                action = policy_fns_apply(i, agent_params, unflatten_obs[i])
            else:
                # Use current policy_params for other agents
                action = policy_fns_apply(i, policy_params[i], unflatten_obs[i])
            agent_actions.append(action)
        
        # Flatten all actions
        flatten_actions = jnp.concatenate(agent_actions, axis=-1)
        
        # Get Q-value using the critic
        q_value = critic_fn(
            critic_params, obs=transitions.obs, actions=flatten_actions
        )
        q_value = jnp.squeeze(q_value, axis=-1)       

        # Policy loss is negative Q-value (we want to maximize Q)
        policy_loss = -jnp.mean(q_value)
        
        return policy_loss
    
    # Compute losses and gradients for each agent
    policy_losses = []
    policy_gradients = []
    
    for agent_idx in range(num_agents):
        # Compute loss and gradient for this agent
        agent_loss, agent_gradient = jax.value_and_grad(single_agent_policy_loss)(
            policy_params[agent_idx], agent_idx
        )

        policy_losses.append(agent_loss)
        policy_gradients.append(agent_gradient)
    
    return policy_losses, policy_gradients


def maddpg_policy_loss_fn_v2(
    policy_params: List[Params],
    critic_params: Params,
    transitions: Transition,

    policy_fns_apply: Callable[[int, Params, Observation], jnp.ndarray],  # Changed type
    critic_fn: Callable[[Params, Observation, Action], jnp.ndarray],
    unflatten_obs_fn: Callable[
        [Observation],
        dict[int, jnp.ndarray]
    ],
    unflatten_actions_fn: Callable[
        [Observation],
        dict[int, jnp.ndarray]
    ],
) -> Tuple[List[jnp.ndarray], List[Params]]:
    """Policy loss function for TD3 agent."""

    unflatten_obs = unflatten_obs_fn(transitions.obs)
    unflatten_actions = unflatten_actions_fn(transitions.actions)
    num_agents = len(policy_params)
    
    def single_agent_policy_loss(agent_params: Params, agent_idx: int) -> jnp.ndarray:
        """Compute policy loss for a single agent"""
        
        new_unflatten_agent_actions =  unflatten_actions.copy()
        new_unflatten_agent_actions[agent_idx] = policy_fns_apply(agent_idx, agent_params, unflatten_obs[agent_idx])
        # Flatten all actions
        flatten_actions = jnp.concatenate([a for a in new_unflatten_agent_actions.values()], axis=-1)
        
        # Get Q-value using the critic
        q_value = critic_fn(
            critic_params, obs=transitions.obs, actions=flatten_actions
        )
        q_value = jnp.squeeze(q_value, axis=-1)        

        # Policy loss is negative Q-value (we want to maximize Q)
        policy_loss = -jnp.mean(q_value)
        
        return policy_loss
    
    # Compute losses and gradients for each agent
    policy_losses = []
    policy_gradients = []
    
    for agent_idx in range(num_agents):
        # Compute loss and gradient for this agent
        agent_loss, agent_gradient = jax.value_and_grad(single_agent_policy_loss)(
            policy_params[agent_idx], agent_idx
        )

        policy_losses.append(agent_loss)
        policy_gradients.append(agent_gradient)
    
    return policy_losses, policy_gradients


@partial(jax.jit, static_argnames=("policy_fns_apply", "critic_fn", "unflatten_obs_fn",
                                   "reward_scaling", "discount"))
def maddpg_critic_loss_fn(
    critic_params: Params,
    target_policy_params: List[Params],
    target_critic_params: Params,
    transitions: Transition,
    random_key: RNGKey,

    policy_fns_apply: Callable[[int, Params, Observation], Action],
    critic_fn: Callable[[Params, Observation, Action], jnp.ndarray],
    unflatten_obs_fn: Callable[
        [Observation],
        dict[int, jnp.ndarray]
    ],
    reward_scaling: float,
    discount: float,
) -> jnp.ndarray:
    """Critics loss function for TD3 agent.

    Args:
        critic_params: critic parameters.
        target_policy_params: target policy parameters.
        target_critic_params: target critic parameters.
        policy_fns: forward pass through the neural network defining the policy.
        critic_fn: forward pass through the neural network defining the critic.
        policy_noise: policy noise.
        noise_clip: noise clip.
        reward_scaling: reward scaling coefficient.
        discount: discount factor.
        transitions: collected transitions.

    Returns:
        Return the loss function used to train the critic in TD3.
    """
    unflatten_next_obs = unflatten_obs_fn(transitions.next_obs)

    next_actions = {}
    for agent_idx, (params, agent_obs) in enumerate(
            zip(
            target_policy_params, unflatten_next_obs.values()
        )
    ):
        a = policy_fns_apply(agent_idx, params, agent_obs)
        next_actions[agent_idx] = a

    flatten_next_actions = jnp.concatenate([a for a in next_actions.values()], axis=-1)

    next_q = critic_fn(  # type: ignore
        target_critic_params, obs=transitions.next_obs, actions=flatten_next_actions
    )
    next_v = jnp.squeeze(next_q, axis=-1)
    target_q = jax.lax.stop_gradient(
        transitions.rewards * reward_scaling
        + (1.0 - transitions.dones) * discount * next_v
    )

    q_old_action = critic_fn(  # type: ignore
        critic_params,
        obs=transitions.obs,
        actions=transitions.actions,
    )
    q_old_action = jnp.squeeze(q_old_action, axis=-1)
    q_error = q_old_action - target_q

    # Better bootstrapping for truncated episodes.
    q_error = q_error * (1.0 - transitions.truncations)

    # compute the loss
    q_loss = jnp.mean(jnp.square(q_error), axis=-1)

    return q_loss