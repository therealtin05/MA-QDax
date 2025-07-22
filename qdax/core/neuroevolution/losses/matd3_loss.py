from typing import Callable, Tuple, List, Dict

import jax
import jax.numpy as jnp

from qdax.core.neuroevolution.buffers.buffer import Transition
from qdax.types import Action, Observation, Params, RNGKey

from functools import partial



def make_matd3_loss_fn(
    policy_fns_apply: Callable[[int, Params, Observation], Action],
    critic_fn: Callable[[Params, Observation, Action], jnp.ndarray],
    unflatten_obs_fn: Callable[
        [Observation],
        dict[int, jnp.ndarray]
    ],
    policy_noise: float,
    noise_clip: float,
    reward_scaling: float,
    discount: float,
) -> Tuple[
    Callable[[Params, Params, Transition], jnp.ndarray],
    Callable[[Params, Params, Params, Transition, RNGKey], jnp.ndarray],
]:
    """Creates the loss functions for TD3.

    Args:
        policy_fns_apply: forward pass through the neural network defining the policy.
        critic_fn: forward pass through the neural network defining the critic.
        reward_scaling: value to multiply the reward given by the environment.
        discount: discount factor.
        noise_clip: value that clips the noise to avoid extreme values.
        policy_noise: noise applied to smooth the bootstrapping.

    Returns:
        Return the loss functions used to train the policy and the critic in TD3.
    """

    @jax.jit
    def _policy_loss_fn(
        policy_params: List[Params],
        critic_params: Params,
        transitions: Transition,
    ) -> Tuple[List[jnp.ndarray], List[Params]]:
        """Policy loss function for TD3 agent."""

        unflatten_obs = unflatten_obs_fn(transitions.obs)
        num_agents = len(policy_params)
        
        def single_agent_policy_loss(agent_idx: int, agent_params: Params) -> jnp.ndarray:
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
            
            # Use only the first critic's Q-value (standard in TD3)
            q1_action = jnp.take(q_value, jnp.asarray([0]), axis=-1)
            
            # Policy loss is negative Q-value (we want to maximize Q)
            policy_loss = -jnp.mean(q1_action)
            
            return policy_loss
        
        # Compute losses and gradients for each agent
        policy_losses = []
        policy_gradients = []
        
        for agent_idx in range(num_agents):
            # Create a loss function that only depends on this agent's parameters
            agent_loss_fn = lambda params, idx=agent_idx: single_agent_policy_loss(idx, params)
            
            # Compute loss and gradient for this agent
            agent_loss, agent_gradient = jax.value_and_grad(agent_loss_fn)(policy_params[agent_idx])
            
            policy_losses.append(agent_loss)
            policy_gradients.append(agent_gradient)
        
        return policy_losses, policy_gradients


    @jax.jit
    def _critic_loss_fn(
        critic_params: Params,
        target_policy_params: List[Params],
        target_critic_params: Params,
        transitions: Transition,
        random_key: RNGKey,
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
        # jax.tree_util.tree_map(
        #     lambda x: print(f"obs shape {x.shape}"), 
        #     unflatten_next_obs
        # )
        next_actions = {}
        for agent_idx, (params, agent_obs) in enumerate(
                zip(
                target_policy_params, unflatten_next_obs.values()
            )
        ):
            # jax.debug.print("duma print ra cho tao obs shape: {shape}", shape=agent_obs.shape)
            a = policy_fns_apply(agent_idx, params, agent_obs)
            random_key, subkey = jax.random.split(random_key)
            noise = (jax.random.normal(subkey, shape=a.shape) * policy_noise).clip(-noise_clip, noise_clip)
            a = (a + noise).clip(-1.0, 1.0)
            next_actions[agent_idx] = a

        flatten_next_actions = jnp.concatenate([a for a in next_actions.values()], axis=-1)

        next_q = critic_fn(  # type: ignore
            target_critic_params, obs=transitions.next_obs, actions=flatten_next_actions
        )
        next_v = jnp.min(next_q, axis=-1)
        target_q = jax.lax.stop_gradient(
            transitions.rewards * reward_scaling
            + (1.0 - transitions.dones) * discount * next_v
        )
        q_old_action = critic_fn(  # type: ignore
            critic_params,
            obs=transitions.obs,
            actions=transitions.actions,
        )
        q_error = q_old_action - jnp.expand_dims(target_q, -1)

        # Better bootstrapping for truncated episodes.
        q_error = q_error * jnp.expand_dims(1.0 - transitions.truncations, -1)

        # compute the loss
        q_losses = jnp.mean(jnp.square(q_error), axis=-2)
        q_loss = jnp.sum(q_losses, axis=-1)

        return q_loss

    return _policy_loss_fn, _critic_loss_fn


def matd3_policy_loss_fn(
    policy_params: List[Params],
    critic_params: Params,
    policy_fns_apply: Callable[[int, Params, Observation], jnp.ndarray],  # Changed type
    critic_fn: Callable[[Params, Observation, Action], jnp.ndarray],
    unflatten_obs_fn: Callable[
        [Observation],
        dict[int, jnp.ndarray]
    ],
    transitions: Transition,
) -> Tuple[List[jnp.ndarray], List[Params]]:
    """Policy loss function for TD3 agent."""

    unflatten_obs = unflatten_obs_fn(transitions.obs)
    num_agents = len(policy_params)
    
    def single_agent_policy_loss(agent_idx: int, agent_params: Params) -> jnp.ndarray:
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
        
        # Use only the first critic's Q-value (standard in TD3)
        q1_action = jnp.take(q_value, jnp.asarray([0]), axis=-1)
        
        # Policy loss is negative Q-value (we want to maximize Q)
        policy_loss = -jnp.mean(q1_action)
        
        return policy_loss
    
    # Compute losses and gradients for each agent
    policy_losses = []
    policy_gradients = []
    
    for agent_idx in range(num_agents):
        # Create a loss function that only depends on this agent's parameters
        agent_loss_fn = lambda params, idx=agent_idx: single_agent_policy_loss(idx, params)
        
        # Compute loss and gradient for this agent
        agent_loss, agent_gradient = jax.value_and_grad(agent_loss_fn)(policy_params[agent_idx])
        
        policy_losses.append(agent_loss)
        policy_gradients.append(agent_gradient)
    
    return policy_losses, policy_gradients


@partial(jax.jit, static_argnames=("policy_fns_apply", "critic_fn", "unflatten_obs_fn",
                                   "policy_noise", "noise_clip", "reward_scaling",
                                   "discount"))
def matd3_critic_loss_fn(
    critic_params: Params,
    target_policy_params: List[Params],
    target_critic_params: Params,
    policy_fns_apply: Callable[[int, Params, Observation], Action],
    critic_fn: Callable[[Params, Observation, Action], jnp.ndarray],
    unflatten_obs_fn: Callable[
        [Observation],
        dict[int, jnp.ndarray]
    ],
    policy_noise: float,
    noise_clip: float,
    reward_scaling: float,
    discount: float,
    transitions: Transition,
    random_key: RNGKey,
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
        random_key, subkey = jax.random.split(random_key)
        noise = (jax.random.normal(subkey, shape=a.shape) * policy_noise).clip(-noise_clip, noise_clip)
        a = (a + noise).clip(-1.0, 1.0)
        next_actions[agent_idx] = a

    flatten_next_actions = jnp.concatenate([a for a in next_actions.values()], axis=-1)

    next_q = critic_fn(  # type: ignore
        target_critic_params, obs=transitions.next_obs, actions=flatten_next_actions
    )
    next_v = jnp.min(next_q, axis=-1)
    target_q = jax.lax.stop_gradient(
        transitions.rewards * reward_scaling
        + (1.0 - transitions.dones) * discount * next_v
    )
    q_old_action = critic_fn(  # type: ignore
        critic_params,
        obs=transitions.obs,
        actions=transitions.actions,
    )
    q_error = q_old_action - jnp.expand_dims(target_q, -1)

    # Better bootstrapping for truncated episodes.
    q_error = q_error * jnp.expand_dims(1.0 - transitions.truncations, -1)

    # compute the loss
    q_losses = jnp.mean(jnp.square(q_error), axis=-2)
    q_loss = jnp.sum(q_losses, axis=-1)

    return q_loss