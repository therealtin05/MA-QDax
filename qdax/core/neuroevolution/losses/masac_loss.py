import functools
from typing import Callable, Tuple, Dict, List

import jax
import jax.numpy as jnp
from brax.training.distribution import ParametricDistribution

from qdax.core.neuroevolution.buffers.buffer import Transition
from qdax.types import Action, Observation, Params, RNGKey


def make_masac_loss_fn(
    policy_fns_apply: Callable[[int, Params, Observation], Action],
    critic_fn: Callable[[Params, Observation, Action], jnp.ndarray],
    parametric_action_distributions: List[ParametricDistribution],
    unflatten_obs_fn: Callable[
        [Observation],
        dict[int, jnp.ndarray]
    ],
    reward_scaling: float,
    discount: float,
    action_sizes: Dict[int, int],
    target_entropy_scale: float = 0.5,
) -> Tuple[
    Callable[[List[Params], Params, List[jnp.ndarray], Transition, RNGKey], Tuple[List[jnp.ndarray], List[jnp.ndarray]]],
    Callable[[Params, List[Params], Params, List[Params], Transition, RNGKey], jnp.ndarray],
    Callable[[List[jnp.ndarray], List[Params], Transition, RNGKey], Tuple[List[jnp.ndarray], List[jnp.ndarray]]],
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
        the loss of the entropy parameter auto-tuning
        the loss of the policy
        the loss of the critic
    """

    _policy_loss_fn = functools.partial(
        masac_policy_loss_fn,
        policy_fns_apply=policy_fns_apply,
        critic_fn=critic_fn,
        parametric_action_distributions=parametric_action_distributions,
        unflatten_obs_fn=unflatten_obs_fn
    )

    _critic_loss_fn = functools.partial(
        masac_critic_loss_fn,
        policy_fns_apply=policy_fns_apply,
        critic_fn=critic_fn,
        parametric_action_distributions=parametric_action_distributions,
        unflatten_obs_fn=unflatten_obs_fn,
        reward_scaling=reward_scaling,
        discount=discount,
    )

    _alpha_loss_fn = functools.partial(
        masac_alpha_loss_fn,
        policy_fns_apply=policy_fns_apply,
        parametric_action_distributions=parametric_action_distributions,
        unflatten_obs_fn=unflatten_obs_fn,
        action_sizes=action_sizes,
        target_entropy_scale=target_entropy_scale
    )

    return _policy_loss_fn, _critic_loss_fn, _alpha_loss_fn, 


def masac_policy_loss_fn(
    policy_params: List[Params],
    critic_params: Params,
    alpha: jnp.ndarray,
    transitions: Transition,
    random_key: RNGKey,

    policy_fns_apply: Callable[[int, Params, Observation], Action],
    critic_fn: Callable[[Params, Observation, Action], jnp.ndarray],
    parametric_action_distributions: List[ParametricDistribution],
    unflatten_obs_fn: Callable[[Observation], dict[int, jnp.ndarray]],
) -> Tuple[List[jnp.ndarray], List[Params]]:
    """Policy loss function for MASAC."""

    unflatten_obs = unflatten_obs_fn(transitions.obs)
    num_agents = len(policy_params)
    agent_keys = jax.random.split(random_key, num_agents)
    
    def single_agent_actor_loss(agent_params: Params, agent_idx: int, agent_key: RNGKey) -> jnp.ndarray:
        """Compute policy loss for a single agent"""
        
        # Get all current actions using current policy parameters
        agent_actions = []
        log_prob = None
        subkeys = jax.random.split(agent_key, num_agents)
        for i in range(num_agents):
            if i == agent_idx:
                # Use the agent_params being optimized for this agent
                dist_params = policy_fns_apply(i, agent_params, unflatten_obs[i])
                action = parametric_action_distributions[i].sample_no_postprocessing(
                    dist_params, subkeys[i]  
                )
                log_prob = parametric_action_distributions[i].log_prob(dist_params, action)
                action = parametric_action_distributions[i].postprocess(action)  # Fixed: add postprocessing
            else:
                # Use current policy_params for other agents
                dist_params = policy_fns_apply(i, policy_params[i], unflatten_obs[i])
                action = parametric_action_distributions[i].sample(
                    dist_params, subkeys[i]
                )
            agent_actions.append(action)
        
        # Flatten all actions
        flatten_actions = jnp.concatenate(agent_actions, axis=-1)
        
        q_action = critic_fn(critic_params, transitions.obs, flatten_actions)
        min_q = jnp.min(q_action, axis=-1)
        actor_loss = alpha * log_prob - min_q

        return jnp.mean(actor_loss)

    actor_losses = []
    actor_gradients = []

    for agent_idx in range(num_agents):
        actor_loss, actor_gradient = jax.value_and_grad(  # Fixed: variable names
            single_agent_actor_loss
        )(
            policy_params[agent_idx],
            agent_idx,
            agent_keys[agent_idx]
        )
        actor_losses.append(actor_loss)      # Fixed: variable names
        actor_gradients.append(actor_gradient)  # Fixed: variable names

    return actor_losses, actor_gradients


def masac_policy_loss_fn_v2(
    policy_params: List[Params],
    critic_params: Params,
    alpha: jnp.ndarray,
    transitions: Transition,
    random_key: RNGKey,

    policy_fns_apply: Callable[[int, Params, Observation], Action],
    critic_fn: Callable[[Params, Observation, Action], jnp.ndarray],
    parametric_action_distributions: List[ParametricDistribution],
    unflatten_obs_fn: Callable[[Observation], dict[int, jnp.ndarray]],
    unflatten_actions_fn: Callable[
        [Observation],
        dict[int, jnp.ndarray]
    ],
) -> Tuple[List[jnp.ndarray], List[Params]]:
    """Policy loss function for MASAC."""

    unflatten_obs = unflatten_obs_fn(transitions.obs)
    # For method 2
    unflatten_actions = unflatten_actions_fn(transitions.actions)
    num_agents = len(policy_params)
    agent_keys = jax.random.split(random_key, num_agents)
    
    def single_agent_actor_loss(agent_params: Params, agent_idx: int, agent_key: RNGKey) -> jnp.ndarray:
        """Compute policy loss for a single agent"""
        
        # Get all current actions using current policy parameters
        log_prob = None
        
        # Method 2
        new_unflatten_agent_actions =  unflatten_actions.copy()
        dist_params = policy_fns_apply(agent_idx, agent_params, unflatten_obs[agent_idx])
        action = parametric_action_distributions[agent_idx].sample_no_postprocessing(
            dist_params, agent_key 
        )
        log_prob = parametric_action_distributions[agent_idx].log_prob(dist_params, action)
        action = parametric_action_distributions[agent_idx].postprocess(action)  # Fixed: add postprocessing
        new_unflatten_agent_actions[agent_idx] = action
        # Flatten all actions
        flatten_actions = jnp.concatenate([a for a in new_unflatten_agent_actions.values()], axis=-1)
        
        q_action = critic_fn(critic_params, transitions.obs, flatten_actions)
        min_q = jnp.min(q_action, axis=-1)
        actor_loss = alpha * log_prob - min_q

        return jnp.mean(actor_loss)

    actor_losses = []
    actor_gradients = []

    for agent_idx in range(num_agents):
        actor_loss, actor_gradient = jax.value_and_grad(  # Fixed: variable names
            single_agent_actor_loss
        )(
            policy_params[agent_idx],
            agent_idx,
            agent_keys[agent_idx]
        )
        actor_losses.append(actor_loss)      # Fixed: variable names
        actor_gradients.append(actor_gradient)  # Fixed: variable names

    return actor_losses, actor_gradients


def masac_critic_loss_fn(
    critic_params: Params,
    policy_params: List[Params],
    target_critic_params: Params,
    alpha: jnp.ndarray,  # Changed: now just single alpha across agents
    transitions: Transition,
    random_key: RNGKey,

    policy_fns_apply: Callable[[int, Params, Observation], Action],
    critic_fn: Callable[[Params, Observation, Action], jnp.ndarray],
    parametric_action_distributions: List[ParametricDistribution],
    unflatten_obs_fn: Callable[[Observation], dict[int, jnp.ndarray]],
    reward_scaling: float,
    discount: float,
) -> jnp.ndarray:
    """Critic loss function for MASAC."""

    unflatten_next_obs = unflatten_obs_fn(transitions.next_obs)
    next_actions, next_log_probs = {}, {}
    
    # Split keys for each agent
    agent_keys = jax.random.split(random_key, len(policy_params))

    for agent_idx, (params, agent_obs) in enumerate(
            zip(policy_params, unflatten_next_obs.values())
    ):
        next_dist_params = policy_fns_apply(agent_idx, params, agent_obs)
        next_a = parametric_action_distributions[agent_idx].sample_no_postprocessing(
            next_dist_params, agent_keys[agent_idx]
        )
        next_lp = parametric_action_distributions[agent_idx].log_prob(
            next_dist_params, next_a
        )

        next_a = parametric_action_distributions[agent_idx].postprocess(next_a)

        next_actions[agent_idx] = next_a
        next_log_probs[agent_idx] = next_lp

    flatten_next_actions = jnp.concatenate([a for a in next_actions.values()], axis=-1)
    
    # Optimized: Use single alpha for all log probabilities
    total_log_probs = jnp.sum(jnp.array([lp for lp in next_log_probs.values()]), axis=0)
    total_entropy_term = alpha * total_log_probs

    next_q = critic_fn(target_critic_params, transitions.next_obs, flatten_next_actions)
    next_v = jnp.min(next_q, axis=-1) - total_entropy_term  # Fixed: use weighted entropy

    target_q = jax.lax.stop_gradient(
        transitions.rewards * reward_scaling
        + (1.0 - transitions.dones) * discount * next_v
    )

    q_old_action = critic_fn(critic_params, transitions.obs, transitions.actions)
    q_error = q_old_action - jnp.expand_dims(target_q, -1)
    q_error *= jnp.expand_dims(1 - transitions.truncations, -1)
    q_loss = 0.5 * jnp.mean(jnp.square(q_error))

    return q_loss


def masac_alpha_loss_fn(
    log_alpha: jnp.ndarray,  # Single log_alpha instead of list
    policy_params: List[Params],
    transitions: Transition,
    random_key: RNGKey,

    action_sizes: Dict[int, int],
    policy_fns_apply: Callable[[int, Params, Observation], Action],
    parametric_action_distributions: List[ParametricDistribution],
    unflatten_obs_fn: Callable[[Observation], dict[int, jnp.ndarray]],
    target_entropy_scale: float = 0.5,
) -> jnp.ndarray:  # Single loss
    """
    Alpha loss for single alpha across all agents.
    Target entropy is the sum of all agents' target entropies.
    """

    unflatten_obs = unflatten_obs_fn(transitions.obs)
    agent_keys = jax.random.split(random_key, len(policy_params))
    num_agents = len(policy_params)

    # Calculate combined target entropy for all agents
    total_target_entropy = -target_entropy_scale * sum(action_sizes.values())

    # Get log probabilities from all agents
    all_log_probs = []
    for agent_idx in range(num_agents):
        dist_params = policy_fns_apply(agent_idx, policy_params[agent_idx], unflatten_obs[agent_idx])
        action = parametric_action_distributions[agent_idx].sample_no_postprocessing(
            dist_params, agent_keys[agent_idx],
        )
        log_prob = parametric_action_distributions[agent_idx].log_prob(dist_params, action)
        all_log_probs.append(log_prob)

    # Sum log probabilities across all agents
    total_log_prob = jnp.sum(jnp.array(all_log_probs), axis=0)
    
    alpha = jnp.exp(log_alpha)
    alpha_loss = alpha * jax.lax.stop_gradient(-total_log_prob - total_target_entropy)

    loss = jnp.mean(alpha_loss)

    return loss