from typing import Tuple, Dict

import flax.linen as nn
import jax.numpy as jnp

from qdax.core.neuroevolution.networks.networks import MLP
from qdax.custom_types import Action, Observation


class MultiAgentCritic(nn.Module):
    hidden_layer_size: Tuple[int, ...]
    use_layer_norm: bool = False
    @nn.compact
    def __call__(self, obs: Observation, action: Action) -> jnp.ndarray:
        input_ = jnp.concatenate([obs, action], axis=-1)

        # kernel_init = nn.initializers.variance_scaling(1.0, "fan_in", "uniform")
        # kernel_init = nn.initializers.orthogonal(jnp.sqrt(2))

        value_1 = MLP(
            layer_sizes=self.hidden_layer_size + (1,),
            # kernel_init=kernel_init,
            activation=nn.leaky_relu,
            # kernel_init_final=nn.initializers.orthogonal(0.01)
            use_layer_norm=self.use_layer_norm,
        )(input_)

        value_2 = MLP(
            layer_sizes=self.hidden_layer_size + (1,),
            # kernel_init=kernel_init,
            activation=nn.leaky_relu,
            # kernel_init_final=nn.initializers.orthogonal(0.01)
            use_layer_norm=self.use_layer_norm,
        )(input_)

        return jnp.concatenate([value_1, value_2], axis=-1)

class MultiagentPolicy(nn.Module):
    hidden_layer_size: Tuple[int, ...]
    action_size: int
    independent_std: bool = True
    use_layer_norm: bool = False
    @nn.compact
    def __call__(self, obs: Observation) -> jnp.ndarray:
        # Policy networks typically only take observations, not actions
        # kernel_init = nn.initializers.variance_scaling(1.0, "fan_in", "uniform")
        # kernel_init = nn.initializers.orthogonal(jnp.sqrt(2))
        # Shared trunks
        trunk = MLP(
            layer_sizes=self.hidden_layer_size,
            # kernel_init=kernel_init,
            activation=nn.leaky_relu,
            use_layer_norm=self.use_layer_norm
        )(obs)
        
        if self.use_layer_norm:
            trunk = nn.LayerNorm(use_bias=False, use_scale=False)(trunk)

        # Mean head
        mean = MLP(
            layer_sizes=(self.action_size,),
            # kernel_init=kernel_init,
            final_activation=None,  # No activation for mean
        )(trunk)
        
        if not self.independent_std:
            # Dependent std: separate parameters for each action dimension
            log_std = MLP(
                layer_sizes=(self.action_size,),
                # kernel_init=kernel_init,
                # kernel_init_final=nn.initializers.orthogonal(0.01),
                final_activation=None,
            )(trunk)
        else:
            # Shared std: single parameter broadcasted to all action dimensions
            log_std_scalar = self.param(
                'log_std', 
                nn.initializers.zeros, 
                (self.action_size,)
            )
            # Handle both single sample and batch cases
            if mean.ndim == 1:
                # Single sample case: mean shape is (action_size,)
                log_std = jnp.broadcast_to(log_std_scalar, (self.action_size,))
            else:
                # Batch case: mean shape is (batch_size, action_size)
                batch_size = mean.shape[0]
                log_std = jnp.broadcast_to(log_std_scalar, (batch_size, self.action_size))
        
        # Concatenate mean and log_std
        return jnp.concatenate([mean, log_std], axis=-1)


def make_masac_networks(
    action_sizes: dict[int, int],
    critic_hidden_layer_size: Tuple[int, ...] = (256, 256),
    policy_hidden_layer_size: Tuple[int, ...] = (256, 256),
    per_agent_critics: bool = False,
    independent_std: bool = True,  # Add this parameter
    use_layer_norm: bool = False
) -> Tuple[Dict[int, nn.Module], Dict[int, nn.Module] | nn.Module]:
    """Creates networks used in MASAC.

    Args:
        action_sizes: dictionary mapping agent_idx to action size
        critic_hidden_layer_size: the number of neurons for critic hidden layers.
        policy_hidden_layer_size: the number of neurons for policy hidden layers.
        per_agent_critics: if True, creates separate critic for each agent
        independent_std: if True, learn independent std for each action dimension

    Returns:
        the policy networks (dict mapping agent_idx to policy network)
        the critic network(s) (single critic or dict of per-agent critics)
    """
    # Create policy networks for each agent
    policy = {
        agent_idx: MultiagentPolicy(
            hidden_layer_size=policy_hidden_layer_size,
            action_size=action_size,
            independent_std=independent_std,
            use_layer_norm=use_layer_norm,
        ) for agent_idx, action_size in action_sizes.items()
    }

    if per_agent_critics:
        # Create separate critic for each agent
        critics = {
            agent_idx: MultiAgentCritic(critic_hidden_layer_size, use_layer_norm=use_layer_norm)
            for agent_idx in action_sizes.keys()
        }
        return policy, critics
    else:
        # Create single shared critic
        critic = MultiAgentCritic(critic_hidden_layer_size, use_layer_norm=use_layer_norm)
        return policy, critic
