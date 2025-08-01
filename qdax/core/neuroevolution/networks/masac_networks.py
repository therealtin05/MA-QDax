from typing import Tuple, Dict

import flax.linen as nn
import jax.numpy as jnp

from qdax.core.neuroevolution.networks.networks import MLP
from qdax.custom_types import Action, Observation


class MultiAgentCritic(nn.Module):
    hidden_layer_size: Tuple[int, ...]

    @nn.compact
    def __call__(self, obs: Observation, action: Action) -> jnp.ndarray:
        input_ = jnp.concatenate([obs, action], axis=-1)

        kernel_init = nn.initializers.variance_scaling(1.0, "fan_in", "uniform")

        value_1 = MLP(
            layer_sizes=self.hidden_layer_size + (1,),
            kernel_init=kernel_init,
            activation=nn.relu,
        )(input_)

        value_2 = MLP(
            layer_sizes=self.hidden_layer_size + (1,),
            kernel_init=kernel_init,
            activation=nn.relu,
        )(input_)

        return jnp.concatenate([value_1, value_2], axis=-1)


def make_masac_networks(
    action_sizes: dict[int, int],
    critic_hidden_layer_size: Tuple[int, ...] = (256, 256),
    policy_hidden_layer_size: Tuple[int, ...] = (256, 256),
) -> Tuple[Dict[int, MLP], nn.Module]:
    """Creates networks used in SAC.

    Args:
        action_size: the size of the environment's action space
        critic_hidden_layer_size: the number of neurons for critic hidden layers.
        policy_hidden_layer_size: the number of neurons for policy hidden layers.

    Returns:
        the policy network
        the critic network
    """
    policy = {
        agent_idx: MLP(
        layer_sizes=policy_hidden_layer_size + (2 * action_size,),
        kernel_init=nn.initializers.variance_scaling(1.0, "fan_in", "uniform"),
    ) for agent_idx, action_size in action_sizes.items()
    }

    critic = MultiAgentCritic(critic_hidden_layer_size)

    return (policy, critic)
