from typing import Tuple, Dict

from jax import numpy as jnp

from qdax.core.neuroevolution.networks.networks import MLP, QModule

def make_maddpg_networks(
    action_sizes: dict[int, int],
    critic_hidden_layer_sizes: Tuple[int, ...],
    policy_hidden_layer_sizes: Tuple[int, ...],
) -> Tuple[Dict[int, MLP], QModule]:
    """Creates networks used by the TD3 agent.

    Args:
        action_size: Size the action array used to interact with the environment.
        num_agents: Number of agents in the multiagent system
        critic_hidden_layer_sizes: Number of layers and units per layer used in the
            neural network defining the critic.
        policy_hidden_layer_sizes: Number of layers and units per layer used in the
            neural network defining the policy.

    Returns:
        The neural network defining the policy and the module defininf the critic.
        This module contains two neural networks.
    """

    policy_network = {
        agent_idx: MLP(
        layer_sizes=policy_hidden_layer_sizes + (action_size,),
        final_activation=jnp.tanh,
    ) for agent_idx, action_size in action_sizes.items()
    }

    q_network = QModule(n_critics=1, hidden_layer_sizes=critic_hidden_layer_sizes)

    return (policy_network, q_network)
