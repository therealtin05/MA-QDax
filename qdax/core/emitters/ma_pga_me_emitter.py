from dataclasses import dataclass
from typing import Callable, Tuple, Dict, List
from functools import partial

import flax.linen as nn

from qdax.core.emitters.multi_emitter import MultiEmitter
from qdax.core.emitters.ma_qpg_emitter import QualityMAPGConfig, QualityMAPGEmitter
from qdax.core.emitters.ma_standard_emitters import NaiveMultiAgentMixingEmitter, MultiAgentEmitter
from qdax.environments.multi_agent_wrappers import MultiAgentBraxWrapper
from qdax.types import Params, RNGKey
from qdax.core.emitters.mutation_operators import isoline_variation, polynomial_mutation

@dataclass
class MAPGAMEConfig:
    """Configuration for PGAME Algorithm"""

    env_batch_size: int = 100
    proportion_mutation_ga: float = 0.5

    emitter_type: str = "naive"
    
    variation_percentage: float = 0.3
    crossplay_percentage: float = 0.3
    agents_to_mutate: int = 1
    
    num_critic_training_steps: int = 300
    num_pg_training_steps: int = 100

    # TD3 params
    replay_buffer_size: int = 1000000
    critic_hidden_layer_size: Tuple[int, ...] = (256, 256)
    critic_learning_rate: float = 3e-4
    greedy_learning_rate: float = 3e-4
    policy_learning_rate: float = 1e-3
    noise_clip: float = 0.5
    policy_noise: float = 0.2
    discount: float = 0.99
    reward_scaling: float = 1.0
    batch_size: int = 256
    soft_tau_update: float = 0.005
    policy_delay: int = 2


class MAPGAMEEmitter(MultiEmitter):
    def __init__(
        self,
        config: MAPGAMEConfig,
        policy_network: Dict[int, nn.Module],
        env: MultiAgentBraxWrapper,
        mutation_fn: Callable[[Params, RNGKey], Tuple[Params, RNGKey]],
        variation_fn: Callable[[Params, Params, RNGKey], Tuple[Params, RNGKey]],
    ) -> None:

        self._config = config
        self._policy_network = policy_network
        self._env = env
        self._variation_fn = variation_fn

        ga_batch_size = int(self._config.proportion_mutation_ga * config.env_batch_size)
        qpg_batch_size = config.env_batch_size - ga_batch_size

        qpg_config = QualityMAPGConfig(
            env_batch_size=qpg_batch_size,
            num_critic_training_steps=config.num_critic_training_steps,
            num_pg_training_steps=config.num_pg_training_steps,
            replay_buffer_size=config.replay_buffer_size,
            critic_hidden_layer_size=config.critic_hidden_layer_size,
            critic_learning_rate=config.critic_learning_rate,
            actor_learning_rate=config.greedy_learning_rate,
            policy_learning_rate=config.policy_learning_rate,
            noise_clip=config.noise_clip,
            policy_noise=config.policy_noise,
            discount=config.discount,
            reward_scaling=config.reward_scaling,
            batch_size=config.batch_size,
            soft_tau_update=config.soft_tau_update,
            policy_delay=config.policy_delay,
        )

        # define the quality emitter
        q_emitter = QualityMAPGEmitter(
            config=qpg_config, policy_network=policy_network, env=env
        )


        if self._config.emitter_type == "naive":

            # define the GA emitter
            ga_emitter = NaiveMultiAgentMixingEmitter(
                mutation_fn=mutation_fn,
                variation_fn=variation_fn,
                variation_percentage=self._config.variation_percentage,
                batch_size=ga_batch_size,
                num_agents=len(policy_network),
                agents_to_mutate=self._config.agents_to_mutate
            )

        else:
            # define the GA emitter
            ga_emitter = MultiAgentEmitter(
                mutation_fn=mutation_fn,
                variation_fn=variation_fn,
                variation_percentage=self._config.variation_percentage,
                crossplay_percentage=self._config.crossplay_percentage,
                batch_size=ga_batch_size,
                num_agents=len(policy_network),
                role_preserving=True,
                agents_to_mutate=self._config.agents_to_mutate
            ) 

        super().__init__(emitters=(q_emitter, ga_emitter))
