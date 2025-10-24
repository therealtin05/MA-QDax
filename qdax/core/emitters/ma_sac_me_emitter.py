from dataclasses import dataclass
from typing import Callable, Tuple, Dict, List
from functools import partial

import flax.linen as nn

from qdax.core.emitters.multi_emitter import MultiEmitter
from qdax.core.emitters.ma_qsac_emitter import QualityMASACConfig, QualityMASACEmitter
from qdax.core.emitters.ma_standard_emitters import NaiveMultiAgentMixingEmitter, MultiAgentEmitter
from qdax.environments.multi_agent_wrappers import MultiAgentBraxWrapper
from qdax.types import Params, RNGKey
from qdax.core.emitters.mutation_operators import isoline_variation, polynomial_mutation

@dataclass
class MASACMEConfig:
    """Configuration for SACME Algorithm"""

    env_batch_size: int = 100
    proportion_mutation_ga: float = 0.5

    emitter_type: str = "naive"
    
    variation_percentage: float = 0.3
    crossplay_percentage: float = 0.3
    agents_to_mutate: int = 1
    safe_mutation_on_pg: bool = False
    pg_safe_mutation_percentage: float = 0.5

    # Safe mutate params
    safe_mut_mag: float = 0.1
    safe_mut_val_bound: float = 1000.0
    safe_mut_noise: bool = False

    num_critic_training_steps: int = 300
    num_pg_training_steps: int = 100

    # SAC params
    replay_buffer_size: int = 1000000
    critic_hidden_layer_size: Tuple[int, ...] = (256, 256)
    critic_learning_rate: float = 3e-4
    greedy_learning_rate: float = 3e-4
    policy_learning_rate: float = 1e-3
    alpha_learning_rate: float = 3e-4
    noise_clip: float = 0.5
    policy_noise: float = 0.2
    discount: float = 0.99
    reward_scaling: float = 1.0
    batch_size: int = 256
    soft_tau_update: float = 0.005
    fix_alpha: bool = False
    target_entropy_scale: float = 0.5
    max_grad_norm: float = 30.0
    policy_delay: int = 4

class MASACMEEmitter(MultiEmitter):
    def __init__(
        self,
        config: MASACMEConfig,
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

        qpg_config = QualityMASACConfig(
            num_agents=len(self._env.get_action_sizes()),
            action_sizes=self._env.get_action_sizes(),
            env_batch_size=qpg_batch_size,
            num_critic_training_steps=config.num_critic_training_steps,
            num_pg_training_steps=config.num_pg_training_steps,
            replay_buffer_size=config.replay_buffer_size,
            critic_hidden_layer_size=config.critic_hidden_layer_size,
            critic_learning_rate=config.critic_learning_rate,
            actor_learning_rate=config.greedy_learning_rate,
            policy_learning_rate=config.policy_learning_rate,
            alpha_learning_rate=config.alpha_learning_rate,
            discount=config.discount,
            reward_scaling=config.reward_scaling,
            batch_size=config.batch_size,
            tau=config.soft_tau_update,
            fix_alpha=config.fix_alpha,
            target_entropy_scale=config.target_entropy_scale,
            max_grad_norm=config.max_grad_norm,
            policy_delay=config.policy_delay,
            safe_mut_mag=config.safe_mut_mag,
            safe_mut_val_bound=config.safe_mut_val_bound,
            safe_mut_noise=config.safe_mut_noise,
            safe_mutation_on_pg=config.safe_mutation_on_pg,
            safe_mutation_percentage=config.pg_safe_mutation_percentage
        )

        # define the quality emitter
        q_emitter = QualityMASACEmitter(
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
