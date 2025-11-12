from dataclasses import dataclass
from typing import Callable, Tuple, Optional

import jax
import jax.numpy as jnp

import flax.linen as nn

from qdax.core.emitters.multi_emitter import MultiEmitter
from qdax.core.emitters.ma_cemrl_emitter import MACEMRLConfig, MACEMRLEmitter
from qdax.core.emitters.ma_standard_emitters import NaiveMultiAgentMixingEmitter, MultiAgentEmitter, ProximalMultiAgentEmitter
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.environments.base_wrappers import QDEnv
from qdax.types import Params, RNGKey


@dataclass
class MACEMRLMEConfig:
    """Configuration for PGAME Algorithm"""

    env_batch_size: int = 100
    proportion_mutation_ga: float = 0.5
    num_critic_training_steps: int = 3000
    num_pg_training_steps: int = 100
    num_warmstart_steps: int = 25_600
    
    emitter_type: str = "naive"

    variation_percentage: float = 0.3
    crossplay_percentage: float = 0.3

    # Mix mutate params
    agents_to_mutate: int = 1

    # CEM
    population_size: int = 10
    num_best: Optional[int] = None
    damp_init: float = 1e-3
    damp_final: float = 1e-5
    damp_tau : float = 0.95
    rank_weight_shift: float = 1.0
    mirror_sampling: bool = False
    weighted_update: bool = True
    num_learning_offspring: Optional[int] = None

    # TD3 params
    replay_buffer_size: int = 1000000
    critic_hidden_layer_size: Tuple[int, ...] = (256, 256)
    critic_learning_rate: float = 3e-4
    actor_learning_rate: float = 3e-4
    policy_learning_rate: float = 1e-3
    noise_clip: float = 0.5
    policy_noise: float = 0.2
    discount: float = 0.99
    reward_scaling: float = 1.0
    batch_size: int = 256
    soft_tau_update: float = 0.005
    policy_delay: int = 2
    use_layer_norm: bool = True
    max_grad_norm: float = 100.0 # set to 0.0 means not use grad clip 

class MACEMRLMEEmitter(MultiEmitter):
    def __init__(
        self,
        config: MACEMRLMEConfig,
        policy_network: nn.Module,
        env: QDEnv,
        mutation_fn: Callable[[Params, RNGKey], Tuple[Params, RNGKey]],
        variation_fn: Callable[[Params, Params, RNGKey], Tuple[Params, RNGKey]],
    ) -> None:

        self._config = config
        self._policy_network = policy_network
        self._env = env
        self._variation_fn = variation_fn

        ga_batch_size = int(self._config.proportion_mutation_ga * config.env_batch_size)
        qpg_batch_size = config.env_batch_size - ga_batch_size

        qpg_config = MACEMRLConfig(
            env_batch_size=qpg_batch_size,
            num_critic_training_steps=config.num_critic_training_steps,
            num_pg_training_steps=config.num_pg_training_steps,
            num_warmstart_steps=config.num_warmstart_steps,
            # CEM
            population_size=config.population_size,
            num_best=config.num_best,
            damp_init=config.damp_init,
            damp_final=config.damp_final,
            damp_tau=config.damp_tau,
            rank_weight_shift=config.rank_weight_shift,
            mirror_sampling=config.mirror_sampling,
            weighted_update=config.weighted_update,
            num_learning_offspring=config.num_learning_offspring,
            # TD3 params
            replay_buffer_size=config.replay_buffer_size,
            critic_hidden_layer_size=config.critic_hidden_layer_size,
            critic_learning_rate=config.critic_learning_rate,
            actor_learning_rate=config.actor_learning_rate,
            policy_learning_rate=config.policy_learning_rate,
            noise_clip=config.noise_clip,
            policy_noise=config.policy_noise,
            discount=config.discount,
            reward_scaling=config.reward_scaling,
            batch_size=config.batch_size,
            soft_tau_update=config.soft_tau_update,
            policy_delay=config.policy_delay,
            use_layer_norm=config.use_layer_norm,
            max_grad_norm=config.max_grad_norm,
        )

        # define the quality emitter
        q_emitter = MACEMRLEmitter(
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


        elif self._config.emitter_type == "role_preserving":
            if self._config.crossplay_percentage == 0:
                raise ValueError(
                    "For 'role_preserving' emitter_type, 'crossplay_percentage' must be non-zero."
                )

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
        
        else:
            raise ValueError(
                f"Unknown emitter_type '{self._config.emitter_type}'. "
                "Must be one of ['naive', 'role_preserving']"
            )
                    
            
        super().__init__(emitters=(q_emitter, ga_emitter))
        # ###DEBUG
        # super().__init__(emitters=(q_emitter,))


    def report(self, emitter_state):
        (cemrl_state,ga_state) = emitter_state.emitter_states

        # ### DEBUG
        # (cemrl_state,) = emitter_state.emitter_states
        # Flatten variance PyTree to compute statistics across all parameters
        variance_leaves = jax.tree_util.tree_leaves(cemrl_state.var_actor_params)
        all_variances = jnp.concatenate([v.flatten() for v in variance_leaves])
        
        metrics = {
            "damp": cemrl_state.damp,
            "max_var": jnp.max(all_variances),
            "mean_var": jnp.mean(all_variances),
            "median_var": jnp.median(all_variances),
            "rl_in_elites_percentage": cemrl_state.rl_in_elites_percentage
        } 

        return metrics