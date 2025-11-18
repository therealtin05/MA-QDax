"""Implements the SAC Emitter for multi-agent environments in jax for brax environments,
based on the PGA-ME algorithm structure but adapted for SAC.

In this version:
1. we move the loss functions into a method of the class.
2. Warmstart also added.
"""

from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, Tuple, Dict, List

import flax.linen as nn
import jax
import optax
from jax import numpy as jnp
from brax.training.distribution import NormalTanhDistribution

from qdax.core.containers.repertoire import Repertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.core.neuroevolution.buffers.buffer import QDTransition, ReplayBuffer, Transition
from qdax.core.neuroevolution.losses.masac_loss import make_masac_loss_fn
from qdax.core.neuroevolution.networks.networks import QModule
from qdax.core.neuroevolution.networks.masac_networks import MultiAgentCritic
from qdax.environments.multi_agent_wrappers import MultiAgentBraxWrapper
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, Params, RNGKey
from qdax.core.emitters.mutation_operators import proximal_mutation

from qdax.core.neuroevolution.networks.networks import MLP, QModule
from brax.envs import State as EnvState


@dataclass
class QualityMASACConfig:
    """Configuration for Quality MASAC Emitter"""

    num_agents: int
    action_sizes: Dict[int, int]
    env_batch_size: int = 100
    num_critic_training_steps: int = 300
    num_pg_training_steps: int = 100
    num_warmstart_steps: int = 25_600

    # SAC params
    replay_buffer_size: int = 1000000
    critic_hidden_layer_size: Tuple[int, ...] = (256, 256)
    policy_hidden_layer_size: Tuple[int, ...] = (256, 256)
    critic_learning_rate: float = 3e-4
    actor_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    policy_learning_rate: float = 1e-3
    alpha_learning_rate: float = 3e-4
    discount: float = 0.99
    reward_scaling: float = 1.0
    batch_size: int = 256
    tau: float = 0.005  # SAC uses tau instead of soft_tau_update
    alpha_init: float = 1.0
    fix_alpha: bool = False
    target_entropy_scale: float = 0.5
    max_grad_norm: float = 10.0
    policy_delay: int = 1

    # Safe mutation
    safe_mutation_on_pg: bool = False
    safe_mutation_percentage: float = 0.5
    safe_mut_mag: float = 0.1
    safe_mut_val_bound: float = 1000.0
    safe_mut_noise: bool = False

class QualityMASACEmitterState(EmitterState):
    """Contains training state for the MASAC emitter."""

    critic_params: Params
    critic_optimizer_state: optax.OptState
    actor_params: List[Params]
    actor_opt_state: List[optax.OptState]
    alpha_params: jnp.ndarray  # Temperature parameters per agent
    alpha_opt_state: optax.OptState
    target_critic_params: Params
    replay_buffer: ReplayBuffer
    random_key: RNGKey
    steps: jnp.ndarray

class QualityMASACEmitter(Emitter):
    """
    A SAC-based emitter used to implement Quality-Diversity with SAC for multi-agent environments.
    """

    def __init__(
        self,
        config: QualityMASACConfig,
        policy_network: Dict[int, MLP],
        env: MultiAgentBraxWrapper,
    ) -> None:
        self._config = config
        self._env = env
        self._policy_network = policy_network

        # Get action sizes for parametric distributions
        self._action_sizes = env.get_action_sizes()
        
        # Create parametric action distributions for each agent
        self._parametric_action_distribution = []
        for agent_idx in range(len(self._action_sizes)):
            self._parametric_action_distribution.append(
                NormalTanhDistribution(event_size=self._action_sizes[agent_idx])
            )

        # Init Critics
        critic_network = MultiAgentCritic(self._config.critic_hidden_layer_size)
        self._critic_network = critic_network

        # Init optimizers
        if self._config.max_grad_norm > 0:
            grad_clip = optax.clip_by_global_norm(self._config.max_grad_norm)
            self._actor_optimizer = optax.chain(
                grad_clip, 
                optax.adam(learning_rate=self._config.actor_learning_rate)
            )
            self._critic_optimizer = optax.chain(
                grad_clip,
                optax.adam(learning_rate=self._config.critic_learning_rate)
            )
            self._policies_optimizer = optax.chain(
                grad_clip,
                optax.adam(learning_rate=self._config.policy_learning_rate)
            )
            self._alpha_optimizer = optax.chain(
                grad_clip,
                optax.adam(learning_rate=self._config.alpha_learning_rate)
            )
        else:
            self._actor_optimizer = optax.adam(learning_rate=self._config.actor_learning_rate)
            self._critic_optimizer = optax.adam(learning_rate=self._config.critic_learning_rate)
            self._policies_optimizer = optax.adam(learning_rate=self._config.policy_learning_rate)
            self._alpha_optimizer = optax.adam(learning_rate=self._config.alpha_learning_rate)
            
    @property
    def batch_size(self) -> int:
        """
        Returns:
            the batch size emitted by the emitter.
        """
        return self._config.env_batch_size

    @property
    def use_all_data(self) -> bool:
        """Whether to use all data or not when used along other emitters.

        QualityMASACEmitter uses the transitions from the genotypes that were generated
        by other emitters.
        """
        return False

    def init(
        self, init_genotypes: Genotype, random_key: RNGKey
    ) -> Tuple[QualityMASACEmitterState, RNGKey]:
        """Initializes the emitter state.

        Args:
            init_genotypes: The initial population.
            random_key: A random key.

        Returns:
            The initial state of the QualityMASACEmitter, a new random key.
        """

        observation_size = self._env.observation_size
        action_size = self._env.action_size
        descriptor_size = self._env.state_descriptor_length

        # Initialise critic, greedy actor and population
        random_key, subkey = jax.random.split(random_key)
        fake_obs = jnp.zeros(shape=(observation_size,))
        fake_action = jnp.zeros(shape=(action_size,))
        critic_params = self._critic_network.init(
            subkey, obs=fake_obs, action=fake_action
        )
        target_critic_params = jax.tree_util.tree_map(lambda x: x, critic_params)

        actor_params = jax.tree_util.tree_map(lambda x: x[0], init_genotypes)

        # # Initialize alpha parameters (temperature) for each agent
        # alpha_params = []
        # alpha_opt_state = []
        # for agent_idx in range(len(self._action_sizes)):
        #     log_alpha = jnp.asarray(jnp.log(self._config.alpha_init), dtype=jnp.float32)
        #     alpha_params.append(log_alpha)
        #     alpha_opt_state.append(self._alpha_optimizer.init(log_alpha))

        # Single alpha
        alpha_params = jnp.asarray(jnp.log(self._config.alpha_init), dtype=jnp.float32)
        alpha_opt_state = self._alpha_optimizer.init(alpha_params)
        
        # Prepare init optimizer states
        critic_optimizer_state = self._critic_optimizer.init(critic_params)
        actor_optimizer_state = []
        for agent_idx, params in enumerate(actor_params):
            actor_optimizer_state.append(self._actor_optimizer.init(params))

        # Initialize replay buffer
        dummy_transition = QDTransition.init_dummy(
            observation_dim=observation_size,
            action_dim=action_size,
            descriptor_dim=descriptor_size,
        )

        replay_buffer = ReplayBuffer.init(
            buffer_size=self._config.replay_buffer_size, transition=dummy_transition
        )

        ### WARMSTART
        random_key, subkey = jax.random.split(random_key)
        keys = jax.random.split(subkey, 10) # hard code to 10 paralel env
        reset_fn = jax.jit(jax.vmap(self._env.reset))
        init_states = reset_fn(keys)

        random_key, subkey = jax.random.split(random_key)
        keys = jax.random.split(subkey, 10) # hard code to 10 paralel env

        replay_buffer, _, _ = self.warmstart_buffer(
            replay_buffer=replay_buffer,
            random_key=keys,
            env_state=init_states,
            env_batch_size=10 # hard code to 10 paralel env
        )

        # Initial training state
        random_key, subkey = jax.random.split(random_key)
        emitter_state = QualityMASACEmitterState(
            critic_params=critic_params,
            critic_optimizer_state=critic_optimizer_state,
            actor_params=actor_params,
            actor_opt_state=actor_optimizer_state,
            alpha_params=alpha_params,
            alpha_opt_state=alpha_opt_state,
            target_critic_params=target_critic_params,
            random_key=subkey,
            steps=jnp.array(0),
            replay_buffer=replay_buffer,
        )

        return emitter_state, random_key


    @partial(jax.jit, static_argnames=("self"))
    def warmstart_play_step_fn(
        self,
        env_state: EnvState,
        random_key: RNGKey,
    ):
        """
        Play an environment step and return the updated state and the transition.
        """
        random_key, subkey = jax.random.split(random_key)

        action_sizes = self._env.get_action_sizes()

        keys = jax.random.split(subkey, len(action_sizes))

        actions = {
            agent_idx: jax.random.uniform(agent_key, (size,), minval=-1, maxval=1)
            for (agent_idx, size), agent_key in zip(action_sizes.items(), keys)
        }

        flatten_actions = jnp.concatenate([a for a in actions.values()])

        state_desc = env_state.info["state_descriptor"]
        next_state = self._env.step(env_state, actions)

        transition = QDTransition(
            obs=next_state.obs,
            next_obs=next_state.obs,
            rewards=next_state.reward,
            dones=next_state.done,
            actions=flatten_actions,
            truncations=next_state.info["truncation"],
            state_desc=state_desc,
            next_state_desc=next_state.info["state_descriptor"],
        )

        return next_state, random_key, transition

    @partial(jax.jit, static_argnames=("self", "env_batch_size"),)
    def warmstart_buffer(
        self,
        replay_buffer: ReplayBuffer,
        random_key: RNGKey,
        env_state: EnvState,
        env_batch_size: int,
    ) -> Tuple[ReplayBuffer, EnvState, RNGKey]:
        """Pre-populates the buffer with transitions. Returns the warmstarted buffer
        and the new state of the environment.
        """
        warmstart_play_step_fn = jax.vmap(self.warmstart_play_step_fn, in_axes=(0, 0))
        def _scan_play_step_fn(
            carry: Tuple[EnvState, RNGKey], unused_arg: Any
        ) -> Tuple[Tuple[EnvState, RNGKey], QDTransition]:
            env_state, random_key, transitions = warmstart_play_step_fn(*carry)
            return (env_state, random_key), transitions

        (env_state, random_key), transitions = jax.lax.scan(
            _scan_play_step_fn,
            (env_state, random_key),
            (),
            length=self._config.num_warmstart_steps // env_batch_size,
        )
        replay_buffer = replay_buffer.insert(transitions)

        return replay_buffer, env_state, random_key


    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def emit(
        self,
        repertoire: Repertoire,
        emitter_state: QualityMASACEmitterState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey, jnp.ndarray]:
        """Do a step of SAC-based emission.

        Args:
            repertoire: the current repertoire of genotypes
            emitter_state: the state of the emitter used
            random_key: a random key

        Returns:
            A batch of offspring, the new emitter state and a new key.
        """

        batch_size = self._config.env_batch_size

        # sample parents
        mutation_pg_batch_size = int(batch_size - 1)
        parents, random_key = repertoire.sample(random_key, mutation_pg_batch_size)

        # apply the pg mutation
        offsprings_pg = self.emit_pg(emitter_state, parents)

        # get the actor (greedy actor)
        offspring_actor = self.emit_actor(emitter_state)

        # add dimension for concatenation
        offspring_actor = jax.tree_util.tree_map(
            lambda x: jnp.expand_dims(x, axis=0), offspring_actor
        )

        # gather offspring
        genotypes = jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate([x, y], axis=0),
            offsprings_pg,
            offspring_actor,
        )

        if self._config.safe_mutation_on_pg:
            genotypes, random_key = self._apply_partial_safe_mutation(
                genotypes, emitter_state, random_key
            )

        return genotypes, random_key, jnp.array(0)

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def emit_pg(
        self, emitter_state: QualityMASACEmitterState, parents: Genotype
    ) -> Genotype:
        """Emit the offsprings generated through SAC mutation.

        Args:
            emitter_state: current emitter state, contains critic and
                replay buffer.
            parents: the parents selected to be applied gradients in order
                to mutate towards better performance.

        Returns:
            A new set of offsprings.
        """
        mutation_fn = partial(
            self._mutation_function_pg,
            emitter_state=emitter_state,
        )
        offsprings = jax.vmap(mutation_fn)(parents)

        return offsprings

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def emit_actor(self, emitter_state: QualityMASACEmitterState) -> Genotype:
        """Emit the greedy actor.

        Simply needs to be retrieved from the emitter state.

        Args:
            emitter_state: the current emitter state, it stores the
                greedy actor.

        Returns:
            The parameters of the actor.
        """
        return emitter_state.actor_params


    @partial(jax.jit, static_argnames=("self",))
    def _apply_partial_safe_mutation(
        self,
        genotypes: Genotype,
        emitter_state: QualityMASACEmitterState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        """Apply safe mutation to only a percentage of genotypes to save VRAM."""
        
        # Get batch size from first agent's genotypes
        batch_size = jax.tree_util.tree_leaves(genotypes[0])[0].shape[0]
        
        # Calculate how many genotypes to mutate
        num_to_mutate = int(batch_size * self._config.safe_mutation_percentage)
        
        if num_to_mutate == 0:
            return genotypes, random_key
        
        # Randomly select indices to mutate (this is more memory efficient)
        random_key, subkey = jax.random.split(random_key)
        indices_to_mutate = jax.random.choice(
            subkey, 
            batch_size, 
            shape=(num_to_mutate,), 
            replace=False
        )
        
        # Extract subset of genotypes to mutate
        subset_genotypes = jax.tree_util.tree_map(
            lambda x: x[indices_to_mutate], genotypes
        )
        
        # Apply safe mutation to subset
        mutated_subset, random_key = self._apply_safe_mutation_to_subset(
            subset_genotypes, emitter_state, random_key
        )
        
        # Use scatter update to put mutated genotypes back
        def update_at_indices(original, mutated):
            return original.at[indices_to_mutate].set(mutated)
        
        final_genotypes = jax.tree_util.tree_map(
            update_at_indices,
            genotypes,
            mutated_subset
        )
        
        return final_genotypes, random_key

    @partial(jax.jit, static_argnames=("self",))
    def _apply_safe_mutation_to_subset(
        self,
        subset_genotypes: Genotype,
        emitter_state: QualityMASACEmitterState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        """Apply safe mutation to a subset of genotypes."""
        
        subset_batch_size = jax.tree_util.tree_leaves(subset_genotypes[0])[0].shape[0]
        
        # Sample transitions (use smaller batch size for efficiency)
        safe_mutation_batch_size = min(self._config.batch_size, subset_batch_size)
        transitions, random_key = emitter_state.replay_buffer.sample(
            random_key, safe_mutation_batch_size
        )

        # Unflatten observations
        obs = jax.vmap(self.unflatten_obs_fn)(transitions.obs)
        
        # Apply safe mutation to each agent's subset
        perturbed_genotypes = []
        for agent_idx, (agent_params, agent_obs) in enumerate(
            zip(subset_genotypes, obs.values())
        ):
            new_agent_params, random_key = proximal_mutation(
                agent_params, 
                random_key, 
                self._policy_network[agent_idx].apply, 
                agent_obs,
                mutation_mag=self._config.safe_mut_mag, 
                minval=-self._config.safe_mut_val_bound,
                maxval=self._config.safe_mut_val_bound, 
                mutation_noise=self._config.safe_mut_noise
            )
            perturbed_genotypes.append(new_agent_params)

        return perturbed_genotypes, random_key

    @partial(jax.jit, static_argnames=("self",))
    def state_update(
        self,
        emitter_state: QualityMASACEmitterState,
        repertoire: Optional[Repertoire],
        genotypes: Optional[Genotype],
        fitnesses: Optional[Fitness],
        descriptors: Optional[Descriptor],
        extra_scores: ExtraScores,
    ) -> QualityMASACEmitterState:
        """This function gives an opportunity to update the emitter state
        after the genotypes have been scored.

        Here it is used to fill the Replay Buffer with the transitions
        from the scoring of the genotypes, and then the training of the
        critic/actor happens. Hence the params of critic/actor are updated,
        as well as their optimizer states.

        Args:
            emitter_state: current emitter state.
            repertoire: the current genotypes repertoire
            genotypes: unused here - but compulsory in the signature.
            fitnesses: unused here - but compulsory in the signature.
            descriptors: unused here - but compulsory in the signature.
            extra_scores: extra information coming from the scoring function,
                this contains the transitions added to the replay buffer.

        Returns:
            New emitter state where the replay buffer has been filled with
            the new experienced transitions.
        """
        # get the transitions out of the dictionary
        assert "transitions" in extra_scores.keys(), "Missing transitions or wrong key"
        transitions = extra_scores["transitions"]

        # add transitions in the replay buffer
        replay_buffer = emitter_state.replay_buffer.insert(transitions)
        emitter_state = emitter_state.replace(replay_buffer=replay_buffer)

        def scan_train_critics(
            carry: QualityMASACEmitterState, unused: Any
        ) -> Tuple[QualityMASACEmitterState, Any]:
            emitter_state = carry
            new_emitter_state = self._train_critics(emitter_state)
            return new_emitter_state, ()

        # Train critics and greedy actor
        emitter_state, _ = jax.lax.scan(
            scan_train_critics,
            emitter_state,
            (),
            length=self._config.num_critic_training_steps,
        )

        return emitter_state  # type: ignore

    @partial(jax.jit, static_argnames=("self",))
    def _train_critics(
        self, emitter_state: QualityMASACEmitterState
    ) -> QualityMASACEmitterState:
        """Apply one gradient step to critics and to the greedy actor
        (contained in carry in training_state), then soft update target critics
        and target actor.

        Those updates are similar to those made in SAC.

        Args:
            emitter_state: actual emitter state

        Returns:
            New emitter state where the critic and the greedy actor have been
            updated. Optimizer states have also been updated in the process.
        """

        # Sample a batch of transitions in the buffer
        random_key = emitter_state.random_key
        replay_buffer = emitter_state.replay_buffer
        transitions, random_key = replay_buffer.sample(
            random_key, sample_size=self._config.batch_size
        )

        alpha_params = emitter_state.alpha_params

        # Update Critic
        # alphas = [jnp.exp(log_alpha) for log_alpha in alpha_params]
        
        # Single alpha
        alphas = jnp.exp(alpha_params)
        (
            critic_optimizer_state,
            critic_params,
            target_critic_params,
            random_key,
        ) = self._update_critic(
            emitter_state.critic_params,
            emitter_state.actor_params,
            emitter_state.target_critic_params,
            critic_optimizer_state=emitter_state.critic_optimizer_state,
            alphas=alphas, 
            transitions=transitions,
            random_key=random_key,
        )

        # Update greedy actor (SAC updates every step, no delay)
        (actor_optimizer_state, actor_params, alpha_opt_state, alpha_params, random_key) = jax.lax.cond(
            emitter_state.steps % self._config.policy_delay == 0,
            lambda _: (self._update_actor(
                emitter_state.actor_params,
                emitter_state.actor_opt_state,
                emitter_state.critic_params,
                emitter_state.alpha_params,
                emitter_state.alpha_opt_state,
                transitions,
                random_key=random_key
            )),
            lambda _: (
                emitter_state.actor_opt_state,
                emitter_state.actor_params,
                emitter_state.alpha_opt_state,
                emitter_state.alpha_params,
                random_key
            ),
            operand=None
        )

        # Create new training state
        new_emitter_state = emitter_state.replace(
            critic_params=critic_params,
            critic_optimizer_state=critic_optimizer_state,
            actor_params=actor_params,
            actor_opt_state=actor_optimizer_state,
            alpha_params=alpha_params,
            alpha_opt_state=alpha_opt_state,
            target_critic_params=target_critic_params,
            random_key=random_key,
            steps=emitter_state.steps + 1,
            replay_buffer=replay_buffer,
        )

        return new_emitter_state  # type: ignore

    @partial(jax.jit, static_argnames=("self",))
    def _update_critic(
        self,
        critic_params: Params,
        actor_params: List[Params],
        target_critic_params: Params,
        critic_optimizer_state: optax.OptState,
        alphas: List[jnp.ndarray],
        transitions: QDTransition,
        random_key: RNGKey,
    ) -> Tuple[optax.OptState, Params, Params, RNGKey]:
        """Update the critic parameters using SAC loss."""

        # compute loss and gradients
        random_key, subkey = jax.random.split(random_key)
        critic_loss, critic_gradient = jax.value_and_grad(self._critic_loss_fn)(
            critic_params,
            actor_params,
            target_critic_params,
            alphas,
            transitions,
            subkey,
        )
        
        critic_updates, critic_optimizer_state = self._critic_optimizer.update(
            critic_gradient, critic_optimizer_state
        )

        # update critic
        critic_params = optax.apply_updates(critic_params, critic_updates)

        # Soft update of target critic network
        target_critic_params = jax.tree_map(
            lambda x1, x2: (1.0 - self._config.tau) * x1 + self._config.tau * x2,
            target_critic_params,
            critic_params,
        )

        return critic_optimizer_state, critic_params, target_critic_params, random_key

    @partial(jax.jit, static_argnames=("self",))
    def _update_actor(
        self,
        actor_params: List[Params],
        actor_opt_state: List[optax.OptState],
        critic_params: Params,
        alpha_params: jnp.ndarray,
        alpha_opt_state: optax.OptState,
        transitions: QDTransition,
        random_key: RNGKey,
    ) -> Tuple[List[optax.OptState], List[Params], optax.OptState, Params, RNGKey]:
        """Update the actor parameters using SAC policy loss."""

        # alphas = [jnp.exp(log_alpha) for log_alpha in alpha_params]

        #Single alpha
        alphas = jnp.exp(alpha_params)

        random_key, subkey = jax.random.split(random_key)
        # Update greedy actor
        actor_losses, actor_gradients = self._policy_loss_fn(
            actor_params,
            critic_params,
            alphas,
            transitions,
            subkey,  
        )
        
        new_actor_optimizer_state = []
        new_actor_params = []
        
        for agent_idx, (pol_grad, act_opt_state) in enumerate(
            zip(actor_gradients, actor_opt_state)
        ):
            actor_updates, act_opt_state = self._actor_optimizer.update(
                pol_grad, act_opt_state
            )
            
            updated_params = optax.apply_updates(
                actor_params[agent_idx], actor_updates
            )

            new_actor_params.append(updated_params)
            new_actor_optimizer_state.append(act_opt_state)


        if not self._config.fix_alpha:
            random_key, subkey = jax.random.split(random_key)
            alpha_losses, alpha_gradients = jax.value_and_grad(self._alpha_loss_fn)(
                alpha_params,
                policy_params=actor_params,
                transitions=transitions,
                random_key=subkey,
            )

            ## Handle single alpha
            alpha_updates, new_alpha_opt_state = self._alpha_optimizer.update(
                alpha_gradients, alpha_opt_state
            )

            new_alpha_params = optax.apply_updates(
                    alpha_params, alpha_updates
                )

            ## Handle multi alpha
            # # Update alpha parameters for each agent
            # new_alpha_params = []
            # new_alpha_opt_state = []

            # for agent_idx, (alpha_grad, alpha_opt_state_i) in enumerate(
            #     zip(alpha_gradients, alpha_opt_state)
            # ):
            #     alpha_updates, updated_opt_state = self._alpha_optimizer.update(
            #         alpha_grad, alpha_opt_state_i
            #     )
            #     updated_alpha = optax.apply_updates(
            #         alpha_params[agent_idx], alpha_updates
            #     )
            #     new_alpha_params.append(updated_alpha)
            #     new_alpha_opt_state.append(updated_opt_state)


        else:
            alpha_losses = [jnp.array(0.0) for _ in range(len(self._action_sizes))]
    

        return (
            new_actor_optimizer_state,
            new_actor_params,
            new_alpha_opt_state,
            new_alpha_params,
            random_key
        )

    @partial(jax.jit, static_argnames=("self",))
    def _mutation_function_pg(
        self,
        policy_params: Genotype,
        emitter_state: QualityMASACEmitterState,
    ) -> Genotype:
        """Apply SAC mutation to a policy via multiple steps of gradient descent.

        Args:
            policy_params: a policy, supposed to be a differentiable neural
                network.
            emitter_state: the current state of the emitter, containing among others,
                the replay buffer, the critic.

        Returns:
            The updated params of the neural network.
        """

        # Define new policy optimizer state
        policy_optimizer_state = []
        for agent_idx, params in enumerate(policy_params):
            policy_optimizer_state.append(self._policies_optimizer.init(params))

        def scan_train_policy(
            carry: Tuple[QualityMASACEmitterState, Genotype, List[optax.OptState]],
            unused: Any,
        ) -> Tuple[Tuple[QualityMASACEmitterState, Genotype, List[optax.OptState]], Any]:
            emitter_state, policy_params, policy_optimizer_state = carry
            (
                new_emitter_state,
                new_policy_params,
                new_policy_optimizer_state,
            ) = self._train_policy(
                emitter_state,
                policy_params,
                policy_optimizer_state,
            )
            return (
                new_emitter_state,
                new_policy_params,
                new_policy_optimizer_state,
            ), ()

        (emitter_state, policy_params, policy_optimizer_state,), _ = jax.lax.scan(
            scan_train_policy,
            (emitter_state, policy_params, policy_optimizer_state),
            (),
            length=self._config.num_pg_training_steps,
        )

        return policy_params

    @partial(jax.jit, static_argnames=("self",))
    def _train_policy(
        self,
        emitter_state: QualityMASACEmitterState,
        policy_params: List[Params],
        policy_optimizer_state: List[optax.OptState],
    ) -> Tuple[QualityMASACEmitterState, List[Params], List[optax.OptState]]:
        """Apply one gradient step to a policy (called policy_params).

        Args:
            emitter_state: current state of the emitter.
            policy_params: parameters corresponding to the weights and bias of
                the neural network that defines the policy.

        Returns:
            The new emitter state and new params of the NN.
        """

        # Sample a batch of transitions in the buffer
        random_key = emitter_state.random_key
        replay_buffer = emitter_state.replay_buffer
        transitions, random_key = replay_buffer.sample(
            random_key, sample_size=self._config.batch_size
        )

        # update policy
        policy_optimizer_state, policy_params, random_key = self._update_policy(
            critic_params=emitter_state.critic_params,
            policy_optimizer_state=policy_optimizer_state,
            policy_params=policy_params,
            # alphas=[jnp.exp(log_alpha) for log_alpha in emitter_state.alpha_params],
            alphas=jnp.exp(emitter_state.alpha_params), # Hangle single alpha, above is for multialpha
            transitions=transitions,
            random_key=random_key
        )

        # Create new training state
        new_emitter_state = emitter_state.replace(
            random_key=random_key,
            replay_buffer=replay_buffer,
        )

        return new_emitter_state, policy_params, policy_optimizer_state

    @partial(jax.jit, static_argnames=("self",))
    def _update_policy(
        self,
        critic_params: Params,
        policy_optimizer_state: List[optax.OptState],
        policy_params: List[Params],
        alphas: jnp.ndarray,
        transitions: QDTransition,
        random_key: RNGKey,
    ) -> Tuple[List[optax.OptState], List[Params], RNGKey]:
        """Update policy parameters using SAC policy loss."""

        random_key, subkey = jax.random.split(random_key)
        # compute loss
        _policy_losses, policy_gradients = self._policy_loss_fn(
            policy_params,
            critic_params,
            alphas,
            transitions,
            subkey,
        )
        
        new_policy_optimizer_state = []
        new_policy_params = []

        # Compute gradient and update policies
        for agent_idx, (pol_grad, pol_opt_state) in enumerate(
            zip(policy_gradients, policy_optimizer_state)
        ):
            policy_updates, pol_opt_state = self._policies_optimizer.update(
                pol_grad, pol_opt_state
            )
            
            updated_params = optax.apply_updates(
                policy_params[agent_idx], policy_updates
            )

            new_policy_params.append(updated_params)
            new_policy_optimizer_state.append(pol_opt_state)

        return new_policy_optimizer_state, new_policy_params, random_key

    
    @partial(jax.jit, static_argnames=("self"))
    def unflatten_actions_fn(self, flatten_action: jnp.ndarray) -> dict[int, jax.Array]:
        """Unflatten actions from concatenated form to agent-specific actions."""
        actions = {}
        start = 0
        for agent_idx, size in self._env.get_action_sizes().items():
            end = start + size
            actions[agent_idx] = flatten_action[start:end]
            start = end
        return actions
    
    @partial(jax.jit, static_argnames=("self"))
    def unflatten_obs_fn(self, global_obs: jnp.ndarray) -> dict[int, jnp.ndarray]:
        agent_obs = {}
        for agent_idx, obs_indices in self._env.agent_obs_mapping.items():
                agent_obs[agent_idx] = global_obs[obs_indices]
        
        return agent_obs
    
    @partial(jax.jit, static_argnames=("self"))
    def _policy_loss_fn(
        self,
        policy_params: List[Params],
        critic_params: Params,
        alpha: jnp.ndarray,
        transitions: Transition,
        random_key: RNGKey,
    ) -> Tuple[List[jnp.ndarray], List[Params]]:
        """Policy loss function for MASAC."""

        unflatten_obs = jax.vmap(self.unflatten_obs_fn)(transitions.obs)
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
                    dist_params = self._policy_network[i].apply(agent_params, unflatten_obs[i])
                    action = self._parametric_action_distribution[i].sample_no_postprocessing(
                        dist_params, subkeys[i]  
                    )
                    log_prob = self._parametric_action_distribution[i].log_prob(dist_params, action)
                    action = self._parametric_action_distribution[i].postprocess(action)  # Fixed: add postprocessing
                else:
                    # Use current policy_params for other agents
                    dist_params = self._policy_network[i].apply(policy_params[i], unflatten_obs[i])
                    action = self._parametric_action_distribution[i].sample(
                        dist_params, subkeys[i]
                    )
                agent_actions.append(action)
            
            # Flatten all actions
            flatten_actions = jnp.concatenate(agent_actions, axis=-1)
            
            q_action = self._critic_network.apply(critic_params, transitions.obs, flatten_actions)
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
    

    @partial(jax.jit, static_argnames=("self"))
    def _alpha_loss_fn(
        self,
        log_alpha: jnp.ndarray,  # Single log_alpha instead of list
        policy_params: List[Params],
        transitions: Transition,
        random_key: RNGKey,
    ) -> jnp.ndarray:  # Single loss
        """
        Alpha loss for single alpha across all agents.
        Target entropy is the sum of all agents' target entropies.
        """

        unflatten_obs = jax.vmap(self.unflatten_obs_fn)(transitions.obs)
        agent_keys = jax.random.split(random_key, len(policy_params))
        num_agents = len(policy_params)

        # Calculate combined target entropy for all agents
        total_target_entropy = -self._config.target_entropy_scale * sum(self._env.get_action_sizes().values())

        # Get log probabilities from all agents
        all_log_probs = []
        for agent_idx in range(num_agents):
            dist_params = self._policy_network[agent_idx].apply(policy_params[agent_idx], unflatten_obs[agent_idx])
            action = self._parametric_action_distribution[agent_idx].sample_no_postprocessing(
                dist_params, agent_keys[agent_idx],
            )
            log_prob = self._parametric_action_distribution[agent_idx].log_prob(dist_params, action)
            all_log_probs.append(log_prob)

        # Sum log probabilities across all agents
        total_log_prob = jnp.sum(jnp.array(all_log_probs), axis=0)
        
        alpha = jnp.exp(log_alpha)
        alpha_loss = alpha * jax.lax.stop_gradient(-total_log_prob - total_target_entropy)

        loss = jnp.mean(alpha_loss)

        return loss

    @partial(jax.jit, static_argnames=("self"))
    def _critic_loss_fn(
        self,
        critic_params: Params,
        policy_params: List[Params],
        target_critic_params: Params,
        alpha: jnp.ndarray,  # Changed: now just single alpha across agents
        transitions: Transition,
        random_key: RNGKey,
    ) -> jnp.ndarray:
        """Critic loss function for MASAC."""

        unflatten_next_obs = jax.vmap(self.unflatten_obs_fn)(transitions.next_obs)
        next_actions, next_log_probs = {}, {}
        
        # Split keys for each agent
        agent_keys = jax.random.split(random_key, len(policy_params))
        for agent_idx, (params, agent_obs) in enumerate(
                zip(policy_params, unflatten_next_obs.values())
        ):
            next_dist_params = self._policy_network[agent_idx].apply(params, agent_obs)
            next_a = self._parametric_action_distribution[agent_idx].sample_no_postprocessing(
                next_dist_params, agent_keys[agent_idx]
            )
            next_lp = self._parametric_action_distribution[agent_idx].log_prob(
                next_dist_params, next_a
            )

            next_a = self._parametric_action_distribution[agent_idx].postprocess(next_a)

            next_actions[agent_idx] = next_a
            next_log_probs[agent_idx] = next_lp

        flatten_next_actions = jnp.concatenate([a for a in next_actions.values()], axis=-1)
        
        # Optimized: Use single alpha for all log probabilities
        total_log_probs = jnp.sum(jnp.array([lp for lp in next_log_probs.values()]), axis=0)
        total_entropy_term = alpha * total_log_probs

        next_q = self._critic_network.apply(target_critic_params, transitions.next_obs, flatten_next_actions)
        next_v = jnp.min(next_q, axis=-1) - total_entropy_term  # Fixed: use weighted entropy

        target_q = jax.lax.stop_gradient(
            transitions.rewards * self._config.reward_scaling
            + (1.0 - transitions.dones) * self._config.discount * next_v
        )

        q_old_action = self._critic_network.apply(critic_params, transitions.obs, transitions.actions)
        q_error = q_old_action - jnp.expand_dims(target_q, -1)
        q_error *= jnp.expand_dims(1 - transitions.truncations, -1)
        q_loss = jnp.mean(jnp.square(q_error))

        return q_loss
