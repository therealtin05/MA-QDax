"""Implements the PG Emitter from PGA-ME algorithm in jax for brax environments,
based on:
https://hal.archives-ouvertes.fr/hal-03135723v2/file/PGA_MAP_Elites_GECCO.pdf

Exactly the same as ma_pqg_emitter.py. However the loss is now a method of the class,
in order to mitigate the use of jax.lax.switch -> now can handle env with dif action shape
for each agent
"""

from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, Tuple, Dict, List

import flax.linen as nn
import jax
import optax
from jax import numpy as jnp

from qdax.core.containers.repertoire import Repertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.core.neuroevolution.buffers.buffer import QDTransition, ReplayBuffer
from qdax.core.neuroevolution.networks.networks import QModule
from qdax.environments.multi_agent_wrappers import MultiAgentBraxWrapper
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, Params, RNGKey
from qdax.core.emitters.mutation_operators import proximal_mutation
from qdax.core.neuroevolution.networks.networks import MLP, QModule
from brax.envs import State as EnvState


@dataclass
class QualityMAPGConfig:
    """Configuration for QualityPG Emitter"""

    env_batch_size: int = 100
    num_critic_training_steps: int = 300
    num_pg_training_steps: int = 100
    num_warmstart_steps: int = 25_600

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
    max_grad_norm: float = 100.0

    # Safe mutation
    safe_mutation_on_pg: bool = False
    safe_mutation_percentage: float = 0.5
    safe_mut_mag: float = 0.1
    safe_mut_val_bound: float = 1000.0
    safe_mut_noise: bool = False

class QualityMAPGEmitterState(EmitterState):
    """Contains training state for the learner."""

    critic_params: Params
    critic_optimizer_state: optax.OptState
    actor_params: List[Params]
    actor_opt_state: List[optax.OptState]
    target_critic_params: Params
    target_actor_params: Params
    replay_buffer: ReplayBuffer
    random_key: RNGKey
    steps: jnp.ndarray



class QualityMAPGEmitter(Emitter):
    """
    A policy gradient emitter used to implement the Policy Gradient Assisted MAP-Elites
    (PGA-Map-Elites) algorithm.
    """

    def __init__(
        self,
        config: QualityMAPGConfig,
        policy_network: Dict[int, MLP],
        env: MultiAgentBraxWrapper,
    ) -> None:
        self._config = config
        self._env = env
        self._policy_network = policy_network

        # Init Critics
        critic_network = QModule(
            n_critics=2, hidden_layer_sizes=self._config.critic_hidden_layer_size
        )
        self._critic_network = critic_network
        

        # Init optimizers
        if self._config.max_grad_norm > 0:
            grad_clip = optax.clip_by_global_norm(self._config.max_grad_norm)
            self._actor_optimizer = optax.chain(
                    grad_clip, 
                    optax.adam(
                    learning_rate=self._config.actor_learning_rate
                )
            )
            self._critic_optimizer = optax.chain(
                    grad_clip,
                    optax.adam(
                    learning_rate=self._config.critic_learning_rate
                )
            )
            self._policies_optimizer = optax.chain(
                    grad_clip,
                    optax.adam(
                    learning_rate=self._config.policy_learning_rate
                )
            )
        else:
            self._actor_optimizer = optax.adam(learning_rate=self._config.actor_learning_rate)
            self._critic_optimizer = optax.adam(learning_rate=self._config.critic_learning_rate)
            self._policies_optimizer = optax.adam(learning_rate=self._config.policy_learning_rate)
            
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

        QualityPGEmitter uses the transitions from the genotypes that were generated
        by other emitters.
        """
        return False ### DEBUG

    def init(
        self, init_genotypes: Genotype, random_key: RNGKey
    ) -> Tuple[QualityMAPGEmitterState, RNGKey]:
        """Initializes the emitter state.

        Args:
            init_genotypes: The initial population.
            random_key: A random key.

        Returns:
            The initial state of the PGAMEEmitter, a new random key.
        """

        observation_size = self._env.observation_size
        action_size = self._env.action_size
        descriptor_size = self._env.state_descriptor_length

        # Initialise critic, greedy actor and population
        random_key, subkey = jax.random.split(random_key)
        fake_obs = jnp.zeros(shape=(observation_size,))
        fake_action = jnp.zeros(shape=(action_size,))
        critic_params = self._critic_network.init(
            subkey, obs=fake_obs, actions=fake_action
        )
        target_critic_params = jax.tree_util.tree_map(lambda x: x, critic_params)

        actor_params = jax.tree_util.tree_map(lambda x: x[0], init_genotypes)
        target_actor_params = jax.tree_util.tree_map(lambda x: x[0], init_genotypes)

        # print(f"INIT len actor param {len(actor_params)}, len target actor param {len(target_actor_params)}")

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

        print(f"Complete warmstart current replay buffer size: {replay_buffer.current_size}")

        # Initial training state
        random_key, subkey = jax.random.split(random_key)
        emitter_state = QualityMAPGEmitterState(
            critic_params=critic_params,
            critic_optimizer_state=critic_optimizer_state,
            actor_params=actor_params,
            actor_opt_state=actor_optimizer_state,
            target_critic_params=target_critic_params,
            target_actor_params=target_actor_params,
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
        emitter_state: QualityMAPGEmitterState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey, jnp.ndarray]:
        """Do a step of PG emission.

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
        self, emitter_state: QualityMAPGEmitterState, parents: Genotype
    ) -> Genotype:
        """Emit the offsprings generated through pg mutation.

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
    def emit_actor(self, emitter_state: QualityMAPGEmitterState) -> Genotype:
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
        emitter_state: QualityMAPGEmitterState,
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
        emitter_state: QualityMAPGEmitterState,
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
        emitter_state: QualityMAPGEmitterState,
        repertoire: Optional[Repertoire],
        genotypes: Optional[Genotype],
        fitnesses: Optional[Fitness],
        descriptors: Optional[Descriptor],
        extra_scores: ExtraScores,
    ) -> QualityMAPGEmitterState:
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

        # jax.debug.print("transitions shape {shape}", shape=transitions.obs.shape)

        # add transitions in the replay buffer
        replay_buffer = emitter_state.replay_buffer.insert(transitions)
        emitter_state = emitter_state.replace(replay_buffer=replay_buffer)

        def scan_train_critics(
            carry: QualityMAPGEmitterState, unused: Any
        ) -> Tuple[QualityMAPGEmitterState, Any]:
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
        self, emitter_state: QualityMAPGEmitterState
    ) -> QualityMAPGEmitterState:
        """Apply one gradient step to critics and to the greedy actor
        (contained in carry in training_state), then soft update target critics
        and target actor.

        Those updates are very similar to those made in TD3.

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

        # # Add regular Python print first
        # print("_train_critics called!")

        # print(f"obs shape from transitions _train_critic {transitions.obs.shape}")

        # Update Critic
        (
            critic_optimizer_state,
            critic_params,
            target_critic_params,
            random_key,
        ) = self._update_critic(
            critic_params=emitter_state.critic_params,
            target_critic_params=emitter_state.target_critic_params,
            target_actor_params=emitter_state.target_actor_params,
            critic_optimizer_state=emitter_state.critic_optimizer_state,
            transitions=transitions,
            random_key=random_key,
        )

        # Update greedy actor
        (actor_optimizer_state, actor_params, target_actor_params,) = jax.lax.cond(
            emitter_state.steps % self._config.policy_delay == 0,
            lambda x: self._update_actor(*x),
            lambda _: (
                emitter_state.actor_opt_state,
                emitter_state.actor_params,
                emitter_state.target_actor_params,
            ),
            operand=(
                emitter_state.actor_params,
                emitter_state.actor_opt_state,
                emitter_state.target_actor_params,
                emitter_state.critic_params,
                transitions,
            ),
        )

        # Create new training state
        new_emitter_state = emitter_state.replace(
            critic_params=critic_params,
            critic_optimizer_state=critic_optimizer_state,
            actor_params=actor_params,
            actor_opt_state=actor_optimizer_state,
            target_critic_params=target_critic_params,
            target_actor_params=target_actor_params,
            random_key=random_key,
            steps=emitter_state.steps + 1,
            replay_buffer=replay_buffer,
        )

        return new_emitter_state  # type: ignore

    @partial(jax.jit, static_argnames=("self",))
    def _update_critic(
        self,
        critic_params: Params,
        target_critic_params: Params,
        target_actor_params: List[Params],
        critic_optimizer_state: Params,
        transitions: QDTransition,
        random_key: RNGKey,
    ) -> Tuple[Params, Params, Params, RNGKey]:

        # compute loss and gradients
        random_key, subkey = jax.random.split(random_key)
        critic_loss, critic_gradient = jax.value_and_grad(self._matd3_critic_loss_fn)(
            critic_params,
            target_actor_params,
            target_critic_params,
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
            lambda x1, x2: (1.0 - self._config.soft_tau_update) * x1
            + self._config.soft_tau_update * x2,
            target_critic_params,
            critic_params,
        )

        return critic_optimizer_state, critic_params, target_critic_params, random_key

    @partial(jax.jit, static_argnames=("self",))
    def _update_actor(
        self,
        actor_params: List[Params],
        actor_opt_state: List[optax.OptState],
        target_actor_params: List[Params],
        critic_params: Params,
        transitions: QDTransition,
    ) -> Tuple[List[optax.OptState], List[Params], List[Params]]:

        # Update greedy actor
        actor_losses, actor_gradients = self._matd3_policy_loss_fn(
            actor_params,
            critic_params,
            transitions,
        )
        
        new_actor_optimizer_state = []
        new_actor_params = []
        new_target_actor_params = []
        
        
        for agent_idx, (pol_grad, act_opt_state) in enumerate(
            zip(actor_gradients, actor_opt_state)
        ):
            # print(f'agent_idx {agent_idx}, len target param {len(target_actor_params)}')
            actor_updates, act_opt_state = self._actor_optimizer.update(
                pol_grad, act_opt_state
            )
            
            updated_params = optax.apply_updates(
                actor_params[agent_idx], actor_updates
            )

            new_actor_params.append(updated_params)
            new_actor_optimizer_state.append(act_opt_state)
            
            # Soft update of target greedy actor
            new_target_actor_params.append(
                jax.tree_util.tree_map(
                    lambda x1, x2: (1.0 - self._config.soft_tau_update) * x1
                    + self._config.soft_tau_update * x2,
                    target_actor_params[agent_idx],
                    updated_params,
                )
            )

        return (
            new_actor_optimizer_state,
            new_actor_params,
            new_target_actor_params,
        )

    @partial(jax.jit, static_argnames=("self",))
    def _mutation_function_pg(
        self,
        policy_params: Genotype,
        emitter_state: QualityMAPGEmitterState,
    ) -> Genotype:
        """Apply pg mutation to a policy via multiple steps of gradient descent.
        First, update the rewards to be diversity rewards, then apply the gradient
        steps.

        Args:
            policy_params: a policy, supposed to be a differentiable neural
                network.
            emitter_state: the current state of the emitter, containing among others,
                the replay buffer, the critic.

        Returns:
            The updated params of the neural network.
        """

        # Define new policy optimizer state
        # policy_optimizer_state = self._policies_optimizer.init(policy_params)
        policy_optimizer_state = []
        for agent_idx, params in enumerate(policy_params):
            policy_optimizer_state.append(self._policies_optimizer.init(params))

        def scan_train_policy(
            carry: Tuple[QualityMAPGEmitterState, Genotype, optax.OptState],
            unused: Any,
        ) -> Tuple[Tuple[QualityMAPGEmitterState, Genotype, optax.OptState], Any]:
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
        emitter_state: QualityMAPGEmitterState,
        policy_params: Params,
        policy_optimizer_state: optax.OptState,
    ) -> Tuple[QualityMAPGEmitterState, Params, optax.OptState]:
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
        policy_optimizer_state, policy_params = self._update_policy(
            critic_params=emitter_state.critic_params,
            policy_optimizer_state=policy_optimizer_state,
            policy_params=policy_params,
            transitions=transitions,
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
        transitions: QDTransition,
    ) -> Tuple[optax.OptState, Params]:

        # compute loss
        _policy_losses, policy_gradients = self._matd3_policy_loss_fn(
            policy_params,
            critic_params,
            transitions,
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

        return policy_optimizer_state, policy_params


    @partial(jax.jit, static_argnames=("self"))
    def unflatten_obs_fn(self, global_obs: jnp.ndarray) -> dict[int, jnp.ndarray]:
        agent_obs = {}
        for agent_idx, obs_indices in self._env.agent_obs_mapping.items():
                agent_obs[agent_idx] = global_obs[obs_indices]
        
        return agent_obs
    

    @partial(jax.jit, static_argnames=("self"))
    def _unflatten_actions_fn(self, flatten_action: jnp.ndarray) -> dict[int, jax.Array]:
        """Because the actions in the form of Dict[int, jnp.array] is flatten by 
        flatten_actions = jnp.concatenate([a for a in actions.values()]) so we do this way
        """

        actions = {}
        start = 0
        for agent_idx, size in self._env.get_action_sizes().items():
            end = start + size
            actions[agent_idx] = flatten_action[start:end]
            start = end
        return actions

    @partial(jax.jit, static_argnames=("self"))
    def _unflatten_obs_fn(self, global_obs: jnp.ndarray) -> dict[int, jnp.ndarray]:
        agent_obs = {}
        for agent_idx, obs_indices in self._env.agent_obs_mapping.items():
                agent_obs[agent_idx] = global_obs[obs_indices]
        
        return agent_obs
    

    @partial(jax.jit, static_argnames=("self"))
    def _matd3_critic_loss_fn(
        self,
        critic_params: Params,
        target_policy_params: List[Params],
        target_critic_params: Params,
        transitions: QDTransition,
        random_key: RNGKey,
    ) -> jnp.ndarray:
        """Critics loss function for TD3 agent.

        Args:
            critic_params: critic parameters.
            target_policy_params: target policy parameters.
            target_critic_params: target critic parameters.
            transitions: collected transitions.

        Returns:
            Return the loss function used to train the critic in TD3.
        """
        unflatten_next_obs = jax.vmap(self._unflatten_obs_fn)(transitions.next_obs)

        next_actions = {}
        for agent_idx, (params, agent_obs) in enumerate(
                zip(
                target_policy_params, unflatten_next_obs.values()
            )
        ):
            a = self._policy_network[agent_idx].apply(params, agent_obs)
            random_key, subkey = jax.random.split(random_key)
            noise = (jax.random.normal(subkey, shape=a.shape) * self._config.policy_noise).clip(-self._config.noise_clip, self._config.noise_clip)
            a = (a + noise).clip(-1.0, 1.0)
            next_actions[agent_idx] = a

        flatten_next_actions = jnp.concatenate([a for a in next_actions.values()], axis=-1)

        next_q = self._critic_network.apply(  # type: ignore
            target_critic_params, obs=transitions.next_obs, actions=flatten_next_actions
        )
        next_v = jnp.min(next_q, axis=-1)
        target_q = jax.lax.stop_gradient(
            transitions.rewards * self._config.reward_scaling
            + (1.0 - transitions.dones) * self._config.discount * next_v
        )
        q_old_action = self._critic_network.apply(  # type: ignore
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


    @partial(jax.jit, static_argnames=("self"))
    def _matd3_policy_loss_fn(
        self,
        policy_params: List[Params],
        critic_params: Params,
        transitions: QDTransition,
    ) -> Tuple[List[jnp.ndarray], List[Params]]:
        """Policy loss function for MATD3 agent."""

        unflatten_obs = jax.vmap(self._unflatten_obs_fn)(transitions.obs)
        num_agents = len(policy_params)
        
        def single_agent_policy_loss(agent_params: Params, agent_idx: int) -> jnp.ndarray:
            """Compute policy loss for a single agent"""
            
            # Get all current actions using current policy parameters
            agent_actions = []
            for i in range(num_agents):
                if i == agent_idx:
                    # Use the agent_params being optimized for this agent
                    action = self._policy_network[i].apply(agent_params, unflatten_obs[i])
                else:
                    # Use current policy_params for other agents
                    action = self._policy_network[i].apply(policy_params[i], unflatten_obs[i])
                agent_actions.append(action)
            
            # Flatten all actions
            flatten_actions = jnp.concatenate(agent_actions, axis=-1)
            
            # Get Q-value using the critic
            q_value = self._critic_network.apply(
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
            # Compute loss and gradient for this agent
            agent_loss, agent_gradient = jax.value_and_grad(single_agent_policy_loss)(
                policy_params[agent_idx], agent_idx
            )

            policy_losses.append(agent_loss)
            policy_gradients.append(agent_gradient)
        
        return policy_losses, policy_gradients