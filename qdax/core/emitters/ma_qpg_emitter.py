"""Implements the PG Emitter from PGA-ME algorithm in jax for brax environments,
based on:
https://hal.archives-ouvertes.fr/hal-03135723v2/file/PGA_MAP_Elites_GECCO.pdf
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
from qdax.core.neuroevolution.losses.matd3_loss import make_matd3_loss_fn
from qdax.core.neuroevolution.networks.networks import QModule
from qdax.environments.multi_agent_wrappers import MultiAgentBraxWrapper
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, Params, RNGKey

from qdax.core.neuroevolution.networks.networks import MLP, QModule

@dataclass
class QualityMAPGConfig:
    """Configuration for QualityPG Emitter"""

    env_batch_size: int = 100
    num_critic_training_steps: int = 300
    num_pg_training_steps: int = 100

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
    max_grad_norm: float = 30.0

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
        
        # Set up the losses and optimizers - return the opt states
        self._policy_loss_fn, self._critic_loss_fn = make_matd3_loss_fn(
            policy_fns_apply=self.create_policy_fns,
            critic_fn=self._critic_network.apply,
            unflatten_obs_fn=jax.vmap(self.unflatten_obs_fn),
            unflatten_actions_fn=jax.vmap(self.unflatten_actions_fn),
            reward_scaling=self._config.reward_scaling,
            discount=self._config.discount,
            noise_clip=self._config.noise_clip,
            policy_noise=self._config.policy_noise,
        )

        # Init optimizers
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
        return True

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
        # actor_optimizer_state = self._actor_optimizer.init(actor_params)
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
        target_actor_params: Params,
        critic_optimizer_state: Params,
        transitions: QDTransition,
        random_key: RNGKey,
    ) -> Tuple[Params, Params, Params, RNGKey]:

        # compute loss and gradients
        random_key, subkey = jax.random.split(random_key)
        critic_loss, critic_gradient = jax.value_and_grad(self._critic_loss_fn)(
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
        actor_losses, actor_gradients = self._policy_loss_fn(
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
        _policy_losses, policy_gradients = self._policy_loss_fn(
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
    def unflatten_actions_fn(self, flatten_action: jnp.ndarray) -> dict[int, jax.Array]:
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

    # Create the policy_fns_apply function (this should be created once and reused)
    @partial(jax.jit, static_argnames=("self"))
    def create_policy_fns(self, index, params, obs):
        return jax.lax.switch(index, [pol.apply for pol in self._policy_network.values()], params, obs)