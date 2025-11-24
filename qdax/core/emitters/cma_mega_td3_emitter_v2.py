"""Similar to cma_mega_td3_emitter.py, however now in each iteration greedy_actor is evaluated
and add its transitions to the replay_buffer, however, the greedy_actor are not added to the repertoire!!!
-> check if on-policy transitions help me greedy actor get better performance 
-> ALSO THE EVAL FUNCTION IS NOISE ADDED IN ACTION to enhance exploration
"""


from __future__ import annotations


from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional, Tuple, Any

import jax
import jax.numpy as jnp
import optax

from qdax.core.cmaes import CMAES, CMAESState
from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    get_cells_indices,
)
from qdax.core.neuroevolution.buffers.buffer import QDTransition, ReplayBuffer
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.environments.base_wrappers import QDEnv
from qdax.core.neuroevolution.networks.td3_networks import  make_td3_networks, MLP
from qdax.core.neuroevolution.losses.td3_loss import make_td3_loss_fn, td3_policy_loss_fn
from qdax.types import (
    Centroid,
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    Gradient,
    RNGKey,
    Params
)
from brax.envs import State as EnvState
@dataclass
class CMAMEGATD3Config:
    "Configuration for cma-mega(td3)"
    env_batch_size: int = 100
    rl_update_batch_size: int = 256
    rl_estimate_batch_size: int = 65_536
    learning_rate: float = 1.0
    sigma_g: float = 3.16
    mirrored_sampling: bool = False
    es_noise: float = 0.02

    # TD3 params
    replay_buffer_size: int = 1000000
    num_warmstart_steps: int = 25_600
    num_critic_training_steps: int = 3000
    policy_hidden_layer_size: Tuple[int, ...] = (128, 128)
    actor_hidden_layer_size: Tuple[int, ...] = (128, 128)
    critic_hidden_layer_size: Tuple[int, ...] = (256, 256)
    critic_learning_rate: float = 3e-4
    actor_learning_rate: float = 3e-4
    noise_clip: float = 0.5
    policy_noise: float = 0.2
    discount: float = 0.99
    reward_scaling: float = 1.0
    batch_size: int = 256
    soft_tau_update: float = 0.005
    policy_delay: int = 2
    max_grad_norm: float = 30.0



class CMAMEGATD3EmitterState(EmitterState):
    """
    Emitter state for the CMA-MEGA emitter.

    Args:
        theta: current genotype from where candidates will be drawn.
        theta_grads: normalized fitness and descriptors gradients of theta.
        random_key: a random key to handle stochastic operations. Used for
            state update only, another key is used to emit. This might be
            subject to refactoring discussions in the future.
        cmaes_state: state of the underlying CMA-ES algorithm
        previous_fitnesses: store last fitnesses of the repertoire. Used to
            compute the improvment.
    """

    theta: Genotype
    theta_grads: Gradient
    cmaes_state: CMAESState
    coeffs: jnp.ndarray
    previous_fitnesses: Fitness

    # TD3
    critic_params: Params
    critic_optimizer_state: optax.OptState
    actor_params: Params
    actor_opt_state: optax.OptState
    target_critic_params: Params
    target_actor_params: Params
    replay_buffer: ReplayBuffer

    random_key: RNGKey
    steps: jnp.ndarray = jnp.array(0)
    critic_training_steps: jnp.ndarray = jnp.array(0)

class CMAMEGATD3Emitter(Emitter):
    def __init__(
        self,
        scoring_function: Callable[
            [Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]
        ],
        config: CMAMEGATD3Config,
        num_descriptors: int,
        centroids: Centroid,
        env: QDEnv,
    ):
        """
        Class for the emitter of CMA Mega from "Differentiable Quality Diversity" by
        Fontaine et al.

        Args:
            scoring_function: a function to score individuals, outputing fitness,
                descriptors and extra scores. With this emitter, the extra score
                contains gradients and normalized gradients.
            batch_size: number of solutions sampled at each iteration
            learning_rate: rate at which the mean of the distribution is updated.
            num_descriptors: number of descriptors
            centroids: centroids of the repertoire used to store the genotypes
            sigma_g: standard deviation for the coefficients
            es_batch_size: if we cannot compute exact gradient, we need to estimate gradient through ES,
            mirrored_sampling: flag if apply mirrored sampling for ES gradient estimation
            es_noise: noise for ES to estimate gradients
        """
    
        self._config = config
        self._scoring_function = scoring_function
        self._num_descriptors = num_descriptors
        # weights used to update the gradient direction through a linear combination
        self._weights = jnp.expand_dims(
            jnp.log(config.env_batch_size + 0.5) - jnp.log(jnp.arange(1, config.env_batch_size + 1)), axis=-1
        )
        self._weights = self._weights / (self._weights.sum())

        self._env = env

        # Init Critics

        self._actor_network, self._critic_network, = make_td3_networks(
            action_size=self._env.action_size,
            critic_hidden_layer_sizes=config.critic_hidden_layer_size,
            policy_hidden_layer_sizes=config.actor_hidden_layer_size,
        )
        policy_layer_sizes = self._config.policy_hidden_layer_size + (env.action_size,)
        self._policy_network = MLP(
            layer_sizes=policy_layer_sizes,
            final_activation=jnp.tanh,
        )

        # Set up the losses and optimizers - return the opt states
        self._actor_loss_fn, self._critic_loss_fn = make_td3_loss_fn(
            policy_fn=self._actor_network.apply,
            critic_fn=self._critic_network.apply,
            reward_scaling=self._config.reward_scaling,
            discount=self._config.discount,
            noise_clip=self._config.noise_clip,
            policy_noise=self._config.policy_noise,
        )

        self._policy_loss_fn = partial(
            td3_policy_loss_fn,
            policy_fn=self._policy_network.apply,
            critic_fn=self._critic_network.apply,
        )

        # define a CMAES instance - used to update the coeffs
        self._cmaes = CMAES(
            population_size=config.env_batch_size,
            search_dim=num_descriptors + 1,
            # no need for fitness function in that specific case
            fitness_function=None,  # type: ignore
            num_best=config.env_batch_size,
            init_sigma=config.sigma_g,
            bias_weights=True,
            delay_eigen_decomposition=True,
        )

        self._centroids = centroids

        self._cma_initial_state = self._cmaes.init()

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
        else:
            self._actor_optimizer = optax.adam(learning_rate=self._config.actor_learning_rate)
            self._critic_optimizer = optax.adam(learning_rate=self._config.critic_learning_rate)



    @partial(jax.jit, static_argnames=("self",))
    def init(
        self, init_genotypes: Genotype, random_key: RNGKey
    ) -> Tuple[CMAMEGATD3EmitterState, RNGKey]:
        """
        Initializes the CMA-MEGA emitter.


        Args:
            init_genotypes: initial genotypes to add to the grid.
            random_key: a random key to handle stochastic operations.

        Returns:
            The initial state of the emitter.
        """

        # define init theta as 0
        theta = jax.tree_util.tree_map(
            lambda x: jnp.zeros_like(x[:1, ...]),
            init_genotypes,
        )

        theta_grads_init = jax.tree_util.tree_map(
            lambda x: jnp.ones(x.shape + (1 + self._num_descriptors,)) * jnp.nan, theta
        )

        # Initialize repertoire with default values
        num_centroids = self._centroids.shape[0]
        default_fitnesses = -jnp.inf * jnp.ones(shape=num_centroids)

         # Draw random coefficients - first iteration of cma-es offpsrings will be drawed from this
        default_coeffs, random_key = self._cmaes.sample(
            cmaes_state=self._cma_initial_state, random_key=random_key
        )
        # Make sure the fitness coefficient is positive
        default_coeffs = default_coeffs.at[:, 0].set(jnp.abs(default_coeffs[:, 0]))


        #### TD3
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

        random_key, subkey = jax.random.split(random_key)
        actor_params = self._actor_network.init(subkey, fake_obs)
        target_actor_params = jax.tree_util.tree_map(lambda x: x, actor_params)

        # Prepare init optimizer states
        critic_optimizer_state = self._critic_optimizer.init(critic_params)
        actor_optimizer_state = self._actor_optimizer.init(actor_params)

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
        
        random_key, subkey = jax.random.split(random_key)
        return (
            CMAMEGATD3EmitterState(
                theta=theta,
                theta_grads=theta_grads_init,
                random_key=subkey,
                cmaes_state=self._cma_initial_state,
                previous_fitnesses=default_fitnesses,
                coeffs=default_coeffs,
                critic_params=critic_params,
                critic_optimizer_state=critic_optimizer_state,
                actor_params=actor_params,
                actor_opt_state=actor_optimizer_state,
                target_critic_params=target_critic_params,
                target_actor_params=target_actor_params,
                replay_buffer=replay_buffer,
                steps=jnp.array(0),
                critic_training_steps=jnp.array(0)
            ),
            random_key,
        )


    @partial(jax.jit, static_argnames=("self",),)
    def warmstart_play_step_fn(
        self,
        env_state: EnvState,
        random_key: RNGKey,
    ):
        """
        Play an environment step and return the updated state and the transition.
        """
        random_key, subkey = jax.random.split(random_key)
        actions = jax.random.uniform(subkey, (self._env.action_size,), minval=-1.0, maxval=1.0)

        next_state = self._env.step(env_state, actions)

        transition = QDTransition(
            obs=env_state.obs,
            next_obs=next_state.obs,
            rewards=next_state.reward,
            dones=next_state.done,
            actions=actions,
            truncations=next_state.info["truncation"],
            state_desc=env_state.info["state_descriptor"],
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

    @partial(jax.jit, static_argnames=("self",))
    def emit(
        self,
        repertoire: Optional[MapElitesRepertoire],
        emitter_state: CMAMEGATD3EmitterState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey, jnp.ndarray]:
        """
        Iteratively changing between emit ES samples and CMAES samples to estimate gradient and update theta.
        At the very first time, steps will be 1, it will based on the gradient of ES to update theta by CMAES
        Args:
            repertoire: a repertoire of genotypes (unused).
            emitter_state: the state of the CMA-MEGA emitter.
            random_key: a random key to handle random operations.

        Returns:
            New genotypes and a new random key.
        """

        samples, random_key = jax.lax.cond(
            emitter_state.steps % 2 == 0,
            lambda args: self.emit_es(*args),
            lambda args: self.emit_cmaes(*args),
            (
                repertoire,
                emitter_state,
                random_key
            )
        )

        return samples, random_key, jnp.array(0)


    def emit_cmaes(
        self,
        repertoire: Optional[MapElitesRepertoire],
        emitter_state: CMAMEGATD3EmitterState,
        random_key: RNGKey,
    ):
        """
        Emits new individuals. Interestingly, this method does not directly modifies
        individuals from the repertoire but sample from a distribution. Hence the
        repertoire is not used in the emit function.

        Args:
            repertoire: a repertoire of genotypes (unused).
            emitter_state: the state of the CMA-MEGA emitter.
            random_key: a random key to handle random operations.

        Returns:
            New genotypes and a new random key.
        """
        
        # retrieve elements from the emitter state
        theta = jax.tree_util.tree_map(
            lambda x: jnp.nan_to_num(x), emitter_state.theta
        )

        # get grads - remove nan and first dimension
        grads = jax.tree_util.tree_map(
            lambda x: jnp.nan_to_num(x.squeeze(axis=0)), emitter_state.theta_grads
        )


        # Use the set of coeffs sampled in state_update  
        coeffs = emitter_state.coeffs

        update_grad = jax.tree_util.tree_map(
            lambda x: jnp.tensordot(coeffs, x, axes=[[-1], [-1]]),
            grads
        )

        # Compute new candidates
        new_thetas = jax.tree_util.tree_map(lambda x, y: x + y, theta, update_grad)

        return new_thetas, random_key


    def emit_es(
        self,
        repertoire: Optional[MapElitesRepertoire],
        emitter_state: CMAMEGATD3EmitterState,
        random_key: RNGKey,
    ):
        """
        Emits samples to estimate the gradient through ES

        Args:
            repertoire: a repertoire of genotypes (unused).
            emitter_state: the state of the CMA-MEGA emitter.
            random_key: a random key to handle random operations.

        Returns:
            New genotypes and a new random key.
        """
        
        # Get random keys the same structure as mean
        num_leaves = len(jax.tree_util.tree_leaves(emitter_state.theta))
        random_key, subkey = jax.random.split(random_key)
        keys = jax.random.split(subkey, num_leaves)
        keys_tree = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(emitter_state.theta),
            keys
        )

        if self._config.mirrored_sampling:
            half_noise = jax.tree_util.tree_map(
                lambda m, k: jax.random.normal(k, (self._config.env_batch_size // 2,) + m.shape[1:]) * self._config.es_noise,
                emitter_state.theta, keys_tree
            )
            noise = jax.tree_util.tree_map(
                lambda x: jnp.concatenate([x, -x], axis=0),
                half_noise
            )

        else:
            noise = jax.tree_util.tree_map(
                lambda m, k: jax.random.normal(k, (self._config.env_batch_size,) + m.shape[1:]) * self._config.es_noise,
                emitter_state.theta, keys_tree
            )

        samples = jax.tree_util.tree_map(
            lambda m, n: m + n, 
            emitter_state.theta, noise
        )

        return samples, random_key
    


    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def state_update(
        self,
        emitter_state: CMAMEGATD3EmitterState,
        repertoire: MapElitesRepertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: Optional[ExtraScores] = None,
    ) -> Optional[EmitterState]:
        """
        Updates the CMA-MEGA emitter state.

        Note: in order to recover the coeffs that where used to sample the genotypes,
        we reuse the emitter state's random key in this function.

        Note: we use the update_state function from CMAES, a function that suppose
        that the candidates are already sorted. We do this because we have to sort
        them in this function anyway, in order to apply the right weights to the
        terms when update theta.

        Args:
            emitter_state: current emitter state
            repertoire: the current genotypes repertoire
            genotypes: the genotypes of the batch of emitted offspring (unused).
            fitnesses: the fitnesses of the batch of emitted offspring.
            descriptors: the descriptors of the emitted offspring.
            extra_scores: unused

        Returns:
            The updated emitter state.
        """

        emitter_state = jax.lax.cond(
            emitter_state.steps % 2 == 0,
            lambda args: self._estimate_gradient_ES(*args),
            lambda args: self._update_cmaes(*args),
            (emitter_state, repertoire, genotypes, fitnesses, descriptors, extra_scores)
        )

        return emitter_state

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

        MACEMRLEmitter uses the transitions from the genotypes that were generated
        by other emitters.
        """
        return True


    @partial(jax.jit, static_argnames=("self"))
    def _update_cmaes(
        self,
        emitter_state: CMAMEGATD3EmitterState,
        repertoire: MapElitesRepertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: Optional[ExtraScores] = None,
    ):
        """
        Updates the CMA-MEGA emitter state.

        Note: in order to recover the coeffs that where used to sample the genotypes,
        we reuse the emitter state's random key in this function.

        Note: we use the update_state function from CMAES, a function that suppose
        that the candidates are already sorted. We do this because we have to sort
        them in this function anyway, in order to apply the right weights to the
        terms when update theta.

        Args:
            emitter_state: current emitter state
            repertoire: the current genotypes repertoire
            genotypes: the genotypes of the batch of emitted offspring (unused).
            fitnesses: the fitnesses of the batch of emitted offspring.
            descriptors: the descriptors of the emitted offspring.
            extra_scores: unused

        Returns:
            The updated emitter state.
        """

        print(f"IN CMAES!: cur_step {emitter_state.steps}")

        # get the transitions out of the dictionary
        assert "transitions" in extra_scores.keys(), "Missing transitions or wrong key"
        transitions = extra_scores["transitions"]

        # add transitions in the replay buffer
        replay_buffer = emitter_state.replay_buffer.insert(transitions)
        emitter_state = emitter_state.replace(
            replay_buffer=replay_buffer
        )

        # EVAL GREEDY ACTOR AND SAVE ONLINE TRANSITIONS
        _, _, emitter_state = self.evaluate_greedy_actor(emitter_state)


        def scan_train_critics(
            carry: CMAMEGATD3EmitterState, unused: Any
        ) -> Tuple[CMAMEGATD3EmitterState, Any]:
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

        # retrieve elements from the emitter state
        cmaes_state = emitter_state.cmaes_state
        random_key = emitter_state.random_key

        theta = jax.tree_util.tree_map(
            lambda x: jnp.nan_to_num(x), emitter_state.theta
        ) 

        grads = jax.tree_util.tree_map(
            lambda x: jnp.nan_to_num(x[0]), emitter_state.theta_grads
        )

        # Update the archive and compute the improvements
        indices = get_cells_indices(descriptors, repertoire.centroids)
        improvements = fitnesses - emitter_state.previous_fitnesses[indices]

        # condition for being a new cell
        condition = improvements == jnp.inf

        # criteria: fitness if new cell, improvement else
        ranking_criteria = jnp.where(condition, fitnesses, improvements)

        # make sure to have all the new cells first
        new_cell_offset = jnp.max(ranking_criteria) - jnp.min(ranking_criteria)

        ranking_criteria = jnp.where(
            condition, ranking_criteria + new_cell_offset, ranking_criteria
        )

        # sort indices according to the criteria
        sorted_indices = jnp.flip(jnp.argsort(ranking_criteria))

        # Use the same coeffs that used to emitted the offsprings
        coeffs = emitter_state.coeffs

        # get the gradients that must be applied

        jax.tree_util.tree_map(
            lambda x: print(f"coeffs shape {coeffs}, grads shape {x.shape}"),
            grads
        )

        update_grad = jax.tree_util.tree_map(
            lambda x: jnp.tensordot(coeffs, x, axes=[[-1], [-1]]),
            grads
        )

        jax.tree_util.tree_map(
            lambda x: print(f"update_grad shape {x}"),
            update_grad
        )


        # weight terms - based on improvement rank
        gradient_step = jax.tree_util.tree_map(
            lambda x: jnp.einsum('i,i...->...', 
                                self._weights[sorted_indices].squeeze(), 
                                x),
            update_grad
        )

        # update theta
        theta = jax.tree_util.tree_map(
            lambda x, y: x + self._config.learning_rate * y, theta, gradient_step
        )

        # Update CMA Parameters
        sorted_candidates = coeffs[sorted_indices]
        cmaes_state = self._cmaes.update_state(cmaes_state, sorted_candidates)

        # If no improvement draw randomly and re-initialize parameters
        reinitialize = jnp.all(improvements < 0) + self._cmaes.stop_condition(
            cmaes_state
        )

        # re-sample
        random_theta, random_key = repertoire.sample(random_key, 1)

        # update theta in case of reinit
        theta = jax.tree_util.tree_map(
            lambda x, y: jnp.where(reinitialize, x, y), random_theta, theta
        )

        # update cmaes state in case of reinit
        cmaes_state = jax.tree_util.tree_map(
            lambda x, y: jnp.where(reinitialize, x, y),
            self._cma_initial_state,
            cmaes_state,
        )

        # Draw random coefficients - for new offsprings
        new_coeffs, random_key = self._cmaes.sample(
            cmaes_state=cmaes_state, random_key=random_key
        )
        # Make sure the fitness coefficient is positive
        new_coeffs = new_coeffs.at[:, 0].set(jnp.abs(new_coeffs[:, 0]))

        # create new emitter state
        emitter_state = emitter_state.replace(
            theta=theta,
            theta_grads=emitter_state.theta_grads,
            random_key=random_key,
            cmaes_state=cmaes_state,
            previous_fitnesses=repertoire.fitnesses,
            coeffs=new_coeffs,
            replay_buffer=replay_buffer,
            steps=emitter_state.steps+1
        )

        return emitter_state


    @partial(jax.jit, static_argnames=("self"))
    def _estimate_gradient_ES(
        self,
        emitter_state: CMAMEGATD3EmitterState,
        repertoire: MapElitesRepertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: Optional[ExtraScores] = None,
    ):
        
        print(f"IN ES!: cur_step {emitter_state.steps}")

        ### UPDATE ACTOR CRITIC
        # get the transitions out of the dictionary
        assert "transitions" in extra_scores.keys(), "Missing transitions or wrong key"
        transitions = extra_scores["transitions"]

        # add transitions in the replay buffer
        replay_buffer = emitter_state.replay_buffer.insert(transitions)

        emitter_state = emitter_state.replace(
            replay_buffer=replay_buffer
        )

        # EVAL GREEDY ACTOR AND SAVE ONLINE TRANSITIONS
        _, _, emitter_state = self.evaluate_greedy_actor(emitter_state)

        def scan_train_critics(
            carry: CMAMEGATD3EmitterState, unused: Any
        ) -> Tuple[CMAMEGATD3EmitterState, Any]:
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


        ### GRADIENT ESTIMATION
        mean = emitter_state.theta
        random_key = emitter_state.random_key
        noise = jax.tree_util.tree_map(
            lambda g, m: (g - m) / self._config.es_noise,
            genotypes, mean
        )

        # fitnesses_and_desc = jnp.concatenate([fitnesses[...,None],  descriptors], axis=1) # shape (n_samples, 1+desc_dim)

        # ranking_indices = jnp.argsort(fitnesses_and_desc, axis=0) # shape (n_samples, 1+desc_dim)
        # ranks = jnp.argsort(ranking_indices, axis=0) 
        # ranks = (ranks / (self._config.env_batch_size - 1)) - 0.5

        # gradients = jax.tree_util.tree_map(
        #     lambda n: jnp.sum(
        #         jnp.expand_dims(ranks, axis=[i for i in range(1, n.ndim)]) * n[..., None], 
        #         axis=0,
        #         keepdims=True,
        #     ) / (self._config.es_noise * self._config.env_batch_size),
        #     noise,
        # ) # gradients is now has dim (1, problem_dim, 1+bd_dim, ...)


        ranking_indices = jnp.argsort(descriptors, axis=0) # shape (n_samples, desc_dim)
        ranks = jnp.argsort(ranking_indices, axis=0) 
        ranks = (ranks / (self._config.env_batch_size - 1)) - 0.5

        bd_gradients = jax.tree_util.tree_map(
            lambda n: jnp.sum(
                jnp.expand_dims(ranks, axis=[i for i in range(1, n.ndim)]) * n[..., None], 
                axis=0,
                keepdims=True,
            ) / (self._config.es_noise * self._config.env_batch_size),
            noise,
        ) # gradients is now has dim (1, problem_dim, bd_dim, ...)

        transitions, random_key = replay_buffer.sample(
            random_key, self._config.rl_estimate_batch_size
        )

        f_gradients = jax.vmap(jax.grad(self._policy_loss_fn), in_axes=(0, None, None))( # need this weird vmap bcs the theta shape is (1, param_dim)
            emitter_state.theta,
            emitter_state.critic_params,
            transitions,
        ) # f_gradients has dim (1, problem_dim)

        jax.tree_util.tree_map(
            lambda x: print(f"f_gradients shape: {x.shape}"),
            f_gradients
        )

        gradients = jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate([x[..., None], y], axis=-1),
            f_gradients, bd_gradients
        ) # shape: (1, problem_dim, 1+bd_dim)

        norm_gradients = jax.tree_util.tree_map(
            lambda x: jnp.linalg.norm(x.reshape(-1, x.shape[-1]), axis=0, keepdims=True), # need reshape because neural network
            gradients,
        ) # shape: (1, 1+bd_dim)

        normalized_gradients = jax.tree_util.tree_map(
            lambda x, y: x / (y + 1e-8), gradients, norm_gradients
        )

        jax.tree_util.tree_map(
            lambda x: print("gradient shape at estimate ES",x.shape),
            normalized_gradients
        )

        emitter_state = emitter_state.replace(
            theta_grads=normalized_gradients,
            random_key=random_key,
            steps=emitter_state.steps+1,
            replay_buffer=replay_buffer
        )

        return emitter_state
    
    @partial(jax.jit, static_argnames=("self",))
    def _train_critics(
        self, emitter_state: CMAMEGATD3EmitterState
    ) -> CMAMEGATD3EmitterState:
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
            random_key, sample_size=self._config.rl_update_batch_size
        )

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
            replay_buffer=replay_buffer,
            critic_training_steps=emitter_state.critic_training_steps+1,
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
        actor_params: Params,
        actor_opt_state: optax.OptState,
        target_actor_params: Params,
        critic_params: Params,
        transitions: QDTransition,
    ) -> Tuple[optax.OptState, Params, Params]:

        # Update greedy actor
        policy_loss, policy_gradient = jax.value_and_grad(self._actor_loss_fn)(
            actor_params,
            critic_params,
            transitions,
        )
        (
            policy_updates,
            actor_optimizer_state,
        ) = self._actor_optimizer.update(policy_gradient, actor_opt_state)
        actor_params = optax.apply_updates(actor_params, policy_updates)

        # Soft update of target greedy actor
        target_actor_params = jax.tree_map(
            lambda x1, x2: (1.0 - self._config.soft_tau_update) * x1
            + self._config.soft_tau_update * x2,
            target_actor_params,
            actor_params,
        )

        return (
            actor_optimizer_state,
            actor_params,
            target_actor_params,
        )
    
    @partial(jax.jit, static_argnames=("self"))
    def evaluate_greedy_actor(
        self,
        emitter_state: CMAMEGATD3EmitterState,
    ):
    
        samples = jax.tree_util.tree_map(
            lambda x: jnp.repeat(x[None, ...], self._config.env_batch_size//10, axis=0),
              emitter_state.actor_params
        )
        fitnesses, descriptors, extra_scores, random_key = self._scoring_function(
            samples, emitter_state.random_key
        )
        transitions = extra_scores["transitions"]
        replay_buffer = emitter_state.replay_buffer.insert(transitions)


        fitnesses = jnp.mean(fitnesses, axis=0)
        descriptors = jnp.mean(descriptors, axis=0)

        emitter_state=emitter_state.replace(
            replay_buffer=replay_buffer
        )

        return fitnesses, descriptors, emitter_state