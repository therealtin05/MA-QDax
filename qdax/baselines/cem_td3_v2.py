""" Implements the CEMTD3 algorithm in jax for brax environments, based on:
https://arxiv.org/pdf/1802.09477.pdf

Just like v1, but here we use the replay buffer from evorl, which only store 
transitions before the first termination

In this verion the update mechanism is
for o in learning_offsprings:
    for i -> num_rl_updates
        update critic 
        update o 
"""

from dataclasses import dataclass
from functools import partial
from typing import Callable, Tuple, Optional, Any

import jax
import optax
from brax.envs import Env
from brax.envs import State as EnvState
from jax import numpy as jnp
from flax import linen as nn

from qdax.core.neuroevolution.buffers.buffer import (
    QDTransition,
    Transition,
)
from qdax.core.neuroevolution.buffers.evorl_buffer import (
    ReplayBuffer,
    ReplayBufferState
)
from qdax.core.neuroevolution.losses.td3_loss import (
    td3_critic_loss_fn,
    td3_policy_loss_fn,
)
from qdax.core.neuroevolution.mdp_utils import TrainingState, get_first_episode
from qdax.core.neuroevolution.networks.td3_networks import make_td3_networks
from qdax.core.neuroevolution.sac_td3_utils import generate_unroll
from qdax.types import (
    Action,
    Descriptor,
    Mask,
    Metrics,
    Observation,
    Params,
    Reward,
    RNGKey,
    Genotype,
    Fitness,
    ExtraScores
)


class CEMTD3TrainingState(TrainingState):
    """Contains training state for the learner."""
    mean_policy_params: Params
    var_policy_params: Params
    damp: jnp.ndarray

    critic_optimizer_state: optax.OptState
    critic_params: Params
    target_critic_params: Params

    random_key: RNGKey
    steps: jnp.ndarray
    


@dataclass
class CEMTD3Config:
    """Configuration for the CEMTD3 algorithm"""

    episode_length: int = 1000 
    batch_size: int = 256
    # CEM
    warmup_iters: int = 10 # number of iter update with only CEM
    population_size: int = 10
    num_best: Optional[int] = None
    damp_init: float = 1e-3
    damp_final: float = 1e-5
    damp_tau : float = 0.95
    rank_weight_shift: float = 1.0
    mirror_sampling: bool = False
    weighted_update: bool = True
    num_learning_offspring: Optional[int] = None
    # RL
    num_rl_updates_per_iter: int = 4096 
    soft_tau_update: float = 0.005
    expl_noise: float = 0.1
    critic_hidden_layer_size: Tuple[int, ...] = (256, 256)
    policy_hidden_layer_size: Tuple[int, ...] = (256, 256)
    critic_learning_rate: float = 3e-4
    policy_learning_rate: float = 3e-4
    discount: float = 0.99
    noise_clip: float = 0.5
    policy_noise: float = 0.2
    reward_scaling: float = 1.0
    policy_delay: int = 1
    use_layer_norm: bool = True
    

class CEMTD3:
    """
    A collection of functions that define the Twin Delayed Deep Deterministic Policy
    Gradient agent (CEMTD3), ref: https://arxiv.org/pdf/1802.09477.pdf
    """

    # def __init__(self, config: CEMTD3Config, action_size: int, ):
    def __init__(
        self,
        config: CEMTD3Config,
        action_size: int,
        scoring_function: Callable[
            [Genotype, RNGKey], Tuple[Fitness, Transition, RNGKey]
        ],
        replay_buffer: ReplayBuffer,
        num_eval: int = 20
    ):
        if config.mirror_sampling:
            assert config.population_size % 2 == 0, "pop_size must be even for mirror sampling"
        self._config = config
        self._scoring_function = scoring_function
        self._policy, self._critic, = make_td3_networks(
            action_size=action_size,
            critic_hidden_layer_sizes=self._config.critic_hidden_layer_size,
            policy_hidden_layer_sizes=self._config.policy_hidden_layer_size,
            activation=nn.relu,
            use_layer_norm=self._config.use_layer_norm,
        )

        if self._config.num_best is None:
            self._num_best = self._config.population_size // 2
        else:
            self._num_best = self._config.num_best

        # weights parameters
        if self._config.weighted_update:
            self._weights = jnp.log(
                (self._num_best + self._config.rank_weight_shift) / jnp.arange(start=1, stop=(self._num_best + 1))
            )
        else:
            self._weights = jnp.ones(self._config.num_best)
        # scale weights
        self._weights = self._weights / (self._weights.sum())

        if self._config.num_learning_offspring is None:
            self._num_learning_offspring = self._config.population_size // 2
        else:
            self._num_learning_offspring = self._config.num_learning_offspring

        self._replay_buffer = replay_buffer
        self._num_eval = num_eval

    def init(
        self, random_key: RNGKey, action_size: int, observation_size: int
    ) -> CEMTD3TrainingState:
        """Initialise the training state of the CEMTD3 algorithm, through creation
        of optimizer states and params.

        Args:
            random_key: a random key used for random operations.
            action_size: the size of the action array needed to interact with the
                environment.
            observation_size: the size of the observation array retrieved from the
                environment.

        Returns:
            the initial training state.
        """

        # Initialize critics and policy params
        fake_obs = jnp.zeros(shape=(observation_size,))
        fake_action = jnp.zeros(shape=(action_size,))
        random_key, subkey_1, subkey_2 = jax.random.split(random_key, num=3)
        critic_params = self._critic.init(subkey_1, obs=fake_obs, actions=fake_action)
        mean_policy_params = self._policy.init(subkey_2, fake_obs)
        var_policy_params = jax.tree_util.tree_map(
            lambda x: jnp.full_like(x, self._config.damp_init), 
            mean_policy_params
        )

        # Initialize target networks
        target_critic_params = jax.tree_util.tree_map(
            lambda x: jnp.asarray(x.copy()), critic_params
        )

        # Create and initialize optimizers
        critic_optimizer_state = optax.adam(learning_rate=1.0).init(critic_params)

        # Initial training state
        training_state = CEMTD3TrainingState(
            mean_policy_params=mean_policy_params,
            var_policy_params=var_policy_params,
            damp=jnp.array(self._config.damp_init),
            critic_optimizer_state=critic_optimizer_state,
            critic_params=critic_params,
            target_critic_params=target_critic_params,
            random_key=random_key,
            steps=jnp.array(0),
        )

        return training_state

    @partial(jax.jit, static_argnames=("self", "deterministic"))
    def select_action(
        self,
        obs: Observation,
        policy_params: Params,
        random_key: RNGKey,
        expl_noise: float,
        deterministic: bool = False,
    ) -> Tuple[Action, RNGKey]:
        """Selects an action according to CEMTD3 policy. The action can be deterministic
        or stochastic by adding exploration noise.

        Args:
            obs: agent observation(s)
            policy_params: parameters of the agent's policy
            random_key: jax random key
            expl_noise: exploration noise
            deterministic: whether to select action in a deterministic way.
                Defaults to False.

        Returns:
            an action and an updated training state.
        """

        actions = self._policy.apply(policy_params, obs)
        if not deterministic:
            random_key, subkey = jax.random.split(random_key)
            noise = jax.random.normal(subkey, actions.shape) * expl_noise
            actions = actions + noise
            actions = jnp.clip(actions, -1.0, 1.0)
        return actions, random_key
    
    @partial(jax.jit, static_argnames=("self", "env", "deterministic"))
    def play_step_fn(
        self,
        env_state: EnvState,
        training_state: CEMTD3TrainingState,
        env: Env,
        deterministic: bool = False,
    ) -> Tuple[EnvState, CEMTD3TrainingState, Transition]:
        """Plays a step in the environment. Selects an action according to CEMTD3 rule and
        performs the environment step.

        Args:
            env_state: the current environment state
            training_state: the SAC training state
            env: the environment
            deterministic: whether to select action in a deterministic way.
                Defaults to False.

        Returns:
            the new environment state
            the new CEMTD3 training state
            the played transition
        """

        actions, random_key = self.select_action(
            obs=env_state.obs,
            policy_params=training_state.mean_policy_params,
            random_key=training_state.random_key,
            expl_noise=self._config.expl_noise,
            deterministic=deterministic,
        )
        training_state = training_state.replace(
            random_key=random_key,
        )
        next_env_state = env.step(env_state, actions)
        transition = Transition(
            obs=env_state.obs,
            next_obs=next_env_state.obs,
            rewards=next_env_state.reward,
            dones=next_env_state.done,
            truncations=next_env_state.info["truncation"],
            actions=actions,
        )
        return next_env_state, training_state, transition

    @partial(jax.jit, static_argnames=("self", "env", "deterministic"))
    def play_qd_step_fn(
        self,
        env_state: EnvState,
        training_state: CEMTD3TrainingState,
        env: Env,
        deterministic: bool = False,
    ) -> Tuple[EnvState, CEMTD3TrainingState, QDTransition]:
        """Plays a step in the environment. Selects an action according to CEMTD3 rule and
        performs the environment step.

        Args:
            env_state: the current environment state
            training_state: the CEMTD3 training state
            env: the environment
            deterministic: the whether to select action in a deterministic way.
                Defaults to False.

        Returns:
            the new environment state
            the new CEMTD3 training state
            the played transition
        """
        next_env_state, training_state, transition = self.play_step_fn(
            env_state, training_state, env, deterministic
        )
        actions = transition.actions

        truncations = next_env_state.info["truncation"]
        transition = QDTransition(
            obs=env_state.obs,
            next_obs=next_env_state.obs,
            rewards=next_env_state.reward,
            dones=next_env_state.done,
            actions=actions,
            truncations=truncations,
            state_desc=env_state.info["state_descriptor"],
            next_state_desc=next_env_state.info["state_descriptor"],
        )

        return (
            next_env_state,
            training_state,
            transition,
        )


    @partial(jax.jit, static_argnames=("self",))
    def evaluate(self, training_state: CEMTD3TrainingState):
        random_key = training_state.random_key
        random_key, subkey = jax.random.split(random_key)

        expanded_mean = jax.tree_util.tree_map(
            lambda x: jnp.repeat(x[None, ...], self._num_eval, axis=0),
            training_state.mean_policy_params
        )

        fitnesses, transitions, random_key = self._scoring_function(expanded_mean, subkey)

        mean_fitness = fitnesses.mean()
        std_fitness = fitnesses.std()
        
        # Flatten variance PyTree to compute statistics across all parameters
        variance_leaves = jax.tree_util.tree_leaves(training_state.var_policy_params)
        all_variances = jnp.concatenate([v.flatten() for v in variance_leaves])
        
        training_state = training_state.replace(random_key=random_key)

        metrics = {
            "center_fitness_average": mean_fitness, 
            "center_fitness_std": std_fitness,
            "center_fitness_max": jnp.max(fitnesses),
            "center_fitness_min": jnp.min(fitnesses),
            "damp": training_state.damp,
            "max_var": jnp.max(all_variances),
            "mean_var": jnp.mean(all_variances),
            "median_var": jnp.median(all_variances)
        } 

        return training_state, metrics

    @partial(
        jax.jit,
        static_argnames=(
            "self",
            "play_step_fn",
            "bd_extraction_fn",
        ),
    )
    def eval_qd_policy_fn(
        self,
        training_state: CEMTD3TrainingState,
        eval_env_first_state: EnvState,
        play_step_fn: Callable[
            [EnvState, Params, RNGKey],
            Tuple[EnvState, CEMTD3TrainingState, QDTransition],
        ],
        bd_extraction_fn: Callable[[QDTransition, Mask], Descriptor],
    ) -> Tuple[Reward, Descriptor, Reward, Descriptor]:
        """Evaluates the agent's policy over an entire episode, across all batched
        environments for QD environments. Averaged BDs are returned as well.


        Args:
            training_state: the SAC training state
            eval_env_first_state: the initial state for evaluation
            play_step_fn: the play_step function used to collect the evaluation episode

        Returns:
            the true return averaged over batch dimension, shape: (1,)
            the descriptor averaged over batch dimension, shape: (num_descriptors,)
            the true return per environment, shape: (env_batch_size,)
            the descriptor per environment, shape: (env_batch_size, num_descriptors)

        """

        state, training_state, transitions = generate_unroll(
            init_state=eval_env_first_state,
            training_state=training_state,
            episode_length=self._config.episode_length,
            play_step_fn=play_step_fn,
        )
        transitions = get_first_episode(transitions)
        true_returns = jnp.nansum(transitions.rewards, axis=0)
        true_return = jnp.mean(true_returns, axis=-1)

        transitions = jax.tree_util.tree_map(
            lambda x: jnp.swapaxes(x, 0, 1), transitions
        )
        masks = jnp.isnan(transitions.rewards)
        bds = bd_extraction_fn(transitions, masks)

        mean_bd = jnp.mean(bds, axis=0)
        return true_return, mean_bd, true_returns, bds



    @partial(jax.jit, static_argnames=("self",))
    def sample(
        self, training_state: CEMTD3TrainingState, random_key: RNGKey
    ) -> Tuple[Genotype, RNGKey]:
        """
        Sample a population.

        Args:
            training_state: current state of the algorithm
            random_key: jax random key

        Returns:
            A tuple that contains a batch of population size genotypes and
            a new random key.
        """

        random_key, subkey = jax.random.split(random_key)

        # Get random keys the same structure as mean
        num_leaves = len(jax.tree_util.tree_leaves(training_state.mean_policy_params))
        keys = jax.random.split(subkey, num_leaves)
        keys_tree = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(training_state.mean_policy_params),
            keys
        )
        
        if self._config.mirror_sampling:
            # Sample with keys same structure as mean
            half_samples = jax.tree_util.tree_map(
                lambda m, v, k: m + jax.random.normal(k, (self._config.population_size // 2,) + m.shape) * jnp.sqrt(v),
                training_state.mean_policy_params, training_state.var_policy_params, keys_tree
            )
            mirrored = jax.tree_util.tree_map(
                lambda s, m: 2 * jnp.expand_dims(m, axis=0) - s,
                half_samples,
                training_state.mean_policy_params,
            )
            samples = jax.tree_util.tree_map(
                lambda a, b: jnp.concatenate([a, b], axis=0),
                half_samples,
                mirrored,
            )
        else:
            samples = jax.tree_util.tree_map(
                lambda m, v, k: m + jax.random.normal(k, (self._config.population_size,) + m.shape) * jnp.sqrt(v),
                training_state.mean_policy_params, training_state.var_policy_params, keys_tree
            )

        return samples, random_key


    @partial(jax.jit, static_argnames=("self",))
    def _update_single(
        self,
        critic_params: Params,
        policy_params: Params,
        target_policy_params: Params,
        target_critic_params: Params,
        policy_optimizer_state: optax.OptState,
        critic_optimizer_state: optax.OptState,
        replay_buffer_state: ReplayBufferState,
        random_key: RNGKey,
        cur_step: int,
    ) -> Tuple[Params, Params, Params, Params, optax.OptState, optax.OptState]:
        """Performs a single training step: updates policy params and critic params
        through gradient descent.

        Args:
            training_state: the current training state, containing the optimizer states
                and the params of the policy and critic.
            replay_buffer: the replay buffer, filled with transitions experienced in
                the environment.

        Returns:
            A new training state, the buffer with new transitions and metrics about the
            training process.
        """
        random_key, rb_key, critic_key = jax.random.split(random_key, 3)
        # Sample a batch of transitions in the buffer
        samples = self._replay_buffer.sample(
            replay_buffer_state, rb_key
        )

        # Update Critic
        critic_loss, critic_gradient = jax.value_and_grad(td3_critic_loss_fn)(
            critic_params,
            target_policy_params=target_policy_params,
            target_critic_params=target_critic_params,
            policy_fn=self._policy.apply,
            critic_fn=self._critic.apply,
            policy_noise=self._config.policy_noise,
            noise_clip=self._config.noise_clip,
            reward_scaling=self._config.reward_scaling,
            discount=self._config.discount,
            transitions=samples,
            random_key=critic_key,
        )
        critic_optimizer = optax.adam(learning_rate=self._config.critic_learning_rate)
        critic_updates, critic_optimizer_state = critic_optimizer.update(
            critic_gradient, critic_optimizer_state
        )
        critic_params = optax.apply_updates(
            critic_params, critic_updates
        )
        # Soft update of target critic network
        target_critic_params = jax.tree_util.tree_map(
            lambda x1, x2: (1.0 - self._config.soft_tau_update) * x1
            + self._config.soft_tau_update * x2,
            target_critic_params,
            critic_params,
        )

        # Update policy

        def update_policy_step() -> Tuple[Params, Params, optax.OptState]:
            policy_loss, policy_gradient = jax.value_and_grad(td3_policy_loss_fn)(
                policy_params,
                critic_params=critic_params,
                policy_fn=self._policy.apply,
                critic_fn=self._critic.apply,
                transitions=samples,
            )

            policy_optimizer = optax.adam(
                learning_rate=self._config.policy_learning_rate
            )
            (policy_updates, new_policy_optimizer_state,) = policy_optimizer.update(
                policy_gradient, policy_optimizer_state
            )
            new_policy_params = optax.apply_updates(
                policy_params, policy_updates
            )
            # Soft update of target policy
            new_target_policy_params = jax.tree_util.tree_map(
                lambda x1, x2: (1.0 - self._config.soft_tau_update) * x1
                + self._config.soft_tau_update * x2,
                target_policy_params,
                policy_params,
            )
            return new_policy_params, new_target_policy_params, new_policy_optimizer_state

        # Delayed update
        current_policy_state = (
            policy_params,
            target_policy_params,
            policy_optimizer_state,
        )
        policy_params, target_policy_params, policy_optimizer_state = jax.lax.cond(
            cur_step % self._config.policy_delay == 0,
            lambda _: update_policy_step(),
            lambda _: current_policy_state,
            operand=None,
        )

        return policy_params, critic_params, target_policy_params, \
              target_critic_params, policy_optimizer_state, critic_optimizer_state


    @partial(jax.jit, static_argnames=("self",))
    def _scan_update_single(
        self,
        carry: Tuple[Params, Params, Params, Params, optax.OptState, optax.OptState, ReplayBuffer, RNGKey],
        cur_step: jnp.ndarray,
    ) -> Tuple[Tuple[Params, Params, Params, Params, optax.OptState, optax.OptState, ReplayBuffer, RNGKey], Any]:
        """Function to work with jax.lax.scan
        """
        (
            critic_params, policy_params, target_policy_params, target_critic_params,
                policy_optimizer_state, critic_optimizer_state, replay_buffer_state, random_key,
        ) = carry
        random_key, subkey = jax.random.split(random_key)

        policy_params, critic_params, target_policy_params, \
              target_critic_params, policy_optimizer_state, critic_optimizer_state \
              = self._update_single(
                    critic_params, policy_params, target_policy_params, target_critic_params,
                        policy_optimizer_state, critic_optimizer_state, replay_buffer_state, subkey, cur_step
              )
        return (
            critic_params, policy_params, target_policy_params, target_critic_params,
                policy_optimizer_state, critic_optimizer_state, replay_buffer_state, random_key,
        ), ()


    @partial(jax.jit, static_argnames=("self",))
    def _update_one_offspring(
        self,
        critic_params: Params,
        policy_params: Params,
        target_critic_params: Params,
        critic_optimizer_state: optax.OptState,
        replay_buffer_state: ReplayBufferState,
        random_key: RNGKey,
    ) -> Tuple[Params, Params, Params, optax.OptState, ReplayBufferState, RNGKey]:
        """Function to update one single offspring, we need to init the target policy params
        and policy optimization stat here
        """

        target_policy_params = jax.tree_util.tree_map(
            lambda x: jnp.asarray(x.copy()), policy_params
        )
        policy_optimizer = optax.adam(
            learning_rate=self._config.policy_learning_rate
        )
        policy_optimizer_state = policy_optimizer.init(policy_params)
        (
            critic_params, policy_params, target_policy_params, target_critic_params,
                policy_optimizer_state, critic_optimizer_state, replay_buffer_state, random_key,
        ), () = jax.lax.scan(
            self._scan_update_single,
            (
                critic_params, policy_params, target_policy_params, target_critic_params, 
                    policy_optimizer_state, critic_optimizer_state, replay_buffer_state, random_key,
            ),
            jnp.arange(0, self._config.num_rl_updates_per_iter // self._num_learning_offspring),
            # length=self._config.num_rl_updates_per_iter // self._num_learning_offspring
        )

        return (
            critic_params, policy_params, target_critic_params,
                critic_optimizer_state, replay_buffer_state, random_key,
        )


    @partial(jax.jit, static_argnames=("self",))
    def _scan_update_one_offspring(
        self,
        carry: Tuple[Params, Params, optax.OptState, ReplayBufferState, RNGKey],
        x: Params
    ) -> Tuple[Tuple[Params, Params, optax.OptState, ReplayBufferState, RNGKey], Params]:
        """Help scan through number of offsprings updated through policy gradient
        """
        critic_params, target_critic_params, critic_optimizer_state, replay_buffer_state, random_key = carry
        policy_params = x
        (
            critic_params, policy_params, target_critic_params,
                critic_optimizer_state, replay_buffer_state, random_key,
        ) = self._update_one_offspring(
            critic_params, policy_params, target_critic_params,
            critic_optimizer_state, replay_buffer_state, random_key
        )

        return (
            critic_params, target_critic_params,
                critic_optimizer_state, replay_buffer_state, random_key,
        ), policy_params


    @partial(jax.jit, static_argnames=("self",))
    def _pg_update_population(
        self,
        offsprings: Params,
        training_state: CEMTD3TrainingState,
        replay_buffer_state: ReplayBufferState
    ):
        """Update all the selected offpsrings through policy gradient.
        Here we only choose a subset of offspring to apply to the update.
        First half of population: RL-updated offspring
        Second half of population: untorched offspring
        """
        random_key = training_state.random_key
        
        # Split offsprings: first half updated, second half untouched

        selected_offsprings = jax.tree_map(lambda x: x[:self._num_learning_offspring], offsprings)
        untouched_offsprings = jax.tree_map(lambda x: x[self._num_learning_offspring:], offsprings)


        init_carry = (
            training_state.critic_params, 
            training_state.target_critic_params, 
            training_state.critic_optimizer_state,
            replay_buffer_state,
            random_key
        )
        xs = selected_offsprings
        (
            critic_params,
            target_critic_params,
            critic_optimizer_state,
            replay_buffer_state,
            random_key
        ), updated_selected_offsprings = jax.lax.scan(
            self._scan_update_one_offspring,
            init_carry,
            xs
        )       

        # Concatenate: first half RL-updated, second half untouched
        new_offsprings = jax.tree_map(
            lambda a, b: jnp.concatenate([a, b], axis=0),
            updated_selected_offsprings,
            untouched_offsprings, 
        )
    

        new_training_state = training_state.replace(
            critic_params=critic_params,
            critic_optimizer_state=critic_optimizer_state,
            target_critic_params=target_critic_params,
            random_key=random_key,
        )

        return new_offsprings, new_training_state
        

    @partial(jax.jit, static_argnames=("self",))
    def update(
        self,
        training_state: CEMTD3TrainingState,
        replay_buffer_state: ReplayBufferState
    ) -> Tuple[CEMTD3TrainingState, ReplayBufferState, Metrics]:
        
        # Updated offspring through policy gradient
        random_key = training_state.random_key
        offsprings, random_key = self.sample(training_state, random_key)
        # offsprings, training_state = self._pg_update_population(
        #     offsprings, training_state, replay_buffer
        # )

        def _pg_update_pop():
            new_offsprings, new_training_state = self._pg_update_population(
                offsprings, training_state, replay_buffer_state
            )
            return new_offsprings, new_training_state

        def _dummy_pg_update():
            return offsprings, training_state


        offsprings, training_state = jax.lax.cond(
            training_state.steps >= self._config.warmup_iters,
            lambda _: _pg_update_pop(),
            lambda _: _dummy_pg_update(),
            operand=None,
        )

        # scoring offsprings
        fitnesses, transitions, random_key = self._scoring_function(
            offsprings, random_key
        )
        idx_sorted = jnp.argsort(-fitnesses)
        sorted_candidates = jax.tree_util.tree_map(
            lambda x: x[idx_sorted[: self._num_best]],
            offsprings
        )

        jax.tree_util.tree_map(
            lambda x: print("transitions shape:", x.shape), transitions
        )

        # add transitions in the replay buffer
        # [n_offsprings, T, ...] -> [T, n_offsprings, ...]
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.swapaxes(x, 0, 1), transitions
        )

        is_done = jnp.clip(jnp.cumsum(transitions.dones, axis=0), 0, 1)
        mask = jnp.roll(is_done, 1, axis=0)
        mask = mask.at[:1].set(0)
        mask =jnp.logical_not(mask)
        # trajectory = trajectory.replace(dones=None)
        # [T, n_offsprings, ...] -> [T * n_offsprings, ...]
        transitions, mask = jax.tree_util.tree_map(
            lambda x: jax.lax.collapse(x, 0, 2), (transitions, mask)
        )

        # add transitions in the replay buffer
        replay_buffer_state = self._replay_buffer.add(
            replay_buffer_state, transitions, mask
        )

        # CEM UPDATE
        old_mean = training_state.mean_policy_params

        new_mean = jax.tree_util.tree_map(
            lambda x: jnp.sum(jnp.expand_dims(self._weights, axis=[i for i in range(1, x.ndim)]) * x, axis=0),
            sorted_candidates
        )

        z = jax.tree_util.tree_map(
            lambda s, o: s - o,
            sorted_candidates, old_mean
        )

        new_var = jax.tree_util.tree_map(
            lambda x: jnp.sum(jnp.expand_dims(self._weights, axis=[i for i in range(1, x.ndim)]) * (x * x), axis=0) + training_state.damp,
            z
        )

        new_damp = training_state.damp * self._config.damp_tau + self._config.damp_final * (1 - self._config.damp_tau)

        new_training_state = training_state.replace(
            mean_policy_params=new_mean,
            var_policy_params=new_var,
            damp=new_damp,
            random_key=random_key,
            steps=training_state.steps+1,
        )


        # METRIC TRACKING
        
        # Check how many RL-updated offspring (first half) are in the elite set
        top_indices = idx_sorted[: self._num_best]
        # RL-updated offspring are at indices [0, num_learning_offspring)
        is_in_top = jnp.isin(jnp.arange(0, self._num_learning_offspring), top_indices)
        num_in_top = jnp.sum(is_in_top)
        percentage_in_elites = num_in_top / self._num_best * 100

        metrics = {
            "rl_in_elites_percentage": percentage_in_elites
        } 

        return new_training_state, replay_buffer_state, metrics
    


    def scan_update(self, carry:Tuple[CEMTD3TrainingState,ReplayBufferState], unused:Any):
        training_state, replay_buffer_state = carry
        training_state, replay_buffer_state, metrics = self.update(
            training_state, replay_buffer_state
        )
        return (training_state, replay_buffer_state), metrics
