""" Implements the CEMMATD3 algorithm in jax for brax environments, based on:
https://arxiv.org/pdf/1802.09477.pdf

In this verion the update mechanism is
for o in learning_offsprings:
    for i -> num_rl_updates
        update critic 
        update o 
"""

from dataclasses import dataclass
from functools import partial
from typing import Callable, Tuple, Optional, Any, List

import jax
import optax
from brax.envs import Env
from brax.envs import State as EnvState
from jax import numpy as jnp
from flax import linen as nn

from qdax.core.neuroevolution.buffers.buffer import (
    QDTransition,
    ReplayBuffer,
    Transition,
)
from qdax.environments.multi_agent_wrappers import MultiAgentBraxWrapper
from qdax.core.neuroevolution.losses.matd3_loss import (
    matd3_critic_loss_fn,
    matd3_policy_loss_fn,
)
from qdax.core.neuroevolution.mdp_utils import TrainingState, get_first_episode
from qdax.core.neuroevolution.networks.matd3_networks import make_matd3_networks
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


class CEMMATD3TrainingState(TrainingState):
    """Contains training state for the learner."""
    mean_policy_params: List[Params]
    var_policy_params: List[Params]
    damp: jnp.ndarray

    critic_optimizer_state: optax.OptState
    critic_params: Params
    target_critic_params: Params

    random_key: RNGKey
    steps: jnp.ndarray
    


@dataclass
class CEMMATD3Config:
    """Configuration for the CEMMATD3 algorithm"""

    num_agents: int
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
    max_grad_norm: float = 100.0
    

class CEMMATD3:
    """
    A collection of functions that define the Twin Delayed Deep Deterministic Policy
    Gradient agent (CEMMATD3), ref: https://arxiv.org/pdf/1802.09477.pdf
    """

    # def __init__(self, config: CEMMATD3Config, action_size: int, ):
    def __init__(
        self,
        config: CEMMATD3Config,
        env: MultiAgentBraxWrapper,
        scoring_function: Callable[
            [Genotype, RNGKey], Tuple[Fitness, Transition, RNGKey]
        ],
        num_eval: int = 20
    ):
        if config.mirror_sampling:
            assert config.population_size % 2 == 0, "pop_size must be even for mirror sampling"
        self._config = config
        self._scoring_function = scoring_function
        self._env = env
        self._policy, self._critic, = make_matd3_networks(
            action_sizes=env.get_action_sizes(),
            critic_hidden_layer_sizes=self._config.critic_hidden_layer_size,
            policy_hidden_layer_sizes=self._config.policy_hidden_layer_size,
            use_layer_norm=config.use_layer_norm,
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

        self._num_eval = num_eval

        if self._config.max_grad_norm > 0.0:
            grad_clip = optax.clip_by_global_norm(self._config.max_grad_norm)
            self._policy_optimizer = optax.chain(
                    grad_clip,
                    optax.adam(learning_rate=self._config.policy_learning_rate)
            )
            self._critic_optimizer = optax.chain(
                grad_clip, 
                optax.adam(learning_rate=self._config.critic_learning_rate)
            )
        else:
            self._policy_optimizer = optax.adam(learning_rate=self._config.policy_learning_rate)

            self._critic_optimizer = optax.adam(learning_rate=self._config.critic_learning_rate)
            

    def init(
        self, random_key: RNGKey
    ) -> CEMMATD3TrainingState:
        """Initialise the training state of the CEMMATD3 algorithm, through creation
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

        action_sizes_each_agent = self._env.get_action_sizes()
        observation_sizes_each_agent = self._env.get_obs_sizes()
        observation_size_raw = self._env.observation_size

        # Initialize critics and policy params
        fake_obs = jnp.zeros(shape=(self._env.observation_size,))
        fake_action = jnp.zeros(shape=(self._env.action_size,))
        random_key, subkey_1, subkey_2 = jax.random.split(random_key, 3)
        keys = jax.random.split(subkey_2, num=self._config.num_agents)

        critic_params = self._critic.init(subkey_1, obs=fake_obs, actions=fake_action)

        # mean_policy_params = self._policy.init(subkey_2, fake_obs)
        mean_policy_params = []
        for (agent_idx, agent_policy), agent_keys in zip(
            self._policy.items(), keys
        ):
            fake_batch = jnp.zeros(shape=(observation_sizes_each_agent[agent_idx]), )
            mean_policy_params.append(agent_policy.init(agent_keys, fake_batch)) # list of pytree each pytree represent batch of params for an i-th agent (n_agents = number of multiagent)

        fake_obs = jnp.zeros(shape=(observation_size_raw, ))
        fake_action = jnp.zeros(shape=(sum(action_sizes_each_agent.values()), ))


        var_policy_params = jax.tree_util.tree_map(
            lambda x: jnp.full_like(x, self._config.damp_init), 
            mean_policy_params
        )

        # Initialize target networks
        target_critic_params = jax.tree_util.tree_map(
            lambda x: jnp.asarray(x.copy()), critic_params
        )

        # Create and initialize optimizers
        critic_optimizer_state = self._critic_optimizer.init(critic_params)

        # Initial training state
        training_state = CEMMATD3TrainingState(
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
        obs: dict[int, Observation],
        policy_params: List[Params],
        random_key: RNGKey,
        expl_noise: float,
        deterministic: bool = False,
    ) -> Tuple[dict[int, Action], RNGKey]:
        """Selects an action according to TD3 policy. The action can be deterministic
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

        actions = {
            agent_idx: network.apply(params, agent_obs)
            for (agent_idx, network), params, agent_obs in zip(
                self._policy.items(), policy_params, obs.values()
            )
        }

        random_key, subkey = jax.random.split(random_key)
        keys = jax.random.split(subkey, len(policy_params))

        if not deterministic:
            # random_key, subkey = jax.random.split(random_key)

            for (agent_idx, act), agent_keys in zip(
                actions.items(), keys
            ):
                noise = jax.random.normal(agent_keys, act.shape) * expl_noise
                act = act + noise
                act = jnp.clip(act, -1.0, 1.0)
                actions[agent_idx] = act

        return actions, random_key

    @partial(jax.jit, static_argnames=("self", "env", "deterministic"))
    def play_step_fn(
        self,
        env_state: EnvState,
        training_state: CEMMATD3TrainingState,
        env: MultiAgentBraxWrapper,
        deterministic: bool = False,
    ) -> Tuple[EnvState, CEMMATD3TrainingState, Transition]:
        """Plays a step in the environment. Selects an action according to TD3 rule and
        performs the environment step.

        Args:
            env_state: the current environment state
            training_state: the SAC training state
            env: the environment
            deterministic: whether to select action in a deterministic way.
                Defaults to False.

        Returns:
            the new environment state
            the new TD3 training state
            the played transition
        """

        actions, random_key = self.select_action(
            obs=env.obs(env_state),
            policy_params=training_state.mean_policy_params,
            random_key=training_state.random_key,
            expl_noise=self._config.expl_noise,
            deterministic=deterministic,
        )
        training_state = training_state.replace(
            random_key=random_key,
        )
        next_env_state = env.step(env_state, actions)

        flatten_action = jnp.concatenate([a for a in actions.values()])

        transition = Transition(
            obs=env_state.obs,
            next_obs=next_env_state.obs,
            rewards=next_env_state.reward,
            dones=next_env_state.done,
            truncations=next_env_state.info["truncation"],
            actions=flatten_action,
        )
        return next_env_state, training_state, transition

    @partial(jax.jit, static_argnames=("self", "env", "deterministic"))
    def play_qd_step_fn(
        self,
        env_state: EnvState,
        training_state: CEMMATD3TrainingState,
        env: MultiAgentBraxWrapper,
        deterministic: bool = False,
    ) -> Tuple[EnvState, CEMMATD3TrainingState, QDTransition]:
        """Plays a step in the environment. Selects an action according to TD3 rule and
        performs the environment step.

        Args:
            env_state: the current environment state
            training_state: the TD3 training state
            env: the environment
            deterministic: the whether to select action in a deterministic way.
                Defaults to False.

        Returns:
            the new environment state
            the new TD3 training state
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
    def evaluate(self, training_state: CEMMATD3TrainingState):
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


    @partial(jax.jit, static_argnames=("self",))
    def sample(
        self, training_state: CEMMATD3TrainingState, random_key: RNGKey
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
        policy_params: List[Params],
        target_policy_params: List[Params],
        target_critic_params: Params,
        policy_optimizer_state: List[optax.OptState],
        critic_optimizer_state: optax.OptState,
        replay_buffer: ReplayBuffer,
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

        # Sample a batch of transitions in the buffer
        # random_key = training_state.random_key
        samples, random_key = replay_buffer.sample(
            random_key, sample_size=self._config.batch_size
        )

        # Update Critic
        random_key, subkey = jax.random.split(random_key)
        critic_loss, critic_gradient = jax.value_and_grad(self._matd3_critic_loss_fn)(
            critic_params,
            target_policy_params=target_policy_params,
            target_critic_params=target_critic_params,
            transitions=samples,
            random_key=subkey,
        )
        critic_updates, critic_optimizer_state = self._critic_optimizer.update(
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
        def update_policy_step() -> Tuple[List[Params], List[Params], List[optax.OptState]]:
            policy_losses, policy_gradients = self._matd3_policy_loss_fn(
                policy_params=policy_params,
                critic_params=critic_params,
                transitions=samples,
            )
            new_policy_params = []
            new_target_policy_params = []
            new_policy_optimizer_state = []

            for agent_idx, (pol_grad, pol_opt_state) in enumerate(
                zip(policy_gradients, policy_optimizer_state)
            ):
                policy_updates, pol_opt_state = self._policy_optimizer.update(
                    pol_grad, pol_opt_state
                )
                
                updated_params = optax.apply_updates(
                    policy_params[agent_idx], policy_updates
                )
                new_policy_params.append(updated_params)
                new_policy_optimizer_state.append(pol_opt_state)
                
                # Soft update of target policy
                new_target_policy_params.append(
                    jax.tree_util.tree_map(
                        lambda x1, x2: (1.0 - self._config.soft_tau_update) * x1
                        + self._config.soft_tau_update * x2,
                        target_policy_params[agent_idx],
                        updated_params,
                    )
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
        carry: Tuple[Params, List[Params], List[Params], Params, optax.OptState, List[optax.OptState], ReplayBuffer, RNGKey],
        cur_step: jnp.ndarray,
    ) -> Tuple[Tuple[Params, List[Params], List[Params], Params, optax.OptState, List[optax.OptState], ReplayBuffer, RNGKey], Any]:
        """Function to work with jax.lax.scan
        """
        (
            critic_params, policy_params, target_policy_params, target_critic_params,
                policy_optimizer_state, critic_optimizer_state, replay_buffer, random_key,
        ) = carry
        random_key, subkey = jax.random.split(random_key)

        policy_params, critic_params, target_policy_params, \
              target_critic_params, policy_optimizer_state, critic_optimizer_state \
              = self._update_single(
                    critic_params, policy_params, target_policy_params, target_critic_params,
                        policy_optimizer_state, critic_optimizer_state, replay_buffer, subkey, cur_step
              )
        return (
            critic_params, policy_params, target_policy_params, target_critic_params,
                policy_optimizer_state, critic_optimizer_state, replay_buffer, random_key,
        ), ()


    @partial(jax.jit, static_argnames=("self",))
    def _update_one_offspring(
        self,
        critic_params: Params,
        policy_params: List[Params],
        target_critic_params: Params,
        critic_optimizer_state: optax.OptState,
        replay_buffer: ReplayBuffer,
        random_key: RNGKey,
    ) -> Tuple[Params, List[Params], Params, optax.OptState, ReplayBuffer, RNGKey]:
        """Function to update one single offspring, we need to init the target policy params
        and policy optimization stat here
        """

        target_policy_params = jax.tree_util.tree_map(
            lambda x: jnp.asarray(x.copy()), policy_params
        )

        policy_optimizer_state = []
        for agent_idx, params in enumerate(policy_params):
            policy_optimizer_state.append(self._policy_optimizer.init(params))

        (
            critic_params, policy_params, target_policy_params, target_critic_params,
                policy_optimizer_state, critic_optimizer_state, replay_buffer, random_key,
        ), () = jax.lax.scan(
            self._scan_update_single,
            (
                critic_params, policy_params, target_policy_params, target_critic_params, 
                    policy_optimizer_state, critic_optimizer_state, replay_buffer, random_key,
            ),
            jnp.arange(0, self._config.num_rl_updates_per_iter // self._num_learning_offspring),
        )

        return (
            critic_params, policy_params, target_critic_params,
                critic_optimizer_state, replay_buffer, random_key,
        )


    @partial(jax.jit, static_argnames=("self",))
    def _scan_update_one_offspring(
        self,
        carry: Tuple[Params, Params, optax.OptState, ReplayBuffer, RNGKey],
        x: List[Params]
    ) -> Tuple[Tuple[Params, Params, optax.OptState, ReplayBuffer, RNGKey], Params]:
        """Help scan through number of offsprings updated through policy gradient
        """
        critic_params, target_critic_params, critic_optimizer_state, replay_buffer, random_key = carry
        policy_params = x
        (
            critic_params, policy_params, target_critic_params,
                critic_optimizer_state, replay_buffer, random_key,
        ) = self._update_one_offspring(
            critic_params, policy_params, target_critic_params,
            critic_optimizer_state, replay_buffer, random_key
        )

        return (
            critic_params, target_critic_params,
                critic_optimizer_state, replay_buffer, random_key,
        ), policy_params


    @partial(jax.jit, static_argnames=("self",))
    def _pg_update_population(
        self,
        offsprings: List[Params],
        training_state: CEMMATD3TrainingState,
        replay_buffer: ReplayBuffer,
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
            replay_buffer,
            random_key
        )
        xs = selected_offsprings
        (
            critic_params,
            target_critic_params,
            critic_optimizer_state,
            replay_buffer,
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
        training_state: CEMMATD3TrainingState,
        replay_buffer: ReplayBuffer,
    ) -> Tuple[CEMMATD3TrainingState, ReplayBuffer, Metrics]:
        
        # Updated offspring through policy gradient
        random_key = training_state.random_key
        offsprings, random_key = self.sample(training_state, random_key)

        def _pg_update_pop():
            new_offsprings, new_training_state = self._pg_update_population(
                offsprings, training_state, replay_buffer
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

        # add transitions in the replay buffer
        replay_buffer = replay_buffer.insert(transitions)

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

        return new_training_state, replay_buffer, metrics
    

    def scan_update(self, carry:Tuple[CEMMATD3TrainingState,ReplayBuffer], unused:Any):
        training_state, replay_buffer = carry
        training_state, replay_buffer, metrics = self.update(
            training_state, replay_buffer
        )
        return (training_state, replay_buffer), metrics


    @partial(jax.jit, static_argnames=("self"))
    def _unflatten_obs_fn(self, global_obs: jnp.ndarray) -> dict[int, jnp.ndarray]:
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
    def _matd3_critic_loss_fn(
        self,
        critic_params: Params,
        target_policy_params: List[Params],
        target_critic_params: Params,
        transitions: Transition,
        random_key: RNGKey,
    ) -> jnp.ndarray:
        """Critics loss function for TD3 agent.

        Args:
            critic_params: critic parameters.
            target_policy_params: target policy parameters.
            target_critic_params: target critic parameters.
            policy_fns: forward pass through the neural network defining the policy.
            critic_fn: forward pass through the neural network defining the critic.
            policy_noise: policy noise.
            noise_clip: noise clip.
            reward_scaling: reward scaling coefficient.
            discount: discount factor.
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
            a = self._policy[agent_idx].apply(params, agent_obs)
            random_key, subkey = jax.random.split(random_key)
            noise = (jax.random.normal(subkey, shape=a.shape) * self._config.policy_noise).clip(-self._config.noise_clip, self._config.noise_clip)
            a = (a + noise).clip(-1.0, 1.0)
            next_actions[agent_idx] = a

        flatten_next_actions = jnp.concatenate([a for a in next_actions.values()], axis=-1)

        next_q = self._critic.apply(  # type: ignore
            target_critic_params, obs=transitions.next_obs, actions=flatten_next_actions
        )
        next_v = jnp.min(next_q, axis=-1)
        target_q = jax.lax.stop_gradient(
            transitions.rewards * self._config.reward_scaling
            + (1.0 - transitions.dones) * self._config.discount * next_v
        )
        q_old_action = self._critic.apply(  # type: ignore
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
        transitions: Transition,
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
                    action = self._policy[i].apply(agent_params, unflatten_obs[i])
                else:
                    # Use current policy_params for other agents
                    action = self._policy[i].apply(policy_params[i], unflatten_obs[i])
                agent_actions.append(action)
            
            # Flatten all actions
            flatten_actions = jnp.concatenate(agent_actions, axis=-1)
            
            # Get Q-value using the critic
            q_value = self._critic.apply(
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