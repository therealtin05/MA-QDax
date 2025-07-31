"""
A collection of functions and classes that define the algorithm Soft Actor Critic
(SAC), ref: https://arxiv.org/abs/1801.01290
"""

from dataclasses import dataclass
from functools import partial
from typing import Callable, Tuple, Dict, List

import jax
import jax.numpy as jnp
import optax
from brax.envs import Env
from brax.envs import State as EnvState
from brax.training.distribution import NormalTanhDistribution

from qdax.core.neuroevolution.buffers.buffer import (
    QDTransition,
    ReplayBuffer,
    Transition,
)
from qdax.core.neuroevolution.losses.masac_loss import (
    masac_alpha_loss_fn,
    masac_critic_loss_fn,
    masac_policy_loss_fn,
)
from qdax.core.neuroevolution.mdp_utils import TrainingState, get_first_episode
from qdax.core.neuroevolution.networks.masac_networks import make_masac_networks
from qdax.core.neuroevolution.normalization_utils import (
    RunningMeanStdState,
    normalize_with_rmstd,
    update_running_mean_std,
)
from qdax.core.neuroevolution.sac_td3_utils import generate_unroll
from qdax.custom_types import (
    Action,
    Descriptor,
    Mask,
    Metrics,
    Observation,
    Params,
    Reward,
    RNGKey,
)
from qdax.environments.multi_agent_wrappers import MultiAgentBraxWrapper

class MASacTrainingState(TrainingState):
    """Training state for the MASAC algorithm"""

    policy_optimizer_state: List[optax.OptState]
    policy_params: List[Params]
    critic_optimizer_state: optax.OptState
    critic_params: Params
    alpha_optimizer_state: List[optax.OptState]
    alpha_params: List[Params]
    target_critic_params: Params
    random_key: RNGKey
    steps: jnp.ndarray
    normalization_running_stats: RunningMeanStdState


@dataclass
class MASacConfig:
    """Configuration for the MASAC algorithm."""
    num_agents: int
    batch_size: int
    episode_length: int = 1000
    tau: float = 0.005
    normalize_observations: bool = False
    learning_rate: float = 3e-4
    alpha_init: float = 1.0
    discount: float = 0.99
    reward_scaling: float = 1.0
    critic_hidden_layer_size: Tuple[int, ...] = (256, 256)
    policy_hidden_layer_size: Tuple[int, ...] = (256, 256)
    fix_alpha: bool = False
    max_grad_norm: float = 30.0

class MASAC:
    def __init__(self, config: MASacConfig, action_sizes: Dict[int, int]) -> None:
        self._config = config
        self._action_sizes = action_sizes

        # define the networks
        self._policy, self._critic = make_masac_networks(
            action_sizes=action_sizes,
            critic_hidden_layer_size=self._config.critic_hidden_layer_size,
            policy_hidden_layer_size=self._config.policy_hidden_layer_size,
        )

        self._parametric_action_distribution = []
        
        for agent_idx in range(self._config.num_agents):
            # define the action distribution
            self._parametric_action_distribution.append(NormalTanhDistribution(
                event_size=action_sizes[agent_idx]
                )
            )

    def init(
        self, 
        random_key: RNGKey, 
        action_sizes_each_agent: Dict[int, int],
        observation_size_raw: int,
        observation_sizes_each_agent: Dict[int, int]
    ) -> MASacTrainingState:
        """Initialise the training state of the MASAC algorithm.

        Args:
            random_key: a jax random key
            action_sizes_each_agent: action sizes for each agent
            observation_size_raw: the size of the global observation space
            observation_sizes_each_agent: observation sizes for each agent

        Returns:
            the initial training state of MASAC
        """

        # Initialize policy parameters for each agent
        random_key, subkey = jax.random.split(random_key)
        keys = jax.random.split(subkey, num=self._config.num_agents)

        policy_params = []
        for (agent_idx, agent_policy), agent_keys in zip(
            self._policy.items(), keys
        ):
            fake_batch = jnp.zeros(shape=(observation_sizes_each_agent[agent_idx],))
            policy_params.append(agent_policy.init(agent_keys, fake_batch))

        # Initialize critic parameters
        fake_obs = jnp.zeros(shape=(observation_size_raw,))
        fake_action = jnp.zeros(shape=(sum(action_sizes_each_agent.values()),))

        random_key, subkey = jax.random.split(random_key)
        critic_params = self._critic.init(subkey, fake_obs, fake_action)

        target_critic_params = jax.tree_util.tree_map(
            lambda x: jnp.asarray(x.copy()), critic_params
        )

        # Initialize optimizers with gradient clipping
        grad_clip = optax.clip_by_global_norm(self._config.max_grad_norm)
        
        # Policy optimizers (one per agent)
        policy_optimizer_state = []
        for agent_idx, params in enumerate(policy_params):
            optimizer = optax.chain(grad_clip, optax.adam(learning_rate=1.0))
            policy_optimizer_state.append(optimizer.init(params))

        # Critic optimizer
        critic_optimizer = optax.chain(grad_clip, optax.adam(learning_rate=1.0))
        critic_optimizer_state = critic_optimizer.init(critic_params)

        # Alpha parameters and optimizers (one per agent)
        alpha_params = []
        alpha_optimizer_state = []
        for agent_idx in range(self._config.num_agents):
            log_alpha = jnp.asarray(jnp.log(self._config.alpha_init), dtype=jnp.float32)
            alpha_params.append(log_alpha)
            
            alpha_optimizer = optax.chain(grad_clip, optax.adam(learning_rate=1.0))
            alpha_optimizer_state.append(alpha_optimizer.init(log_alpha))

        # create and retrieve the training state
        training_state = MASacTrainingState(
            policy_optimizer_state=policy_optimizer_state,
            policy_params=policy_params,
            critic_optimizer_state=critic_optimizer_state,
            critic_params=critic_params,
            alpha_optimizer_state=alpha_optimizer_state,
            alpha_params=alpha_params,
            target_critic_params=target_critic_params,
            normalization_running_stats=RunningMeanStdState(
                mean=jnp.zeros(observation_size_raw),
                var=jnp.ones(observation_size_raw),
                count=jnp.zeros(()),
            ),
            random_key=random_key,
            steps=jnp.array(0),
        )

        return training_state

    @partial(jax.jit, static_argnames=("self", "deterministic"))
    def select_action(
        self,
        obs: Dict[int, Observation],
        policy_params: List[Params],
        random_key: RNGKey,
        deterministic: bool = False,
    ) -> Tuple[Dict[int, Action], RNGKey]:
        """Selects an action according to MASAC policy.

        Args:
            obs: agent observations
            policy_params: parameters of the agents' policies
            random_key: jax random key
            deterministic: whether to select action in a deterministic way.

        Returns:
            The selected actions and a new random key.
        """
        actions = {}
        random_key, subkey = jax.random.split(random_key)
        keys = jax.random.split(subkey, len(policy_params))

        for (agent_idx, network), params, obs_i, key in zip(
            self._policy.items(), policy_params, obs.values(), keys
        ):
            dist_params = network.apply(params, obs_i)
            
            if not deterministic:
                action = self._parametric_action_distribution[agent_idx].sample(
                    dist_params, key
                )
            else:
                # The first half of parameters is for mean 
                action = jax.nn.tanh(dist_params[..., : dist_params.shape[-1] // 2])
            
            actions[agent_idx] = action

        return actions, random_key

    @partial(jax.jit, static_argnames=("self", "env", "deterministic"))
    def play_step_fn(
        self,
        env_state: EnvState,
        training_state: MASacTrainingState,
        env: MultiAgentBraxWrapper,
        deterministic: bool = False,
    ) -> Tuple[EnvState, MASacTrainingState, Transition]:
        """Plays a step in the environment. Selects an action according to MASAC rule and
        performs the environment step.

        Args:
            env_state: the current environment state
            training_state: the MASAC training state
            env: the environment
            deterministic: whether to select action in a deterministic way.

        Returns:
            the new environment state
            the new MASAC training state
            the played transition
        """
        random_key = training_state.random_key
        policy_params = training_state.policy_params
        obs = env.obs(env_state)

        if self._config.normalize_observations:
            # Note: This would need to be adapted for multi-agent observations
            normalized_obs = normalize_with_rmstd(
                env_state.obs, training_state.normalization_running_stats
            )
            normalization_running_stats = update_running_mean_std(
                training_state.normalization_running_stats, env_state.obs
            )
        else:
            normalized_obs = obs
            normalization_running_stats = training_state.normalization_running_stats

        actions, random_key = self.select_action(
            obs=normalized_obs,
            policy_params=policy_params,
            random_key=random_key,
            deterministic=deterministic,
        )

        training_state = training_state.replace(
            random_key=random_key,
            normalization_running_stats=normalization_running_stats,
        )

        next_env_state = env.step(env_state, actions)
        flatten_actions = jnp.concatenate([a for a in actions.values()])

        transition = Transition(
            obs=env_state.obs,
            next_obs=next_env_state.obs,
            rewards=next_env_state.reward,
            dones=next_env_state.done,
            actions=flatten_actions,
            truncations=next_env_state.info["truncation"],
        )

        return next_env_state, training_state, transition

    @partial(jax.jit, static_argnames=("self", "env", "deterministic"))
    def play_qd_step_fn(
        self,
        env_state: EnvState,
        training_state: MASacTrainingState,
        env: MultiAgentBraxWrapper,
        deterministic: bool = False,
    ) -> Tuple[EnvState, MASacTrainingState, QDTransition]:
        """Plays a step in the environment for QD environments."""

        next_env_state, training_state, transition = self.play_step_fn(
            env_state, training_state, env, deterministic
        )

        qd_transition = QDTransition(
            obs=env_state.obs,
            next_obs=next_env_state.obs,
            rewards=next_env_state.reward,
            dones=next_env_state.done,
            actions=transition.actions,
            truncations=next_env_state.info["truncation"],
            state_desc=env_state.info["state_descriptor"],
            next_state_desc=next_env_state.info["state_descriptor"],
        )

        return next_env_state, training_state, qd_transition

    @partial(
        jax.jit,
        static_argnames=("self", "play_step_fn"),
    )
    def eval_policy_fn(
        self,
        training_state: MASacTrainingState,
        eval_env_first_state: EnvState,
        play_step_fn: Callable[
            [EnvState, MASacTrainingState],
            Tuple[EnvState, MASacTrainingState, Transition],
        ],
    ) -> Tuple[Reward, Reward]:
        """Evaluates the agent's policy over an entire episode."""

        state, training_state, transitions = generate_unroll(
            init_state=eval_env_first_state,
            training_state=training_state,
            episode_length=self._config.episode_length,
            play_step_fn=play_step_fn,
        )

        transitions = get_first_episode(transitions)
        true_returns = jnp.nansum(transitions.rewards, axis=0)
        true_return = jnp.mean(true_returns, axis=-1)

        return true_return, true_returns

    @partial(
        jax.jit,
        static_argnames=("self", "play_step_fn", "bd_extraction_fn"),
    )
    def eval_qd_policy_fn(
        self,
        training_state: MASacTrainingState,
        eval_env_first_state: EnvState,
        play_step_fn: Callable[
            [EnvState, MASacTrainingState],
            Tuple[EnvState, MASacTrainingState, QDTransition],
        ],
        bd_extraction_fn: Callable[[QDTransition, Mask], Descriptor],
    ) -> Tuple[Reward, Descriptor, Reward, Descriptor]:
        """Evaluates the agent's policy for QD environments."""

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

    @partial(jax.jit, static_argnames=("self", "unflatten_obs_fn"))
    def update(
        self,
        training_state: MASacTrainingState,
        replay_buffer: ReplayBuffer,
        unflatten_obs_fn: Callable[[Observation], Dict[int, jnp.ndarray]],
    ) -> Tuple[MASacTrainingState, ReplayBuffer, Metrics]:
        """Performs a training step to update the policy and the critic parameters.

        Args:
            training_state: the current MASAC training state
            replay_buffer: the replay buffer
            unflatten_obs_fn: function to unflatten observations

        Returns:
            the updated MASAC training state
            the replay buffer
            the training metrics
        """

        # sample a batch of transitions in the buffer
        random_key = training_state.random_key
        transitions, random_key = replay_buffer.sample(
            random_key,
            sample_size=self._config.batch_size,
        )

        unflatten_obs_fn_vmap = jax.vmap(unflatten_obs_fn)

        # Create the policy_fns_apply function
        def create_policy_fns(index, params, obs):
            return jax.lax.switch(index, [pol.apply for pol in self._policy.values()], params, obs)
        
        policy_fns_apply = jax.jit(create_policy_fns)

        # normalise observations if necessary
        if self._config.normalize_observations:
            normalization_running_stats = training_state.normalization_running_stats
            normalized_obs = normalize_with_rmstd(
                transitions.obs, normalization_running_stats
            )
            normalized_next_obs = normalize_with_rmstd(
                transitions.next_obs, normalization_running_stats
            )
            transitions = transitions.replace(
                obs=normalized_obs, next_obs=normalized_next_obs
            )

        # update alpha (temperature parameter)
        alpha_params = training_state.alpha_params
        alpha_optimizer_state = training_state.alpha_optimizer_state
        alpha_losses = []

        if not self._config.fix_alpha:
            random_key, subkey = jax.random.split(random_key)
            alpha_losses, alpha_gradients = masac_alpha_loss_fn(
                log_alphas=training_state.alpha_params,
                policy_fns_apply=policy_fns_apply,
                parametric_action_distributions=self._parametric_action_distribution,
                unflatten_obs_fn=unflatten_obs_fn_vmap,
                action_sizes=self._action_sizes,
                policy_params=training_state.policy_params,
                transitions=transitions,
                random_key=subkey,
            )

            # Update alpha parameters for each agent
            new_alpha_params = []
            new_alpha_optimizer_state = []
            grad_clip = optax.clip_by_global_norm(self._config.max_grad_norm)
            alpha_optimizer = optax.chain(grad_clip, optax.adam(learning_rate=self._config.learning_rate))

            for agent_idx, (alpha_grad, alpha_opt_state) in enumerate(
                zip(alpha_gradients, alpha_optimizer_state)
            ):
                alpha_updates, updated_opt_state = alpha_optimizer.update(
                    alpha_grad, alpha_opt_state
                )
                updated_alpha = optax.apply_updates(
                    training_state.alpha_params[agent_idx], alpha_updates
                )
                new_alpha_params.append(updated_alpha)
                new_alpha_optimizer_state.append(updated_opt_state)

            alpha_params = new_alpha_params
            alpha_optimizer_state = new_alpha_optimizer_state
        else:
            alpha_losses = [jnp.array(0.0) for _ in range(self._config.num_agents)]

        # update critic
        random_key, subkey = jax.random.split(random_key)
        alphas = [jnp.exp(log_alpha) for log_alpha in alpha_params]
        
        critic_loss, critic_gradient = jax.value_and_grad(masac_critic_loss_fn)(
            training_state.critic_params,
            policy_fns_apply=policy_fns_apply,
            critic_fn=self._critic.apply,
            parametric_action_distributions=self._parametric_action_distribution,
            unflatten_obs_fn=unflatten_obs_fn_vmap,
            reward_scaling=self._config.reward_scaling,
            discount=self._config.discount,
            policy_params=training_state.policy_params,
            target_critic_params=training_state.target_critic_params,
            alphas=alphas, 
            transitions=transitions,
            random_key=subkey,
        )

        grad_clip = optax.clip_by_global_norm(self._config.max_grad_norm)
        critic_optimizer = optax.chain(grad_clip, optax.adam(learning_rate=self._config.learning_rate))
        critic_updates, critic_optimizer_state = critic_optimizer.update(
            critic_gradient, training_state.critic_optimizer_state
        )
        critic_params = optax.apply_updates(
            training_state.critic_params, critic_updates
        )
        
        # Soft update target critic
        target_critic_params = jax.tree_util.tree_map(
            lambda x1, x2: (1.0 - self._config.tau) * x1 + self._config.tau * x2,
            training_state.target_critic_params,
            critic_params,
        )

        # update actors
        random_key, subkey = jax.random.split(random_key)
        policy_losses, policy_gradients = masac_policy_loss_fn(
            policy_params=training_state.policy_params,
            policy_fns_apply=policy_fns_apply,
            critic_fn=self._critic.apply,
            parametric_action_distributions=self._parametric_action_distribution,
            unflatten_obs_fn=unflatten_obs_fn_vmap,
            critic_params=training_state.critic_params,
            alphas=alphas,
            transitions=transitions,
            random_key=subkey,
        )

        # Update policy parameters for each agent
        new_policy_params = []
        new_policy_optimizer_state = []
        grad_clip = optax.clip_by_global_norm(self._config.max_grad_norm)
        policy_optimizer = optax.chain(grad_clip, optax.adam(learning_rate=self._config.learning_rate))

        for agent_idx, (pol_grad, pol_opt_state) in enumerate(
            zip(policy_gradients, training_state.policy_optimizer_state)
        ):
            policy_updates, updated_opt_state = policy_optimizer.update(
                pol_grad, pol_opt_state
            )
            updated_params = optax.apply_updates(
                training_state.policy_params[agent_idx], policy_updates
            )
            new_policy_params.append(updated_params)
            new_policy_optimizer_state.append(updated_opt_state)

        # create new training state
        new_training_state = MASacTrainingState(
            policy_optimizer_state=new_policy_optimizer_state,
            policy_params=new_policy_params,
            critic_optimizer_state=critic_optimizer_state,
            critic_params=critic_params,
            alpha_optimizer_state=alpha_optimizer_state,
            alpha_params=alpha_params,
            normalization_running_stats=training_state.normalization_running_stats,
            target_critic_params=target_critic_params,
            random_key=random_key,
            steps=training_state.steps + 1,
        )

        metrics = {
            "actor_losses": policy_losses,  # List of losses per agent
            "critic_loss": critic_loss,
            "alpha_losses": alpha_losses,  # List of alpha losses per agent
            "alphas": alphas,  # Current alpha values
            "obs_mean": jnp.mean(transitions.obs),
            "obs_std": jnp.std(transitions.obs),
        }
        
        return new_training_state, replay_buffer, metrics
