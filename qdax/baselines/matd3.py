import jax
import jax.numpy as jnp
import optax
from qdax.core.neuroevolution.mdp_utils import TrainingState, get_first_episode
from typing import List, Any
from brax.v1.envs import Env
from brax.v1.envs import State as EnvState
from qdax.environments.multi_agent_wrappers import MultiAgentBraxWrapper
from qdax.core.neuroevolution.buffers.buffer import (
    QDTransition,
    ReplayBuffer,
    Transition,
)
from qdax.core.neuroevolution.networks.matd3_networks import make_matd3_networks
from qdax.core.neuroevolution.mdp_utils import TrainingState, get_first_episode
from qdax.core.neuroevolution.losses.matd3_loss import (
    matd3_critic_loss_fn,
    matd3_policy_loss_fn,
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

from typing import Any, Callable, Tuple, Dict, List
from functools import partial
from dataclasses import dataclass



class MATD3TrainingState(TrainingState):
    """Contains training state for the learner."""

    policy_optimizer_state: List[optax.OptState]
    policy_params: List[Params]
    critic_optimizer_state: optax.OptState
    critic_params: Params
    target_policy_params: List[Params]
    target_critic_params: Params
    random_key: RNGKey
    steps: jnp.ndarray


@dataclass
class MATD3Config:
    """Configuration for the TD3 algorithm"""

    num_agents: int
    episode_length: int = 1000
    batch_size: int = 256
    policy_delay: int = 2
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



class MATD3:
    """
    A collection of functions that define the Twin Delayed Deep Deterministic Policy
    Gradient agent (TD3), ref: https://arxiv.org/pdf/1802.09477.pdf
    """

    def __init__(self, config: MATD3Config, action_sizes: dict[int, int]):
        self._config = config

        self._policy, self._critic, = make_matd3_networks(
            action_sizes=action_sizes,
            critic_hidden_layer_sizes=self._config.critic_hidden_layer_size,
            policy_hidden_layer_sizes=self._config.policy_hidden_layer_size,
        )

    def init(
        self, random_key: RNGKey, action_sizes_each_agent: dict[int, int], 
        observation_size_raw: int,  observation_sizes_each_agent: dict[int, int]
    ) -> MATD3TrainingState:
        """Initialise the training state of the TD3 algorithm, through creation
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

        random_key, subkey = jax.random.split(random_key)
        keys = jax.random.split(subkey, num=self._config.num_agents)


        policy_params = []
        for (agent_idx, agent_policy), agent_keys in zip(
            self._policy.items(), keys
        ):
            fake_batch = jnp.zeros(shape=(observation_sizes_each_agent[agent_idx]), )
            policy_params.append(agent_policy.init(agent_keys, fake_batch)) # list of pytree each pytree represent batch of params for an i-th agent (n_agents = number of multiagent)

        fake_obs = jnp.zeros(shape=(observation_size_raw, ))
        fake_action = jnp.zeros(shape=(sum(action_sizes_each_agent.values()), ))


        random_key, subkey = jax.random.split(random_key)
        critic_params = self._critic.init(subkey, fake_obs, fake_action) # list of pytree each pytree represent batch of params for an i-th agent (n_agents = number of multiagent)

        # Initialize target networks
        target_critic_params = jax.tree_util.tree_map(
            lambda x: jnp.asarray(x.copy()), critic_params
        )
        target_policy_params = jax.tree_util.tree_map(
            lambda x: jnp.asarray(x.copy()), policy_params
        )

        # Create and initialize optimizers
        critic_optimizer_state = optax.adam(learning_rate=1.0).init(critic_params)

        policy_optimizer_state = []
        for agent_idx, params in enumerate(policy_params):
            policy_optimizer_state.append(optax.adam(learning_rate=1.0).init(params))

        # Initial training state
        training_state = MATD3TrainingState(
            policy_optimizer_state=policy_optimizer_state,
            policy_params=policy_params,
            critic_optimizer_state=critic_optimizer_state,
            critic_params=critic_params,
            target_policy_params=target_policy_params,
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
        training_state: MATD3TrainingState,
        env: MultiAgentBraxWrapper,
        deterministic: bool = False,
    ) -> Tuple[EnvState, MATD3TrainingState, Transition]:
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
            policy_params=training_state.policy_params,
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
        training_state: MATD3TrainingState,
        env: MultiAgentBraxWrapper,
        deterministic: bool = False,
    ) -> Tuple[EnvState, MATD3TrainingState, QDTransition]:
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

    @partial(
        jax.jit,
        static_argnames=(
            "self",
            "play_step_fn",
        ),
    )
    def eval_policy_fn(
        self,
        training_state: MATD3TrainingState,
        eval_env_first_state: EnvState,
        play_step_fn: Callable[
            [EnvState, Params, RNGKey],
            Tuple[EnvState, Params, RNGKey, Transition],
        ],
    ) -> Tuple[Reward, Reward]:
        """Evaluates the agent's policy over an entire episode, across all batched
        environments.

        Args:
            training_state: TD3 training state.
            eval_env_first_state: the first state of the environment.
            play_step_fn: function defining how to play a step in the env.

        Returns:
            true return averaged over batch dimension, shape: (1,)
            true return per env, shape: (env_batch_size,)
        """
        # TODO: this generate unroll shouldn't take a random key
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
        static_argnames=(
            "self",
            "play_step_fn",
            "bd_extraction_fn",
        ),
    )
    def eval_qd_policy_fn(
        self,
        training_state: MATD3TrainingState,
        eval_env_first_state: EnvState,
        play_step_fn: Callable[
            [EnvState, Params, RNGKey],
            Tuple[EnvState, MATD3TrainingState, QDTransition],
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

    @partial(jax.jit, static_argnames=("self", "unflatten_obs_fn"))
    # Update the MATD3.update method to use the new approach
    def update(
        self,
        training_state: MATD3TrainingState,
        replay_buffer: ReplayBuffer,
        unflatten_obs_fn: Callable[
            [Observation],
            dict[int, jnp.ndarray]
        ],
    ) -> Tuple[MATD3TrainingState, ReplayBuffer, Metrics]:
        """Updated update method using the new policy_fns approach"""

        # Sample a batch of transitions in the buffer
        random_key = training_state.random_key
        samples, random_key = replay_buffer.sample(
            random_key, sample_size=self._config.batch_size
        )

        unflatten_obs_fn_vmap = jax.vmap(unflatten_obs_fn)
        
        # Create the policy_fns_apply function (this should be created once and reused)
        def create_policy_fns(index, params, obs):
            return jax.lax.switch(index, [pol.apply for pol in self._policy.values()], params, obs)
        
        policy_fns_apply = jax.jit(create_policy_fns)

        # Update Critic
        random_key, subkey = jax.random.split(random_key)
        critic_loss, critic_gradient = jax.value_and_grad(matd3_critic_loss_fn)(
            training_state.critic_params,
            target_policy_params=training_state.target_policy_params,
            target_critic_params=training_state.target_critic_params,
            policy_fns_apply=policy_fns_apply,
            critic_fn=self._critic.apply,
            unflatten_obs_fn=unflatten_obs_fn_vmap,
            policy_noise=self._config.policy_noise,
            noise_clip=self._config.noise_clip,
            reward_scaling=self._config.reward_scaling,
            discount=self._config.discount,
            transitions=samples,
            random_key=subkey,
        )
        
        # ... rest of critic update code remains the same ...
        critic_optimizer = optax.adam(learning_rate=self._config.critic_learning_rate)
        critic_updates, critic_optimizer_state = critic_optimizer.update(
            critic_gradient, training_state.critic_optimizer_state
        )
        critic_params = optax.apply_updates(
            training_state.critic_params, critic_updates
        )
        
        # Soft update of target critic network
        target_critic_params = jax.tree_util.tree_map(
            lambda x1, x2: (1.0 - self._config.soft_tau_update) * x1
            + self._config.soft_tau_update * x2,
            training_state.target_critic_params,
            critic_params,
        )

        # Update policy
        policy_losses, policy_gradients = matd3_policy_loss_fn(
            policy_params=training_state.policy_params,
            critic_params=training_state.critic_params,
            policy_fns_apply=policy_fns_apply,
            critic_fn=self._critic.apply,
            unflatten_obs_fn=unflatten_obs_fn_vmap,
            transitions=samples,
        )

        def update_policy_step() -> Tuple[List[Params], List[Params], List[optax.OptState]]:
            policy_params = []
            target_policy_params = []
            policy_optimizer_state = []
            policy_optimizer = optax.adam(
                learning_rate=self._config.policy_learning_rate
            )
            
            for agent_idx, (pol_grad, pol_opt_state) in enumerate(
                zip(policy_gradients, training_state.policy_optimizer_state)
            ):
                policy_updates, pol_opt_state = policy_optimizer.update(
                    pol_grad, pol_opt_state
                )
                
                updated_params = optax.apply_updates(
                    training_state.policy_params[agent_idx], policy_updates
                )
                policy_params.append(updated_params)
                policy_optimizer_state.append(pol_opt_state)
                
                # Soft update of target policy
                target_policy_params.append(
                    jax.tree_util.tree_map(
                        lambda x1, x2: (1.0 - self._config.soft_tau_update) * x1
                        + self._config.soft_tau_update * x2,
                        training_state.target_policy_params[agent_idx],
                        updated_params,
                    )
                )
            
            return policy_params, target_policy_params, policy_optimizer_state

        # ... rest of the update method remains the same ...
        # Delayed update
        current_policy_state = (
            training_state.policy_params,
            training_state.target_policy_params,
            training_state.policy_optimizer_state,
        )
        policy_params, target_policy_params, policy_optimizer_state = jax.lax.cond(
            training_state.steps % self._config.policy_delay == 0,
            lambda _: update_policy_step(),
            lambda _: current_policy_state,
            operand=None,
        )

        # Create new training state
        new_training_state = training_state.replace(
            critic_params=critic_params,
            critic_optimizer_state=critic_optimizer_state,
            policy_params=policy_params,
            policy_optimizer_state=policy_optimizer_state,
            target_critic_params=target_critic_params,
            target_policy_params=target_policy_params,
            random_key=random_key,
            steps=training_state.steps + 1,
        )

        metrics = {
            "actor_losses": policy_losses,  # List of losses per agent
            "critic_loss": critic_loss,
        }

        return new_training_state, replay_buffer, metrics
