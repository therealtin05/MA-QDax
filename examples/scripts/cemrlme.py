
import os

from IPython.display import clear_output
import functools
import time

import jax
import jax.numpy as jnp

from typing import Optional, Tuple, Callable, Any

from baselines.qdax.core.map_elites import MAPElites
from baselines.qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
from baselines.qdax import environments
from baselines.qdax.tasks.brax_envs import scoring_function_brax_envs as scoring_function
from baselines.qdax.core.neuroevolution.buffers.buffer import QDTransition
from baselines.qdax.core.neuroevolution.networks.networks import MLP
from baselines.qdax.core.emitters.mutation_operators import isoline_variation
from baselines.qdax.utils.plotting import plot_map_elites_results

from baselines.qdax.core.emitters.cemrl_emitter import CEMRLEmitter
from baselines.qdax.core.emitters.cemrl_me_emitter import CEMRLMEConfig, CEMRLMEEmitter
from baselines.qdax.utils.metrics import CSVLogger, default_qd_metrics
from baselines.qdax.environments import get_feat_mean

from baselines.qdax.core.neuroevolution.buffers.buffer import Transition

#@title QD Training Definitions Fields
#@markdown ---
env_name = 'walker2d_feet_contact'#@param['ant_uni', 'hopper_uni', 'walker_uni', 'halfcheetah_uni', 'humanoid_uni', 'ant_omni', 'humanoid_omni']
env_batch_size = 100 #@param {type:"number"}
episode_length=1000
num_iterations = 4000 #@param {type:"integer"}
seed = 5 #@param {type:"integer"}
policy_hidden_layer_sizes = (256, 256) #@param {type:"raw"}
iso_sigma = 0.005 #@param {type:"number"}
line_sigma = 0.05 #@param {type:"number"}
num_init_cvt_samples = 50000 #@param {type:"integer"}
num_centroids = 1024 #@param {type:"integer"}
min_bd = 0. #@param {type:"number"}
max_bd = 1.0 #@param {type:"number"}

proportion_mutation_ga: float = 0.5
num_critic_training_steps: int = 3000
num_pg_training_steps: int = 100

num_warmstart_steps: int = 25_600

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
use_layer_norm: bool = False
#@markdown ---

# Init environment
env = environments.create(env_name, episode_length=episode_length)

# Init a random key
random_key = jax.random.PRNGKey(seed)

# Init policy network
policy_layer_sizes = policy_hidden_layer_sizes + (env.action_size,)
policy_network = MLP(
    layer_sizes=policy_layer_sizes,
    # kernel_init=jax.nn.initializers.lecun_uniform(),
    final_activation=jnp.tanh,
    use_layer_norm=use_layer_norm,
)

# Init population of controllers
random_key, subkey = jax.random.split(random_key)
keys = jax.random.split(subkey, num=env_batch_size)
fake_batch = jnp.zeros(shape=(env_batch_size, env.observation_size))
init_variables = jax.vmap(policy_network.init)(keys, fake_batch)



# Create the initial environment states
random_key, subkey = jax.random.split(random_key)
keys = jnp.repeat(jnp.expand_dims(subkey, axis=0), repeats=env_batch_size, axis=0)
reset_fn = jax.jit(jax.vmap(env.reset))
init_states = reset_fn(keys)


# Define the fonction to play a step with the policy in the environment
def play_step_fn(
  env_state,
  policy_params,
  random_key,
):
    """
    Play an environment step and return the updated state and the transition.
    """

    actions = policy_network.apply(policy_params, env_state.obs)
    
    state_desc = env_state.info["state_descriptor"]
    next_state = env.step(env_state, actions)

    transition = QDTransition(
        obs=env_state.obs,
        next_obs=next_state.obs,
        rewards=next_state.reward,
        dones=next_state.done,
        actions=actions,
        truncations=next_state.info["truncation"],
        state_desc=state_desc,
        next_state_desc=next_state.info["state_descriptor"],
        desc=jnp.zeros(env.behavior_descriptor_length, ) * jnp.nan,
        desc_prime=jnp.zeros(env.behavior_descriptor_length, ) * jnp.nan,
    )

    return next_state, policy_params, random_key, transition

# Prepare the scoring function
bd_extraction_fn = get_feat_mean
scoring_fn = functools.partial(
    scoring_function,
    init_states=init_states,
    episode_length=episode_length,
    play_step_fn=play_step_fn,
    behavior_descriptor_extractor=bd_extraction_fn,
)

# Get minimum reward value to make sure qd_score are positive
reward_offset = environments.reward_offset[env_name]

# Define a metrics function
metrics_function = functools.partial(
    default_qd_metrics,
    qd_offset=reward_offset * episode_length,
)


# Define the PG-emitter config
cemrl_emitter_config = CEMRLMEConfig(
    env_batch_size=env_batch_size,
    proportion_mutation_ga=proportion_mutation_ga,
    num_critic_training_steps=num_critic_training_steps,
    num_pg_training_steps=num_pg_training_steps,
    num_warmstart_steps=num_warmstart_steps,

    # CEM
    population_size=population_size,
    num_best=num_best,
    damp_init=damp_init,
    damp_final=damp_final,
    damp_tau=damp_tau,
    rank_weight_shift=rank_weight_shift,
    mirror_sampling=mirror_sampling,
    weighted_update=weighted_update,
    num_learning_offspring=num_learning_offspring,

    # TD3 params
    replay_buffer_size=replay_buffer_size,
    critic_hidden_layer_size=critic_hidden_layer_size,
    critic_learning_rate=critic_learning_rate,
    actor_learning_rate=actor_learning_rate,
    policy_learning_rate=policy_learning_rate,
    noise_clip=noise_clip,
    policy_noise=policy_noise,
    discount=discount,
    reward_scaling=reward_scaling,
    batch_size=batch_size,
    soft_tau_update=soft_tau_update,
    policy_delay=policy_delay,
    use_layer_norm=use_layer_norm,
)


# Get the emitter
variation_fn = functools.partial(
    isoline_variation, iso_sigma=iso_sigma, line_sigma=line_sigma
)

pg_emitter = CEMRLMEEmitter(
    config=cemrl_emitter_config,
    policy_network=policy_network,
    env=env,
    variation_fn=variation_fn,
)

# Change the init_variables to follow the initial distribution
mean = jax.tree_util.tree_map(lambda x: x[0], init_variables)
var = jax.tree_util.tree_map(
    lambda x: jnp.full_like(x, damp_init), 
    mean
)

random_key, subkey = jax.random.split(random_key)
init_cem_offsprings, random_key = CEMRLEmitter.sample_cem_offsprings(
    mean, var, subkey, population_size, mirror_sampling
)

init_variables = jax.tree_util.tree_map(
    lambda x, y: x.at[jnp.arange(0,population_size)].set(y),
    init_variables, init_cem_offsprings
)
# mean will be the (population_size+1)-th element
init_variables = jax.tree_util.tree_map(
    lambda x, y: x.at[population_size].set(y),
    init_variables, mean
)


# Instantiate MAP Elites
map_elites = MAPElites(
    scoring_function=scoring_fn,
    emitter=pg_emitter,
    metrics_function=metrics_function,
)

# Compute the centroids
centroids, random_key = compute_cvt_centroids(
    num_descriptors=env.behavior_descriptor_length,
    num_init_cvt_samples=num_init_cvt_samples,
    num_centroids=num_centroids,
    minval=min_bd,
    maxval=max_bd,
    random_key=random_key,
)

# compute initial repertoire
repertoire, emitter_state, random_key = map_elites.init(
    init_variables, centroids, random_key
)



log_period = 10
num_loops = int(num_iterations / log_period)

csv_logger = CSVLogger(
    "cemrlme-logs.csv",
    header=["loop", "iteration", "qd_score", "max_fitness", "mean_fitness", "coverage",\
             "rl_in_elites_percentage", "damp", "max_var", "mean_var", "median_var", "time"]
)
all_metrics = {}

# main loop
map_elites_scan_update = map_elites.scan_update
for i in range(num_loops):
    start_time = time.time()
    # main iterations
    (repertoire, emitter_state, random_key,), metrics = jax.lax.scan(
        map_elites_scan_update,
        (repertoire, emitter_state, random_key),
        (),
        length=log_period,
    )
    timelapse = time.time() - start_time

    # log metrics
    logged_metrics = {"time": timelapse, "loop": 1+i, "iteration": 1 + i*log_period}
    for key, value in metrics.items():

        # take last value
        logged_metrics[key] = value[-1]

        # take all values
        if key in all_metrics.keys():
            all_metrics[key] = jnp.concatenate([all_metrics[key], value])
        else:
            all_metrics[key] = value

    # log emitter metrics
    emitter_metrics = pg_emitter.report(emitter_state=emitter_state)
    for key, value in emitter_metrics.items():
        logged_metrics[key] = value

        # # take all values
        # if key in all_metrics.keys():
        #     all_metrics[key] = jnp.concatenate([all_metrics[key], value])
        # else:
        #     all_metrics[key] = value

    csv_logger.log(logged_metrics)