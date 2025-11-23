import jax
import jax.numpy as jnp
import pytest

from qdax.core.neuroevolution.buffers.buffer import Transition
from qdax.core.neuroevolution.buffers.trajectory_buffer_v2 import TrajectoryBuffer


class TestTrajectoryBufferV2:
    """Test suite for TrajectoryBuffer v2."""
    
    @pytest.fixture
    def setup_buffer(self):
        """Setup a basic buffer configuration."""
        obs_dim = 4
        action_dim = 2
        buffer_size = 12
        episode_length = 4
        env_batch_size = 3
        
        # Create a sample transition for initialization
        dummy_transition = Transition(
            obs=jnp.zeros(obs_dim),
            next_obs=jnp.zeros(obs_dim),
            rewards=jnp.array(0.0),
            dones=jnp.array(0.0),
            actions=jnp.zeros(action_dim),
            truncations=jnp.array(0.0),
        )
        
        buffer = TrajectoryBuffer.init(
            buffer_size=buffer_size,
            transition=dummy_transition,
            env_batch_size=env_batch_size,
            episode_length=episode_length,
        )
        
        return buffer, obs_dim, action_dim, env_batch_size, episode_length
    
    def test_buffer_initialization(self, setup_buffer):
        """Test that buffer initializes correctly."""
        buffer, obs_dim, action_dim, env_batch_size, episode_length = setup_buffer
        
        assert buffer.buffer_size == 12
        assert buffer.episode_length == episode_length
        assert buffer.env_batch_size == env_batch_size
        assert buffer.num_trajectories == 3
        assert buffer.current_position == 0
        assert buffer.current_size == 0
        assert jnp.all(jnp.isnan(buffer.data))
    
    def test_insert_single_batch(self, setup_buffer):
        """Test inserting a single batch of transitions."""
        buffer, obs_dim, action_dim, env_batch_size, episode_length = setup_buffer
        
        # Create transitions for one step
        transitions = Transition(
            obs=jnp.ones((env_batch_size, obs_dim)),
            next_obs=jnp.ones((env_batch_size, obs_dim)) * 2,
            rewards=jnp.ones(env_batch_size),
            dones=jnp.zeros(env_batch_size),
            actions=jnp.ones((env_batch_size, action_dim)),
            truncations=jnp.zeros(env_batch_size),
        )
        
        buffer = buffer.insert(transitions)
        
        assert buffer.current_size == env_batch_size
        assert buffer.current_position == env_batch_size
        assert jnp.all(buffer.timestep_positions == 1)
    
    def test_insert_complete_episode(self, setup_buffer):
        """Test inserting a complete episode."""
        buffer, obs_dim, action_dim, env_batch_size, episode_length = setup_buffer
        
        # Insert episode_length steps with done=1 at the end
        for step in range(episode_length):
            dones = jnp.where(
                step == episode_length - 1,
                jnp.ones(env_batch_size),
                jnp.zeros(env_batch_size)
            )
            transitions = Transition(
                obs=jnp.ones((env_batch_size, obs_dim)) * step,
                next_obs=jnp.ones((env_batch_size, obs_dim)) * (step + 1),
                rewards=jnp.ones(env_batch_size) * (step + 1),
                dones=dones,
                actions=jnp.ones((env_batch_size, action_dim)),
                truncations=jnp.zeros(env_batch_size),
            )
            buffer = buffer.insert(transitions)
        
        assert buffer.current_size == episode_length * env_batch_size
        assert jnp.all(buffer.timestep_positions == 0)  # Reset after done
        assert jnp.all(buffer.trajectory_positions == 1)  # Moved to next episode
    
    def test_sample_basic(self, setup_buffer):
        """Test basic sampling."""
        buffer, obs_dim, action_dim, env_batch_size, episode_length = setup_buffer
        
        # Insert some transitions
        transitions = Transition(
            obs=jnp.ones((env_batch_size * 2, obs_dim)),
            next_obs=jnp.ones((env_batch_size * 2, obs_dim)) * 2,
            rewards=jnp.ones(env_batch_size * 2),
            dones=jnp.zeros(env_batch_size * 2),
            actions=jnp.ones((env_batch_size * 2, action_dim)),
            truncations=jnp.zeros(env_batch_size * 2),
        )
        buffer = buffer.insert(transitions)
        
        # Sample
        key = jax.random.PRNGKey(0)
        sample_size = 4
        sampled_transitions, _ = buffer.sample(key, sample_size)
        
        assert sampled_transitions.obs.shape == (sample_size, obs_dim)
        assert sampled_transitions.rewards.shape == (sample_size,)
    
    def test_sample_n_step(self, setup_buffer):
        """Test n-step sampling."""
        buffer, obs_dim, action_dim, env_batch_size, episode_length = setup_buffer
        
        # Insert complete episodes with varying rewards
        for episode in range(2):
            for step in range(episode_length):
                dones = jnp.where(
                    step == episode_length - 1,
                    jnp.ones(env_batch_size),
                    jnp.zeros(env_batch_size)
                )
                transitions = Transition(
                    obs=jnp.ones((env_batch_size, obs_dim)) * (episode * 10 + step),
                    next_obs=jnp.ones((env_batch_size, obs_dim)) * (episode * 10 + step + 1),
                    rewards=jnp.ones(env_batch_size) * (step + 1),
                    dones=dones,
                    actions=jnp.ones((env_batch_size, action_dim)),
                    truncations=jnp.zeros(env_batch_size),
                )
                buffer = buffer.insert(transitions)
        
        # Sample n-step transitions
        key = jax.random.PRNGKey(42)
        sample_size = 2
        n_step = 3
        sampled_transitions, _ = buffer.sample_n_step(key, sample_size, n_step)
        
        # Check shapes
        assert sampled_transitions.obs.shape == (sample_size, n_step, obs_dim)
        assert sampled_transitions.rewards.shape == (sample_size, n_step)
        assert sampled_transitions.actions.shape == (sample_size, n_step, action_dim)
    
    def test_sample_n_step_edge_cases(self, setup_buffer):
        """Test n-step sampling with n_step=1."""
        buffer, obs_dim, action_dim, env_batch_size, episode_length = setup_buffer
        
        # Insert transitions
        transitions = Transition(
            obs=jnp.ones((env_batch_size * episode_length, obs_dim)),
            next_obs=jnp.ones((env_batch_size * episode_length, obs_dim)) * 2,
            rewards=jnp.ones(env_batch_size * episode_length),
            dones=jnp.zeros(env_batch_size * episode_length),
            actions=jnp.ones((env_batch_size * episode_length, action_dim)),
            truncations=jnp.zeros(env_batch_size * episode_length),
        )
        buffer = buffer.insert(transitions)
        
        # Sample with n_step=1 (should behave like regular sampling but with extra dim)
        key = jax.random.PRNGKey(0)
        sampled_transitions, _ = buffer.sample_n_step(key, sample_size=4, n_step=1)
        
        assert sampled_transitions.obs.shape == (4, 1, obs_dim)
    
    def test_compute_returns(self, setup_buffer):
        """Test return computation."""
        buffer, obs_dim, action_dim, env_batch_size, episode_length = setup_buffer
        
        # Insert episode with known rewards
        for step in range(episode_length):
            dones = jnp.where(
                step == episode_length - 1,
                jnp.ones(env_batch_size),
                jnp.zeros(env_batch_size)
            )
            transitions = Transition(
                obs=jnp.ones((env_batch_size, obs_dim)),
                next_obs=jnp.ones((env_batch_size, obs_dim)),
                rewards=jnp.ones(env_batch_size) * (step + 1),  # Rewards: 1, 2, 3, 4
                dones=dones,
                actions=jnp.ones((env_batch_size, action_dim)),
                truncations=jnp.zeros(env_batch_size),
            )
            buffer = buffer.insert(transitions)
        
        # Expected return: 1 + 2 + 3 + 4 = 10 for each trajectory
        expected_return = 10.0
        
        # Check that returns are computed correctly
        valid_returns = buffer.returns[~jnp.isinf(buffer.returns)]
        valid_returns = valid_returns[~jnp.isnan(valid_returns)]
        
        assert jnp.allclose(valid_returns, expected_return)
    
    def test_sample_with_returns(self, setup_buffer):
        """Test sampling with returns."""
        buffer, obs_dim, action_dim, env_batch_size, episode_length = setup_buffer
        
        # Insert complete episode
        for step in range(episode_length):
            dones = jnp.where(
                step == episode_length - 1,
                jnp.ones(env_batch_size),
                jnp.zeros(env_batch_size)
            )
            transitions = Transition(
                obs=jnp.ones((env_batch_size, obs_dim)),
                next_obs=jnp.ones((env_batch_size, obs_dim)),
                rewards=jnp.ones(env_batch_size),
                dones=dones,
                actions=jnp.ones((env_batch_size, action_dim)),
                truncations=jnp.zeros(env_batch_size),
            )
            buffer = buffer.insert(transitions)
        
        # Sample with returns
        key = jax.random.PRNGKey(0)
        sampled_transitions, sampled_returns, _ = buffer.sample_with_returns(key, sample_size=4)
        
        assert sampled_transitions.obs.shape == (4, obs_dim)
        assert sampled_returns.shape == (4,)
        # All returns should be 4.0 (sum of 1+1+1+1)
        assert jnp.allclose(sampled_returns, 4.0)
