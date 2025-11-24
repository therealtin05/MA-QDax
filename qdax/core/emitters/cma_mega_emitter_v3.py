"""Similar to cma_mega_emitter_v2.py, however now, the coeffs ensured to be correct by
kept in training state, not by using same random seed (probably wrong in v2)
-> prevent wrong coeffs 
"""

from __future__ import annotations

from functools import partial
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp

from qdax.core.cmaes import CMAES, CMAESState
from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    get_cells_indices,
)
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.types import (
    Centroid,
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    Gradient,
    RNGKey,
)


class CMAMEGAState(EmitterState):
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
    random_key: RNGKey
    cmaes_state: CMAESState
    coeffs: jnp.ndarray
    previous_fitnesses: Fitness
    steps: jnp.ndarray = jnp.array(0)


class CMAMEGAEmitter(Emitter):
    def __init__(
        self,
        scoring_function: Callable[
            [Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]
        ],
        batch_size: int,
        learning_rate: float,
        num_descriptors: int,
        centroids: Centroid,
        sigma_g: float,
        exact_gradient: bool = True,
        es_batch_size: Optional[int] = None,
        mirrored_sampling: bool = False,
        es_noise: float = 0.02,
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
            exact_gradient: a flag if the gradient for fitness and BD differential thus exactly derived,
            es_batch_size: if we cannot compute exact gradient, we need to estimate gradient through ES,
            mirrored_sampling: flag if apply mirrored sampling for ES gradient estimation
            es_noise: noise for ES to estimate gradients
        """

        self._scoring_function = scoring_function
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._num_descriptors = num_descriptors
        if es_batch_size != None:
            self._es_batch_size = es_batch_size
        else:
            self._es_batch_size = batch_size // 2
        self._exact_gradient = exact_gradient
        self._mirrored_sampling = mirrored_sampling
        self._es_noise = es_noise
        # weights used to update the gradient direction through a linear combination
        self._weights = jnp.expand_dims(
            jnp.log(batch_size + 0.5) - jnp.log(jnp.arange(1, batch_size + 1)), axis=-1
        )
        self._weights = self._weights / (self._weights.sum())

        # define a CMAES instance - used to update the coeffs
        self._cmaes = CMAES(
            population_size=batch_size,
            search_dim=num_descriptors + 1,
            # no need for fitness function in that specific case
            fitness_function=None,  # type: ignore
            num_best=batch_size,
            init_sigma=sigma_g,
            bias_weights=True,
            delay_eigen_decomposition=True,
        )

        self._centroids = centroids

        self._cma_initial_state = self._cmaes.init()

    @partial(jax.jit, static_argnames=("self",))
    def init(
        self, init_genotypes: Genotype, random_key: RNGKey
    ) -> Tuple[CMAMEGAState, RNGKey]:
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


        # return the initial state
        random_key, subkey = jax.random.split(random_key)
        return (
            CMAMEGAState(
                theta=theta,
                theta_grads=theta_grads_init,
                random_key=subkey,
                cmaes_state=self._cma_initial_state,
                coeffs=default_coeffs,
                previous_fitnesses=default_fitnesses,
                steps=jnp.array(0),
            ),
            random_key,
        )

    @partial(jax.jit, static_argnames=("self",))
    def emit(
        self,
        repertoire: Optional[MapElitesRepertoire],
        emitter_state: CMAMEGAState,
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
        emitter_state: CMAMEGAState,
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

        cmaes_state = emitter_state.cmaes_state

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
        emitter_state: CMAMEGAState,
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

        if self._mirrored_sampling:
            half_noise = jax.tree_util.tree_map(
                lambda m, k: jax.random.normal(k, (self._es_batch_size // 2,) + m.shape[1:]) * self._es_noise,
                emitter_state.theta, keys_tree
            )
            noise = jax.tree_util.tree_map(
                lambda x: jnp.concatenate([x, -x], axis=0),
                half_noise
            )

        else:
            noise = jax.tree_util.tree_map(
                lambda m, k: jax.random.normal(k, (self._es_batch_size,) + m.shape[1:]) * self._es_noise,
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
        emitter_state: CMAMEGAState,
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
        return self._batch_size
    

    @partial(jax.jit, static_argnames=("self"))
    def _update_cmaes(
        self,
        emitter_state: CMAMEGAState,
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
            lambda x, y: x + self._learning_rate * y, theta, gradient_step
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
        emitter_state = CMAMEGAState(
            theta=theta,
            theta_grads=emitter_state.theta_grads,
            random_key=random_key,
            cmaes_state=cmaes_state,
            coeffs=new_coeffs,
            previous_fitnesses=repertoire.fitnesses,
            steps=emitter_state.steps+1
        )

        return emitter_state

    @partial(jax.jit, static_argnames=("self"))
    def _estimate_gradient_ES(
        self,
        emitter_state: CMAMEGAState,
        repertoire: MapElitesRepertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: Optional[ExtraScores] = None,
    ):
        
        print(f"IN ES!: cur_step {emitter_state.steps}")

        mean = emitter_state.theta
        random_key = emitter_state.random_key
        noise = jax.tree_util.tree_map(
            lambda g, m: (g - m) / self._es_noise,
            genotypes, mean
        )

        fitnesses_and_desc = jnp.concatenate([fitnesses[...,None],  descriptors], axis=1) # shape (n_samples, 1+desc_dim)

        ranking_indices = jnp.argsort(fitnesses_and_desc, axis=0) # shape (n_samples, 1+desc_dim)
        ranks = jnp.argsort(ranking_indices, axis=0) 
        ranks = (ranks / (self._es_batch_size - 1)) - 0.5

        gradients = jax.tree_util.tree_map(
            lambda n: jnp.sum(
                jnp.expand_dims(ranks, axis=[i for i in range(1, n.ndim)]) * n[..., None], 
                axis=0,
                keepdims=True,
            ) / (self._es_noise * self._es_batch_size),
            noise,
        ) # gradients is now has dim (1, problem_dim, 1+bd_dim, ...)

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
            steps=emitter_state.steps+1
        )

        return emitter_state