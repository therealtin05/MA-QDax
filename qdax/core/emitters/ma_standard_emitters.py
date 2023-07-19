import random
from functools import partial
from typing import Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp

from qdax.core.containers.repertoire import Repertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.types import Genotype, RNGKey


class NaiveMultiAgentMixingEmitter(Emitter):
    def __init__(
        self,
        mutation_fn: Callable[[Genotype, RNGKey], Tuple[Genotype, RNGKey]],
        variation_fn: Callable[[Genotype, Genotype, RNGKey], Tuple[Genotype, RNGKey]],
        variation_percentage: float,
        batch_size: int,
        num_agents: int,
        agents_to_mutate: int,
    ) -> None:
        self._mutation_fn = mutation_fn
        self._variation_fn = variation_fn
        self._variation_percentage = variation_percentage
        self._batch_size = batch_size
        self._num_agents = num_agents
        self._agents_to_mutate = agents_to_mutate

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def emit(
        self,
        repertoire: Repertoire,
        emitter_state: Optional[EmitterState],
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey, None]:
        """
        Emitter that performs both mutation and variation. Two batches of
        variation_percentage * batch_size genotypes are sampled in the
        repertoire, copied and cross-over to obtain new offsprings. One batch
        of (1.0 - variation_percentage) * batch_size genotypes are sampled in
        the repertoire, copied and mutated.

        Note: this emitter has no state. A fake none state must be added
        through a function redefinition to make this emitter usable with
        MAP-Elites.

        Params:
            repertoire: the MAP-Elites repertoire to sample from
            emitter_state: void
            random_key: a jax PRNG random key

        Returns:
            a batch of offsprings
            a new jax PRNG key
        """
        # The indices of agents to not vary
        agent_indices = random.sample(
            range(self._num_agents), self._num_agents - self._agents_to_mutate
        )

        n_variation = int(self._batch_size * self._variation_percentage)
        n_mutation = self._batch_size - n_variation

        if n_variation > 0:
            x1, random_key = repertoire.sample(random_key, n_variation)
            x2, random_key = repertoire.sample(random_key, n_variation)

            x_variation, random_key = self._variation_fn(x1, x2, random_key)

            # Put back agents in their original positions (x_variation is a list)
            for i in agent_indices:
                x_variation[i] = x1[i]

        if n_mutation > 0:
            x1, random_key = repertoire.sample(random_key, n_mutation)
            x_mutation, random_key = self._mutation_fn(x1, random_key)

            # Put back agents in their original positions
            for i in agent_indices:
                x_mutation[i] = x1[i]

        if n_variation == 0:
            genotypes = x_mutation
        elif n_mutation == 0:
            genotypes = x_variation
        else:
            genotypes = jax.tree_util.tree_map(
                lambda x_1, x_2: jnp.concatenate([x_1, x_2], axis=0),
                x_variation,
                x_mutation,
            )

        return genotypes, random_key, None

    @property
    def batch_size(self) -> int:
        """
        Returns:
            the batch size emitted by the emitter.
        """
        return self._batch_size


class MultiAgentEmitter(Emitter):
    def __init__(
        self,
        mutation_fn: Callable[[Genotype, RNGKey], Tuple[Genotype, RNGKey]],
        variation_fn: Callable[[Genotype, Genotype, RNGKey], Tuple[Genotype, RNGKey]],
        variation_percentage: float,
        crossplay_percentage: float,
        batch_size: int,
        num_agents: int,
        role_preserving: bool = True,
        **kwargs: Dict,
    ) -> None:
        self._mutation_fn = mutation_fn
        self._variation_fn = variation_fn
        self._variation_percentage = variation_percentage
        self._crossplay_percentage = crossplay_percentage
        self._batch_size = batch_size
        self._num_agents = num_agents
        self._role_preserving = role_preserving

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def emit(
        self,
        repertoire: Repertoire,
        emitter_state: Optional[EmitterState],
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey, jnp.ndarray]:
        """
        Emitter that performs both mutation and variation. Two batches of
        variation_percentage * batch_size genotypes are sampled in the
        repertoire, copied and cross-over to obtain new offsprings. One batch
        of (1.0 - variation_percentage) * batch_size genotypes are sampled in
        the repertoire, copied and mutated.

        Note: this emitter has no state. A fake none state must be added
        through a function redefinition to make this emitter usable with
        MAP-Elites.

        Params:
            repertoire: the MAP-Elites repertoire to sample from
            emitter_state: void
            random_key: a jax PRNG random key

        Returns:
            a batch of offsprings
            a new jax PRNG key
        """
        # The indices of agents to vary
        assert (
            0 <= self._variation_percentage + self._crossplay_percentage <= 1.0
        ), "The sum of variation and crossplay percentages must be between 0 and 1"

        n_variation = int(self._batch_size * self._variation_percentage)
        n_crossplay = int(self._batch_size * self._crossplay_percentage)
        n_mutation = self._batch_size - n_variation - n_crossplay
        x_variation = None
        x_mutation = None
        x_crossplay = None

        if n_variation > 0:
            x1, random_key = repertoire.sample(random_key, n_variation)
            x2, random_key = repertoire.sample(random_key, n_variation)
            x_variation, random_key = self._variation_fn(x1, x2, random_key)

        if n_mutation > 0:
            x1, random_key = repertoire.sample(random_key, n_mutation)
            x_mutation, random_key = self._mutation_fn(x1, random_key)

        if n_crossplay > 0:
            # TODO: this is not efficient, we should sample only once
            x_crossplay = [None] * self._num_agents
            for i in range(self._num_agents):
                x1, random_key = repertoire.sample(random_key, n_crossplay)

                x_crossplay[i] = (
                    x1[i]
                    if self._role_preserving
                    else x1[random.randint(0, self._num_agents - 1)]
                )

        x_values = [x for x in [x_variation, x_mutation, x_crossplay] if x is not None]
        genotypes = jax.tree_util.tree_map(
            lambda *x: jnp.concatenate(x, axis=0), *x_values
        )
        operation_history = jnp.concatenate(
            [
                jnp.zeros(n_variation, dtype=jnp.int32),
                jnp.ones(n_mutation, dtype=jnp.int32),
                2 * jnp.ones(n_crossplay, dtype=jnp.int32),
            ],
            axis=0,
        )

        return genotypes, random_key, operation_history

    @property
    def batch_size(self) -> int:
        """
        Returns:
            the batch size emitted by the emitter.
        """
        return self._batch_size
