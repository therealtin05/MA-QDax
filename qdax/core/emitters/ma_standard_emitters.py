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
    ) -> Tuple[Genotype, RNGKey]:
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

        return genotypes, random_key

    @property
    def batch_size(self) -> int:
        """
        Returns:
            the batch size emitted by the emitter.
        """
        return self._batch_size


class RolePreservingMultiAgentMixingEmitter(Emitter):
    def __init__(
        self,
        mutation_fn: Callable[[Genotype, RNGKey], Tuple[Genotype, RNGKey]],
        variation_fn: Callable[[Genotype, Genotype, RNGKey], Tuple[Genotype, RNGKey]],
        variation_percentage: float,
        batch_size: int,
        num_agents: int,
        **kwargs: Dict,
    ) -> None:
        self._mutation_fn = mutation_fn
        self._variation_fn = variation_fn
        self._variation_percentage = variation_percentage
        self._batch_size = batch_size
        self._num_agents = num_agents

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def emit(
        self,
        repertoire: Repertoire,
        emitter_state: Optional[EmitterState],
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
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

        n_variation = int(self._batch_size * self._variation_percentage)
        n_mutation = self._batch_size - n_variation
        x_variation = None
        x_mutation = None

        if n_variation > 0:
            # FIXME: this is not efficient, we should sample only once
            for i in range(self._num_agents):
                x1, random_key = repertoire.sample(random_key, n_variation)
                x2, random_key = repertoire.sample(random_key, n_variation)

                x_variation_, random_key = self._variation_fn(x1, x2, random_key)

                if x_variation is None:
                    x_variation = x_variation_
                else:
                    x_variation[i] = x_variation_[i]

        if n_mutation > 0:
            # FIXME: this is not efficient, we should sample only once
            for i in range(self._num_agents):
                x1, random_key = repertoire.sample(random_key, n_mutation)
                x_mutation_, random_key = self._mutation_fn(x1, random_key)

                if x_mutation is None:
                    x_mutation = x_mutation_
                else:
                    x_mutation[i] = x_mutation_[i]

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

        return genotypes, random_key

    @property
    def batch_size(self) -> int:
        """
        Returns:
            the batch size emitted by the emitter.
        """
        return self._batch_size


class SharedPoolMultiAgentMixingEmitter(Emitter):
    def __init__(
        self,
        mutation_fn: Callable[[Genotype, RNGKey], Tuple[Genotype, RNGKey]],
        variation_fn: Callable[[Genotype, Genotype, RNGKey], Tuple[Genotype, RNGKey]],
        variation_percentage: float,
        batch_size: int,
        num_agents: int,
        **kwargs: Dict,
    ) -> None:
        self._mutation_fn = mutation_fn
        self._variation_fn = variation_fn
        self._variation_percentage = variation_percentage
        self._batch_size = batch_size
        self._num_agents = num_agents

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def emit(
        self,
        repertoire: Repertoire,
        emitter_state: Optional[EmitterState],
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
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

        n_variation = int(self._batch_size * self._variation_percentage)
        n_mutation = self._batch_size - n_variation
        x_variation = None
        x_mutation = None

        if n_variation > 0:
            # FIXME: this is not efficient, we should sample only once
            for i in range(self._num_agents):
                x1, random_key = repertoire.sample(random_key, n_variation)
                x2, random_key = repertoire.sample(random_key, n_variation)

                x_variation_, random_key = self._variation_fn(x1, x2, random_key)

                if x_variation is None:
                    x_variation = x_variation_

                # Sample an agent from the tuple
                j = random.randint(0, self._num_agents - 1)
                x_variation[i] = x_variation_[j]

        if n_mutation > 0:
            # FIXME: this is not efficient, we should sample only once
            for i in range(self._num_agents):
                x1, random_key = repertoire.sample(random_key, n_mutation)
                x_mutation_, random_key = self._mutation_fn(x1, random_key)

                if x_mutation is None:
                    x_mutation = x_mutation_

                # Sample an agent from the tuple
                j = random.randint(0, self._num_agents - 1)
                x_mutation[i] = x_mutation_[j]

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

        return genotypes, random_key

    @property
    def batch_size(self) -> int:
        """
        Returns:
            the batch size emitted by the emitter.
        """
        return self._batch_size
