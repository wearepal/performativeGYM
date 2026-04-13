from collections.abc import Callable, Sequence
from typing import Protocol, TypeVar
import time

import jax
import jax.numpy as jnp
from jax import Array
from numpy import typing as npt
from tqdm.auto import tqdm

from performative_gym.optimizers import LossFn, Optimizers

Y = TypeVar("Y", contravariant=True, bound=Array | None)


class PlotArgs(Protocol):
    n: int
    optimizer: Optimizers


def acc_fn(output: Array, labels: Array) -> Array:
    preds = output > 0.5
    correct = jnp.sum(preds == labels)
    return correct / len(labels)


def initialize_params(n_features: tuple[int], seed: int) -> Array:
    return jax.random.normal(jax.random.PRNGKey(seed), n_features)

def weight_norm(params: Array) -> Array:
    """Compute total Frobenius norm of all weight matrices."""

    def norm_fn(p: Array) -> Array:
        if isinstance(p, jnp.ndarray):
            return jnp.sum(jnp.square(p))
        return 0.0

    squared_norm = jax.tree_util.tree_reduce(
        lambda acc, x: acc + x, jax.tree_util.tree_map(norm_fn, params), initializer=0.0
    )
    return jnp.sqrt(squared_norm)
