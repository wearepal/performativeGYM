from collections.abc import Callable, Sequence
from typing import Protocol, TypeVar

import jax
import jax.numpy as jnp
from jax import Array
from numpy import typing as npt
from tqdm.auto import tqdm

from .optimizers import LossFn, Optimizers

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


def loss_values_diag(
    shift_data_distribution: Callable[[Array, int], tuple[Array, Y]],
    loss_fn: LossFn[Y],
    args: PlotArgs,
    x_domain: Sequence[Array],
) -> list[Array]:
    print("Calculating loss\n")
    losses = []
    with tqdm(total=len(x_domain)) as pbar:
        for p_p in x_domain:
            x, y = shift_data_distribution(p_p, args.n)
            losses.append(jnp.mean(loss_fn(p_p, x=x, y=y)))
            pbar.update(1)

    return losses


def loss_values(
    shift_data_distribution: Callable[[Array, int], tuple[Array, Y]],
    loss_fn: LossFn[Y],
    n: int,
    x_domain: npt.NDArray,
    y_domain: npt.NDArray,
) -> list[list[Array]]:
    print("Calculating loss\n")
    losses_2d = []
    with tqdm(total=len(x_domain) ** 2) as pbar:
        for p_p in x_domain:
            losses = []
            for p in y_domain:
                x, y = shift_data_distribution(p_p, n)
                losses.append(jnp.mean(loss_fn(p, x=x, y=y)))
                pbar.update(1)
            losses_2d.append(losses)
            if p_p == -1 + 0.01 * losses.index(min(losses)):
                print(f"\n{p_p} is a stable point with loss {min(losses)}")
    return losses_2d


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
