from . import BaseOptimizer, Optimizer, LossFn, Generic, Y

from collections.abc import Callable, Sequence

import optax
import jax.numpy as jnp
import jax

from jax import Array, grad

#Todo: RRM to use adam, adagrad, etc..
# Todo: implement regularized version
class RRM(Optimizer[Y], Generic[Y]):
    grads: Array

    def __init__(
            self,
            params: Array,
            lr: float,
            loss_fn: LossFn[Y],
            proj_fn: Callable[[Array], Array],
            tol: float,
    ):
        super().__init__(params, lr, loss_fn, proj_fn)
        self.tol = tol

    def compute_mean(self, params_list: Sequence[Array]):
        # Use tree_map to compute the mean across all corresponding elements
        return jax.tree_util.tree_map(
            lambda *arrays: jnp.mean(jnp.stack(arrays), axis=0)
            if all(isinstance(a, jnp.ndarray) for a in arrays)
            else arrays[0],
            *params_list,
        )

    def step(self, params: Array, x: Array, y: Y) -> Array:
        total_diff = jnp.finfo(
            jnp.float64
        ).max  # initial value for grads so it enters in while loop
        history_grads = []
        j = 0
        while total_diff > self.tol:
            grads = grad(lambda params: jnp.mean(self.loss_fn(params, x, y)))(
                self.current_params
            )
            params_new = jax.tree_util.tree_map(
                lambda x, y: self.proj_fn(x - self.lr * y)
                if isinstance(x, jnp.ndarray)
                else x,
                params,
                grads,
            )

            diff = jax.tree_util.tree_map(
                lambda x, y: jnp.linalg.norm(x - y)
                if isinstance(x, jnp.ndarray)
                else x,
                params_new,
                params,
            )
            total_diff = sum(
                jnp.sum(leaf)
                for leaf in jax.tree_util.tree_leaves(diff)
                if isinstance(leaf, jnp.ndarray)
            )

            params = params_new
            j += 1
            history_grads.append(grads)

        self.current_params = params
        self.params_history.append(self.current_params)
        self.grads = self.compute_mean(history_grads)
        self.i += 1
        return self.current_params
