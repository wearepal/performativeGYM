from typing import cast

from . import BaseOptimizer, Optimizer, LossFn, Generic, Y

from collections.abc import Callable

import optax
import jax.numpy as jnp

from jax import Array, grad


class PerfGDReparam(Optimizer[Y], Generic[Y]):  # Especial Gradient Descent
    grads: Array

    def __init__(
            self,
            params: Array,
            lr: float,
            loss_fn: LossFn[Y],
            proj_fn: Callable[[Array], Array],
            distr_shift: Callable[[Array], tuple[Array, Y]],
            base_optimizer: BaseOptimizer = "GD",
            momentum: float = 0,
    ):
        super().__init__(params, lr, loss_fn, proj_fn)
        self.distr_shift = distr_shift
        if base_optimizer == "GD":
            self.optimizer = optax.sgd(learning_rate=lr, momentum=momentum)
        elif base_optimizer == "adam":
            self.optimizer = optax.adam(learning_rate=lr)
        elif base_optimizer == "adamw":
            self.optimizer = optax.adamw(learning_rate=lr)
        elif base_optimizer == "adagrad":
            self.optimizer = optax.adagrad(learning_rate=lr)
        self.opt_state = self.optimizer.init(params)

    def step(self, params: Array, x: Array, y: Y) -> Array:
        def decoupled_loss(p_p: Array, p: Array) -> Array:
            x, y = self.distr_shift(p_p)
            return jnp.mean(self.loss_fn(p, x, y))

        def performative_optimal(params: Array) -> Array:
            return decoupled_loss(params, params)

        self.grads = grad(lambda params: performative_optimal(params))(params)

        # self.current_params = jax.tree_util.tree_map(lambda x, y: self.proj_fn(x - self.lr * y) if isinstance(x, jnp.ndarray) else x, params, self.grads)

        updates, self.opt_state = self.optimizer.update(
            self.grads, self.opt_state, params
        )
        current_params = optax.apply_updates(params, updates)
        self.current_params = cast(Array, current_params)

        self.params_history.append(self.current_params)
        self.i += 1
        # self.lr = self.lr/self.i
        return self.current_params
