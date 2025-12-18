from .optimizer import BaseOptimizer, Optimizer, LossFn, Generic, Y

from collections.abc import Callable

import optax
import jax.numpy as jnp

from jax import Array, grad

class RGD(Optimizer[Y], Generic[Y]):
    grads: Array
    last_update: Array

    def __init__(
            self,
            params: Array,
            lr: float,
            loss_fn: LossFn[Y],
            proj_fn: Callable[[Array], Array] = (lambda params: params),
            base_optimizer: BaseOptimizer = "GD",
            momentum: float = 0,
    ):
        super().__init__(params, lr, loss_fn, proj_fn)
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
        self.grads = grad(lambda params: jnp.mean(self.loss_fn(params, x, y)))(
            self.current_params
        )
        # self.current_params = jax.tree_util.tree_map(lambda x, y: self.proj_fn(x - self.lr * y) if isinstance(x, jnp.ndarray) else x, params, self.grads)

        updates, self.opt_state = self.optimizer.update(
            self.grads, self.opt_state, params
        )
        self.current_params = self.proj_fn(optax.apply_updates(params, updates))

        self.params_history.append(self.current_params)
        self.i += 1
        return self.current_params
