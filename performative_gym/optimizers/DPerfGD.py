from typing import cast

from .optimizer import BaseOptimizer, Optimizer, LossFn, Generic, Y

from collections.abc import Callable

import optax
import jax.numpy as jnp
import jax

from jax import Array, grad

__all__ = ["DPerfGD"]

#TODO: implement Reg, Barrier version

class DPerfGD(Optimizer[Y], Generic[Y]):  # Decoupled Gradient Descent
    grads: Array
    last_update: Array

    def __init__(
            self,
            params: Array,
            lr: float,
            loss_fn: LossFn[Y],
            proj_fn: Callable[[Array], Array],
            distr_map: Callable[[Array], tuple[Array, Y]],
            reg: float = 0,
            base_optimizer: BaseOptimizer = "GD",
            momentum: float = 0,
    ):
        super().__init__(params, lr, loss_fn, proj_fn)
        self.reg = reg
        self.distr_map = distr_map
        self.current_p_d = params
        self.p_d_history = [params]

        if base_optimizer == "GD":
            self.optimizer_M = optax.sgd(learning_rate=lr, momentum=momentum)
            self.optimizer_D = optax.sgd(learning_rate=lr, momentum=momentum)
        elif base_optimizer == "adam":
            self.optimizer_M = optax.adam(learning_rate=lr)
            self.optimizer_D = optax.adam(learning_rate=lr)
        elif base_optimizer == "adamw":
            self.optimizer_M = optax.adamw(learning_rate=lr)
            self.optimizer_D = optax.adamw(learning_rate=lr)
        elif base_optimizer == "adagrad":
            self.optimizer_M = optax.adagrad(learning_rate=lr)
            self.optimizer_D = optax.adagrad(learning_rate=lr)

        self.opt_state_M = self.optimizer_M.init(params)
        self.opt_state_D = self.optimizer_D.init(params)

    def step(self, params: Array, x: Array, y: Y) -> Array:

        def decoupled_loss(p_m: Array, p_d: Array) -> Array:
            x_0, y_0 = self.distr_map(p_d)

            return jnp.mean(
                self.loss_fn(p_m, x=x_0, y=y_0)
            ) + self.reg * jnp.sum(jnp.abs(p_m - p_d + 1e-8))

        grad_M = grad(lambda p: decoupled_loss(p, self.current_p_d))(params)
        grad_D = grad(lambda p_p: decoupled_loss(params, p_p))(self.current_p_d)

        updates_M, self.opt_state_M = self.optimizer_M.update(
            grad_M, self.opt_state_M, params
        )
        current_params = self.proj_fn(optax.apply_updates(params, updates_M))
        self.current_params = cast(Array, current_params)

        updates_D, self.opt_state_D = self.optimizer_D.update(
            grad_D, self.opt_state_D, self.current_p_d
        )
        current_p_d = self.proj_fn(optax.apply_updates(self.current_p_d, updates_D))
        self.current_p_d = cast(Array, current_p_d)

        self.grads = jax.tree_util.tree_map(lambda x, y: x + y, grad_M, grad_D)
        self.params_history.append(self.current_params)
        self.p_d_history.append(self.current_p_d)
        self.i += 1

        return self.current_params
