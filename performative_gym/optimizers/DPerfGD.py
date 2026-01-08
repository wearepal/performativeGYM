from typing import cast

from .optimizer import BaseOptimizer, Optimizer, LossFn, Generic, Y

from collections.abc import Callable

import optax
import jax.numpy as jnp
import jax

from jax import Array, grad

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
            distr_shift: Callable[[Array], tuple[Array, Y]],
            reg: float = 0,
            base_optimizer: BaseOptimizer = "GD",
            momentum: float = 0,
            rho: float = 0,
    ):
        super().__init__(params, lr, loss_fn, proj_fn)
        self.reg = reg
        self.distr_shif = distr_shift
        self.current_p_d = params
        self.p_d_history = [params]
        self.rho = rho

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
        def _sam_perturb(grads: Array, params: Array):
            """Apply SAM perturbation to parameters."""

            def perturb(p: Array, g: Array) -> Array:
                g_norm = jnp.linalg.norm(g)
                scale = self.rho / (g_norm + 1e-12)
                return p + scale * g

            return jax.tree_util.tree_map(perturb, params, grads)

        def decoupled_loss(p_m: Array, p_d: Array) -> Array:
            x_0, y_0 = self.distr_shif(p_d)
            # return jnp.mean(self.loss_fn(p_m, x=x, y=y)) + self.reg * jnp.linalg.norm(p_m - p_d + 1e-8)**2 #figure out why gradient of 0 is Nan
            # figure out why gradient of 0 is Nan
            reg_term = jax.tree_util.tree_map(lambda a, b: jnp.sum((a - b) ** 2), p_m, p_d)
            return jnp.mean(
                self.loss_fn(p_m, x=x_0, y=y_0)
            ) + self.reg * jax.tree_util.tree_reduce(lambda x, y: x + y, reg_term)

        grad_M = grad(lambda p: decoupled_loss(p, self.current_p_d))(params)
        grad_D = grad(lambda p_p: decoupled_loss(params, p_p))(self.current_p_d)

        if self.rho != 0:
            perturbed_M = _sam_perturb(grad_M, params)
            perturbed_D = _sam_perturb(grad_D, self.current_p_d)
            grad_M = grad(lambda p: decoupled_loss(p, self.current_p_d))(perturbed_M)
            grad_D = grad(lambda p_p: decoupled_loss(params, p_p))(perturbed_D)

        # self.current_params = jax.tree_util.tree_map(lambda p_m, grads_m: self.proj_fn(p_m - self.lr * grads_m) if isinstance(p_m, jnp.ndarray) else p_m, params, grad_M)
        # self.current_p_d = jax.tree_util.tree_map(lambda p_d, grads_d: self.proj_fn(p_d - self.lr * grads_d) if isinstance(p_d, jnp.ndarray) else p_d, self.current_p_d, grad_D)

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

class DecCostDPerfGD(Optimizer[Y], Generic[Y]):  # Decoupled Gradient Descent
    grads: Array
    last_update: Array

    def __init__(
            self,
            params: Array,
            lr: float,
            loss_fn: LossFn[Y],
            h: Callable[[Array, Array], [Array]],
            proj_fn: Callable[[Array], Array],
            distr_shift: Callable[[Array], tuple[Array, Y]],
            reg: float = 0,
            base_optimizer: BaseOptimizer = "GD",
            momentum: float = 0,
            rho: float = 0,
    ):
        super().__init__(params, lr, loss_fn, proj_fn)
        self.h = h
        self.reg = reg
        self.distr_shif = distr_shift
        self.current_p_d = params
        self.p_d_history = [params]
        self.rho = rho

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
        def _sam_perturb(grads: Array, params: Array):
            """Apply SAM perturbation to parameters."""

            def perturb(p: Array, g: Array) -> Array:
                g_norm = jnp.linalg.norm(g)
                scale = self.rho / (g_norm + 1e-12)
                return p + scale * g

            return jax.tree_util.tree_map(perturb, params, grads)

        def decoupled_loss(p_m: Array, p_d: Array) -> Array:
            x_0, y_0 = self.distr_shif(p_d)
            return jnp.mean(
                self.loss_fn(p_m, x=x_0, y=y_0)
            ) + self.reg * jnp.sum(jnp.abs(self.h(p_m, x_0) - self.h(p_d, x_0) + 1e-8))

        grad_M = grad(lambda p: decoupled_loss(p, self.current_p_d))(params)
        grad_D = grad(lambda p_p: decoupled_loss(params, p_p))(self.current_p_d)

        if self.rho != 0:
            perturbed_M = _sam_perturb(grad_M, params)
            perturbed_D = _sam_perturb(grad_D, self.current_p_d)
            grad_M = grad(lambda p: decoupled_loss(p, self.current_p_d))(perturbed_M)
            grad_D = grad(lambda p_p: decoupled_loss(params, p_p))(perturbed_D)

        # self.current_params = jax.tree_util.tree_map(lambda p_m, grads_m: self.proj_fn(p_m - self.lr * grads_m) if isinstance(p_m, jnp.ndarray) else p_m, params, grad_M)
        # self.current_p_d = jax.tree_util.tree_map(lambda p_d, grads_d: self.proj_fn(p_d - self.lr * grads_d) if isinstance(p_d, jnp.ndarray) else p_d, self.current_p_d, grad_D)

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

class BarierDPerfGD(Optimizer[Y], Generic[Y]):  # Decoupled Gradient Descent
    grads: Array
    last_update: Array

    def __init__(
            self,
            params: Array,
            lr: float,
            loss_fn: LossFn[Y],
            proj_fn: Callable[[Array], Array],
            distr_shift: Callable[[Array], tuple[Array, Y]],
            reg: float = 0,
            base_optimizer: BaseOptimizer = "GD",
            momentum: float = 0,
            rho: float = 0,
    ):
        super().__init__(params, lr, loss_fn, proj_fn)
        self.reg = reg
        self.distr_shif = distr_shift
        self.current_p_d = params
        self.p_d_history = [params]
        self.rho = rho

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

        def log_barrier_term(p_m, p_d, reg=0.9):
            reg_term = jax.tree_util.tree_map(lambda a, b: jnp.sum((a - b) ** 2), p_m, p_d)

            reg_term = jax.tree_util.tree_reduce(lambda x, y: x + y, reg_term)

            g = reg_term - reg
            if g >= 0:
                return jnp.inf
            return jnp.log(-g)

        def decoupled_loss(p_m: Array, p_d: Array) -> Array:
            x_0, y_0 = self.distr_shif(p_d)
            # return jnp.mean(self.loss_fn(p_m, x=x, y=y)) + self.reg * jnp.linalg.norm(p_m - p_d + 1e-8)**2 #figure out why gradient of 0 is Nan
            # figure out why gradient of 0 is Nan
            return jnp.mean(
                self.loss_fn(p_m, x=x_0, y=y_0)
            )

        grad_M = grad(lambda p: decoupled_loss(p, self.current_p_d) - log_barrier_term(p, self.current_p_d, self.reg) )(
            params)  # - self.reg * grad_barrier_term_M(params, self.current_p_d)
        grad_D = grad(lambda p_p: decoupled_loss(params, p_p) - log_barrier_term(params, p_p, self.reg))(
            self.current_p_d)  # - self.reg * grad_barrier_term_D(params, self.current_p_d)

        # self.current_params = jax.tree_util.tree_map(lambda p_m, grads_m: self.proj_fn(p_m - self.lr * grads_m) if isinstance(p_m, jnp.ndarray) else p_m, params, grad_M)
        # self.current_p_d = jax.tree_util.tree_map(lambda p_d, grads_d: self.proj_fn(p_d - self.lr * grads_d) if isinstance(p_d, jnp.ndarray) else p_d, self.current_p_d, grad_D)

        updates_M, self.opt_state_M = self.optimizer_M.update(
            grad_M, self.opt_state_M, params
        )
        updates_D, self.opt_state_D = self.optimizer_D.update(
            grad_D, self.opt_state_D, self.current_p_d
        )

        min_lr = 1e-8
        lr = self.lr
        while lr > min_lr:
            # Scale updates by current lr
            scaled_updates_M = jax.tree.map(lambda u: lr * u, updates_M)
            scaled_updates_D = jax.tree.map(lambda u: lr * u, updates_D)

            # Apply updates and project
            candidate_params = self.proj_fn(optax.apply_updates(params, scaled_updates_M))
            candidate_p_d = self.proj_fn(optax.apply_updates(self.current_p_d, scaled_updates_D))

            reg_term = jax.tree_util.tree_map(lambda a, b: jnp.sum((a - b) ** 2), candidate_params, candidate_p_d)
            reg_term = jax.tree_util.tree_reduce(lambda x, y: x + y, reg_term)

            if reg_term < self.reg:
                break
            else:
                lr *= 0.1

        self.current_params = cast(Array, candidate_params)
        self.current_p_d = cast(Array, candidate_p_d)

        self.grads = jax.tree_util.tree_map(lambda x, y: x + y, grad_M, grad_D)
        self.params_history.append(self.current_params)
        self.p_d_history.append(self.current_p_d)
        self.i += 1

        return self.current_params
