from abc import abstractmethod
from collections.abc import Callable, Sequence
from typing import Generic, Literal, Protocol, TypeAlias, TypeVar, cast

from jax import Array, grad, jacobian
import jax.numpy as jnp
import jax

import optax

__all__ = [
    "DFO",
    "DPerfGD",
    "Objective",
    "Optimizer",
    "Optimizers",
    "PerfGDReinforce",
    "PerfGDReparam",
    "RGD",
    "RRM",
    "RegRRM",
]

Optimizers: TypeAlias = Literal[
    "RGD",
    "PerfGDReparam",
    "DPerfGD",
    "RRM",
    "PerfGDReinforce",
    "RegRRM",
    "TwoStage",
    "DFO",
]
Objective: TypeAlias = Literal["stability", "optimality"]
Y = TypeVar("Y", contravariant=True, bound=Array | None)


class LossFn(Protocol[Y]):
    def __call__(self, params: Array, x: Array, y: Y) -> Array: ...


class Optimizer(Generic[Y]):
    def __init__(
        self,
        params: Array,
        lr: float,
        loss_fn: LossFn[Y],
        proj_fn: Callable[[Array], Array] = (lambda params: params),
    ):
        self.current_params = params
        self.lr = lr
        self.loss_fn = loss_fn
        self.proj_fn = proj_fn
        self.params_history = [params]

        self.i = 0

    @abstractmethod
    def step(self, params: Array, x: Array, y: Y) -> Array:
        pass


class RGD(Optimizer[Y], Generic[Y]):
    grads: Array
    last_update: Array

    def __init__(
        self,
        params: Array,
        lr: float,
        loss_fn: LossFn[Y],
        proj_fn: Callable[[Array], Array] = (lambda params: params),
        base_optimizer: str = "GD",
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
            # diff = jnp.linalg.norm(params_new - params)
            params = params_new
            j += 1
            history_grads.append(grads)

        self.current_params = params
        self.params_history.append(self.current_params)
        self.grads = self.compute_mean(history_grads)
        self.i += 1
        return self.current_params


class RegRRM(Optimizer[Y], Generic[Y]):
    grads: Array

    def __init__(
        self,
        params: Array,
        lr: float,
        loss_fn: LossFn[Y],
        proj_fn: Callable[[Array], Array],
        tol: float,
        reg: float,
    ):
        super().__init__(params, lr, loss_fn, proj_fn)
        self.tol = tol
        self.reg = reg

    def compute_mean(self, params_list: list[Array]):
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
        while total_diff > self.tol:
            grads = grad(
                lambda params: jnp.mean(self.loss_fn(params, x, y))
                + self.reg
                * jnp.linalg.norm(params - self.params_history[-1] + 1e-8) ** 2
            )(self.current_params)

            params_new = jax.tree_util.tree_map(
                lambda x, y: self.proj_fn(x - self.lr * y)
                if isinstance(x, jnp.ndarray)
                else x,
                params,
                grads,
            )

            # diff = jnp.linalg.norm(params_new - params)
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

            history_grads.append(grads)

        self.current_params = params
        self.grads = self.compute_mean(history_grads)
        self.params_history.append(self.current_params)
        self.i += 1
        return self.current_params


class PerfGDReinforce(Optimizer[Y], Generic[Y]):
    grads: Array

    def __init__(
        self,
        params: Array,
        lr: float,
        loss_fn: LossFn[Y],
        proj_fn: Callable[[Array], Array],
        f_fn: LossFn[Y],
        H: int,
        prob_distr: Callable[[Array, Y, Array, Array], Array],
    ):
        super().__init__(params, lr, loss_fn, proj_fn)
        self.f_fn = f_fn
        self.H = H
        self.prob_distr = prob_distr
        self.f_history: list[Array] = []

    def delta_f_theta(self):
        # Estimating the second part of the performative gradient
        delta_theta = (
            jnp.array(self.params_history[self.i - self.H : self.i])
            - self.params_history[self.i]
        ).T
        delta_f = (
            jnp.array(self.f_history[self.i - self.H : self.i]) - self.f_history[self.i]
        ).T
        delta_f_theta = delta_f @ jnp.linalg.pinv(delta_theta)
        return delta_f_theta

    def _grad2(self, params: Array, x: Array, y: Y) -> Array:
        loss_ft = self.loss_fn(params, x, y)
        delta_f_theta = self.delta_f_theta()
        jacobians = jacobian(
            lambda mean: jnp.squeeze(self.prob_distr(x, y, mean, params))
        )(self.f_fn(params, x, y))

        # for pricing and binary classification
        perf_gradients = delta_f_theta @ jnp.mean(jacobians * loss_ft, axis=0)

        return perf_gradients

    def step(self, params: Array, x: Array, y: Y) -> Array:
        self.f_history.append(self.f_fn(params, x, y))

        if self.i < self.H:
            grads = grad(lambda params: jnp.mean(self.loss_fn(params, x, y)))(
                self.current_params
            )
        else:
            grad1 = grad(lambda params: jnp.mean(self.loss_fn(params, x, y)))(
                self.current_params
            )

            grad2 = self._grad2(params, x, y)
            grads = jax.tree_util.tree_map(
                lambda g1, g2: g1 + g2
                if isinstance(g1, jnp.ndarray) and isinstance(g2, jnp.ndarray)
                else g1,
                grad1,
                grad2,
            )
        self.grads = grads
        # self.current_params = jnp.squeeze(self.proj_fn(params - self.lr * grads))

        self.current_params = jax.tree_util.tree_map(
            lambda x, y: self.proj_fn(x - self.lr * y)
            if isinstance(x, jnp.ndarray)
            else x,
            params,
            grads,
        )

        self.params_history.append(self.current_params)
        self.i += 1
        return self.current_params


class PerfGDReparam(Optimizer[Y], Generic[Y]):  # Especial Gradient Descent
    grads: Array

    def __init__(
        self,
        params: Array,
        lr: float,
        loss_fn: LossFn[Y],
        proj_fn: Callable[[Array], Array],
        distr_shift: Callable[[Array], tuple[Array, Y]],
        base_optimizer: str = "GD",
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
        base_optimizer: str = "GD",
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
            return jnp.mean(
                self.loss_fn(p_m, x=x_0, y=y_0)
            )  # + self.reg * jnp.sum(jnp.abs(p_m - p_d + 1e-8))

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
        current_params = optax.apply_updates(params, updates_M)
        self.current_params = cast(Array, current_params)

        updates_D, self.opt_state_D = self.optimizer_D.update(
            grad_D, self.opt_state_D, self.current_p_d
        )
        current_p_d = optax.apply_updates(self.current_p_d, updates_D)
        self.current_p_d = cast(Array, current_p_d)

        self.grads = grad_M + grad_D
        self.params_history.append(self.current_params)
        self.p_d_history.append(self.current_p_d)
        self.i += 1

        return self.current_params


class DFO(Optimizer[Y], Generic[Y]):
    def __init__(
        self,
        params: Array,
        lr: float,
        loss_fn: LossFn[Y],
        proj_fn: Callable[[Array], Array],
        shift_data_distribution: Callable,
        seed: int,
        samples: int = 10,
        delta: float = 0.1,
    ):
        super().__init__(params, lr, loss_fn, proj_fn)
        self.distr_shift = shift_data_distribution
        self.delta = delta
        self.samples = samples
        self.seed = seed

    def step(self, params: Array, x: Array, y: Y) -> Array:
        def sample_unit_sphere(
            dim: tuple[int, ...], num_samples: int, seed: int
        ) -> Array:
            """
            Generate samples uniformly on the unit sphere S^{d-1}.
            """
            samples = jax.random.normal(
                jax.random.PRNGKey(seed), shape=(num_samples, *dim)
            )
            samples /= jnp.linalg.norm(
                samples, axis=tuple(range(1, len(samples.shape))), keepdims=True
            )
            return samples

        def decoupled_loss(p_p: Array, p: Array) -> Array:
            x, y = self.distr_shift(p_p)
            return jnp.mean(self.loss_fn(p, x, y))

        def performative_risk(params: Array):
            return decoupled_loss(params, params)

        u_samples = jax.tree_util.tree_map(
            lambda params: sample_unit_sphere(params.shape, self.samples, self.seed),
            params,
        )

        perturbed_params = jax.tree_util.tree_map(
            lambda u_samples, params: params + self.delta * u_samples, u_samples, params
        )
        risks = jax.vmap(performative_risk)(perturbed_params)

        grads = jax.tree_util.tree_map(
            lambda u_samples: jnp.mean(
                risks.reshape((self.samples,) + (1,) * (u_samples.ndim - 1))
                * u_samples,
                axis=0,
            ),
            u_samples,
        )

        # Update parameters using the computed gradients
        updated_params = jax.tree_util.tree_map(
            lambda p, g: self.proj_fn(p - self.lr * g)
            if isinstance(p, jnp.ndarray)
            else p,
            params,
            grads,
        )

        # Update history and iteration count
        self.params_history.append(updated_params)
        self.i += 1

        return updated_params
