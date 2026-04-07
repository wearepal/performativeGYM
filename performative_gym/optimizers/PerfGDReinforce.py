from .optimizer import BaseOptimizer, Optimizer, LossFn, Generic, Y

from collections.abc import Callable, Sequence

import optax
import jax.numpy as jnp
import jax

from jax import Array, grad, jacobian

#TODO: implement adagrad, etc.

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
                jnp.array(self.params_history[self.i - self.H: self.i])
                - self.params_history[self.i]
        ).T
        delta_f = (
                jnp.array(self.f_history[self.i - self.H: self.i]) - self.f_history[self.i]
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
