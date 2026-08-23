from .optimizer import BaseOptimizer, Optimizer, LossFn, Generic, Y

from collections.abc import Callable
from typing import cast

import jax.numpy as jnp
import jax
import optax

from jax import Array, grad, jacobian

__all__ = ["PerfGDReinforce"]


class PerfGDReinforce(Optimizer[Y], Generic[Y]):
    """
    Implementation of the Performative Gradient Descent (PerfGD) algorithm
    introduced in Izzo et al. (2021).

    PerfGD is designed for settings with performative distribution shift, where
    deploying model parameters ``theta`` changes the data distribution
    ``D(theta)``. Unlike repeated gradient descent, which only accounts for the
    direct effect of the parameters on the loss, PerfGD estimates the full
    performative gradient by combining:

    - the standard gradient term
      :math:`\\nabla_1 L(\\theta) = \\mathbb{E}_{D(\\theta)}[\\nabla \\ell(z; \\theta)]`,
    - and a second term :math:`\\nabla_2 L(\\theta)` capturing how the induced
      distribution changes with ``theta``.

    In the parametric setting considered in the paper, the distribution map is
    assumed to admit a known density of the form :math:`p(z; f(\\theta))`,
    where ``f(theta)`` is an unknown quantity that can be estimated from data.
    PerfGD then:

    1. estimates :math:`f(\\theta_t)` from the current batch,
    2. approximates :math:`\\mathrm{d}f / \\mathrm{d}\\theta` using finite
       differences over the previous ``H`` iterates,
    3. uses a REINFORCE-style estimator for the distribution-shift term
       :math:`\\nabla_2 L`.

    This variant corresponds to the REINFORCE estimator described in the paper:

    .. math::

        \\nabla_2 L(\\theta)
        =
        \\mathbb{E}_{D(\\theta)}
        \\left[
            \\ell(z; \\theta)
            \\left(\\frac{\\mathrm{d}f}{\\mathrm{d}\\theta}\\right)^\\top
            \\partial_2 \\log p(z; f(\\theta))
        \\right].

    The final update combines both terms through a base optimizer such as GD,
    Adam, or Adagrad.

    Parameters
    ----------
    params : Array
        Initial model parameters :math:`\\theta`.

    lr : float
        Learning rate used by the base optimizer.

    loss_fn : LossFn[Y]
        Pointwise loss function :math:`\\ell(z; \\theta)`. This is used both to
        compute the empirical estimate of :math:`\\nabla_1 L` and inside the
        REINFORCE estimator for :math:`\\nabla_2 L`.

    proj_fn : Callable[[Array], Array]
        Projection operator applied after each update. This is typically used to
        enforce that the iterates remain in a feasible parameter set, e.g. a
        closed convex domain.

    f_fn : LossFn[Y]
        Estimator for the distribution parameter :math:`f(\\theta)` from a batch
        of samples. In the notation of the paper, this plays the role of
        :math:`\\hat{f}`. For example, in Gaussian models this may be the sample
        mean or another sufficient-statistic estimator.

    H : int
        Gradient-estimation horizon used to approximate
        :math:`\\mathrm{d}f / \\mathrm{d}\\theta` via finite differences from the
        previous ``H`` iterates. In the paper, larger values of ``H`` help form
        an overdetermined system and reduce sensitivity to estimation noise.

    prob_distr : Callable[[Array, Y, Array, Array], Array]
        Function implementing the score term associated with the parametric
        distribution model, i.e. the derivative of the log-density with respect
        to the distribution parameter. It is used inside the REINFORCE-style
        estimator of :math:`\\nabla_2 L`.

    base_optimizer : BaseOptimizer, default="GD"
        Base optimization method used to apply the estimated performative
        gradient. Supported options are:

        - ``"GD"``: gradient descent,
        - ``"adam"``: Adam,
        - ``"adamw"``: AdamW,
        - ``"adagrad"``: Adagrad.

    momentum : float, default=0
        Momentum parameter used when ``base_optimizer="GD"``. Ignored by the
        other optimizers.

    Attributes
    ----------
    f_history : list[Array]
        History of estimated values of :math:`f(\\theta_t)`, used to construct
        the finite-difference approximation of
        :math:`\\mathrm{d}f / \\mathrm{d}\\theta`.

    optimizer
        Optax optimizer instance used to apply parameter updates.

    opt_state
        Internal state of the Optax optimizer.

    grads : Array
        Most recent estimate of the performative gradient.

    Methods
    -------
    step(params: Array, x: Array, y: Y) -> Array
        Perform one optimization step using the batch ``(x, y)`` sampled from the
        current performative distribution, update the internal optimizer state, and
        return the new parameters.

    """
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
            base_optimizer: BaseOptimizer = "GD",
            momentum: float = 0,
    ):
        super().__init__(params, lr, loss_fn, proj_fn)
        self.f_fn = f_fn
        self.H = H
        self.prob_distr = prob_distr
        self.f_history: list[Array] = []

        if base_optimizer == "GD":
            self.optimizer = optax.sgd(learning_rate=lr, momentum=momentum)
        elif base_optimizer == "adam":
            self.optimizer = optax.adam(learning_rate=lr)
        elif base_optimizer == "adamw":
            self.optimizer = optax.adamw(learning_rate=lr)
        elif base_optimizer == "adagrad":
            self.optimizer = optax.adagrad(learning_rate=lr)
        self.opt_state = self.optimizer.init(params)


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
        # `delta_f_theta` is the Jacobian df/dtheta of shape (dim(f), dim(theta)),
        # so it has to be transposed to contract over f and leave theta free.
        perf_gradients = delta_f_theta.T @ jnp.mean(jacobians * loss_ft, axis=0)

        return perf_gradients

    def step(self, params: Array, x: Array, y: Y) -> Array:
        self.f_history.append(self.f_fn(params, x, y))

        if self.i < self.H: #Warm-up phase
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

        updates, self.opt_state = self.optimizer.update(
            self.grads, self.opt_state, params
        )
        current_params = optax.apply_updates(params, updates)
        self.current_params = cast(Array, current_params)

        self.params_history.append(self.current_params)
        self.i += 1
        return self.current_params
