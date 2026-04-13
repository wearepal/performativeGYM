from typing import cast

from .optimizer import BaseOptimizer, Optimizer, LossFn, Generic, Y

from collections.abc import Callable

import optax
import jax.numpy as jnp

from jax import Array, grad


class PerfGDReparam(Optimizer[Y], Generic[Y]):
    """
    Implementation of the algorithm introduced in Cyffers et al. (2025).

    This method follows the Performative Gradient Descent (PerfGD) framework, but
    replaces the REINFORCE estimator with the reparameterization trick to estimate
    the performative gradient.

    The key idea is to express the performative distribution as a transformation of
    a parameter-independent base distribution. Specifically, samples from the
    performative distribution are generated as:

        z = \\varphi(z_0, \\theta),   \\quad z_0 \\sim \\mathcal{D}_0

    where \\mathcal{D}_0 is a fixed base distribution (capturing the data before
    performativity), and \\varphi encodes how the model parameters \\theta transform
    the data distribution (push-forward map).

    Under this formulation, the performative risk can be written as:

        \\mathcal{PR}(\\theta)
        =
        \\mathbb{E}_{z_0 \\sim \\mathcal{D}_0}
        \\big[
            \\ell(\\theta; \\varphi(z_0, \\theta))
        \\big]

    Since the expectation no longer depends on \\theta through the distribution, the
    gradient can be computed using the multivariate chain rule:

    \\begin{align}
    \\nabla_\\theta \\mathcal{PR}(\\theta)
    &= \\mathbb{E}_{z_0 \\sim \\mathcal{D}_0}
    \\Big[
        \\nabla_\\theta \\ell(\\theta; \\varphi(z_0, \\theta))
    \\Big] \\\\
    &= \\mathbb{E}_{z_0 \\sim \\mathcal{D}_0}
    \\left[
        \\left.\\frac{\\partial \\ell(\\theta; z)}{\\partial z}\\right|_{z=\\varphi(z_0, \\theta)}
        \\frac{\\partial \\varphi(z_0; \\theta)}{\\partial \\theta}
        +
        \\left.\\frac{\\partial \\ell(\\theta; z)}{\\partial \\theta}\\right|_{z=\\varphi(z_0, \\theta)}
    \\right].
    \\end{align}

    This provides a lower-variance gradient estimator compared to REINFORCE, as it
    avoids score-function estimation and instead relies on direct differentiation
    through the transformation \\varphi.

    The final update combines both terms through a base optimizer such as GD,
    Adam, or Adagrad.

    Parameters
    ----------
    params : Array
        Initial model parameters :math:`\\theta`.

    lr : float
        Learning rate used by the base optimizer.

    loss_fn : LossFn[Y]
        Pointwise loss function :math:`\\ell(z; \\theta)`.

    proj_fn : Callable[[Array], Array]
        Projection operator applied after each update. This is typically used to
        enforce that the iterates remain in a feasible parameter set, e.g. a
        closed convex domain.

    distr_map : Callable[[Array], tuple[Array, Y]]
        Distribution map defining the performative data-generating process.
        Given parameters :math:`\\theta`, it returns a batch of samples
        :math:`(x, y) \\sim D(\\theta)`. This corresponds to sampling from the
        push-forward distribution induced by the current model parameters.

    base_optimizer : BaseOptimizer, default="GD"
        Base optimization method used to apply the estimated performative
        gradient. Supported options are:

        - ``"GD"``: gradient descent
        - ``"adam"``: Adam
        - ``"adamw"``: AdamW
        - ``"adagrad"``: Adagrad

    momentum : float, default=0
        Momentum parameter used when ``base_optimizer="GD"``.
        Ignored by the other optimizers.

    Attributes
    ----------
    current_params : Array
        Parameters at the current iteration.

    params_history : list[Array]
        History of parameter values across iterations.

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
            distr_map: Callable[[Array], tuple[Array, Y]],
            base_optimizer: BaseOptimizer = "GD",
            momentum: float = 0,
    ):
        super().__init__(params, lr, loss_fn, proj_fn)

        self.distr_map = distr_map

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
        def performative_risk(params: Array) -> Array:
            x, y = self.distr_map(params)
            return jnp.mean(self.loss_fn(params, x, y))

        self.grads = grad(performative_risk)(params)

        updates, self.opt_state = self.optimizer.update(
            self.grads, self.opt_state, params
        )
        current_params = optax.apply_updates(params, updates)
        self.current_params = cast(Array, current_params)

        self.params_history.append(self.current_params)
        self.i += 1

        return self.current_params
