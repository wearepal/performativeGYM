from .optimizer import BaseOptimizer, Optimizer, LossFn, Generic, Y

from collections.abc import Callable

import optax
import jax.numpy as jnp

from jax import Array, grad

class RGD(Optimizer[Y], Generic[Y]):
    """
    Implementation of Repeated Gradient Descent (RGD), as introduced
    by [Perdomo et al., 2020](https://arxiv.org/abs/2002.06673).

    At each iteration, RGD treats the current deployed data distribution as fixed
    and performs a gradient-based update of the model parameters using the
    empirical risk on data sampled from that distribution. In other words, if the
    current parameters are :math:`\\theta_t` and the observed data are sampled from
    :math:`D(\\theta_t)`, then RGD updates the model using only the gradient of the
    loss with respect to the model parameters, ignoring the dependence of the data
    distribution on :math:`\\theta_t`.

    Given a batch :math:`(x, y) \\sim D(\\theta_t)`, the update is based on the
    gradient

    .. math::

        \\nabla_1 L(\\theta_t)
        =
        \\mathbb{E}_{D(\\theta_t)}
        \\left[
            \\nabla_\\theta \\ell(\\theta_t; x, y)
        \\right],

    which is approximated in practice by the empirical batch average:

    .. math::

        \\widehat{\\nabla}_1 L(\\theta_t)
        =
        \\nabla_\\theta
        \\left(
            \\frac{1}{n} \\sum_{i=1}^n \\ell(\\theta_t; x_i, y_i)
        \\right).

    The parameters are then updated with the chosen base optimizer and projected
    back onto the feasible set if a projection operator is provided.

    Unlike performative gradient methods such as PerfGD, RGD does not account for
    the indirect effect of the model parameters on the data distribution. As a
    result, it can only converge to a performatively stable point rather than a
    performatively optimal one.

    Parameters
    ----------
    params : Array
        Initial model parameters :math:`\\theta`.

    lr : float
        Learning rate used by the base optimizer.

    loss_fn : LossFn[Y]
        Pointwise loss function. Given parameters and a batch of inputs and labels,
        it returns the corresponding losses used to form the empirical objective.

    proj_fn : Callable[[Array], Array], default=lambda params: params
        Projection operator applied after each update. This can be used to enforce
        constraints on the parameter space, such as box constraints or
        normalization. By default, no projection is applied.

    base_optimizer : BaseOptimizer, default="GD"
        Base optimization method used to apply the gradient update. Supported
        options are:

        - ``"GD"``: gradient descent
        - ``"adam"``: Adam
        - ``"adamw"``: AdamW
        - ``"adagrad"``: Adagrad

    momentum : float, default=0
        Momentum parameter used when ``base_optimizer="GD"``. Ignored by the other
        optimizers.

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
        Most recent gradient of the empirical risk with respect to the model
        parameters.

    last_update : Array
        Most recent parameter update applied by the optimizer, if tracked by the
        implementation.

    Methods
    -------
    step(params: Array, x: Array, y: Y) -> Array
        Perform one optimization step using the batch ``(x, y)`` sampled from the
        current performative distribution, update the internal optimizer state, and
        return the new parameters.
    """

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

        self.grads = grad(lambda p: jnp.mean(self.loss_fn(p, x, y)))(
            self.current_params
        )

        updates, self.opt_state = self.optimizer.update(
            self.grads, self.opt_state, params
        )
        self.current_params = self.proj_fn(optax.apply_updates(params, updates))

        self.params_history.append(self.current_params)
        self.i += 1
        return self.current_params
