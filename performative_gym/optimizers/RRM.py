from .optimizer import BaseOptimizer, Optimizer, LossFn, Generic, Y

from collections.abc import Callable, Sequence

import optax
import jax.numpy as jnp
import jax

from jax import Array, grad

class RRM(Optimizer[Y], Generic[Y]):
    """
    Implementation of Repeated Risk Minimization (RRM), as introduced in the
    performative prediction literature by [Perdomo et al., 2020](https://arxiv.org/abs/2002.06673).

    At each iteration, RRM treats the current deployed data distribution as fixed
    and approximately minimizes the empirical risk on data sampled from that
    distribution. If the current parameters are :math:`\\theta_t` and the observed
    data are sampled from :math:`D(\\theta_t)`, then RRM computes the next iterate
    by solving, up to numerical tolerance, the optimization problem

    .. math::

        \\theta_{t+1}
        \\approx
        \\arg\\min_{\\theta} \\, \\frac{1}{n} \\sum_{i=1}^n \\ell(\\theta; x_i, y_i),

    where :math:`(x_i, y_i)_{i=1}^n` is the batch sampled from the current
    performative distribution.

    In this implementation, the inner minimization is carried out by repeated
    gradient steps on the empirical risk until the difference between two
    successive parameter values falls below a prescribed tolerance. The final
    iterate is then taken as the output of the RRM update.

    Unlike performative gradient methods such as PerfGD, RRM does not account for
    how changing the model parameters modifies the future data distribution. As a
    result, it is designed to converge to a performatively stable point rather than
    a performatively optimal one.

    Parameters
    ----------
    params : Array
        Initial model parameters :math:`\\theta`.

    lr : float
        Step size used in the inner gradient-based minimization of the empirical
        risk.

    loss_fn : LossFn[Y]
        Pointwise loss function. Given parameters and a batch of inputs and labels,
        it returns the corresponding losses used to form the empirical objective.

    proj_fn : Callable[[Array], Array]
        Projection operator applied after each inner update. This can be used to
        enforce constraints on the parameter space, such as box constraints or
        normalization.

    tol : float
        Tolerance used to terminate the inner optimization loop. The inner loop
        stops when the norm of the difference between successive parameter values
        is smaller than ``tol``.

    Attributes
    ----------
    current_params : Array
        Parameters at the current iteration.

    params_history : list[Array]
        History of parameter values across outer iterations.

    grads : Array
        Mean of the gradients computed during the inner optimization loop of the
        most recent RRM step.

    tol : float
        Convergence tolerance for the inner minimization procedure.

    Methods
    -------
    step(params: Array, x: Array, y: Y) -> Array
        Perform one RRM update by approximately minimizing the empirical risk on
        the batch ``(x, y)`` until convergence of the inner loop, and return the
        resulting parameters.

    Notes
    -----
    For each outer iteration, RRM solves a static empirical risk minimization
    problem based on data sampled from the current performative distribution. This
    mirrors the original RRM procedure, where the model is repeatedly retrained to
    optimality on the data induced by the current deployment before being deployed
    again.
    """
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

    def _compute_mean(self, params_list: Sequence[Array]):
        # Use tree_map to compute the mean across all corresponding elements
        return jax.tree_util.tree_map(
            lambda *arrays: jnp.mean(jnp.stack(arrays), axis=0)
            if all(isinstance(a, jnp.ndarray) for a in arrays)
            else arrays[0],
            *params_list,
        )

    def _compute_diff(self, params1: Array, params2: Array):

        diff = jax.tree_util.tree_map(
            lambda x, y: jnp.linalg.norm(x - y)
            if isinstance(x, jnp.ndarray)
            else x,
            params1,
            params2,
        )

        total_diff = sum(
            jnp.sum(leaf)
            for leaf in jax.tree_util.tree_leaves(diff)
            if isinstance(leaf, jnp.ndarray)
        )
        return total_diff

    def step(self, params: Array, x: Array, y: Y) -> Array:

        total_diff = jnp.finfo(
            jnp.float64
        ).max  # initial value for grads so it enters in while loop

        history_grads = []
        j = 0

        while total_diff > self.tol: #Updates until convergence (difference is small)

            grads = grad(lambda p: jnp.mean(self.loss_fn(p, x, y)))(
                self.current_params
            )

            params_new = jax.tree_util.tree_map(
                lambda x, y: self.proj_fn(x - self.lr * y)
                if isinstance(x, jnp.ndarray)
                else x,
                params,
                grads,
            )

            total_diff = self._compute_diff(params_new, params)

            params = params_new
            j += 1
            history_grads.append(grads)

        self.current_params = params
        self.params_history.append(self.current_params)
        self.grads = self._compute_mean(history_grads)
        self.i += 1
        return self.current_params
