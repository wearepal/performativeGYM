from .optimizer import Optimizer, LossFn, Generic, Y

from collections.abc import Callable

import jax.numpy as jnp
import jax

from jax import Array

__all__ = ["DFO"]


class DFO(Optimizer[Y], Generic[Y]):
    """
    Implementation of the Derivative-Free Optimization (DFO) algorithm introduced in
    Flaxman et al. (2004).

    This method optimizes the performative risk $PR(\theta)$, thus is supposed to converge to the optimal point.
    It relies on a stochastic approximation of the gradient obtained by smoothing the objective function.

    Given a smoothing parameter $\delta > 0$, define the smoothed objective:
    \[
    \hat{PR}(\theta) = \mathbb{E}_{u}[PR(\theta + \delta u)],
    \]
    where $u$ is a random perturbation vector, sampled uniformly from the unit sphere.

    The gradient of the smoothed objective satisfies:
    \[
    \nabla \hat{PR}(\theta) = \frac{1}{\delta} \mathbb{E}_{u}[PR(\theta + \delta u)\, u].
    \]

    In practice, the expectation is approximated using $S$ independent samples
    $u_1, \dots, u_S$, yielding the estimator:
    \[
    \widehat{\nabla} PR(\theta)
    =
    \frac{1}{S} \sum_{i=1}^S PR(\theta + \delta u_i)\, u_i.
    \]

    This estimator is unbiased for the gradient of the smoothed objective $\hat{PR}$
    and enables gradient-based updates without requiring explicit differentiation of $PR$.

    Parameters
    ----------
    params : Array
        Initial model parameters $\theta \in \mathbb{R}^d$.

    lr : float
        Learning rate $\eta$ used in the parameter update step.

    loss_fn : LossFn[Y]
        Loss function used to compute the performative risk. It must accept
        parameters and data samples, and return a scalar loss value.

    proj_fn : Callable[[Array], Array]
        Projection operator applied to the parameters after each update.
        This can be used to enforce constraints (e.g., bounded domain or
        normalization).

    distr_map : Callable
        Function defining the distribution map. Given the current
        parameters, it returns samples from the induced distribution
        $D(\theta)$.

    seed : int
        Random seed used to initialize the pseudo-random number generator.

    samples : int, default=10
        Number of perturbation directions used to estimate the gradient
        (Monte Carlo sample size).

    delta : float, default=0.1
        Smoothing parameter $\delta > 0$ controlling the magnitude of the
        perturbations applied to the parameters.

    Methods
    -------
    step(params: Array, x: Array, y: Y) -> Array
        Perform one optimization step using the batch ``(x, y)`` sampled from the
        current performative distribution, update the internal optimizer state, and
        return the new parameters.

    """
    def __init__(
            self,
            params: Array,
            lr: float,
            loss_fn: LossFn[Y],
            proj_fn: Callable[[Array], Array],
            distr_map: Callable,
            seed: int,
            samples: int = 10,
            delta: float = 0.1,
    ):
        super().__init__(params, lr, loss_fn, proj_fn)
        self.distr_map = distr_map
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
            x, y = self.distr_map(p_p)
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
