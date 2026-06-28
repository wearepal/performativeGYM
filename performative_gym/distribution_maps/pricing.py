import jax
import jax.numpy as jnp
from jax import Array

from functools import cached_property

from .distribution_map import DistributionMap

__all__ = ["Pricing"]

class Pricing(DistributionMap):
    """
    Pricing Gaussian distribution map.

    This class implements a performative distribution inspired by the pricing
    model in Izzo et al. (2021), where the deployed parameter \\theta represents
    a price vector and the data corresponds to demand.

    The distribution is defined as

    .. math::

        \\mathcal{D}(\\theta) = \\mathcal{N}(\\mu_0 - \\epsilon \\theta, \\Sigma),

    where :math:`\\mu_0 \\in \\mathbb{R}^d` is a baseline demand vector,
    :math:`\\epsilon > 0` controls the sensitivity of demand to price, and
    :math:`\\Sigma` is a fixed covariance matrix.

    Equivalently, samples are generated via the reparameterization

    .. math::

        z = \\mu_0 - \\epsilon \\theta + \\xi,
        \\qquad
        \\xi \\sim \\mathcal{N}(0, \\Sigma).

    This distribution models a standard economic effect: increasing prices
    (larger \\theta) decreases demand. The parameter \\epsilon controls how
    strongly demand reacts to price changes.

    Parameters
    ----------
    n : int, default=1000
        Number of samples drawn each time :meth:`sample` is called.

    epsilon : float, default=1.5
        Sensitivity parameter controlling how strongly the mean demand depends
        on the price vector \\theta.

    seed : int, default=3
        Random seed used for sampling.

    mu0 : float, default=6
        Baseline level for the demand vector.

    d : int, default=1
        Dimension of the parameter and data space.

    Attributes
    ----------
    mu_0 : Array of shape (d,)
        Baseline demand vector. It is initialized as a constant vector plus a
        small random perturbation for each product.

    cov : Array of shape (d, d)
        Covariance matrix of the Gaussian distribution. In this implementation,
        it is the identity matrix.

    Methods
    -------
    sample(params: Array) -> tuple[Array, None]
        Draw ``n`` samples from :math:`\\mathcal{D}(\\theta)`.

        The returned samples have shape ``(n, d)`` and satisfy

        .. math::

            z_i \\sim \\mathcal{N}(\\mu_0 - \\epsilon \\theta, \\Sigma),
            \\qquad i = 1, \\dots, n.

        No labels are associated with this distribution map, so the second
        element of the returned tuple is ``None``.
    """

    def __init__(
            self,
            n: int = 1000,
            epsilon: float = 1.5,
            seed: int = 3,
            mu0: float = 6,
            d: int = 1,
    ):
        super().__init__(n, epsilon, seed)

        self.mu0 = mu0
        self.d = d

    @cached_property
    def mu_0(self) -> Array:
        return self.mu0 * jnp.ones((self.d)) + jax.random.uniform(
            jax.random.PRNGKey(3), (self.d,)
        )

    @cached_property
    def cov(self) -> Array:
        return jnp.diag(jnp.ones(self.d))

    def sample(
            self,
            params: Array,
    ) -> tuple[Array, None]:

        mean = self.mu_0 - self.epsilon * params

        return jax.random.multivariate_normal(
            jax.random.PRNGKey(self.seed), mean, self.cov, shape=(self.n,)
        ), None
