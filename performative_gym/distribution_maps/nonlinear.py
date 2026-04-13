import jax
import jax.numpy as jnp
from jax import Array

from .distribution_map import DistributionMap


class NonLinear(DistributionMap):
    """
    Nonlinear Gaussian distribution map.

    This class implements a one-dimensional performative distribution of the form

    .. math::

        \\mathcal{D}(\\theta) = \\mathcal{N}(\\sqrt{A_1 \\theta + A_0}, \\sigma^2),

    where :math:`\\theta` denotes the deployed model parameter, and the mean
    depends nonlinearly on :math:`\\theta` through a square-root transformation.

    Equivalently, samples are generated via the reparameterization

    .. math::

        z = \\sqrt{A_1 \\theta + A_0} + \\sigma \\xi,
        \\qquad
        \\xi \\sim \\mathcal{N}(0,1).

    Parameters
    ----------
    n : int, default=10000
        Number of samples drawn each time :meth:`sample` is called.

    epsilon : float, optional
        Unused placeholder parameter inherited from :class:`DistributionMap`.

    seed : int, default=3
        Random seed used to generate the Gaussian noise.

    A0 : float, default=1
        Offset term ensuring positivity of the argument inside the square root.

    A1 : float, default=1
        Scaling term controlling how strongly the mean depends on \\theta.

    STD : float, default=1
        Standard deviation :math:`\\sigma` of the Gaussian noise.

    Methods
    -------
    sample(params: Array) -> tuple[Array, None]
        Draw ``n`` samples from :math:`\\mathcal{D}(\\theta)`.

        The returned samples have shape ``(n, 1)`` and satisfy

        .. math::

            z_i \\sim \\mathcal{N}(\\sqrt{A_1 \\theta + A_0}, \\sigma^2),
            \\qquad i = 1, \\dots, n.

        No labels are associated with this distribution map, so the second
        element of the returned tuple is ``None``.
    """

    def __init__(
            self,
            n: int = 10000,
            epsilon: float = None,
            seed: int = 3,
            A0: float = 1,
            A1: float = 1,
            STD: float = 1,
    ):
        super().__init__(n, epsilon, seed)

        self.A0, self.A1 = A0, A1
        self.STD = STD

    def sample(
            self,
            params: Array,
    ) -> tuple[Array, None]:

        z_0 = jax.random.normal(jax.random.PRNGKey(self.seed), (self.n,))

        mean = jnp.sqrt(self.A1 * params + self.A0)

        z = mean + z_0 * self.STD

        return jnp.expand_dims(z, axis=1), None

class FixedNonLinear(DistributionMap):
    def __init__(
            self,
            n: int = 10000,
            epsilon: float = None,
            seed: int = 3,
            A0: float = 1,
            A1: float = 1,
            STD: float = 1,
    ):
        super().__init__(n, epsilon, seed)

        self.A0, self.A1 = A0, A1
        self.STD = STD

        self.z_0 = jax.random.normal(jax.random.PRNGKey(self.seed), (self.n,))

    def sample(self,
               params: Array,
               ):

        mean = jnp.sqrt(self.A1 * params + self.A0)

        z = mean + self.z_0 * self.STD

        return jnp.expand_dims(z, axis=1), None
