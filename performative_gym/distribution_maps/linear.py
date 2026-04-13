import jax
import jax.numpy as jnp
from jax import Array

from .distribution_map import DistributionMap

class Linear(DistributionMap):
    """
    Linear Gaussian distribution map.

    This class implements a one-dimensional performative distribution of the form

    .. math::

        \\mathcal{D}(\\theta) = \\mathcal{N}(A_1 \\theta + A_0, \\sigma^2),

    where :math:`\\theta` denotes the deployed model parameter, :math:`A_0` and
    :math:`A_1` control the affine dependence of the mean on :math:`\\theta`, and
    :math:`\\sigma = \\texttt{STD}` is the standard deviation.

    Equivalently, samples are generated through the reparameterization

    .. math::

        z = (A_1 \\theta + A_0) + \\sigma \\xi,
        \\qquad
        \\xi \\sim \\mathcal{N}(0, 1).

    Parameters
    ----------
    n : int, default=10000
        Number of samples drawn each time :meth:`sample` is called.

    epsilon : float, optional
        Unused placeholder parameter inherited from :class:`DistributionMap`.
        Included for interface compatibility.

    seed : int, default=3
        Random seed used to generate the Gaussian noise.

    A0 : float, default=5
        Intercept term :math:`A_0` in the mean function
        :math:`\\mu(\\theta) = A_1 \\theta + A_0`.

    A1 : float, default=1
        Slope term :math:`A_1` controlling how strongly the mean depends on the
        deployed parameter :math:`\\theta`.

    STD : float, default=1
        Standard deviation :math:`\\sigma` of the Gaussian noise.

    Attributes
    ----------
    n : int
        Number of samples generated per call.

    epsilon : float or None
        Placeholder parameter inherited from the base class.

    seed : int
        Random seed used for sampling.

    A0 : float
        Intercept of the affine mean function.

    A1 : float
        Slope of the affine mean function.

    STD : float
        Standard deviation of the Gaussian distribution.

    Methods
    -------
    sample(params: Array) -> tuple[Array, None]
        Draw ``n`` independent samples from :math:`\\mathcal{D}(\\theta)`, where
        ``params`` plays the role of :math:`\\theta`.

        The returned samples have shape ``(n, 1)`` and correspond to

        .. math::

            z_i \\overset{\\text{iid}}{\\sim}
            \\mathcal{N}(A_1 \\theta + A_0, \\sigma^2),
            \\qquad i = 1, \\dots, n.

        No labels are associated with this distribution map, so the second element
        of the returned tuple is ``None``.

    """
    def __init__(
            self,
            n: int = 10000,
            epsilon: float = None,
            seed: int = 3,
            A0: float = 5,
            A1: float = 1,
            STD: float = 1,
    ):

        super().__init__(n, epsilon, seed)

        self.A0, self.A1 = A0, A1
        self.STD = STD


    def sample(self,
               params: Array,
               ) -> tuple[Array, None]:  # MUST return size (n,d), None

        z_0 = jax.random.normal(jax.random.PRNGKey(self.seed), (self.n,))

        z = (self.A1 * params + self.A0) + z_0 * self.STD

        return jnp.expand_dims(z, axis=1), None


class FixedLinear(DistributionMap):
    def __init__(
            self,
            n: int = 10000,
            epsilon: float = None,
            seed: int = 3,
            A0: float = 5,
            A1: float = 1,
            STD: float = 1,
    ):
        super().__init__(n, epsilon, seed)

        self.A0, self.A1 = A0, A1
        self.STD = STD

        self.z_0 = jax.random.normal(jax.random.PRNGKey(self.seed), (self.n,))

    def sample(self,
               params: Array,
               )-> tuple[Array, None]:  # MUST return size (n,d), None
        z = (self.A1 * params + self.A0) + self.z_0 * self.STD

        return jnp.expand_dims(z, axis=1), None
