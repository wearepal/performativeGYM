import jax
import jax.numpy as jnp
from jax import Array

from .distribution_map import DistributionMap

class Mixture(DistributionMap):
    """
    Mixture of Gaussians distribution map.

    This class implements a performative distribution of the form

    .. math::

        \\mathcal{D}(\\theta)
        =
        \\gamma \\, \\mathcal{N}(A_1 \\theta + A_0, \\mathrm{STD}_A^2)
        +
        (1-\\gamma) \\, \\mathcal{N}(B_1 \\theta + B_0, \\mathrm{STD}_B^2),

    where :math:`\\theta` denotes the deployed model parameter and
    :math:`\\gamma \\in [0,1]` is the mixture weight.

    The two components represent different subpopulations that respond
    differently to the deployed model. Their means depend affinely on
    :math:`\\theta`:

    Parameters
    ----------
    n : int
        Number of samples drawn each time :meth:`sample` is called.

    epsilon : float, optional
        Unused placeholder parameter inherited from :class:`DistributionMap`.

    seed : int, default=3
        Random seed used to generate samples.

    A0, A1 : float
        Parameters defining the affine mean of the first component:
        :math:`\\mu_A(\\theta) = A_1 \\theta + A_0`.

    STD_A : float
        Standard deviation of the first Gaussian component.

    B0, B1 : float
        Parameters defining the affine mean of the second component:
        :math:`\\mu_B(\\theta) = B_1 \\theta + B_0`.

    STD_B : float
        Standard deviation of the second Gaussian component.

    gamma : float, default=0.5
        Mixture weight :math:`\\gamma` for the first component.

    Methods
    -------
    sample(params: Array) -> tuple[Array, None]
        Draw ``n`` samples from :math:`\\mathcal{D}(\\theta)`.

        Returns an array of shape ``(n, 1)`` containing the samples.
        No labels are associated with this distribution map, so the second
        element of the returned tuple is ``None``.
    """

    def __init__(
            self,
            n: int,
            epsilon: float = None,
            seed: int = 3,
            A0: float = -0.5,
            A1: float = 1,
            STD_A: float = 1,
            B0: float = 1,
            B1: float = -0.3,
            STD_B: float = 0.25,
            gamma: float = 0.5,
    ):
        super().__init__(n, epsilon, seed)

        self.A0, self.A1, self.B0, self.B1 = A0, A1, B0, B1
        self.STD_A, self.STD_B = STD_A, STD_B
        self.gamma = gamma

    def mean_i(self, a0: float, a1: float, params: Array) -> Array:
        return a1 * params + a0

    def sample(
            self,
            params: Array,
    ) -> tuple[Array, None]:

        key = jax.random.PRNGKey(self.seed)
        key_a, key_b, key_mix = jax.random.split(key, 3)

        z_0_A = jax.random.normal(key_a, (self.n,))
        z_0_B = jax.random.normal(key_b, (self.n,))

        z_A = self.mean_i(self.A0, self.A1, params) + z_0_A * self.STD_A
        z_B = self.mean_i(self.B0, self.B1, params) + z_0_B * self.STD_B

        mixture_mask = jax.random.bernoulli(key_mix, p=self.gamma, shape=(self.n,))
        z = jnp.where(mixture_mask, z_A, z_B)

        return jnp.expand_dims(z, axis=1), None
class FixedMixture(DistributionMap):
    def __init__(
            self,
            n: int,
            epsilon: float,
            seed: int,
            A0: float = -0.5,
            A1: float = 1,
            STD_A: float = 1,
            B0: float = 1,
            B1: float = -0.3,
            STD_B: float = 0.25,
            gamma: float = 0.5,
    ):
        super().__init__(n, epsilon, seed)

        self.A0, self.A1, self.B0, self.B1 = A0, A1, B0, B1
        self.STD_A, self.STD_B = STD_A, STD_B
        self.gamma = gamma

        self.key = jax.random.PRNGKey(self.seed)
        self.key_a, self.key_b, self.key_mix = jax.random.split(self.key, 3)

        self.z_o_A = jax.random.normal(self.key_a, (self.n,))
        self.z_o_B = jax.random.normal(self.key_b, (self.n,))

    def mean_i(self, a0: float, a1: float, params: Array) -> Array:
        return a1 * params + a0

    def sample(self,
               params: Array,
               ):

        z_A = self.mean_i(self.A0, self.A1, params) + self.z_o_A * self.STD_A
        z_B = self.mean_i(self.B0, self.B1, params) + self.z_o_B * self.STD_B

        mixture_mask = jax.random.bernoulli(self.key_mix, p=self.gamma, shape=(self.n,))
        z = jnp.where(mixture_mask, z_A, z_B)

        return jnp.expand_dims(self.sigma * z_A + (1 - self.sigma) * z_B, axis=1), None

