from typing import Callable

from jax import Array, grad
import jax.numpy as jnp
import jax


from .distribution_map import DistributionMap
from .datasets import CreditDataset

__all__ = ["StrategicClassification"]

class StrategicClassification(DistributionMap):
    """
    Strategic classification distribution map.

    This class implements a performative distribution map motivated by the
    strategic classification framework of Perdomo et al. (2020). In this
    setting, individuals modify their features in response to the deployed
    classifier in order to improve their predicted outcome.

    Let :math:`x_0` denote the original (pre-strategic) feature vector and let
    :math:`h_\\theta(x)` be a score function induced by model parameters
    :math:`\\theta`. The strategic response is modeled as a first-order feature
    update of the form

    .. math::

        x = x_0 - \\epsilon \\, \\nabla_x h_\\theta(x_0),

    while the label remains unchanged:

    .. math::

        y = y_0.

    Therefore, the induced performative distribution is obtained by pushing the
    original dataset through the map

    .. math::

        \\varphi_\\theta(x_0, y_0)
        =
        \\bigl(x_0 - \\epsilon \\, \\nabla_x h_\\theta(x_0),\\; y_0\\bigr).

    The parameter :math:`\\epsilon > 0` controls the strength of the strategic
    response. Larger values of :math:`\\epsilon` correspond to stronger feature
    manipulation in the direction that most improves the score function.

    In this implementation, the base dataset is fixed and loaded once during
    initialization. Calling :meth:`sample` returns the strategically modified
    features together with the original labels.

    Parameters
    ----------
    n : int
        Number of data points retained from the base dataset.

    epsilon : float
        Magnitude of the strategic response. This scales the feature update
        :math:`\\nabla_x h_\\theta(x)`.

    seed : int
        Random seed used when initializing the underlying dataset.

    dataset_name : str
        Name of the dataset used as the pre-strategic population. Currently,
        only ``"giveMeSomeCredit"`` is implemented.

    h : Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
        Score function :math:`h_\\theta(x)` defining the strategic response.
        It must accept model parameters and a batch of features, and return a
        scalar score for each input. The gradient of this score with respect to
        the features determines the direction of strategic manipulation.

    Attributes
    ----------
    dataset_name : str
        Name of the underlying dataset.

    h : Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
        Score function used to define the strategic response.

    datafile : str
        Name of the data file used to load the dataset.

    dataset : CreditDataset
        Dataset object containing the original, pre-strategic data.

    x_0 : Array
        Original feature vectors before strategic modification.

    y_0 : Array
        Original labels. These remain unchanged under the strategic response.

    Methods
    -------
    sample(params: Array) -> tuple[Array, Array]
        Return the strategically modified features and original labels.

        For each original feature vector :math:`x_{0,i}`, the returned feature
        vector is

        .. math::

            x_i
            =
            x_{0,i} - \\epsilon \\, \\nabla_x h_\\theta(x_{0,i}),

        and the labels satisfy

        .. math::

            y_i = y_{0,i}.

        The returned arrays therefore represent a sample from the performative
        distribution induced by the current model parameters.

    Notes
    -----
    This implementation uses automatic differentiation to compute
    :math:`\\nabla_x h_\\theta(x)` and vectorizes the feature update across the
    dataset using :func:`jax.vmap`.

    The model assumes a differentiable score function and a first-order
    approximation of strategic behavior. It captures the intuition that agents
    locally modify their features in the direction that most improves their
    score under the current deployed model.

    Every column of the feature matrix is shifted, so the features must consist
    only of genuine, manipulable quantities. In particular, a model's intercept
    must be part of :math:`\\theta` and must not be encoded as a constant
    feature column, which would otherwise be shifted like a real feature. For
    this reason, :class:`~.CreditDataset` does not append a bias column.

    At present, only the ``"giveMeSomeCredit"`` dataset is supported.
    """

    def __init__(self,
                 n: int,
                 epsilon: float,
                 seed: int,
                 dataset_name: str,
                 h: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]):

        super().__init__(n, epsilon, seed)
        self.dataset_name = dataset_name
        self.h = h

        if self.dataset_name == 'giveMeSomeCredit':

            self.datafile = 'credit_data.zip'

            self.dataset = CreditDataset(datafile=self.datafile, seed=self.seed)
            self.x_0, self.y_0 = self.dataset.features[:n], self.dataset.labels[:n]

        else:

            raise NotImplementedError("Dataset not implemented")

    def sample(self,
               params: Array,
               ):

        grad_h = grad(lambda x: jnp.squeeze(self.h(params, jnp.expand_dims(x, axis=0))))
        return self.x_0 - self.epsilon * jax.vmap(grad_h)(self.x_0), self.y_0
