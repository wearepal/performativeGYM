from typing import Generic, Literal, Protocol, TypeAlias, TypeVar
from abc import abstractmethod
from collections.abc import Callable

from jax import Array

Optimizers: TypeAlias = Literal[
    "RGD",
    "PerfGDReparam",
    "DPerfGD",
    "BarierDPerfGD",
    "DecCostDPerfGD",
    "RRM",
    "PerfGDReinforce",
    "RegRRM",
    "TwoStage",
    "DFO",
]

Objective: TypeAlias = Literal["stability", "optimality"]
BaseOptimizer: TypeAlias = Literal["GD", "adam", "adamw", "adagrad"]
Y = TypeVar("Y", contravariant=True, bound=Array | None)

class LossFn(Protocol[Y]):
    def __call__(self, params: Array, x: Array, y: Y) -> Array: ...

class Optimizer(Generic[Y]):
    """
    Abstract base class for optimizers used in performative prediction.

    This class defines the common interface and shared state for all optimization
    algorithms in the library. Each optimizer maintains the current model
    parameters and updates them iteratively based on data sampled from a
    performative distribution.

    Subclasses must implement the ``step`` method, which specifies how the
    parameters are updated given a batch of data.

    Parameters
    ----------
    params : Array
        Initial model parameters :math:`\\theta`.

    lr : float
        Learning rate used by the optimizer.

    loss_fn : LossFn[Y]
        Pointwise loss function. Given parameters and a batch of inputs and labels,
        it returns the corresponding losses used to form the empirical objective.

    proj_fn : Callable[[Array], Array], default=lambda params: params
        Projection operator applied after each update. This can be used to enforce
        constraints on the parameter space (e.g., bounded domains or normalization).
        By default, no projection is applied.

    Attributes
    ----------
    current_params : Array
        Current value of the model parameters.

    lr : float
        Learning rate of the optimizer.

    loss_fn : LossFn[Y]
        Loss function used to compute gradients or objectives.

    proj_fn : Callable[[Array], Array]
        Projection operator applied after each update.

    params_history : list[Array]
        History of parameter values across iterations.

    i : int
        Iteration counter (number of optimization steps performed).

    Methods
    -------
    step(params: Array, x: Array, y: Y) -> Array
        Perform one optimization step using the batch ``(x, y)`` and return the
        updated parameters. This method must be implemented by subclasses.

    Notes
    -----
    This class does not implement any optimization logic by itself. Concrete
    optimizers (e.g., RGD, RRM, PerfGD) must inherit from this class and define
    their own update rules within the ``step`` method.
    """

    def __init__(
            self,
            params: Array,
            lr: float,
            loss_fn: LossFn[Y],
            proj_fn: Callable[[Array], Array] = (lambda params: params),
    ):
        self.current_params = params
        self.lr = lr
        self.loss_fn = loss_fn
        self.proj_fn = proj_fn
        self.params_history = [params]

        self.i = 0

    @abstractmethod
    def step(self, params: Array, x: Array, y: Y) -> Array:
        pass
