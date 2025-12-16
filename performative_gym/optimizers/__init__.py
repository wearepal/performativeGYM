from typing import Generic, Literal, Protocol, TypeAlias, TypeVar
from abc import abstractmethod
from collections.abc import Callable

from jax import Array

from .DFO import DFO
from .DPerfGD import DPerfGD, BarierDPerfGD
from .PerfGDReinforce import PerfGDReinforce
from .PerfGDReparam import PerfGDReparam
from .RGD import RGD
from .RRM import RRM

__all__ = [
    "BaseOptimizer",
    "DFO",
    "DPerfGD",
    "BarierDPerfGD",
    "Objective",
    "Optimizer",
    "Optimizers",
    "PerfGDReinforce",
    "PerfGDReparam",
    "RGD",
    "RRM",
]


Optimizers: TypeAlias = Literal[
    "RGD",
    "PerfGDReparam",
    "DPerfGD",
    "BarierDPerfGD",
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
