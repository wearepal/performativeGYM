from abc import abstractmethod

from jax import Array

__all__ = ["DistributionMap"]

class DistributionMap:
    def __init__(
            self,
            n: int,
            epsilon: float,
            seed: int,
    ):
        self.n = n
        self.epsilon = epsilon
        self.seed = seed

        self.x_0 = None
        self.y_0 = None

    @abstractmethod
    def sample(self,
               params: Array,
               ):
        pass

