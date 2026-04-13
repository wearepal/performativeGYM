# performativeGYM

performativeGYM is a library for simulating *performative prediction*, a machine learning setting that was introduced by [Perdomo et al. (2020)](https://proceedings.mlr.press/v119/perdomo20a.html). In performative prediction, the act of making predictions is affecting the world, such that the distribution of data encountered doesn’t match the training distribution anymore. An example is classifier used by a bank to make lending decisions, which has the effect that bank customers try to “game” the classifier in order to improve their chances of success.

The code in this project is split into two parts: the library itself, in the directory `performative_gym/`, which contains important definitions and implementations of proposed methods, and on the other hand, the `examples/` directory, which contains the implementations of concrete performative prediction scenarios.

## The library

**PerformativeGym** is structured around two core components: *optimizers* and *distribution maps*. Users can easily select from the provided implementations—or define their own—to pair a distribution map with an optimizer, enabling flexible experimentation with minimal setup.

The library is written in [JAX](https://jax.dev). 

### Optimizers

The library contains implementations of many algorithms that have been proposed or used as baselines in the literature of performative prediction:

- Repeated Gradient Descent (RGD) from [Perdomo et al., 2020](https://arxiv.org/abs/2002.06673)
- Repeated Risk Minimization (RRM) from [Perdomo et al., 2020](https://arxiv.org/abs/2002.06673)
- Performative Gradient Descent with REINFORCE (PerfGD) from [Izzo et al., 2021](https://arxiv.org/abs/2102.07698)
- PerfGD with reparametrization trick from [Cyffers et al., 2021](https://arxiv.org/abs/2411.02023)
- Decoupled Performative Gradient Descent (DPerfGD) from [Sanguino et al., 2025](https://arxiv.org/abs/2506.09044)
- Derivative Free Optimization (DFO) from [Flaxman et al., 2004](https://arxiv.org/abs/cs/0408007)

All these methods are implemented as subclasses of the following abstract base class:

```python
class Optimizer:
    def __init__(
        self,
        params: Array,
        lr: float,
        loss_fn: Callable[[Array, Array, Array], Array],
        proj_fn: Callable[[Array], Array] = (lambda params: params),
    ):
        self.current_params = params
        self.lr = lr
        self.loss_fn = loss_fn
        self.proj_fn = proj_fn
        self.params_history = [params]
        self.i = 0

    @abstractmethod
    def step(self, params: Array, x: Array, y: Array) -> Array:
        pass
```

In every call to the `.step()` method, the methods need to update the given parameters, for the features `x` and labels `y`, and need to return the new parameters. The methods are given an initial set of parameters, a loss function, and a projection function (which projects parameter values into the allowed range of parameter values). These three things are specific to the concrete setting in which the experiment is run. Some methods need even more information than that; for example, many need the distribution shift function as a differentiable function.

It is possible to define new optimizers following this structure.

### Distribution maps

In addition, PerformativeGym implements several distribution maps commonly studied in the literature, providing ready-to-use benchmarks for experimentation.

Concretely, it includes:

- Linear: $\mathcal{D}(\theta) = \mathcal{N}(a_1 \theta + a_0, \sigma)$
- Nonlinear from [Izzo et al., 2021](https://arxiv.org/abs/2102.07698): $\mathcal{D}(\theta) = \mathcal{N}(\sqrt{a_1 \theta + a_0}, \sigma)$
- Mixture of gaussians from [Izzo et al., 2021](https://arxiv.org/abs/2102.07698):  $\mathcal{D}(\theta) = \gamma \mathcal{N}(a_1 \theta + a_0, \sigma_a) + (1-\gamma) \mathcal{N}(b_1 \theta + b_0, \sigma_b)$
- Pricing from [Izzo et al., 2021](https://arxiv.org/abs/2102.07698): $\mathcal{D}(\theta) = \mathcal{N}(\mu_0 - \epsilon\theta, \sigma)$
- Strategic classification for performative prediction from [Perdomo et al., 2020](https://arxiv.org/abs/2002.06673): $(x, y) = (x_0-\epsilon\nabla_x f_\theta(x), y)$ 

All distribution maps in the library are implemented as subclasses of the abstract class `DistributionMap`. 
This base class defines a common interface and shared attributes, ensuring consistency across different implementations.

```python
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
    def sample(self, params: Array):
        pass
```

Each distribution map must implement the sample method, which generates data given the current model parameters.
The base class also standardizes key configuration variables such as the number of samples (`n`), the stregth of the performative effect (`epsilon`), and the random seed.

This modular design makes it straightforward to implement custom distribution maps: users only need to inherit
from `DistributionMap` and define their own sampling logic, while automatically benefiting from a consistent
interface that integrates seamlessly with the rest of the library.


## The examples
In the examples directory, several concrete scenarios are implemented, which can be run with any of the methods defined in the library. As mentioned above, each scenario needs to define, at minimum, the initial parameters, the loss function and the projection function. In addition, many methods also need a differentiable distribution shift function.

Here is a minimal example where the model is a simple linear model with a 1D weight vector and the data is sampled from a Gaussian which has a linear dependency on the weight vector:

```python
from dataclasses import dataclass
from functools import cached_property

import jax.numpy as jnp
from jax import Array

from performative_gym import RGD, Linear
from performative_gym.utils import initialize_params

@dataclass
class Minimal:
    A0: float = 5
    A1: float = 1
    STD: float = 1
    n: int = 10000
    iterations: int = 30
    seed: int = 0
    lr: float = 0.1

    @cached_property
    def distribution_map(self):
        return Linear(
            n=self.n,
            seed=self.seed,
            A0=self.A0,
            A1=self.A1,
            STD=self.STD,
        )
    
    def loss_fn(self, params: Array, x: Array, y: None) -> Array:
        # Simple linear loss function
        return params * x

    def proj_fn(self, params: Array) -> Array:
        return jnp.clip(params, -1.0, 1.0)

    def initial_params(self):
        return initialize_params((1,), self.seed) 

    def train(self) -> RGD:
        params = self.initial_params()
        method = RGD(params, lr=self.lr, loss_fn=self.loss_fn, proj_fn=self.proj_fn)

        for i in range(self.iterations):
            z = self.distribution_map.sample(params)
            # Perform gradient descent step
            params = method.step(params, x=z, y=None)
            # Compute current loss
            current_loss = jnp.mean(self.loss_fn(params, x=z, y=None))
            print(f"Iteration {i + 1}/{self.iterations}, Loss: {current_loss:.4f}")
        return method
```

It is not necessary to use `dataclasses` for this, but it is convenient. The model and the data can be anything, as long as the loss function, the projection function and the shift function can handle them.

The existing examples are:

- `credit.py`: [GiveMeSomeCredit](https://www.kaggle.com/c/GiveMeSomeCredit/overview)
- `linear.py`: 1D Gaussian with linear dependency on the model weights
- `nonlinear.py`: 1D Gaussian with non-linear dependency on the model weights
- `mixture.py`: a mixture of Gaussians
- `pricing.py`: multivariate Gaussian
- `cosine.py`: 1D Gaussian with a cosine loss function

If you want to run these examples, see below for the instructions.
## Usage

### Install dependencies

With `uv`:
```sh
uv sync
```

With `pip`:
```sh
pip install -e .
```

### Run examples

With `uv`:
```sh
uv run python examples/linear.py
```

With `pip`:
```sh
python examples/linear.py
```

If you supply the `--help` flag, a help message is printed with information about the available commandline arguments.

## License
This project is licensed under the Apache License 2.0. See the LICENSE file for details.
