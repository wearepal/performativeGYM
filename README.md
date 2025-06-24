# performativeGYM

performativeGYM is a library for simulating *performative prediction*, a machine learning setting that was introduced by [Perdomo et al. (2020)](https://proceedings.mlr.press/v119/perdomo20a.html). In performative prediction, the act of making predictions is affecting the world, such that the distribution of data encountered doesn’t match the training distribution anymore. An example is classifier used by a bank to make lending decisions, which has the effect that bank customers try to “game” the classifier in order to improve their chances of success.

The code in this project is split into two parts: the library itself, in the directory `performative_gym/`, which contains important definitions and implementations of proposed methods, and on the other hand, the `examples/` directory, which contains the implementations of concrete performative prediction scenarios.

### The library
The library is written in [JAX](https://jax.dev). It contains implementations of many algorithms that have been proposed in the literature:

- RGD (Perdomo et al., 2020)
- RRM (Perdomo et al., 2020)
- RegRRM (regularized RRM)
- PerfGD (REINFORCE) ([Izzo et al., 2021](https://arxiv.org/abs/2102.07698))
- PerfGD (reparam)
- DPerfGD ([Sanguino et al., 2025](https://arxiv.org/abs/2506.09044))
- DFO ([Flaxman et al., 2004](https://arxiv.org/abs/cs/0408007))

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

### The examples
In the examples directory, several concrete scenarios are implemented, which can be run with any of the methods defined in the library. As mentioned above, each scenario needs to define, at minimum, the initial parameters, the loss function and the projection function. In addition, many methods also need a differentiable distribution shift function.

Here is a minimal example where the model is a simple linear model with a 1D weight vector and the data is sampled from a Gaussian which has a linear dependency on the weight vector:

```python
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from performative_gym import RGD
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

    def loss_fn(self, params: Array, x: Array, y: None) -> Array:
        # Simple linear loss function
        return params * x

    def proj_fn(self, params: Array) -> Array:
        return jnp.clip(params, -1.0, 1.0)

    def shift_data_distribution(self, params: Array, n: int) -> Array:
        # Normal distribution with mean A1 * params + A0 and std STD
        z = jax.random.normal(jax.random.PRNGKey(self.seed), (n,))
        return jnp.expand_dims((self.A1 * params + self.A0) + z * self.STD, axis=1)

    def initial_params(self):
        return 0.85 + initialize_params((1,), self.seed) * 0.1

    def train(self) -> RGD:
        params = self.initial_params()
        method = RGD(params, lr=self.lr, loss_fn=self.loss_fn, proj_fn=self.proj_fn)

        for i in range(self.iterations):
            z = self.shift_data_distribution(params, self.n)
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
his project is licensed under the Apache License 2.0. See the LICENSE file for details.
