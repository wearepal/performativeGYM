import time
from dataclasses import asdict, dataclass
from typing import Literal

import haiku as hk
import jax
import jax.numpy as jnp
import tyro
from jax import Array, grad
from jax.typing import ArrayLike
from optax.losses import sigmoid_binary_cross_entropy  # type: ignore
from tqdm.auto import tqdm

from examples.datasets import CreditDataset
from performative_gym import (
    DFO,
    RGD,
    RRM,
    DPerfGD,
    Optimizer,
    Optimizers,
    PerfGDReinforce,
    PerfGDReparam,
)
from performative_gym.logger import Log, Logger
from performative_gym.optimizers import BaseOptimizer
from performative_gym.utils import acc_fn, initialize_params, weight_norm

jax.config.update("jax_enable_x64", True)

"""Original Perdomo et al. Experiment"""


@dataclass
class Credit:
    """Run the credit experiment with the specified optimizer."""

    epsilon: float = 10
    """Epsilon for the data distribution shift; higher values lead to more significant shifts."""
    n: int = 120_000
    """Number of samples to use for the experiment; 120,000 is the default size for the credit dataset."""
    iterations: int = 5000
    seed: int = 10
    optimizer: Optimizers = "DPerfGD"
    base_optimizer: BaseOptimizer = "GD"
    momentum: float = 0
    logging: Literal["offline", "wandb", "mlflow"] = "offline"
    output_file: str = "credit"
    model: Literal["NN", "logistic_regression"] = "NN"
    """Model type to use for the experiment, either 'NN' for a neural network or 'logistic_regression'."""
    lr: float = 0.1
    """Learning rate for the optimizer."""
    reg: float = 0
    """Regularization parameter for the logistic regression model."""
    rho: float = 0
    datafile: str = "credit_data.zip"
    """Data file containing the credit dataset."""

    def loss_fn(self, params: Array, x: Array, y: Array) -> Array:  # Size (n, 1)
        match self.model:
            case "NN":
                return jnp.expand_dims(
                    sigmoid_binary_cross_entropy(self.h(params, x), y),
                    axis=1,
                )
            case "logistic_regression":
                return jnp.expand_dims(
                    sigmoid_binary_cross_entropy(self.h(params, x), y)
                    + (self.reg / 2) * jnp.linalg.norm(params) ** 2,
                    axis=1,
                )

    def accuracy(self, params: Array, x: Array, y: Array) -> Array:
        return acc_fn(jax.nn.sigmoid(self.h(params, x)), y)

    def init_model(self) -> Array:
        match self.model:
            case "NN":

                def forward(x: Array) -> Array:
                    mlp = hk.Sequential([hk.Linear(100), jax.nn.relu, hk.Linear(1)])
                    return mlp(x).squeeze()

                model = hk.without_apply_rng(hk.transform(forward))
                self.h = jax.jit(model.apply)
                params = model.init(jax.random.PRNGKey(self.seed), jnp.zeros((1, 11)))
                return params
            case "logistic_regression":
                self.h = lambda params, x: x @ params
                return initialize_params((11,), self.seed)

    def init_data(self) -> None:
        self.dataset = CreditDataset(datafile=self.datafile, seed=self.seed)

    def proj_fn(self, params: Array) -> Array:
        def fn(x: Array) -> Array:
            norm = jnp.linalg.norm(x, ord=2)  # Compute L2 norm of theta
            scale = jnp.minimum(1.0, 10.0 / norm)  # Scale factor to enforce constraint
            return x * scale

        return jax.tree_util.tree_map(fn, params)

    def shift_data_distribution(
        self, params: Array, n: int
    ) -> tuple[Array, Array]:  # MUST return size (n,d)
        z, y = self.dataset.features[:n], self.dataset.labels[:n]
        grad_h = grad(lambda x: jnp.squeeze(self.h(params, jnp.expand_dims(x, axis=0))))
        return z - self.epsilon * jax.vmap(grad_h)(z), y  # type: ignore

    def prob_distr(self, x: Array, y: Array, mean: Array, params: Array) -> Array:
        def normal(x: Array, mean: Array, std: ArrayLike) -> Array:
            z = jax.scipy.stats.multivariate_normal.pdf(x, mean=mean, cov=std)
            return z

        def log_distr(distr: Array) -> Array:
            epsilon = 1e-12
            return jnp.log(jnp.clip(distr, a_min=epsilon, a_max=None))

        cov = jnp.diag(jnp.ones(x.shape[1]))
        return log_distr(normal(x, mean, cov))

    def f_fn(self, params: Array, x: Array, y: Array) -> Array:
        return jnp.mean(x, axis=0)

    def decoupled_loss(self, p_p: Array, p: Array) -> Array:
        self.init_model()
        x, y = self.shift_data_distribution(p_p, self.n)
        return jnp.mean(self.loss_fn(p, x=x, y=y))

    """
    params = jnp.array([-2/3.])
    grad1 = grad(lambda p: decoupled_loss(params, p))(params)
    grad2 = grad(lambda p_p: decoupled_loss(p_p, params))(params)
    """

    def train(self, optimizer_name: Optimizers) -> Optimizer:
        self.init_data()
        start_time = time.time()
        match self.logging:
            case "wandb":
                log_type = Log.WANDB
            case "mlflow":
                log_type = Log.ML_FLOW
            case "offline":
                log_type = Log.OFFLINE

        logger = Logger(
            project="decoupled-loss",
            group=f"{self.output_file}",
            name=optimizer_name
            + f"_model_{self.model}_{self.base_optimizer}_M{self.momentum}_S{self.rho}_{self.seed}",
            config=asdict(self),
            log_type=log_type,
        )

        try:
            params = self.init_model()
            match optimizer_name:
                case "RGD":
                    optimizer = RGD(
                        params,
                        lr=self.lr,
                        loss_fn=self.loss_fn,
                        proj_fn=self.proj_fn,
                        base_optimizer=self.base_optimizer,
                        momentum=self.momentum,
                    )
                case "PerfGDReparam":
                    optimizer = PerfGDReparam(
                        params,
                        lr=self.lr,
                        loss_fn=self.loss_fn,
                        proj_fn=self.proj_fn,
                        distr_shift=(lambda p: self.shift_data_distribution(p, self.n)),
                        base_optimizer=self.base_optimizer,
                        momentum=self.momentum,
                    )
                case "DPerfGD":
                    optimizer = DPerfGD(
                        params,
                        lr=self.lr,
                        loss_fn=self.loss_fn,
                        proj_fn=self.proj_fn,
                        distr_shift=(lambda p: self.shift_data_distribution(p, self.n)),
                        base_optimizer=self.base_optimizer,
                        momentum=self.momentum,
                        rho=self.rho,
                    )
                case "RRM":
                    optimizer = RRM(
                        params,
                        lr=self.lr,
                        loss_fn=self.loss_fn,
                        proj_fn=self.proj_fn,
                        tol=0.001,
                    )
                case "PerfGDReinforce":
                    optimizer = PerfGDReinforce(
                        params,
                        lr=self.lr,
                        f_fn=self.f_fn,
                        loss_fn=self.loss_fn,
                        proj_fn=self.proj_fn,
                        H=4,
                        prob_distr=self.prob_distr,
                    )
                case "DFO":
                    optimizer = DFO(
                        params,
                        lr=self.lr,
                        loss_fn=self.loss_fn,
                        proj_fn=self.proj_fn,
                        shift_data_distribution=(
                            lambda params: self.shift_data_distribution(params, self.n)
                        ),
                        seed=self.seed,
                    )

                case _:
                    print("Optimizer choice unknown")
                    exit()
            if log_type is not Log.ML_FLOW:
                logger.log(
                    {"p_o": jax.tree_util.tree_map(lambda x: x.tolist(), params)},
                    step=0,
                )
            with tqdm(total=self.iterations) as pbar:
                for i in range(self.iterations):
                    x, y = (
                        self.shift_data_distribution(optimizer.current_p_d, self.n)
                        if optimizer_name == "DPerfGD"
                        else self.shift_data_distribution(params, self.n)
                    )
                    # Perform gradient descent step
                    params = optimizer.step(params, x=x, y=y)
                    current_loss = jnp.mean(self.loss_fn(params, x=x, y=y))
                    logger.log(
                        {
                            "iteration": i,
                            "losses": current_loss.item(),
                            "accuracy": self.accuracy(params, x, y).item(),
                            "grads": weight_norm(
                                grad(lambda p: self.decoupled_loss(p, p))(params)
                            ).item(),
                            "grads_D": weight_norm(
                                grad(lambda p: self.decoupled_loss(p, params))(params)
                            ).item(),
                            "grads_M": weight_norm(
                                grad(lambda p_p: self.decoupled_loss(params, p_p))(
                                    params
                                )
                            ).item(),
                        },
                        step=i,
                    )

                    """
                    grad2 = grad(lambda p_p: decoupled_loss(p_p, params))(params)
                    grad1 = grad(lambda p: decoupled_loss(params, p))(params)
                    print(grad1, grad2)
                    """

                    pbar.set_description(
                        "Performative_loss: {0:.4f} Accuracy: {1:.2f}%".format(
                            current_loss.item(),
                            self.accuracy(params, x, y).item() * 100,
                        )
                    )

                    # print(f'Iteration {i+1} - loss: {current_loss:.4f} params: {params} ')
                    pbar.update(1)

            logger.log({"time": time.time() - start_time}, step=0)
            if log_type is not Log.ML_FLOW:
                logger.log(
                    {"p_f": jax.tree_util.tree_map(lambda x: x.tolist(), params)},
                    step=0,
                )

            return optimizer

        finally:
            logger.finish()


if __name__ == "__main__":
    args = tyro.cli(Credit, use_underscores=True)
    start_time = time.time()
    args.train(optimizer_name=args.optimizer)
    print(f"credit with {args.optimizer} in {time.time() - start_time} s")
