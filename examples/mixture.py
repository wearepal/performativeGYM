import time
from dataclasses import asdict, dataclass
from functools import cached_property

import jax
import jax.numpy as jnp
import numpy as np
import tyro
import wandb
from jax import Array
from jax.typing import ArrayLike
from tqdm.auto import tqdm

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
from performative_gym.utils import initialize_params, loss_values


@dataclass
class Mixture:
    """Argument parser for configuration options."""

    # Configuration options with default values
    A0: float = -0.5
    A1: float = 1
    STD_A: float = 1
    B0: float = 1
    B1: float = -0.3
    STD_B: float = 0.25
    sigma: float = 0.5
    n: int = 10000
    iterations: int = 100
    seed: int = 0
    optimizer: Optimizers = "RRM"
    lr: float = 0.1
    log_wandb: bool = False

    @cached_property
    def params_opt(self) -> float:
        return -(self.sigma * self.A0 + (1 - self.sigma) * self.B0) / (
            2 * (self.sigma * self.A1 + (1 - self.sigma) * self.B1)
        )

    @cached_property
    def params_stab(self) -> float:
        return -(self.sigma * self.A0 + (1 - self.sigma) * self.B0) / (
            self.sigma * self.A1 + (1 - self.sigma) * self.B1
        )

    def mean_i(self, a0: float, a1: float, params: Array) -> Array:
        return a1 * params + a0

    def loss_fn(self, params: Array, x: Array, y: None) -> Array:
        return params * x

    def proj_fn(self, params: Array) -> Array:
        return jnp.clip(params, -1.0, 1.0)

    def shift_data_distribution(self, params: Array, n: int) -> tuple[Array, None]:
        z1 = jax.random.normal(jax.random.PRNGKey(3), (n,))
        z2 = jax.random.normal(jax.random.PRNGKey(3), (n,))

        n1 = self.mean_i(self.A0, self.A1, params) + z1 * self.STD_A
        n2 = self.mean_i(self.B0, self.B1, params) + z2 * self.STD_B

        return jnp.expand_dims(self.sigma * n1 + (1 - self.sigma) * n2, axis=1), None

    def prob_distr(self, x: Array, y: None, mean: Array, params: Array) -> Array:
        def normal(x: Array, mean: Array, std: ArrayLike) -> Array:
            z = jax.scipy.stats.norm.pdf(x, loc=mean, scale=std)
            return z

        def log_distr(distr: Array) -> Array:
            return jnp.log(distr)

        return log_distr(
            self.sigma * normal(x, mean, self.STD_A)
            + (1 - self.sigma) * normal(x, mean, self.STD_B)
        )

    def decoupled_loss(self, p_p: Array, p: Array) -> Array:
        x, y = self.shift_data_distribution(p_p, self.n)
        return jnp.mean(self.loss_fn(p, x=x, y=y))

    """
    params = jnp.array([-2/3.])
    grad1 = grad(lambda p: decoupled_loss(params, p))(params)
    grad2 = grad(lambda p_p: decoupled_loss(p_p, params))(params)
    """

    def f_fn(self, params: Array, x: Array, y: None) -> Array:
        return jnp.mean(x, axis=0)

    def init_model(self):
        return 0.85 + initialize_params((1,), self.seed) * 0.1

    def log_decoupled_landscape(self):
        logger = Logger(
            project="decoupled-loss",
            group="landscape",
            name="mixture",
            config=asdict(self),
            log_type=Log.WANDB if self.log_wandb else Log.OFFLINE,
        )
        x = np.arange(-1, 1.01, 0.01)
        y = np.arange(-1, 1.01, 0.01)
        landscape = loss_values(
            self.shift_data_distribution, self.loss_fn, self.n, x, y
        )
        logger.log(
            {
                "landscape": wandb.Table(data=landscape)
                if logger.log_type is Log.WANDB
                else np.array(landscape).tolist(),
                "x": x.tolist(),
                "y": y.tolist(),
            },
            step=0,
        )
        logger.finish()

    def train(self, optimizer_name: Optimizers) -> Optimizer:
        start_time = time.time()

        logger = Logger(
            project="PerfGD",
            group="mixture",
            name=f"{optimizer_name}_{self.seed}",
            config=asdict(self),
            log_type=Log.WANDB if self.log_wandb else Log.OFFLINE,
        )

        try:
            params = self.init_model()
            match optimizer_name:
                case "RGD":
                    optimizer = RGD(
                        params, lr=self.lr, loss_fn=self.loss_fn, proj_fn=self.proj_fn
                    )
                case "RRM":
                    optimizer = RRM(
                        params,
                        lr=self.lr,
                        loss_fn=self.loss_fn,
                        proj_fn=self.proj_fn,
                        tol=0.0001,
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
                case "PerfGDReparam":
                    optimizer = PerfGDReparam(
                        params,
                        lr=self.lr,
                        loss_fn=self.loss_fn,
                        proj_fn=self.proj_fn,
                        distr_shift=(lambda p: self.shift_data_distribution(p, self.n)),
                    )
                case "DPerfGD":
                    optimizer = DPerfGD(
                        params,
                        lr=self.lr,
                        loss_fn=self.loss_fn,
                        proj_fn=self.proj_fn,
                        distr_shift=(lambda p: self.shift_data_distribution(p, self.n)),
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

            """
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(np.arange(-1, 1.01, 0.01), np.arange(-1, 1.01, 0.01),
                                   np.array(losses_2d), rstride=1, cstride=1,
                                   linewidth=0, antialiased=False, cmap='viridis')
            fig.colorbar(surf)
            plt.show()
            """
            with tqdm(total=self.iterations) as pbar:
                for i in range(self.iterations):
                    x, y = self.shift_data_distribution(params, self.n)
                    # Perform gradient descent step
                    logger.log(
                        {
                            "iteration": i,
                            "p_d": params.item(),
                            "p_m": params.item(),
                            "losses": jnp.mean(self.loss_fn(params, x=x, y=y)).item(),
                        },
                        step=i,
                    )

                    params = optimizer.step(params, x=x, y=y)
                    logger.log(
                        {
                            "iteration": i + 1,
                            "p_d": optimizer.params_history[i].item(),
                            "p_m": params.item(),
                            "losses": jnp.mean(self.loss_fn(params, x=x, y=y)).item(),
                        },
                        step=i,
                    )

                    # Compute current loss
                    current_loss = jnp.mean(self.loss_fn(params, x=x, y=y))
                    pbar.set_description(
                        "Performative_loss: {0:.4f} params: {1:.2f} params_opt: {2:.4f} params_stab: {3:.4f}".format(
                            current_loss.item(),
                            params.item(),
                            self.params_opt,
                            self.params_stab,
                        )
                    )

                    pbar.update(1)
            logger.log({"time": time.time() - start_time}, step=0)

            return optimizer

        finally:
            logger.finish()


if __name__ == "__main__":
    start_time = time.time()
    args = tyro.cli(Mixture, use_underscores=True)
    args.train(optimizer_name=args.optimizer)
    print(f"non-linear with {args.optimizer} in {time.time() - start_time} s")
