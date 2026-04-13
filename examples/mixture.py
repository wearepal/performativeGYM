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
    Mixture,
)
from performative_gym.logger import Log, Logger
from performative_gym.utils import initialize_params


@dataclass
class MixtureExp:
    """Run the mixture experiment with the specified optimizer."""

    A0: float = -0.5
    A1: float = 1
    STD_A: float = 1
    B0: float = 1
    B1: float = -0.3
    STD_B: float = 0.25
    gamma: float = 0.5
    n: int = 10000
    iterations: int = 100
    seed: int = 0
    optimizer: Optimizers = "RRM"
    lr: float = 0.1
    log_wandb: bool = False

    @cached_property
    def params_opt(self) -> float:
        return -(self.gamma * self.A0 + (1 - self.gamma) * self.B0) / (
                2 * (self.gamma * self.A1 + (1 - self.gamma) * self.B1)
        )

    @cached_property
    def params_stab(self) -> float:
        return -(self.gamma * self.A0 + (1 - self.gamma) * self.B0) / (
                self.gamma * self.A1 + (1 - self.gamma) * self.B1
        )

    @cached_property
    def distribution_map(self):
        return Mixture(
            n=self.n,
            seed=self.seed,
            A0=self.A0,
            A1=self.A1,
            STD_A=self.STD_A,
            B0=self.B0,
            B1=self.B1,
            STD_B=self.STD_B,
            gamma=self.gamma,
        )

    def loss_fn(self, params: Array, x: Array, y: None) -> Array:
        return params * x

    def proj_fn(self, params: Array) -> Array:
        return jnp.clip(params, -1.0, 1.0)

    def prob_distr(self, x: Array, y: None, mean: Array, params: Array) -> Array:
        def normal_pdf(x: Array, mean: Array, std: ArrayLike) -> Array:
            z = jax.scipy.stats.norm.pdf(x, loc=mean, scale=std)
            return z

        return jnp.log(
            self.gamma * normal_pdf(x, mean, self.STD_A)
            + (1 - self.gamma) * normal_pdf(x, mean, self.STD_B)
        )

    def f_fn(self, params: Array, x: Array, y: None) -> Array:
        return jnp.mean(x, axis=0)

    def init_model(self):
        return initialize_params((1,), self.seed)

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
                    optimizer = PerfGDReparam(params, lr=self.lr, loss_fn=self.loss_fn, proj_fn=self.proj_fn,
                                              distr_map=self.distribution_map.sample)
                case "DPerfGD":
                    optimizer = DPerfGD(params, lr=self.lr, loss_fn=self.loss_fn, proj_fn=self.proj_fn,
                                        distr_map=self.distribution_map.sample)
                case "DFO":
                    optimizer = DFO(params, lr=self.lr, loss_fn=self.loss_fn, proj_fn=self.proj_fn,
                                    distr_map=self.distribution_map.sample, seed=self.seed)

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
                    x, y = self.distribution_map.sample(params)
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
    args = tyro.cli(MixtureExp, use_underscores=True)
    args.train(optimizer_name=args.optimizer)
    print(f"non-linear with {args.optimizer} in {time.time() - start_time} s")
