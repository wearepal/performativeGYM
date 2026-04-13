import time
from dataclasses import asdict, dataclass
from functools import cached_property

import jax
import jax.numpy as jnp
import tyro

from jax import Array, grad

from tqdm import tqdm

from performative_gym import (
    DFO,
    RGD,
    RRM,
    DPerfGD,
    Optimizer,
    Optimizers,
    PerfGDReinforce,
    PerfGDReparam,
    Linear,
)
from performative_gym.logger import Log, Logger
from performative_gym.utils import initialize_params


@dataclass
class CosineExp:
    """Run the cosine experiment with the specified optimizer."""

    A1: float = 1
    STD: float = 1
    n: int = 10000
    iterations: int = 100
    seed: int = 0
    optimizer: Optimizers = "PerfGDReparam"
    base_optimizer: str = "GD"
    momentum: float = 1

    lr: float = 0.05
    log_wandb: bool = False

    @cached_property
    def params_opt(self) -> float:
        return 3.42

    @cached_property
    def params_stab(self) -> float:
        return jnp.pi

    @cached_property
    def distribution_map(self):
        return Linear(
            n=self.n,
            seed=self.seed,
            A0=0,
            A1=self.A1,
            STD=self.STD,
        )

    def loss_fn(self, params: Array, x: Array, y: None) -> Array:  # Size (n, 1)
        return jnp.cos(params) * x

    def proj_fn(self, params: Array) -> Array:
        return jnp.clip(params, -1.0, 1.0)

    def prob_distr(self, x: Array, y: None, mean: Array, params: Array) -> Array:
        return jnp.log(jax.scipy.stats.norm.pdf(x, loc=mean, scale=self.STD))

    def f_fn(self, params: Array, x: Array, y: None) -> Array:
        return jnp.mean(x, axis=0)

    def decoupled_loss(self, p_p: Array, p: Array) -> Array:
        x, y = self.distribution_map.sample(p_p)
        return jnp.mean(self.loss_fn(p, x=x, y=y))

    def init_model(self):
        return initialize_params((1,), self.seed)   # changes the std of initialization

    def train(self, optimizer_name: Optimizers) -> Optimizer:
        start_time = time.time()

        logger = Logger(
            project="PerfGD",
            group="cosine",
            name=f"{optimizer_name}"
            + f"_{self.base_optimizer}_{self.momentum}_{self.seed}",
            config=asdict(self),
            log_type=Log.WANDB if self.log_wandb else Log.OFFLINE,
        )

        try:
            params = self.init_model()
            match optimizer_name:
                case "RGD":
                    optimizer = RGD(
                        params,
                        lr=self.lr,
                        loss_fn=self.loss_fn,
                        proj_fn=self.proj_fn
                    )
                case "PerfGDReparam":
                    optimizer = PerfGDReparam(
                        params,
                        lr=self.lr,
                        loss_fn=self.loss_fn,
                        proj_fn=self.proj_fn,
                        distr_map=self.distribution_map.sample,
                        base_optimizer=self.base_optimizer,
                        momentum=self.momentum)
                case "DPerfGD":
                    optimizer = DPerfGD(
                        params,
                        lr=self.lr,
                        loss_fn=self.loss_fn,
                        proj_fn=self.proj_fn,
                        distr_map=self.distribution_map.sample)
                case "RRM":
                    optimizer = RRM(
                        params,
                        lr=self.lr,
                        loss_fn=self.loss_fn,
                        proj_fn=self.proj_fn,
                        tol=0.01,
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
                    optimizer = DFO(params, lr=self.lr, loss_fn=self.loss_fn, proj_fn=self.proj_fn,
                                    distr_map=self.distribution_map.sample, seed=self.seed)

                case _:
                    print("Optimizer choice unknown")
                    exit()

            with tqdm(total=self.iterations) as pbar:
                for i in range(self.iterations):

                    z, _ = self.distribution_map.sample(params)

                    logger.log(
                        {
                            "iteration": i,
                            "p_d": params.item(),
                            "p_m": params.item(),
                            "losses": jnp.mean(
                                self.loss_fn(params, x=z, y=None)
                            ).item(),
                        },
                        step=i,
                    )
                    # Perform gradient descent step
                    params = optimizer.step(params, x=z, y=None)
                    # Compute metrics
                    logger.log(
                        {
                            "iteration": i + 1,
                            "p_d": optimizer.params_history[i].item(),
                            "p_m": params.item(),
                            "losses": jnp.mean(
                                self.loss_fn(params, x=z, y=None)
                            ).item(),
                            "grads": (
                                grad(lambda p: self.decoupled_loss(p, p))(params)
                            ).item(),
                            "grads_D": (
                                grad(lambda p: self.decoupled_loss(p, params))(params)
                            ).item(),
                            "grads_M": (
                                grad(lambda p_p: self.decoupled_loss(params, p_p))(
                                    params
                                )
                            ).item(),
                        },
                        step=i,
                    )
                    current_loss = jnp.mean(self.loss_fn(params, x=z, y=None))

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
    args = tyro.cli(CosineExp, use_underscores=True)
    start_time = time.time()
    args.train(optimizer_name=args.optimizer)
    print(f"cosine with {args.optimizer} in {time.time() - start_time} s")
