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
    # BarierDPerfGD,
    Optimizer,
    Optimizers,
    PerfGDReinforce,
    PerfGDReparam,
    Pricing,
)
from performative_gym.logger import Log, Logger
from performative_gym.utils import initialize_params, weight_norm

jax.config.update("jax_enable_x64", True)

from jax import grad


@dataclass
class PricingExp:
    """Argument parser for configuration options."""

    # Configuration options with default values
    mu0: float = 6
    epsilon: float = 1.5
    n: int = 1000
    d: int = 1
    iterations: int = 100
    seed: int = 10
    optimizer: Optimizers = "DPerfGD"
    reg: float = 0
    lr: float = 0.1
    log_wandb: bool = False

    @cached_property
    def mu_0(self) -> Array:
        return self.mu0 * jnp.ones((self.d)) + jax.random.uniform(
            jax.random.PRNGKey(3), (self.d,)
        )

    @cached_property
    def cov(self) -> Array:
        return jnp.diag(jnp.ones(self.d))

    @cached_property
    def params_opt(self) -> Array:
        return self.mu_0 / (2 * self.epsilon)

    @cached_property
    def params_stab(self) -> Array:
        return self.mu_0 / self.epsilon

    @cached_property
    def distribution_map(self):
        return Pricing(
            n=self.n,
            epsilon=self.epsilon,
            seed=self.seed,
            mu0=self.mu0,
            d=self.d,
        )

    def loss_fn(self, params: Array, x: Array, y: None) -> Array:
        return jnp.expand_dims(-params @ x.T, axis=1)

    def proj_fn(self, params: Array) -> Array:
        return jnp.clip(params, 0.0, 5.0)

    def prob_distr(self, x: Array, y: None, mean: Array, params: Array) -> Array:
        def normal(x: Array, mean: Array, std: ArrayLike) -> Array:
            z = jax.scipy.stats.multivariate_normal.pdf(x, mean=mean, cov=std)
            return z

        def log_distr(distr: Array) -> Array:
            return jnp.log(distr)

        return log_distr(normal(x, mean, self.cov))

    def f_fn(self, params: Array, x: Array, y: None) -> Array:
        return jnp.mean(x, axis=0)

    def init_model(self):
        return initialize_params((self.d,), self.seed) + 2.5

    def decoupled_loss(self, p_p: Array, p: Array) -> Array:
        x, y = self.distribution_map.sample(p_p)
        return jnp.mean(self.loss_fn(p, x=x, y=y))

    def train(self, optimizer_name: Optimizers) -> Optimizer:
        start_time = time.time()

        logger = Logger(
            project="PerfGD",
            group="pricing",
            name=optimizer_name + f"_{self.d}d_{self.lr}lr_{self.seed}",
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
                case "PerfGDReparam":
                    optimizer = PerfGDReparam(params, lr=self.lr, loss_fn=self.loss_fn, proj_fn=self.proj_fn,
                                              distr_map=self.distribution_map.sample)
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
                        H=14,
                        prob_distr=self.prob_distr,
                    )
                case "DPerfGD":
                    optimizer = DPerfGD(params, lr=self.lr, loss_fn=self.loss_fn, proj_fn=self.proj_fn,
                                        distr_map=self.distribution_map.sample, reg=self.reg)
                case "BarierDPerfGD":
                    optimizer = BarierDPerfGD(
                        params,
                        lr=self.lr,
                        loss_fn=self.loss_fn,
                        proj_fn=self.proj_fn,
                        distr_shift=self.distribution_map.sample,
                        reg=self.reg,
                    )
                case "DFO":
                    optimizer = DFO(params, lr=self.lr, loss_fn=self.loss_fn, proj_fn=self.proj_fn,
                                    distr_map=self.shift_data_distribution, seed=self.seed)

                case _:
                    print("Optimizer choice unknown")
                    exit()

            with tqdm(total=self.iterations) as pbar:
                for i in range(self.iterations):
                    x, y = (
                        self.distribution_map.sample(optimizer.current_p_d)
                        if optimizer_name in ["DPerfGD", "BarierDPerfGD"]
                        else self.distribution_map.sample(params)
                    )
                    logger.log(
                        {
                            "iteration": i,
                            "p_d": optimizer.current_p_d.tolist()
                            if optimizer_name in ["DPerfGD", "BarierDPerfGD"]
                            else params.tolist(),
                            "p_m": params.tolist(),
                            "losses": jnp.mean(self.loss_fn(params, x=x, y=y)).item(),
                        },
                        step=i,
                    )

                    # Perform gradient descent step
                    params = optimizer.step(params, x=x, y=y)
                    logger.log(
                        {
                            "iteration": i + 1,
                            "p_d": optimizer.current_p_d.tolist()
                            if optimizer_name in ["DPerfGD", "BarierDPerfGD"]
                            else optimizer.params_history[i].tolist(),
                            "p_m": params.tolist(),
                            "losses": jnp.mean(self.loss_fn(params, x=x, y=y)).item(),
                            "dist_params": jnp.linalg.norm(
                                params - self.params_opt
                            ).item(),
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

                    # Compute current loss
                    current_loss = jnp.mean(self.loss_fn(params, x=x, y=y))

                    pbar.set_description(
                        "Performative_loss: {0:.4f} dist_params: {1:.2f}".format(
                            current_loss.item(),
                            jnp.linalg.norm(params - self.params_opt).item(),
                        )
                    )
                    """
                    logger.log({
                        'Loss': current_loss.item(),
                        'dist_params': jnp.linalg.norm(params - self.params_opt).item(),
                        'grads': jnp.linalg.norm(optimizer.grads).item(),
                    })
                    """
                    pbar.update(1)

            # print(f'params: {params}')
            # print(
            #    f'theta_opt: {self.mu_0 / (2 * self.epsilon)}, theta_stab: {self.mu_0 / (self.epsilon)}'
            # )
            logger.log({"time": time.time() - start_time}, step=0)

            return optimizer

        finally:
            logger.finish()


if __name__ == "__main__":
    start_time = time.time()
    args = tyro.cli(PricingExp, use_underscores=True)
    args.train(optimizer_name=args.optimizer)
    print(f"non-linear with {args.optimizer} in {time.time() - start_time} s")
