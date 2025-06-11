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
from performative_gym.logger import Logger
from performative_gym.utils import initialize_params, loss_values, weight_norm

jax.config.update("jax_enable_x64", True)

from jax import grad


@dataclass
class Pricing:
    """Argument parser for configuration options."""

    # Configuration options with default values
    mu0: float = 6
    epsilon: float = 1.5
    n: int = 1000
    d: int = 100
    iterations: int = 100
    seed: int = 0
    optimizer: Optimizers = "RGD"
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

    def loss_fn(self, params: Array, x: Array, y: None) -> Array:
        return jnp.expand_dims(-params @ x.T, axis=1)
        # return - params @ x.T

    def proj_fn(self, params: Array) -> Array:
        return jnp.clip(params, 0.0, 5.0)

    def shift_data_distribution(self, params: Array, n: int) -> tuple[Array, None]:
        mean = self.mu_0 - self.epsilon * params
        return jax.random.multivariate_normal(
            jax.random.PRNGKey(3), mean, self.cov, shape=(n,)
        ), None

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
        return initialize_params((self.d,), self.seed)

    def decoupled_loss(self, p_p: Array, p: Array) -> Array:
        x, y = self.shift_data_distribution(p_p, self.n)
        return jnp.mean(self.loss_fn(p, x=x, y=y))

    def log_decoupled_landscape(self):
        logger = Logger(
            project="decoupled-loss",
            group="landscape",
            name="pricing",
            config=asdict(self),
            upload=self.log_wandb,
        )
        x = np.arange(0, 5.01, 0.01)
        x = x.reshape(x.shape[0], 1)
        y = np.arange(0, 5.01, 0.01)
        y = y.reshape(y.shape[0], 1)
        landscape = loss_values(
            self.shift_data_distribution, self.loss_fn, self.n, x, y
        )
        logger.log(
            {
                "landscape": wandb.Table(data=landscape)
                if logger.upload
                else np.array(landscape).tolist(),
                "x": x.tolist(),
                "y": y.tolist(),
            }
        )
        logger.finish()

    """
    params = jnp.array([params_opt])
    grad1 = grad(lambda p: decoupled_loss(params, p))(params)
    grad2 = grad(lambda p_p: decoupled_loss(p_p, params))(params)
    """

    def train(self, optimizer_name: Optimizers) -> Optimizer:
        start_time = time.time()

        logger = Logger(
            project="PerfGD",
            group="pricing_grads",
            name=optimizer_name + f"_{self.d}d_{self.seed}",
            config=asdict(self),
            upload=self.log_wandb,
        )
        try:
            params = self.init_model()
            match optimizer_name:
                case "RGD":
                    optimizer = RGD(
                        params, lr=self.lr, loss_fn=self.loss_fn, proj_fn=self.proj_fn
                    )
                case "PerfGDReparam":
                    optimizer = PerfGDReparam(
                        params,
                        lr=self.lr,
                        loss_fn=self.loss_fn,
                        proj_fn=self.proj_fn,
                        distr_shift=(lambda p: self.shift_data_distribution(p, self.n)),
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
                        H=14,
                        prob_distr=self.prob_distr,
                    )
                case "DPerfGD":
                    optimizer = DPerfGD(
                        params,
                        lr=self.lr,
                        loss_fn=self.loss_fn,
                        proj_fn=self.proj_fn,
                        distr_shift=(lambda p: self.shift_data_distribution(p, self.n)),
                        reg=self.d / 25,
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

            with tqdm(total=self.iterations) as pbar:
                for i in range(self.iterations):
                    x, y = (
                        self.shift_data_distribution(optimizer.current_p_d, self.n)
                        if optimizer_name == "DPerfGD"
                        else self.shift_data_distribution(params, self.n)
                    )
                    logger.log(
                        {
                            "iteration": i,
                            "p_d": optimizer.current_p_d.tolist()
                            if optimizer_name == "DPerfGD"
                            else params.tolist(),
                            "p_m": params.tolist(),
                            "losses": jnp.mean(self.loss_fn(params, x=x, y=y)).item(),
                        }
                    )

                    # Perform gradient descent step
                    params = optimizer.step(params, x=x, y=y)
                    logger.log(
                        {
                            "iteration": i + 1,
                            "p_d": optimizer.current_p_d.tolist()
                            if optimizer_name == "DPerfGD"
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
                        }
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
            logger.log({"time": time.time() - start_time})

            return optimizer

        finally:
            logger.finish()


if __name__ == "__main__":
    start_time = time.time()
    args = tyro.cli(Pricing, use_underscores=True)
    args.train(optimizer_name=args.optimizer)
    print(f"non-linear with {args.optimizer} in {time.time() - start_time} s")
