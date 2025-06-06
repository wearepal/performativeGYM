from dataclasses import asdict, dataclass
from functools import cached_property
import time

import jax
from jax import Array, grad
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np
from tqdm.auto import tqdm
import tyro

from optimizers import DPerfGD, PerfGDReparam, PerfGDReinforce, RGD, RRM, Optimizer, Optimizers, RegRRM, DFO
from examples.utils import initialize_params, loss_values
from examples.logger import Logger
import wandb

@dataclass
class Cosine:
    """Argument parser for configuration options."""

    A1: float = 1
    STD: float = 1
    n: int = 10000
    iterations: int = 100
    seed: int = 0
    optimizer: Optimizers = 'PerfGDReparam'
    base_optimizer: str = 'GD'
    momentum: float = 1

    lr: float = 0.05
    log_wandb: bool = False

    @cached_property
    def params_opt(self) -> float:
        return 3.42

    @cached_property
    def params_stab(self) -> float:
        return jnp.pi

    def loss_fn(self, params: Array, x: Array, y: None) -> Array:  # Size (n, 1)
        #return - (1 - jnp.cos(params)) * x
        return jnp.cos(params) * x

    def proj_fn(self, params: Array) -> Array:
        return jnp.clip(params, -1.0, 1.0)

    def shift_data_distribution(
        self, params: Array, n: int
    ) -> tuple[Array, None]:  # MUST return size (n,d)
        z = jax.random.normal(jax.random.PRNGKey(self.seed), (n,))
        return jnp.expand_dims((self.A1 * params) + z * self.STD, axis=1), None

    def prob_distr(self, x: Array, y: None, mean: Array, params: Array) -> Array:
        def normal(x: Array, mean: Array, std: ArrayLike) -> Array:
            z = jax.scipy.stats.norm.pdf(x, loc=mean, scale=std)
            return z

        def log_distr(distr: Array) -> Array:
            return jnp.log(distr)

        return log_distr(normal(x, mean, self.STD))

    def f_fn(self, params: Array, x: Array, y: None) -> Array:
        return jnp.mean(x, axis=0)

    def decoupled_loss(self, p_p: Array, p: Array) -> Array:
        x, y = self.shift_data_distribution(p_p, self.n)
        return jnp.mean(self.loss_fn(p, x=x, y=y))

    def init_model(self):
        params_0 = initialize_params((1,), self.seed) * 3/2 * jnp.pi
        return params_0 #changes the std of initialization

    '''
    params = jnp.array([-2/3.])
    grad1 = grad(lambda p: decoupled_loss(params, p))(params)
    grad2 = grad(lambda p_p: decoupled_loss(p_p, params))(params)
    '''
    def log_decoupled_landscape(self):
        logger = Logger(
            project="decoupled-loss", group="landscape", name='cosine', config=asdict(self), upload=self.log_wandb
        )
        x = np.arange(-3/2*jnp.pi, 3/2*jnp.pi, 0.01)
        y = np.arange(-3/2*jnp.pi, 3/2*jnp.pi, 0.01)
        landscape = loss_values(self.shift_data_distribution, self.loss_fn, self.n, x, y)
        logger.log({
            'landscape': wandb.Table(data=landscape) if logger.upload else np.array(landscape).tolist(),
            'x': x.tolist(),
            'y': y.tolist()
        })
        logger.finish()

    def train(
        self, optimizer_name: Optimizers
    ) -> tuple[Optimizer, float]:
        start_time = time.time()

        logger = Logger(
            project="PerfGD", group="cosine", name=f'{optimizer_name}'+ f'_{self.base_optimizer}_{self.momentum}_{self.seed}', config=asdict(self), upload=self.log_wandb
        )

        try:
            params = self.init_model()
            match optimizer_name:
                case 'RGD':
                    optimizer = RGD(params, lr=self.lr, loss_fn=self.loss_fn, proj_fn=self.proj_fn)
                case 'PerfGDReparam':
                    optimizer = PerfGDReparam(
                        params,
                        lr=self.lr,
                        loss_fn=self.loss_fn,
                        proj_fn=self.proj_fn,
                        distr_shift=(lambda p: self.shift_data_distribution(p, self.n)),
                        base_optimizer=self.base_optimizer,
                        momentum=self.momentum
                    )
                case 'DPerfGD':
                    optimizer = DPerfGD(
                        params,
                        lr=self.lr,
                        loss_fn=self.loss_fn,
                        proj_fn=self.proj_fn,
                        distr_shift=(lambda p: self.shift_data_distribution(p, self.n)),
                    )
                case 'RRM':
                    optimizer = RRM(
                        params, lr=self.lr, loss_fn=self.loss_fn, proj_fn=self.proj_fn, tol=0.01
                    )
                case 'RegRRM':
                    optimizer = RegRRM(
                        params, lr=self.lr, loss_fn=self.loss_fn, proj_fn=self.proj_fn, tol=0.01, reg=10
                    )
                case 'PerfGDReinforce':
                    optimizer = PerfGDReinforce(
                        params,
                        lr=self.lr,
                        f_fn=self.f_fn,
                        loss_fn=self.loss_fn,
                        proj_fn=self.proj_fn,
                        H=4,
                        prob_distr=self.prob_distr,
                    )
                case 'TwoStage':
                    optimizer = TwoStage(
                        params,
                        lr=self.lr,
                        loss_fn=self.loss_fn,
                        proj_fn=self.proj_fn,
                        init_model=self.init_model,
                        shift_data_distribution=(lambda params, n: self.shift_data_distribution(params, n)),
                        n=self.n
                    )
                case 'DFO':
                    optimizer = DFO(
                        params,
                        lr=self.lr,
                        loss_fn=self.loss_fn,
                        proj_fn=self.proj_fn,
                        shift_data_distribution=(lambda params: self.shift_data_distribution(params, self.n)),
                        seed=self.seed
                    )

                case _:
                    print('Optimizer choice unknown')
                    exit()

            losses = []
            losses_p_p = []
            with tqdm(total=self.iterations) as pbar:
                for i in range(self.iterations):
                    z, _ = self.shift_data_distribution(params, self.n)
                    losses_p_p.append(jnp.mean(self.loss_fn(params, x=z, y=None)))
                    logger.log({
                        'iteration': i,
                        'p_d': params.item(),
                        'p_m': params.item(),
                        'losses': jnp.mean(self.loss_fn(params, x=z, y=None)).item()
                    })
                    # Perform gradient descent step
                    params = optimizer.step(params, x=z, y=None)
                    # Compute current loss
                    logger.log({
                        'iteration': i + 1,
                        'p_d': optimizer.params_history[i].item(),
                        'p_m': params.item(),
                        'losses': jnp.mean(self.loss_fn(params, x=z, y=None)).item(),
                        'grads': (grad(lambda p: self.decoupled_loss(p, p))(params)).item(),
                        'grads_D': (grad(lambda p: self.decoupled_loss(p, params))(params)).item(),
                        'grads_M': (grad(lambda p_p: self.decoupled_loss(params, p_p))(params)).item()
                    })
                    current_loss = jnp.mean(self.loss_fn(params, x=z, y=None))
                    losses.append(current_loss)
                    '''
                    grad2 = grad(lambda p_p: decoupled_loss(p_p, params))(params)
                    grad1 = grad(lambda p: decoupled_loss(params, p))(params)
                    print(grad1, grad2)
                    '''

                    pbar.set_description(
                        'Performative_loss: {0:.4f} params: {1:.2f} params_opt: {2:.4f} params_stab: {3:.4f}'.format(
                            current_loss.item(),
                            params.item(),
                            self.params_opt,
                            self.params_stab,
                        )
                    )
                    # print(f'Iteration {i+1} - loss: {current_loss:.4f} params: {params} ')
                    pbar.update(1)

            logger.log({
                'time': time.time() - start_time
            })
            return optimizer

        finally:
            logger.finish()


if __name__ == '__main__':
    args = tyro.cli(Cosine, use_underscores=True)
    start_time = time.time()
    args.train(optimizer_name=args.optimizer)
    print(f'cosine with {args.optimizer} in {time.time() - start_time} s')