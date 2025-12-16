from . import BaseOptimizer, Optimizer, LossFn, Generic, Y

from collections.abc import Callable

import optax
import jax.numpy as jnp
import jax

from jax import Array, grad


class DFO(Optimizer[Y], Generic[Y]):
    def __init__(
            self,
            params: Array,
            lr: float,
            loss_fn: LossFn[Y],
            proj_fn: Callable[[Array], Array],
            shift_data_distribution: Callable,
            seed: int,
            samples: int = 10,
            delta: float = 0.1,
    ):
        super().__init__(params, lr, loss_fn, proj_fn)
        self.distr_shift = shift_data_distribution
        self.delta = delta
        self.samples = samples
        self.seed = seed

    def step(self, params: Array, x: Array, y: Y) -> Array:
        def sample_unit_sphere(
                dim: tuple[int, ...], num_samples: int, seed: int
        ) -> Array:
            """
            Generate samples uniformly on the unit sphere S^{d-1}.
            """
            samples = jax.random.normal(
                jax.random.PRNGKey(seed), shape=(num_samples, *dim)
            )
            samples /= jnp.linalg.norm(
                samples, axis=tuple(range(1, len(samples.shape))), keepdims=True
            )
            return samples

        def decoupled_loss(p_p: Array, p: Array) -> Array:
            x, y = self.distr_shift(p_p)
            return jnp.mean(self.loss_fn(p, x, y))

        def performative_risk(params: Array):
            return decoupled_loss(params, params)

        u_samples = jax.tree_util.tree_map(
            lambda params: sample_unit_sphere(params.shape, self.samples, self.seed),
            params,
        )

        perturbed_params = jax.tree_util.tree_map(
            lambda u_samples, params: params + self.delta * u_samples, u_samples, params
        )
        risks = jax.vmap(performative_risk)(perturbed_params)

        grads = jax.tree_util.tree_map(
            lambda u_samples: jnp.mean(
                risks.reshape((self.samples,) + (1,) * (u_samples.ndim - 1))
                * u_samples,
                axis=0,
            ),
            u_samples,
        )

        # Update parameters using the computed gradients
        updated_params = jax.tree_util.tree_map(
            lambda p, g: self.proj_fn(p - self.lr * g)
            if isinstance(p, jnp.ndarray)
            else p,
            params,
            grads,
        )

        # Update history and iteration count
        self.params_history.append(updated_params)
        self.i += 1

        return updated_params
