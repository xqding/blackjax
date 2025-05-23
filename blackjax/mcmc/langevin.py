from typing import NamedTuple, Callable
import jax
import jax.numpy as jnp
from blackjax.types import ArrayTree, Array, ArrayLikeTree, PRNGKey
import blackjax.mcmc.metrics as metrics
from blackjax.base import SamplingAlgorithm

__all__ = [
    "LangevinState",
    "LangevinInfo",
    "init",
    "build_kernel",
    "as_top_level_api",
]


class LangevinState(NamedTuple):
    position: ArrayTree
    momentum: ArrayTree
    logdensity: float
    logdensity_grad: ArrayTree


class LangevinInfo(NamedTuple):
    step_size: float
    kBT: float
    friction_coefficient: float
    num_integration_steps: int


def init(
    rng_key,
    position: ArrayLikeTree,
    inverse_mass_matrix: Array,
    kbT: float,
    logdensity_grad_fn: Callable,
    constraint_position: Callable = None,
    constraint_momentum: Callable = None,
):
    if constraint_position is not None:
        position = constraint_position(position)

    metric = metrics.default_metric(inverse_mass_matrix)

    momentum = metric.sample_momentum(rng_key, position)
    momentum = jax.tree.map(lambda x: jnp.sqrt(kbT) * x, momentum)

    if constraint_momentum is not None:
        momentum = constraint_momentum(position, momentum)

    logdensity, logdensity_grad = logdensity_grad_fn(position)

    return LangevinState(
        position,
        momentum,
        logdensity,
        logdensity_grad,
    )


def build_kernel(
    step_size: float,
    save_freq: int,
    kbT: float,
    friction_coefficient: float,
    logdensity_grad_fn: Callable,
    inverse_mass_matrix: Array,
    constraint_position: Callable = None,
    constraint_momentum: Callable = None,
):
    alpha = jnp.exp(-friction_coefficient * step_size)
    metric = metrics.default_metric(inverse_mass_matrix)
    kinetic_energy_grad_fn = jax.grad(metric.kinetic_energy)

    def one_step(rng_key, state: LangevinState, batch: tuple = ()):
        ## position(t), momentum(t - step_size/2), logdensity_grad(t)
        position, momentum, _, logdensity_grad = state

        # update momentum(t + step_size/2)
        momentum = jax.tree_util.tree_map(
            lambda m, g: m + step_size * g,
            momentum,
            logdensity_grad,
        )

        if constraint_momentum is not None:
            momentum = constraint_momentum(position, momentum)

        # update position(t + step_size/2)
        kinetic_energy_grad = kinetic_energy_grad_fn(momentum)

        position = jax.tree_util.tree_map(
            lambda p, g: p + step_size / 2 * g,
            position,
            kinetic_energy_grad,
        )

        # momentum_prime(t + step_size/2)
        random_momentum = metric.sample_momentum(rng_key, position)
        momentum_prime = jax.tree_util.tree_map(
            lambda old_m, random_m: alpha * old_m
            + jnp.sqrt(kbT * (1 - alpha**2)) * random_m,
            momentum,
            random_momentum,
        )

        # position(t + step_size)
        kinetic_energy_grad = kinetic_energy_grad_fn(momentum_prime)
        position = jax.tree_util.tree_map(
            lambda p, g: p + step_size / 2 * g,
            position,
            kinetic_energy_grad,
        )

        if constraint_position is not None:
            position = constraint_position(position)

        logdensity, logdensity_grad = logdensity_grad_fn(position, *batch)
        return LangevinState(
            position,
            momentum_prime,
            logdensity,
            logdensity_grad,
        )

    def kernel(rng_key: PRNGKey, state: LangevinState):
        keys = jax.random.split(rng_key, save_freq)
        _, state = jax.lax.while_loop(
            lambda loop_state: loop_state[0] < save_freq,
            lambda loop_state: (
                loop_state[0] + 1,
                one_step(keys[loop_state[0]], loop_state[1]),
            ),
            (0, state),
        )

        info = LangevinInfo(
            step_size=step_size,
            kBT=kbT,
            friction_coefficient=friction_coefficient,
            num_integration_steps=save_freq,
        )

        return state, info

    return kernel


def as_top_level_api(
    logdensity_grad_fn: Callable,
    step_size: float,
    save_freq: int,
    kbT: float,
    friction_coefficient: float,
    inverse_mass_matrix: Array,
):
    """Langevin middle integrator as described in
    http://docs.openmm.org/latest/userguide/theory/04_integrators.html#langevinmiddleintegrator
    """

    def init_fn(position: ArrayTree, rng_key):
        return init(
            rng_key,
            position,
            inverse_mass_matrix,
            kbT,
            logdensity_grad_fn,
        )

    def step_fn(rng_key, state: LangevinState):
        kernel = build_kernel(
            step_size,
            save_freq,
            kbT,
            friction_coefficient,
            logdensity_grad_fn,
            inverse_mass_matrix,
        )
        return kernel(rng_key, state)

    return SamplingAlgorithm(init_fn, step_fn)
