import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
from blackjax.mcmc import langevin
from blackjax.util import run_inference_algorithm
from blackjax.base import SamplingAlgorithm
import matplotlib as mpl
import matplotlib.pyplot as plt
from sys import exit

mpl.use("Agg")
jax.config.update("jax_platform_name", "cpu")


def logdensity(x):
    return stats.norm.logpdf(x, loc=1.0, scale=2.0)


step_size = 0.01
kbT = 1.0
friction_coefficient = 1
inverse_mass_matrix = jnp.array([1.0])

key = jax.random.PRNGKey(0)
subkey, key = jax.random.split(key)

initial_position = jnp.array(1.0)
initial_state = langevin.init(
    rng_key=subkey,
    position=initial_position,
    inverse_mass_matrix=inverse_mass_matrix,
    kbT=kbT,
    logdensity_grad_fn=jax.value_and_grad(logdensity),
)

langevin_kernel = langevin.build_kernel(
    step_size=step_size,
    save_freq=100,
    kbT=kbT,
    friction_coefficient=friction_coefficient,
    logdensity_grad_fn=jax.value_and_grad(logdensity),
    inverse_mass_matrix=inverse_mass_matrix,
)

def inference_loop(rng_key, kernel, initial_state, num_samples):
    @jax.jit
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states

num_samples = 1000
states = inference_loop(key, langevin_kernel, initial_state, num_samples)


positions = states.position.reshape(-1)
momentums = states.momentum.reshape(-1)

fig = plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(positions, bins=30, density=True)
plt.title("Position Distribution")
plt.xlabel("Position")
plt.ylabel("Density")

plt.subplot(1, 2, 2)
plt.hist(momentums, bins=30, density=True)
plt.title("Momentum Distribution")
plt.xlabel("Momentum")
plt.ylabel("Density")
plt.tight_layout()
plt.savefig("langevin_distribution.png")

fig = plt.figure(figsize=(10, 5))
plt.plot(positions, label="Position")
plt.plot(momentums, label="Momentum")
plt.title("Position and Momentum Over Time")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.legend()
plt.savefig("langevin_time_series.png")

exit()

init_fn, step_fn = langevin(
    logdensity_grad_fn=jax.value_and_grad(logdensity),
    step_size=step_size,
    kbT=kbT,
    friction_coefficient=friction_coefficient,
    inverse_mass_matrix=inverse_mass_matrix,
)


x0 = jnp.array([1.0])


state = init_fn(x0, key)
print("Initial state:", state)


state, history = run_inference_algorithm(
    key,
    SamplingAlgorithm(init_fn, step_fn),
    num_steps = 100,
    initial_state = state,
    progress_bar=True,    
)

exit()



