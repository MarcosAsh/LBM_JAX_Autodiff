# First Optimization: Inlet Velocity for Target Drag

This page shows the core differentiable simulation workflow. Given a
sphere in a flow, we find the inlet velocity that produces a specific
target drag coefficient. One line of `jax.grad` replaces what would
otherwise be a parameter sweep over hundreds of simulations.

## The problem

You have a sphere in a wind tunnel. You want the drag coefficient to be
exactly 1.5. What inlet velocity achieves that?

Without gradients, you would sweep over velocities, run a simulation for
each, and interpolate. With gradients, you define a loss function and
let gradient descent find the answer.

## Setup

```python
import jax
import jax.numpy as jnp
import numpy as np

from jax_lbm.core.lattice import D3Q19_OPPOSITE
from jax_lbm.core.equilibrium import equilibrium
from jax_lbm.core.collision import bgk
from jax_lbm.core.streaming import stream
from jax_lbm.core.boundary import zou_he_inlet, zou_he_outlet
from jax_lbm.core.forces import momentum_exchange, drag_coefficient, projected_area
from jax_lbm.geometry.primitives import create_sphere

nx, ny, nz = 48, 24, 24
center = (0.0, 0.0, 0.0)
radius = 0.5
solid, q = create_sphere(nx, ny, nz, center, radius)
area = jnp.maximum(projected_area(solid, axis=0), 1.0)
opp_arr = jnp.array([int(x) for x in np.asarray(D3Q19_OPPOSITE)])
```

## Define a differentiable simulation

The function takes inlet velocity as input and returns the drag coefficient.
Everything inside is pure JAX, so `jax.grad` can differentiate through it.

```python
def simulate_cd(u_inlet):
    scale_x = nx / 8.0
    D_lattice = 2.0 * radius * scale_x
    nu = u_inlet * D_lattice / 50.0  # Re = 50
    tau = 3.0 * nu + 0.5

    rho = jnp.ones((nx, ny, nz))
    u = jnp.zeros((3, nx, ny, nz)).at[0].set(u_inlet)
    f = equilibrium(rho, u)
    vel = jnp.array([u_inlet, 0.0, 0.0])

    def step_fn(f_in, _):
        f_c = bgk(f_in, tau)
        f_bounced = f_c[opp_arr]
        is_solid = (solid >= 1)[jnp.newaxis, :, :, :]
        f_c = jnp.where(is_solid, f_bounced, f_c)
        f_post = f_c
        f_s = stream(f_c, q, periodic_yz=True)
        f_s = zou_he_inlet(f_s, vel)
        f_s = zou_he_outlet(f_s, rho_out=1.0)
        return f_s, f_post

    f_final, f_posts = jax.lax.scan(step_fn, f, None, length=30)
    f_post_last = jax.tree.map(lambda x: x[-1], f_posts)
    force = momentum_exchange(f_post_last, f_final, q)
    return drag_coefficient(force, u_inlet, area)
```

## Define the loss and take the gradient

```python
target_cd = 1.5

def loss(u_inlet):
    cd = simulate_cd(u_inlet)
    return (cd - target_cd) ** 2

loss_and_grad = jax.value_and_grad(loss)
```

That's it. `jax.value_and_grad` returns both the loss value and its
gradient with respect to `u_inlet`. The gradient tells us: if I increase
the inlet velocity by a small amount, how much does the squared error
change?

## Run gradient descent

```python
u_inlet = jnp.float32(0.05)
lr = 0.001

for i in range(20):
    l, g = loss_and_grad(u_inlet)
    cd = simulate_cd(u_inlet)
    print(f"iter {i:2d}: u={float(u_inlet):.5f}, Cd={float(cd):.4f}, loss={float(l):.6f}")
    u_inlet = u_inlet - lr * g
    u_inlet = jnp.clip(u_inlet, 0.01, 0.15)
```

The optimizer converges in a handful of iterations. Each iteration
requires one forward simulation (to compute Cd) and one backward pass
(to compute dCd/du), which JAX fuses into a single efficient program.

## What just happened

The gradient `dLoss/d(u_inlet)` was computed by backpropagating through:

1. The drag coefficient formula (trivial)
2. The momentum exchange force summation (sum over boundary links)
3. 30 timesteps of streaming with Bouzidi bounce-back
4. 30 timesteps of BGK collision
5. The equilibrium initialization

Every one of these operations is a pure JAX function, so autodiff traces
through the entire chain automatically. No adjoint equations to derive, no
sensitivity analysis to implement by hand.

## What to try next

- Increase `n_steps` for a more converged flow (the gradient becomes
  more accurate but the backward pass uses more memory)
- Try optimizing `tau` (viscosity) instead of inlet velocity
- Move on to [shape optimization](../examples/shape_optimization.md)
  where the gradient flows through the Bouzidi wall distances into
  obstacle geometry
