# JAX LBM

A differentiable 3D Lattice Boltzmann solver in JAX for inverse fluid design.

Backpropagate through hundreds of simulation timesteps with `jax.grad` and
optimize boundary conditions, obstacle geometry, or viscosity to hit a
target flow field or drag coefficient.

```python
import jax
import jax.numpy as jnp
from jax_lbm import LBMState, LBMParams, step, equilibrium
from jax_lbm.geometry.primitives import create_sphere
from jax_lbm.core.forces import momentum_exchange, drag_coefficient, projected_area

# Set up a sphere in a wind tunnel.
nx, ny, nz = 64, 32, 32
solid, q = create_sphere(nx, ny, nz, center=(0.0, 0.0, 0.0), radius=0.5)
rho = jnp.ones((nx, ny, nz))
u = jnp.zeros((3, nx, ny, nz))
f = equilibrium(rho, u)

state = LBMState(f=f, solid=solid, q=q)
params = LBMParams(tau=jnp.float32(0.6), inlet_velocity=jnp.array([0.05, 0.0, 0.0]))

# Simulate and compute drag.
final = jax.jit(lambda s: step(s, params))(state)
```

## What makes this different

Most LBM solvers are forward-only: you set up a geometry, press play, and
watch the flow develop. If you want to optimize something, you run hundreds
of simulations with different parameters and hope to find a good one.

This solver is **differentiable**. JAX's automatic differentiation traces
through every collision, streaming step, and boundary condition to give you
exact gradients of any output (drag, lift, velocity field) with respect to
any input (inlet velocity, viscosity, obstacle shape). One forward pass plus
one backward pass replaces hundreds of trial-and-error simulations.

The key technical contribution is making **Bouzidi interpolated bounce-back**
differentiable. This is the 2nd-order accurate boundary scheme that places
walls at fractional lattice positions. The wall distances depend on geometry,
and gradients flow through them cleanly, enabling shape optimization with
accurate wall treatment.

## Features

- **D3Q19 lattice** with BGK and MRT collision operators
- **Bouzidi interpolated bounce-back** with differentiable fractional wall
  distances from ray-sphere intersection
- **Zou-He boundary conditions** for velocity inlets and pressure outlets
- **Smagorinsky subgrid-scale model** for under-resolved turbulence
- **Momentum exchange force computation** with pressure/friction decomposition
- **Sigmoid-smoothed solids** for gradient-friendly shape representation
- **Gradient checkpointing** via `jax.checkpoint` for memory-bounded backprop
  through long simulations
- Validated against the [reference C/GLSL solver](https://github.com/MarcosAsh/Lattice_Fluid_Dynamics)
  and experimental data (Clift et al. 1978 for sphere drag)

## Quick links

- [Installation](getting_started/installation.md) -- get running in 2 minutes
- [First simulation](getting_started/first_simulation.md) -- channel flow in 30 lines
- [First optimization](getting_started/first_optimization.md) -- optimize inlet velocity for target Cd
- [LBM theory](guides/lattice_boltzmann.md) -- the physics, for people who know PDEs
- [API reference](api/core/lattice.md) -- every function, documented
