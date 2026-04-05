"""Differentiable 3D Lattice Boltzmann solver in JAX.

A from-scratch D3Q19 LBM implementation designed for inverse fluid design.
Every function is pure and JIT-compilable, so you can backpropagate through
hundreds of timesteps with ``jax.grad`` and optimize boundary conditions,
obstacle geometry, or viscosity to hit a target flow field or drag coefficient.

Quick start::

    from jax_lbm import LBMState, LBMParams, step, equilibrium
    import jax.numpy as jnp

    nx, ny, nz = 64, 32, 32
    params = LBMParams(tau=0.8, inlet_velocity=jnp.array([0.05, 0.0, 0.0]))
    rho = jnp.ones((nx, ny, nz))
    u = jnp.zeros((3, nx, ny, nz))
    f = equilibrium(rho, u)
    state = LBMState(f=f, solid=jnp.zeros((nx, ny, nz), dtype=jnp.int32),
                     q=-jnp.ones((19, nx, ny, nz)))
    state = step(state, params)
"""

from jax_lbm.core.lattice import (
    D3Q19_VELOCITIES,
    D3Q19_WEIGHTS,
    D3Q19_OPPOSITE,
    CS2,
)
from jax_lbm.core.equilibrium import (
    equilibrium,
    compute_density,
    compute_velocity,
)
from jax_lbm.core.collision import bgk, mrt, smagorinsky_tau
from jax_lbm.core.streaming import stream
from jax_lbm.core.boundary import zou_he_inlet, zou_he_outlet
from jax_lbm.core.forces import momentum_exchange, drag_coefficient
from jax_lbm.state import LBMState, LBMParams, step, simulate

__all__ = [
    "D3Q19_VELOCITIES",
    "D3Q19_WEIGHTS",
    "D3Q19_OPPOSITE",
    "CS2",
    "equilibrium",
    "compute_density",
    "compute_velocity",
    "bgk",
    "mrt",
    "smagorinsky_tau",
    "stream",
    "zou_he_inlet",
    "zou_he_outlet",
    "momentum_exchange",
    "drag_coefficient",
    "LBMState",
    "LBMParams",
    "step",
    "simulate",
]
