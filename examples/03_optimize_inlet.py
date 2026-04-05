"""Optimize inlet velocity to hit a target drag coefficient.

The simplest differentiable optimization: given a sphere in a flow,
find the inlet velocity that produces a specific target Cd. This is a
scalar-to-scalar optimization problem and a good smoke test for the
gradient pipeline.

Usage:
    python examples/03_optimize_inlet.py
"""

import jax
import jax.numpy as jnp
import numpy as np

from jax_lbm.core.lattice import D3Q19_OPPOSITE
from jax_lbm.core.equilibrium import equilibrium, compute_density, compute_velocity
from jax_lbm.core.collision import bgk
from jax_lbm.core.streaming import stream
from jax_lbm.core.boundary import zou_he_inlet, zou_he_outlet
from jax_lbm.core.forces import momentum_exchange, drag_coefficient, projected_area
from jax_lbm.geometry.primitives import create_sphere


def main():
    nx, ny, nz = 48, 24, 24
    n_steps = 30

    # Fixed geometry.
    center = (0.0, 0.0, 0.0)
    radius = 0.5
    solid, q = create_sphere(nx, ny, nz, center, radius)
    area = jnp.maximum(projected_area(solid, axis=0), 1.0)

    opp_arr = jnp.array([int(x) for x in np.asarray(D3Q19_OPPOSITE)])

    def simulate_cd(u_inlet):
        """Run forward sim and return Cd as a function of inlet velocity."""
        scale_x = nx / 8.0
        D_lattice = 2.0 * radius * scale_x
        re = 50.0
        nu = u_inlet * D_lattice / re
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

        f_final, f_posts = jax.lax.scan(step_fn, f, None, length=n_steps)
        f_post_last = jax.tree.map(lambda x: x[-1], f_posts)
        force = momentum_exchange(f_post_last, f_final, q)
        return drag_coefficient(force, u_inlet, area)

    # Loss: (Cd - target)^2
    target_cd = 1.5

    def loss(u_inlet):
        cd = simulate_cd(u_inlet)
        return (cd - target_cd) ** 2

    loss_and_grad = jax.value_and_grad(loss)

    # Gradient descent.
    u_inlet = jnp.float32(0.05)
    lr = 0.001
    n_iters = 20

    print("Inlet Velocity Optimization")
    print(f"Target Cd: {target_cd}")
    print(f"Grid: {nx}x{ny}x{nz}, steps per eval: {n_steps}")
    print()
    print(f"{'Iter':>4s}  {'u_inlet':>8s}  {'Cd':>10s}  {'Loss':>12s}  {'dL/du':>10s}")
    print("-" * 50)

    for i in range(n_iters):
        l, g = loss_and_grad(u_inlet)
        cd = simulate_cd(u_inlet)
        print(f"{i:4d}  {float(u_inlet):8.5f}  {float(cd):10.6f}  {float(l):12.8f}  {float(g):10.4f}")

        u_inlet = u_inlet - lr * g
        u_inlet = jnp.clip(u_inlet, 0.01, 0.15)

    print()
    print(f"Final inlet velocity: {float(u_inlet):.5f}")
    print(f"Final Cd: {float(simulate_cd(u_inlet)):.6f}")


if __name__ == "__main__":
    main()
