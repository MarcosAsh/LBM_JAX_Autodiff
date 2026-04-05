"""Shape optimization: minimize drag on a sphere by adjusting its radius.

This example demonstrates the core capability of the differentiable LBM
solver. We set up a sphere in a flow, compute the drag coefficient, and
use jax.grad to find how Cd changes with respect to the sphere radius.
Then we run gradient descent to find the radius that minimizes drag.

This is a toy problem (the answer is "make the sphere smaller"), but the
same machinery applies to more complex parameterizations: level sets,
free-form deformation, or mesh vertex positions.

The key insight: because the Bouzidi q-values depend on the sphere
radius through the ray-sphere intersection formula, and because the
streaming step is differentiable through those q-values, the gradient
of Cd with respect to radius is well-defined and accurate.

Usage:
    python examples/04_shape_optimization.py
"""

import jax
import jax.numpy as jnp
import numpy as np

from jax_lbm.core.lattice import Q
from jax_lbm.core.equilibrium import equilibrium, compute_density, compute_velocity
from jax_lbm.core.collision import bgk
from jax_lbm.core.streaming import stream
from jax_lbm.core.boundary import zou_he_inlet, zou_he_outlet
from jax_lbm.core.forces import momentum_exchange, drag_coefficient, projected_area
from jax_lbm.geometry.smooth import (
    smooth_sphere_geometry,
    soft_bounce_back,
)


def simulate_cd(radius, nx, ny, nz, n_steps, u_inlet, tau):
    """Forward simulation: compute Cd as a differentiable function of radius."""
    porosity, solid, q = smooth_sphere_geometry(
        nx, ny, nz, center=(0.0, 0.0, 0.0), radius=radius, sharpness=20.0,
    )

    rho = jnp.ones((nx, ny, nz))
    u = jnp.zeros((3, nx, ny, nz)).at[0].set(u_inlet)
    f = equilibrium(rho, u)
    vel = jnp.array([u_inlet, 0.0, 0.0])

    def step_fn(f_in, _):
        f_c = bgk(f_in, tau)
        f_c = soft_bounce_back(f_c, porosity)
        f_post = f_c
        f_s = stream(f_c, q, periodic_yz=True)
        f_s = zou_he_inlet(f_s, vel)
        f_s = zou_he_outlet(f_s, rho_out=1.0)
        return f_s, f_post

    f_final, f_posts = jax.lax.scan(step_fn, f, None, length=n_steps)
    f_post_last = jax.tree.map(lambda x: x[-1], f_posts)

    force = momentum_exchange(f_post_last, f_final, q)
    area = jnp.maximum(projected_area(solid, axis=0), 1.0)
    return drag_coefficient(force, u_inlet, area)


def main():
    # Grid and physics setup.
    nx, ny, nz = 48, 24, 24
    u_inlet = 0.05
    n_steps = 30  # short for demo

    # Re = 50 on this grid.
    D_lattice = 48 * 0.5 / 4.0  # 6 cells diameter for radius=0.5
    re = 50.0
    nu = u_inlet * D_lattice / re
    tau = jnp.float32(3.0 * nu + 0.5)

    print("Differentiable LBM Shape Optimization")
    print(f"Grid: {nx}x{ny}x{nz}, Re={re:.0f}, tau={float(tau):.4f}")
    print(f"Steps per evaluation: {n_steps}")
    print()

    # Gradient function.
    cd_and_grad = jax.value_and_grad(
        lambda r: simulate_cd(r, nx, ny, nz, n_steps, u_inlet, tau)
    )

    # Gradient descent on the radius.
    radius = jnp.float32(0.5)
    lr = 0.01
    n_iters = 15

    print(f"{'Iter':>4s}  {'Radius':>8s}  {'Cd':>10s}  {'dCd/dr':>10s}")
    print("-" * 40)

    for i in range(n_iters):
        cd, grad = cd_and_grad(radius)
        print(f"{i:4d}  {float(radius):8.4f}  {float(cd):10.6f}  {float(grad):10.6f}")

        # Gradient descent step (minimize Cd).
        radius = radius - lr * grad

        # Clamp to physical range.
        radius = jnp.clip(radius, 0.2, 1.0)

    print()
    print(f"Final radius: {float(radius):.4f}")
    print(f"Final Cd: {float(cd):.6f}")

    # Verify that Cd decreased.
    cd_initial = simulate_cd(jnp.float32(0.5), nx, ny, nz, n_steps, u_inlet, tau)
    cd_final = simulate_cd(radius, nx, ny, nz, n_steps, u_inlet, tau)
    print(f"Cd reduction: {float(cd_initial):.6f} -> {float(cd_final):.6f}")

    if float(cd_final) < float(cd_initial):
        print("Optimization successful: Cd decreased.")
    else:
        print("Warning: Cd did not decrease. May need more steps or different learning rate.")


if __name__ == "__main__":
    main()
