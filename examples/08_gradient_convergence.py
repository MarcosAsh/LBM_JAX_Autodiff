"""Grid convergence of dCd/dr -- does the gradient converge with refinement?

Computes the gradient of drag coefficient with respect to sphere radius
at multiple grid resolutions. If the method is correct, the gradient
should converge to a stable value as the grid is refined. This is the
standard way to verify that a numerical gradient is trustworthy.

We also compare the autodiff gradient against finite differences at each
resolution to check consistency.

Usage:
    python examples/08_gradient_convergence.py
"""

import jax
import jax.numpy as jnp

from jax_lbm.core.equilibrium import equilibrium, compute_density, compute_velocity
from jax_lbm.core.collision import bgk
from jax_lbm.core.streaming import stream
from jax_lbm.core.boundary import zou_he_inlet, zou_he_outlet
from jax_lbm.core.forces import momentum_exchange, drag_coefficient, projected_area
from jax_lbm.geometry.smooth import smooth_sphere_geometry, soft_bounce_back


def _simulate_cd(radius, nx, ny, nz, n_steps, u_inlet, tau):
    """Differentiable simulation returning Cd."""
    porosity, solid, q = smooth_sphere_geometry(
        nx, ny, nz, center=(0.0, 0.0, 0.0), radius=radius, sharpness=20.0,
    )

    rho = jnp.ones((nx, ny, nz))
    u = jnp.zeros((3, nx, ny, nz)).at[0].set(u_inlet)
    f = equilibrium(rho, u)
    vel = jnp.array([u_inlet, 0.0, 0.0])

    def step_fn(carry, _):
        f_in, _ = carry
        f_c = bgk(f_in, tau)
        f_c = soft_bounce_back(f_c, porosity)
        f_post = f_c
        f_s = stream(f_c, q, periodic_yz=True)
        f_s = zou_he_inlet(f_s, vel)
        f_s = zou_he_outlet(f_s, rho_out=1.0)
        rho_s = compute_density(f_s)
        u_s = compute_velocity(f_s, rho_s)
        u_mag = jnp.sqrt(jnp.sum(u_s * u_s, axis=0))
        bad = (rho_s < 0.3) | (rho_s > 2.0) | (u_mag > 0.4) | jnp.isnan(rho_s)
        f_eq_r = equilibrium(jnp.where(bad, 1.0, rho_s),
                             jnp.where(bad[None], 0.0, u_s))
        f_s = jnp.where(bad[None], f_eq_r, f_s)
        return (f_s, f_post), None

    (f_final, f_post), _ = jax.lax.scan(step_fn, (f, f), None, length=n_steps)

    force = momentum_exchange(f_post, f_final, q)
    area = jnp.maximum(projected_area(solid, axis=0), 1.0)
    return drag_coefficient(force, u_inlet, area)


def main():
    u_inlet = 0.05
    re = 50.0
    radius = jnp.float32(0.5)
    eps = 1e-3  # for finite differences

    # Grid resolutions: (nx, ny, nz, n_steps)
    # More steps at finer grids so the flow has time to develop.
    configs = [
        (32, 16, 16, 80),
        (48, 24, 24, 120),
        (64, 32, 32, 160),
        (96, 48, 48, 200),
    ]

    print("Grid Convergence of dCd/dr")
    print("=" * 70)
    print()
    print(f"Sphere radius={float(radius)}, Re={re}, U={u_inlet}")
    print()
    print(f"{'Grid':>12s}  {'D (cells)':>10s}  {'tau':>8s}  "
          f"{'Cd':>10s}  {'dCd/dr (AD)':>12s}  {'dCd/dr (FD)':>12s}")
    print("-" * 70)

    for nx, ny, nz, n_steps in configs:
        D_lattice = nx * 0.5 / 4.0  # sphere diameter in lattice cells
        nu = u_inlet * D_lattice / re
        tau = jnp.float32(3.0 * nu + 0.5)

        cd_fn = lambda r: _simulate_cd(r, nx, ny, nz, n_steps, u_inlet, tau)
        cd_and_grad = jax.jit(jax.value_and_grad(cd_fn))

        # AD gradient.
        cd_val, grad_ad = cd_and_grad(radius)

        # FD gradient.
        cd_plus = jax.jit(cd_fn)(radius + eps)
        cd_minus = jax.jit(cd_fn)(radius - eps)
        grad_fd = (float(cd_plus) - float(cd_minus)) / (2.0 * eps)

        grid_str = f"{nx}x{ny}x{nz}"
        print(f"{grid_str:>12s}  {D_lattice:10.1f}  {float(tau):8.4f}  "
              f"{float(cd_val):10.6f}  {float(grad_ad):12.6f}  {grad_fd:12.6f}")

    print()
    print("If dCd/dr converges as the grid refines, the gradient is trustworthy.")
    print("The AD and FD columns should roughly agree at each resolution.")


if __name__ == "__main__":
    main()
