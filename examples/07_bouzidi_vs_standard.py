"""Bouzidi vs standard bounce-back: gradient quality comparison.

This is the key evidence for the paper's novel claim. We run the same
shape optimization problem twice:

1. With proper Bouzidi q-values from ray-sphere intersection (2nd order).
2. With q=0.5 everywhere (standard midpoint bounce-back, 1st order).

Both use the same grid, same physics, same optimizer. The difference is
only in the wall treatment. If Bouzidi gradients are better, the
optimization should converge faster and/or to a lower Cd.

Usage:
    python examples/07_bouzidi_vs_standard.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_lbm.core.lattice import D3Q19_VELOCITIES, D3Q19_OPPOSITE, Q
from jax_lbm.core.equilibrium import equilibrium, compute_density, compute_velocity
from jax_lbm.core.collision import bgk
from jax_lbm.core.streaming import stream
from jax_lbm.core.boundary import zou_he_inlet, zou_he_outlet
from jax_lbm.core.forces import momentum_exchange, drag_coefficient, projected_area
from jax_lbm.geometry.smooth import (
    signed_distance_sphere,
    porosity_from_sdf,
    soft_bounce_back,
)


def _build_geometry(radius, nx, ny, nz, use_bouzidi):
    """Build smooth sphere geometry with either Bouzidi or standard q.

    When use_bouzidi=True, q values come from differentiable ray-sphere
    intersection. When False, q=0.5 everywhere (standard bounce-back).
    """
    sdf = signed_distance_sphere(nx, ny, nz, (0.0, 0.0, 0.0), radius)
    porosity = porosity_from_sdf(sdf, sharpness=20.0)
    solid = (porosity > 0.5).astype(jnp.int32)

    scale_x, scale_y, scale_z = nx / 8.0, ny / 4.0, nz / 4.0
    cx, cy, cz = 0.0, 0.0, 0.0

    gx = jnp.arange(nx, dtype=jnp.float32)
    gy = jnp.arange(ny, dtype=jnp.float32)
    gz = jnp.arange(nz, dtype=jnp.float32)
    gx3d, gy3d, gz3d = jnp.meshgrid(gx, gy, gz, indexing="ij")

    wx = (gx3d + 0.5) / scale_x - 4.0
    wy = (gy3d + 0.5) / scale_y - 2.0
    wz = (gz3d + 0.5) / scale_z - 2.0

    e_np = np.asarray(D3Q19_VELOCITIES)
    q = -jnp.ones((Q, nx, ny, nz))

    for i in range(1, Q):
        ex_i, ey_i, ez_i = int(e_np[i, 0]), int(e_np[i, 1]), int(e_np[i, 2])

        nbx = (gx3d + ex_i).astype(jnp.int32)
        nby = (gy3d + ey_i).astype(jnp.int32)
        nbz = (gz3d + ez_i).astype(jnp.int32)

        in_bounds = (
            (nbx >= 0) & (nbx < nx) & (nby >= 0) & (nby < ny)
            & (nbz >= 0) & (nbz < nz)
        )
        nbx_safe = jnp.clip(nbx, 0, nx - 1)
        nby_safe = jnp.clip(nby, 0, ny - 1)
        nbz_safe = jnp.clip(nbz, 0, nz - 1)

        nb_solid = solid[nbx_safe, nby_safe, nbz_safe]
        is_boundary = (solid == 0) & (nb_solid == 1) & in_bounds

        if use_bouzidi:
            # Differentiable ray-sphere intersection for exact q.
            ray_dx = jnp.float32(ex_i) / scale_x
            ray_dy = jnp.float32(ey_i) / scale_y
            ray_dz = jnp.float32(ez_i) / scale_z
            ox, oy, oz = wx - cx, wy - cy, wz - cz
            a = ray_dx**2 + ray_dy**2 + ray_dz**2
            b = 2.0 * (ox * ray_dx + oy * ray_dy + oz * ray_dz)
            c = ox**2 + oy**2 + oz**2 - radius**2
            disc = b * b - 4.0 * a * c
            sqrt_disc = jnp.sqrt(jnp.maximum(disc, 1e-10))
            t = (-b - sqrt_disc) / (2.0 * a + 1e-20)
            valid = (disc >= 0.0) & (t > 0.0) & (t <= 1.0)
            q_val = jnp.where(valid, t, 0.5)
            q_val = jnp.clip(q_val, 0.01, 0.99)
        else:
            # Standard bounce-back: q=0.5 everywhere.
            q_val = 0.5

        q = q.at[i].set(jnp.where(is_boundary, q_val, -1.0))

    return porosity, solid, q


def _run_optimization(use_bouzidi, n_iters=25):
    """Run shape optimization and return the Cd history."""
    nx, ny, nz = 48, 24, 24
    u_inlet = 0.05
    n_steps = 150

    D_lattice = 48 * 0.5 / 4.0
    re = 50.0
    nu = u_inlet * D_lattice / re
    tau = jnp.float32(3.0 * nu + 0.5)

    def simulate_cd(radius):
        porosity, solid, q = _build_geometry(radius, nx, ny, nz, use_bouzidi)

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

    loss_and_grad = jax.jit(jax.value_and_grad(simulate_cd))

    # Warm up JIT.
    radius = jnp.float32(0.5)
    loss_and_grad(radius)

    # Adam optimizer.
    optimizer = optax.adam(learning_rate=0.003)
    opt_state = optimizer.init(radius)

    cd_history = []
    grad_history = []

    for i in range(n_iters):
        cd, grad = loss_and_grad(radius)
        cd_history.append(float(cd))
        grad_history.append(float(grad))

        updates, opt_state = optimizer.update(grad, opt_state)
        radius = optax.apply_updates(radius, updates)
        radius = jnp.clip(radius, 0.2, 0.8)

    return cd_history, grad_history


def main():
    print("Bouzidi vs Standard Bounce-Back: Gradient Quality Comparison")
    print("=" * 62)
    print()

    print("Running with Bouzidi interpolated bounce-back (2nd order)...")
    cd_bouzidi, grad_bouzidi = _run_optimization(use_bouzidi=True)
    print(f"  Final Cd: {cd_bouzidi[-1]:.6f}")
    print()

    print("Running with standard bounce-back (q=0.5, 1st order)...")
    cd_standard, grad_standard = _run_optimization(use_bouzidi=False)
    print(f"  Final Cd: {cd_standard[-1]:.6f}")
    print()

    # Print comparison.
    print(f"{'Iter':>4s}  {'Cd (Bouzidi)':>12s}  {'Cd (Standard)':>14s}  "
          f"{'|grad| (Bouzidi)':>16s}  {'|grad| (Standard)':>18s}")
    print("-" * 72)
    for i in range(len(cd_bouzidi)):
        print(f"{i:4d}  {cd_bouzidi[i]:12.6f}  {cd_standard[i]:14.6f}  "
              f"{abs(grad_bouzidi[i]):16.6f}  {abs(grad_standard[i]):18.6f}")

    print()
    print("Summary:")
    print(f"  Bouzidi best Cd:  {min(cd_bouzidi):.6f}")
    print(f"  Standard best Cd: {min(cd_standard):.6f}")

    # Average gradient magnitude (measure of signal quality).
    avg_grad_b = sum(abs(g) for g in grad_bouzidi) / len(grad_bouzidi)
    avg_grad_s = sum(abs(g) for g in grad_standard) / len(grad_standard)
    print(f"  Bouzidi avg |grad|:  {avg_grad_b:.4f}")
    print(f"  Standard avg |grad|: {avg_grad_s:.4f}")

    if min(cd_bouzidi) < min(cd_standard):
        improvement = (1.0 - min(cd_bouzidi) / min(cd_standard)) * 100
        print(f"  Bouzidi achieves {improvement:.1f}% lower drag than standard")
    else:
        print("  Standard achieved equal or lower drag (unexpected)")


if __name__ == "__main__":
    main()
