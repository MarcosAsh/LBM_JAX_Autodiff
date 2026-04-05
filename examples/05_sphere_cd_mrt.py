"""Sphere drag at Re=100 with MRT collision -- validation against Clift et al. (1978).

Same setup as 02_sphere_drag.py but uses the multiple-relaxation-time (MRT)
collision operator instead of BGK. MRT damps non-physical ghost modes
aggressively while keeping the physical viscosity at the correct rate,
which gives better accuracy near solid boundaries and a more converged
drag coefficient on this relatively coarse grid.

Target Cd for a sphere at Re=100 is 1.09 (Clift, Grace & Weber, 1978).

Usage:
    python examples/05_sphere_cd_mrt.py
"""

import jax
import jax.numpy as jnp
import numpy as np

from jax_lbm.core.lattice import D3Q19_OPPOSITE
from jax_lbm.core.equilibrium import equilibrium, compute_density, compute_velocity
from jax_lbm.core.collision import mrt
from jax_lbm.core.streaming import stream
from jax_lbm.core.boundary import zou_he_inlet, zou_he_outlet
from jax_lbm.core.forces import momentum_exchange, drag_coefficient, projected_area
from jax_lbm.geometry.primitives import create_sphere


def main():
    nx, ny, nz = 128, 64, 64
    u_inlet = 0.05
    center = (0.0, 0.0, 0.0)
    radius = 0.5

    # Compute physical parameters.
    scale_x = nx / 8.0
    D_lattice = 2.0 * radius * scale_x
    re = 100.0
    nu = u_inlet * D_lattice / re
    tau = 3.0 * nu + 0.5

    print("Sphere Drag at Re=100 (MRT collision)")
    print(f"Grid: {nx}x{ny}x{nz}")
    print(f"Sphere: center={center}, radius={radius}, D={D_lattice:.1f} cells")
    print(f"Re={re:.0f}, nu={nu:.6f}, tau={tau:.4f}")
    print()

    # Create sphere geometry.
    solid, q = create_sphere(nx, ny, nz, center, radius)
    n_solid = int(jnp.sum(solid == 1))
    area = projected_area(solid, axis=0)
    print(f"Solid cells: {n_solid}, projected area: {float(area):.1f}")
    print()

    # Initialize with uniform flow.
    rho = jnp.ones((nx, ny, nz))
    u = jnp.zeros((3, nx, ny, nz)).at[0].set(u_inlet)
    f = equilibrium(rho, u)
    vel = jnp.array([u_inlet, 0.0, 0.0])

    opp_arr = jnp.array([int(x) for x in np.asarray(D3Q19_OPPOSITE)])
    is_solid = (solid >= 1)[jnp.newaxis, :, :, :]
    tau_val = jnp.float32(tau)

    # Equilibrium state used by the stability guard to reset blown-up cells.
    rho_ref = jnp.ones((nx, ny, nz))
    u_ref = jnp.zeros((3, nx, ny, nz)).at[0].set(u_inlet)
    f_eq_ref = equilibrium(rho_ref, u_ref)

    def stability_guard(f_in):
        """Reset cells to equilibrium if density, velocity, or NaN is out of range.

        Thresholds:
            rho < 0.3 or rho > 2.0
            |u| > 0.4
            any NaN in the 19 distributions
        """
        rho_loc = compute_density(f_in)
        u_loc = compute_velocity(f_in, rho_loc)
        speed = jnp.sqrt(jnp.sum(u_loc ** 2, axis=0))

        bad = (
            (rho_loc < 0.3)
            | (rho_loc > 2.0)
            | (speed > 0.4)
            | jnp.any(jnp.isnan(f_in), axis=0)
        )
        # Broadcast (nx, ny, nz) mask to (19, nx, ny, nz).
        bad_broad = bad[jnp.newaxis, :, :, :]
        return jnp.where(bad_broad, f_eq_ref, f_in)

    def step_fn(carry, _):
        f_in, _ = carry

        # Stability guard: clamp before collision.
        f_in = stability_guard(f_in)

        # MRT collision.
        f_c = mrt(f_in, tau_val)

        # Bounce-back on solid cells.
        f_c = jnp.where(is_solid, f_c[opp_arr], f_c)
        f_post = f_c

        # Streaming.
        f_s = stream(f_c, q, periodic_yz=True)

        # Boundary conditions.
        f_s = zou_he_inlet(f_s, vel)
        f_s = zou_he_outlet(f_s, rho_out=1.0)

        return (f_s, f_post), None

    n_steps = 4000
    chunk = 200

    print(f"Running {n_steps} steps (MRT collision)...")
    print(f"{'Step':>6s}  {'Cd':>10s}  {'Fx':>12s}  {'max|u|':>10s}")
    print("-" * 44)

    for start in range(0, n_steps, chunk):
        length = min(chunk, n_steps - start)
        (f, f_post), _ = jax.lax.scan(step_fn, (f, f), None, length=length)

        force = momentum_exchange(f_post, f, q)
        cd = drag_coefficient(force, u_inlet, area)
        rho_now = compute_density(f)
        u_now = compute_velocity(f, rho_now)
        max_u = float(jnp.max(jnp.sqrt(jnp.sum(u_now ** 2, axis=0))))
        print(f"{start + length:6d}  {float(cd):10.4f}  {float(force[0]):12.6f}  {max_u:10.6f}")

    cd_final = float(cd)
    cd_ref = 1.09
    rel_err = abs(cd_final - cd_ref) / cd_ref

    print()
    print(f"Final Cd: {cd_final:.4f}")
    print(f"Clift reference: {cd_ref}")
    print(f"Relative error: {rel_err:.1%}")

    if rel_err < 0.10:
        print("PASS -- within 10% of reference")
    else:
        print(f"MISS -- relative error {rel_err:.1%} exceeds 10% threshold")


if __name__ == "__main__":
    main()
