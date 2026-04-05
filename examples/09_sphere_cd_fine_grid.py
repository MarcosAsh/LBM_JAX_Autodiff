"""Sphere Cd on a finer grid (256x128x128) for tight Clift validation.

The 128x64x64 grid gives a sphere diameter of 16 cells and Cd within
10% of Clift. This script doubles the resolution to D=32 cells, which
should reduce the geometric discretization error and bring Cd within
5% of the 1.09 reference.

Uses MRT collision for stability at tau=0.524.

Usage:
    python examples/09_sphere_cd_fine_grid.py
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
    nx, ny, nz = 256, 128, 128
    u_inlet = 0.05
    center = (0.0, 0.0, 0.0)
    radius = 0.5

    scale_x = nx / 8.0
    D_lattice = 2.0 * radius * scale_x
    re = 100.0
    nu = u_inlet * D_lattice / re
    tau = 3.0 * nu + 0.5

    print("Sphere Drag at Re=100 -- Fine Grid (MRT)")
    print(f"Grid: {nx}x{ny}x{nz}")
    print(f"Sphere: D={D_lattice:.0f} cells")
    print(f"Re={re:.0f}, nu={nu:.6f}, tau={tau:.4f}")
    print()

    solid, q = create_sphere(nx, ny, nz, center, radius)
    area = projected_area(solid, axis=0)
    print(f"Solid cells: {int(jnp.sum(solid == 1))}, projected area: {float(area):.0f}")

    rho = jnp.ones((nx, ny, nz))
    u = jnp.zeros((3, nx, ny, nz)).at[0].set(u_inlet)
    f = equilibrium(rho, u)
    vel = jnp.array([u_inlet, 0.0, 0.0])

    opp_arr = jnp.array([int(x) for x in np.asarray(D3Q19_OPPOSITE)])
    is_solid = (solid >= 1)[jnp.newaxis, :, :, :]
    tau_val = jnp.float32(tau)

    # Stability guard reference state.
    f_eq_ref = equilibrium(rho, u)

    def step_fn(carry, _):
        f_in, _ = carry
        # Stability guard.
        rho_in = compute_density(f_in)
        u_in = compute_velocity(f_in, rho_in)
        speed = jnp.sqrt(jnp.sum(u_in ** 2, axis=0))
        bad = (rho_in < 0.3) | (rho_in > 2.0) | (speed > 0.4) | jnp.isnan(rho_in)
        f_in = jnp.where(bad[None], f_eq_ref, f_in)

        f_c = mrt(f_in, tau_val)
        f_c = jnp.where(is_solid, f_c[opp_arr], f_c)
        f_post = f_c
        f_s = stream(f_c, q, periodic_yz=True)
        f_s = zou_he_inlet(f_s, vel)
        f_s = zou_he_outlet(f_s, rho_out=1.0)
        return (f_s, f_post), None

    n_steps = 6000
    chunk = 500

    print(f"\nRunning {n_steps} MRT steps...")
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
    print(f"\nFinal Cd: {cd_final:.4f}")
    print(f"Clift reference: 1.09")
    print(f"Relative error: {abs(cd_final - 1.09) / 1.09:.1%}")


if __name__ == "__main__":
    main()
