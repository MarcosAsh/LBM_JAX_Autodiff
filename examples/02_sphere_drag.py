"""Sphere drag at Re=100 -- validation against Clift et al. (1978).

Places a sphere in a wind tunnel and measures the drag coefficient as
the simulation approaches steady state. The target Cd for a sphere at
Re=100 is 1.09 from experimental data.

Usage:
    python examples/02_sphere_drag.py
"""

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

    print("Sphere Drag at Re=100")
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
    tau_val = jnp.float32(tau)

    n_steps = 4000
    print_interval = 200
    force_interval = 100

    print(f"Running {n_steps} steps...")
    print(f"{'Step':>6s}  {'Cd':>10s}  {'Fx':>12s}  {'max|u|':>10s}")
    print("-" * 44)

    for i in range(n_steps):
        f = bgk(f, tau_val)

        f_bounced = f[opp_arr]
        is_solid = (solid >= 1)[jnp.newaxis, :, :, :]
        f = jnp.where(is_solid, f_bounced, f)

        f_post = f

        f = stream(f, q, periodic_yz=True)
        f = zou_he_inlet(f, vel)
        f = zou_he_outlet(f, rho_out=1.0)

        if (i + 1) % force_interval == 0:
            force = momentum_exchange(f_post, f, q)
            cd = drag_coefficient(force, u_inlet, area)

            if (i + 1) % print_interval == 0:
                rho_now = compute_density(f)
                u_now = compute_velocity(f, rho_now)
                max_u = float(jnp.max(jnp.sqrt(jnp.sum(u_now ** 2, axis=0))))
                print(f"{i+1:6d}  {float(cd):10.4f}  {float(force[0]):12.6f}  {max_u:10.6f}")

    print()
    print(f"Final Cd: {float(cd):.4f}")
    print(f"Clift reference: 1.09")
    print(f"Relative error: {abs(float(cd) - 1.09) / 1.09:.1%}")


if __name__ == "__main__":
    main()
