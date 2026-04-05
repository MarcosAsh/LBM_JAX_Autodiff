"""Channel flow (Poiseuille) -- the simplest LBM simulation.

Drives flow through a rectangular channel with no-slip walls at the top
and bottom (y-direction). The steady-state velocity profile is a parabola,
which we can compare against the analytic solution.

Usage:
    python examples/01_channel_flow.py
"""

import jax.numpy as jnp
import numpy as np

from jax_lbm.core.lattice import D3Q19_OPPOSITE
from jax_lbm.core.equilibrium import equilibrium, compute_density, compute_velocity
from jax_lbm.core.collision import bgk
from jax_lbm.core.streaming import stream
from jax_lbm.core.boundary import zou_he_inlet, zou_he_outlet
from jax_lbm.geometry.bouzidi import compute_q_from_solid


def main():
    nx, ny, nz = 60, 20, 4
    u_inlet = 0.05
    tau = jnp.float32(0.8)
    nu = (1.0 / 3.0) * (float(tau) - 0.5)
    n_steps = 3000

    print("Poiseuille Channel Flow")
    print(f"Grid: {nx}x{ny}x{nz}, tau={float(tau):.2f}, nu={nu:.4f}")
    print(f"Inlet velocity: {u_inlet}")
    print()

    # Walls at y=0 and y=ny-1.
    solid = jnp.zeros((nx, ny, nz), dtype=jnp.int32)
    solid = solid.at[:, 0, :].set(1)
    solid = solid.at[:, ny - 1, :].set(1)
    q = compute_q_from_solid(solid, nx, ny, nz, default_q=0.5)

    # Initialize at rest.
    rho = jnp.ones((nx, ny, nz))
    u = jnp.zeros((3, nx, ny, nz))
    f = equilibrium(rho, u)
    vel = jnp.array([u_inlet, 0.0, 0.0])

    import jax

    opp_arr = jnp.array([int(x) for x in np.asarray(D3Q19_OPPOSITE)])
    is_solid = (solid >= 1)[jnp.newaxis, :, :, :]

    def step_fn(f_in, _):
        f_c = bgk(f_in, tau)
        f_c = jnp.where(is_solid, f_c[opp_arr], f_c)
        f_c = stream(f_c, q, periodic_yz=True)
        f_c = zou_he_inlet(f_c, vel)
        f_c = zou_he_outlet(f_c, rho_out=1.0)
        return f_c, None

    # Run in chunks of 500 to print progress.
    print(f"Running {n_steps} steps...")
    chunk = 500
    for start in range(0, n_steps, chunk):
        length = min(chunk, n_steps - start)
        f, _ = jax.lax.scan(step_fn, f, None, length=length)
        rho_now = compute_density(f)
        u_now = compute_velocity(f, rho_now)
        print(f"  Step {start + length}: max|u_x|={float(jnp.max(jnp.abs(u_now[0]))):.5f}")

    # Extract profile at midpoint.
    rho_final = compute_density(f)
    u_final = compute_velocity(f, rho_final)
    x_mid = nx // 2
    ux_profile = jnp.mean(u_final[0, x_mid, :, :], axis=-1)

    print()
    print("Velocity profile at x = nx/2:")
    print(f"  {'y':>3s}  {'u_x (LBM)':>12s}  {'u_x (analytic)':>14s}")

    # Analytic parabola.
    H = float(ny - 2)
    u_max = float(jnp.max(ux_profile[1:-1]))
    for j in range(ny):
        ux_sim = float(ux_profile[j])
        if j == 0 or j == ny - 1:
            ux_ana = 0.0
        else:
            y = j - 0.5  # distance from bottom wall midpoint
            ux_ana = u_max * 4.0 * y * (H - y) / (H * H)
        print(f"  {j:3d}  {ux_sim:12.6f}  {ux_ana:14.6f}")


if __name__ == "__main__":
    main()
