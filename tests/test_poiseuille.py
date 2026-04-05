"""Poiseuille channel flow validation against the analytic solution.

A long channel with no-slip walls on top and bottom (y-direction), driven
by a Zou-He velocity inlet. The steady-state velocity profile is a parabola:

    u_x(y) = 4 * u_max * y * (H - y) / H^2

where H is the channel height and u_max is the centerline velocity.

This is the first quantitative physics validation beyond unit tests. The
LBM solution should converge to the analytic profile as the simulation
reaches steady state.

For a channel of height H lattice cells (measured between the wall
midpoints, so H = ny - 2 for walls at y=0 and y=ny-1), driven by a
uniform inlet velocity U, the centerline velocity overshoots U slightly
during the transient but settles to approximately U at steady state in a
long enough channel.

We measure the velocity profile at the channel midpoint (x = nx/2) and
compare against the fitted parabola. The L2 error should be below 5%
for a reasonable grid.
"""

import jax
import jax.numpy as jnp
import pytest

from jax_lbm.core.lattice import Q
from jax_lbm.core.equilibrium import equilibrium, compute_density, compute_velocity
from jax_lbm.core.collision import bgk
from jax_lbm.core.streaming import stream
from jax_lbm.core.boundary import zou_he_inlet, zou_he_outlet
from jax_lbm.geometry.bouzidi import compute_q_from_solid


def _run_poiseuille(nx, ny, nz, u_inlet, tau, n_steps):
    """Run a Poiseuille channel flow simulation and return the velocity profile.

    Sets up a channel with solid walls at y=0 and y=ny-1, fluid everywhere
    else. Zou-He inlet at x=0, pressure outlet at x=nx-1. Periodic in z.

    Returns the x-velocity profile at x=nx//2 averaged over z.
    """
    # Solid walls at top and bottom of Y.
    solid = jnp.zeros((nx, ny, nz), dtype=jnp.int32)
    solid = solid.at[:, 0, :].set(1)       # bottom wall
    solid = solid.at[:, ny - 1, :].set(1)  # top wall

    # Bouzidi q for wall-adjacent fluid cells.
    q = compute_q_from_solid(solid, nx, ny, nz, default_q=0.5)

    # Initialize at rest.
    rho = jnp.ones((nx, ny, nz))
    u = jnp.zeros((3, nx, ny, nz))
    f = equilibrium(rho, u)

    vel = jnp.array([u_inlet, 0.0, 0.0])

    import numpy as np
    from jax_lbm.core.lattice import D3Q19_OPPOSITE
    opp_arr = jnp.array([int(x) for x in np.asarray(D3Q19_OPPOSITE)])
    is_solid = (solid >= 1)[jnp.newaxis, :, :, :]

    def step_fn(f_in, _):
        f_c = bgk(f_in, tau)
        f_c = jnp.where(is_solid, f_c[opp_arr], f_c)
        f_c = stream(f_c, q, periodic_yz=True)
        f_c = zou_he_inlet(f_c, vel)
        f_c = zou_he_outlet(f_c, rho_out=1.0)
        return f_c, None

    f, _ = jax.lax.scan(step_fn, f, None, length=n_steps)

    # Extract velocity profile at channel midpoint.
    rho_final = compute_density(f)
    u_final = compute_velocity(f, rho_final)

    # Average ux over z at x = nx // 2.
    x_mid = nx // 2
    ux_profile = jnp.mean(u_final[0, x_mid, :, :], axis=-1)  # (ny,)

    return ux_profile


class TestPoiseuille:
    """Poiseuille channel flow validation."""

    def test_parabolic_profile(self):
        """Steady-state velocity profile should be approximately parabolic.

        We fit a parabola to the simulated profile (excluding wall cells)
        and check that the relative L2 error is below 10%. On a coarse
        grid (ny=16) with only 2000 steps, we don't expect perfect
        agreement, but the shape should clearly be parabolic.
        """
        nx, ny, nz = 40, 16, 4
        u_inlet = 0.05
        tau = jnp.float32(0.8)
        n_steps = 2000

        ux = _run_poiseuille(nx, ny, nz, u_inlet, tau, n_steps)

        # Fluid cells are y=1 to y=ny-2.
        ux_fluid = ux[1:-1]
        n_fluid = ny - 2  # number of fluid cells

        # Analytic parabola: u(y) = u_max * 4 * y * (H - y) / H^2
        # where y is measured from 0 to H across the fluid region,
        # with y=0 at the first fluid cell center (y_index=1, so half a
        # cell from the wall midpoint). H = n_fluid (wall-to-wall distance
        # in lattice units, between midpoints of wall cells).
        H = float(n_fluid)
        y = jnp.linspace(0.5, H - 0.5, n_fluid)  # cell centers
        u_max = float(jnp.max(ux_fluid))
        u_analytic = u_max * 4.0 * y * (H - y) / (H * H)

        # Relative L2 error.
        l2_error = float(jnp.sqrt(
            jnp.sum((ux_fluid - u_analytic) ** 2) / jnp.sum(u_analytic ** 2)
        ))

        # The profile should be parabolic. Accept 10% error on this
        # coarse grid with a short run.
        assert l2_error < 0.10, f"Poiseuille L2 error {l2_error:.3f} exceeds 10%"

    def test_no_slip_at_walls(self):
        """Velocity at wall cells should be zero (or very close)."""
        nx, ny, nz = 40, 16, 4
        u_inlet = 0.05
        tau = jnp.float32(0.8)
        n_steps = 1000

        ux = _run_poiseuille(nx, ny, nz, u_inlet, tau, n_steps)

        # Wall cells at y=0 and y=ny-1 should have near-zero velocity.
        assert jnp.abs(ux[0]) < 0.01, f"Bottom wall velocity {float(ux[0]):.4f}"
        assert jnp.abs(ux[-1]) < 0.01, f"Top wall velocity {float(ux[-1]):.4f}"

    def test_symmetric_profile(self):
        """The profile should be symmetric about the channel centerline."""
        nx, ny, nz = 40, 16, 4
        u_inlet = 0.05
        tau = jnp.float32(0.8)
        n_steps = 1500

        ux = _run_poiseuille(nx, ny, nz, u_inlet, tau, n_steps)

        ux_fluid = ux[1:-1]
        n = len(ux_fluid)

        # Compare first half against reversed second half.
        first_half = ux_fluid[:n // 2]
        second_half = ux_fluid[n // 2:][::-1]
        max_asymmetry = float(jnp.max(jnp.abs(first_half - second_half)))
        mean_velocity = float(jnp.mean(ux_fluid))

        # Relative asymmetry should be small.
        rel_asymmetry = max_asymmetry / (mean_velocity + 1e-10)
        assert rel_asymmetry < 0.15, f"Asymmetry {rel_asymmetry:.3f} exceeds 15%"
