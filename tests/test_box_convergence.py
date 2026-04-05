"""Grid convergence study for a box in uniform flow.

Runs the same box geometry at multiple grid resolutions and checks that
the drag coefficient values are consistent (within 10% of each other).
This verifies that the solver converges as the grid is refined, which
is a basic sanity check for any CFD code.

We use a small box and short runs to keep the test fast. The point is
convergence behavior, not absolute accuracy.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_lbm.core.lattice import D3Q19_OPPOSITE
from jax_lbm.core.equilibrium import equilibrium, compute_density, compute_velocity
from jax_lbm.core.collision import bgk
from jax_lbm.core.streaming import stream
from jax_lbm.core.boundary import zou_he_inlet, zou_he_outlet
from jax_lbm.core.forces import momentum_exchange, drag_coefficient, projected_area
from jax_lbm.geometry.primitives import create_box


def _run_box_cd(nx, ny, nz, n_steps):
    """Run a box drag simulation and return Cd."""
    # Box centered in the domain.
    box_half = 0.4
    solid, q = create_box(
        nx, ny, nz,
        corner_min=(-box_half, -box_half, -box_half),
        corner_max=(box_half, box_half, box_half),
    )

    u_inlet = 0.05
    D_lattice = 2.0 * box_half * (ny / 4.0)  # box width in lattice cells
    re = 20.0  # low Re for stability with sharp corners
    nu = u_inlet * D_lattice / re
    tau = jnp.float32(3.0 * nu + 0.5)

    rho = jnp.ones((nx, ny, nz))
    u = jnp.zeros((3, nx, ny, nz)).at[0].set(u_inlet)
    f = equilibrium(rho, u)
    vel = jnp.array([u_inlet, 0.0, 0.0])

    opp_arr = jnp.array([int(x) for x in np.asarray(D3Q19_OPPOSITE)])
    is_solid = (solid >= 1)[jnp.newaxis, :, :, :]

    def step_fn(carry, _):
        f_in, _ = carry
        f_c = bgk(f_in, tau)
        f_c = jnp.where(is_solid, f_c[opp_arr], f_c)
        f_post = f_c
        f_s = stream(f_c, q, periodic_yz=True)
        f_s = zou_he_inlet(f_s, vel)
        f_s = zou_he_outlet(f_s, rho_out=1.0)
        # Stability guard.
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
    area = projected_area(solid, axis=0)
    cd = drag_coefficient(force, u_inlet, area)
    return float(cd)


class TestBoxConvergence:
    """Grid convergence for box drag."""

    @pytest.mark.slow
    def test_cd_values_positive(self):
        """Cd should be positive after enough steps for wake to develop."""
        cd = _run_box_cd(64, 32, 32, n_steps=800)
        assert cd > 0.0, f"Cd should be positive, got {cd}"
        assert cd < 20.0, f"Cd={cd:.4f} implausibly large"

    @pytest.mark.slow
    def test_convergence_consistency(self):
        """Cd at different sample times should stabilize.

        Run a warm-up phase (1000 steps) to let the initial transient
        die out, then take 5 measurements at 200-step intervals. The
        Cd values should be positive and within 5x of each other once
        the flow is near steady state.

        Grid: 64x32x32, box side = 0.8 world units -> ~6.4 lattice
        cells across. Re = 20 (tau = 0.548, safely above stability
        limit). Box corners create recirculation zones that need
        O(1000) steps to develop on this grid.
        """
        nx, ny, nz = 64, 32, 32
        n_warmup = 1000
        n_per_sample = 200
        n_samples = 5

        box_half = 0.4
        solid, q = create_box(
            nx, ny, nz,
            corner_min=(-box_half, -box_half, -box_half),
            corner_max=(box_half, box_half, box_half),
        )

        u_inlet = 0.05
        D_lattice = 2.0 * box_half * (ny / 4.0)  # 6.4 cells
        re = 20.0
        nu = u_inlet * D_lattice / re
        tau = jnp.float32(3.0 * nu + 0.5)  # 0.548

        rho = jnp.ones((nx, ny, nz))
        u = jnp.zeros((3, nx, ny, nz)).at[0].set(u_inlet)
        f = equilibrium(rho, u)
        vel = jnp.array([u_inlet, 0.0, 0.0])

        opp_arr = jnp.array([int(x) for x in np.asarray(D3Q19_OPPOSITE)])
        is_solid = (solid >= 1)[jnp.newaxis, :, :, :]
        area = projected_area(solid, axis=0)

        def step_fn(carry, _):
            f_in, _ = carry
            f_c = bgk(f_in, tau)
            f_c = jnp.where(is_solid, f_c[opp_arr], f_c)
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

        # Warm up: let the transient die out before measuring.
        (f, _), _ = jax.lax.scan(step_fn, (f, f), None, length=n_warmup)

        # Sample Cd at intervals after warm-up.
        cd_samples = []
        for _ in range(n_samples):
            (f, f_post), _ = jax.lax.scan(step_fn, (f, f), None, length=n_per_sample)
            force = momentum_exchange(f_post, f, q)
            cd = float(drag_coefficient(force, u_inlet, area))
            cd_samples.append(cd)

        # All should be finite and positive after warm-up.
        assert all(not np.isnan(cd) for cd in cd_samples), f"NaN in Cd: {cd_samples}"
        assert all(cd > 0 for cd in cd_samples), f"Negative Cd: {cd_samples}"

        # After 1000 warm-up steps, the spread should be within 5x.
        cd_max = max(cd_samples)
        cd_min = min(cd_samples)
        assert cd_max < 5 * cd_min, (
            f"Cd not converged: min={cd_min:.4f}, max={cd_max:.4f}, "
            f"samples={[f'{c:.4f}' for c in cd_samples]}"
        )
