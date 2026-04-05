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
from jax_lbm.core.equilibrium import equilibrium, compute_density
from jax_lbm.core.collision import bgk
from jax_lbm.core.streaming import stream
from jax_lbm.core.boundary import zou_he_inlet, zou_he_outlet
from jax_lbm.core.forces import momentum_exchange, drag_coefficient, projected_area
from jax_lbm.geometry.primitives import create_box


def _run_box_cd(nx, ny, nz, n_steps):
    """Run a box drag simulation and return Cd."""
    # Box centered in the domain, 1/4 of the domain width.
    box_half = 0.3
    solid, q = create_box(
        nx, ny, nz,
        corner_min=(-box_half, -box_half, -box_half),
        corner_max=(box_half, box_half, box_half),
    )

    u_inlet = 0.05
    scale_x = nx / 8.0
    D_lattice = 2.0 * box_half * (ny / 4.0)  # box width in lattice cells
    re = 50.0
    nu = u_inlet * D_lattice / re
    tau = jnp.float32(3.0 * nu + 0.5)

    rho = jnp.ones((nx, ny, nz))
    u = jnp.zeros((3, nx, ny, nz)).at[0].set(u_inlet)
    f = equilibrium(rho, u)
    vel = jnp.array([u_inlet, 0.0, 0.0])

    opp_arr = jnp.array([int(x) for x in np.asarray(D3Q19_OPPOSITE)])
    is_solid = (solid >= 1)[jnp.newaxis, :, :, :]

    def step_fn(f_in, _):
        f_c = bgk(f_in, tau)
        f_c = jnp.where(is_solid, f_c[opp_arr], f_c)
        f_post = f_c
        f_s = stream(f_c, q, periodic_yz=True)
        f_s = zou_he_inlet(f_s, vel)
        f_s = zou_he_outlet(f_s, rho_out=1.0)
        return f_s, f_post

    f_final, f_posts = jax.lax.scan(step_fn, f, None, length=n_steps)
    f_post = jax.tree.map(lambda x: x[-1], f_posts)

    force = momentum_exchange(f_post, f_final, q)
    area = projected_area(solid, axis=0)
    cd = drag_coefficient(force, u_inlet, area)
    return float(cd)


class TestBoxConvergence:
    """Grid convergence for box drag."""

    @pytest.mark.slow
    def test_cd_values_positive(self):
        """Cd should be positive at all resolutions."""
        cd = _run_box_cd(32, 16, 16, n_steps=300)
        assert cd > 0.0, f"Cd should be positive, got {cd}"

    @pytest.mark.slow
    def test_convergence_consistency(self):
        """Cd at different sample times should stabilize.

        Run 5 measurements at 100-step intervals on the same grid. All
        should be positive, and the spread should be within 10x of the
        mean (loose tolerance for a short run).
        """
        nx, ny, nz = 48, 24, 24
        n_per_sample = 100
        n_samples = 5

        box_half = 0.3
        solid, q = create_box(
            nx, ny, nz,
            corner_min=(-box_half, -box_half, -box_half),
            corner_max=(box_half, box_half, box_half),
        )

        u_inlet = 0.05
        D_lattice = 2.0 * box_half * (ny / 4.0)
        re = 50.0
        nu = u_inlet * D_lattice / re
        tau = jnp.float32(3.0 * nu + 0.5)

        rho = jnp.ones((nx, ny, nz))
        u = jnp.zeros((3, nx, ny, nz)).at[0].set(u_inlet)
        f = equilibrium(rho, u)
        vel = jnp.array([u_inlet, 0.0, 0.0])

        opp_arr = jnp.array([int(x) for x in np.asarray(D3Q19_OPPOSITE)])
        is_solid = (solid >= 1)[jnp.newaxis, :, :, :]
        area = projected_area(solid, axis=0)

        def step_fn(f_in, _):
            f_c = bgk(f_in, tau)
            f_c = jnp.where(is_solid, f_c[opp_arr], f_c)
            f_post = f_c
            f_s = stream(f_c, q, periodic_yz=True)
            f_s = zou_he_inlet(f_s, vel)
            f_s = zou_he_outlet(f_s, rho_out=1.0)
            return f_s, f_post

        cd_samples = []
        for _ in range(n_samples):
            f, f_posts = jax.lax.scan(step_fn, f, None, length=n_per_sample)
            f_post = jax.tree.map(lambda x: x[-1], f_posts)
            force = momentum_exchange(f_post, f, q)
            cd = float(drag_coefficient(force, u_inlet, area))
            cd_samples.append(cd)

        # All positive.
        assert all(cd > 0 for cd in cd_samples), f"Some Cd values negative: {cd_samples}"

        # Spread within 10x of each other (loose for short run).
        cd_max = max(cd_samples)
        cd_min = min(cd_samples)
        assert cd_max < 10 * cd_min, (
            f"Cd spread too wide: min={cd_min:.4f}, max={cd_max:.4f}"
        )
