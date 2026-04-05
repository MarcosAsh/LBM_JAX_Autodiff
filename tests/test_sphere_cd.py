"""Sphere drag coefficient at Re=100 -- validation against Clift et al. (1978).

This is the primary integration test for the solver. A sphere is placed
in a wind tunnel (128x64x64 grid), the simulation runs to approximate
steady state, and the drag coefficient is compared against the
experimentally established value of Cd = 1.09 for a sphere at Re=100.

The reference C/GLSL solver uses a sphere diameter of ~16 lattice cells
on a 128x64x64 grid with 4000 timesteps. We replicate that setup here
but use fewer steps to keep test time manageable -- the Cd should be
in the right ballpark after enough steps for the wake to develop.

Grid setup (matching the reference solver):
- World coordinates: x in [-4, 4], y in [-2, 2], z in [-2, 2]
- Sphere center at (0, 0, 0), radius 0.5 in world units
- At 128x64x64, the sphere diameter is ~16 lattice cells
- U = 0.05 (inlet velocity), Re = U * D / nu = 100
- D = 16 cells, so nu = U * D / Re = 0.05 * 16 / 100 = 0.008
- tau = 3 * nu + 0.5 = 0.524

References:
    Clift, R., Grace, J.R. and Weber, M.E. (1978). "Bubbles, Drops,
    and Particles." Academic Press. Table 5.2: Cd = 1.09 at Re=100.
"""

import jax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_lbm.core.lattice import D3Q19_OPPOSITE, Q
from jax_lbm.core.equilibrium import equilibrium, compute_density, compute_velocity
from jax_lbm.core.collision import bgk
from jax_lbm.core.streaming import stream
from jax_lbm.core.boundary import zou_he_inlet, zou_he_outlet
from jax_lbm.core.forces import momentum_exchange, drag_coefficient, projected_area
from jax_lbm.geometry.primitives import create_sphere


def _run_sphere_drag(nx, ny, nz, n_steps, u_inlet=0.05, re=100.0):
    """Run a sphere drag simulation and return the drag coefficient.

    Args:
        nx, ny, nz: Grid dimensions.
        n_steps: Number of timesteps.
        u_inlet: Inlet velocity magnitude.
        re: Reynolds number.

    Returns:
        Tuple of (cd, force_x) -- drag coefficient and streamwise force.
    """
    # Sphere geometry.
    center = (0.0, 0.0, 0.0)
    radius = 0.5  # world units

    solid, q = create_sphere(nx, ny, nz, center, radius)

    # Physical parameters.
    # D in lattice units: the sphere diameter spans roughly (2*radius) / (8/nx) = nx*radius/4
    # For nx=128, radius=0.5: D = 128*0.5/4 = 16 cells
    scale_x = nx / 8.0
    D_lattice = 2.0 * radius * scale_x
    nu = u_inlet * D_lattice / re
    tau = 3.0 * nu + 0.5

    # Initialize.
    rho = jnp.ones((nx, ny, nz))
    u = jnp.zeros((3, nx, ny, nz))
    u = u.at[0].set(u_inlet)
    f = equilibrium(rho, u)

    vel = jnp.array([u_inlet, 0.0, 0.0])

    opp_arr = jnp.array([int(x) for x in np.asarray(D3Q19_OPPOSITE)])
    is_solid = (solid >= 1)[jnp.newaxis, :, :, :]
    tau_val = jnp.float32(tau)

    def step_fn(f_in, _):
        f_c = bgk(f_in, tau_val)
        f_c = jnp.where(is_solid, f_c[opp_arr], f_c)
        f_post = f_c
        f_s = stream(f_c, q, periodic_yz=True)
        f_s = zou_he_inlet(f_s, vel)
        f_s = zou_he_outlet(f_s, rho_out=1.0)
        return f_s, f_post

    f, f_posts = jax.lax.scan(step_fn, f, None, length=n_steps)
    f_post = jax.tree.map(lambda x: x[-1], f_posts)

    # Compute force and Cd.
    force = momentum_exchange(f_post, f, q)
    area = projected_area(solid, axis=0)
    cd = drag_coefficient(force, u_inlet, area)

    return float(cd), float(force[0])


class TestSphereCd:
    """Sphere drag coefficient validation."""

    @pytest.mark.slow
    def test_sphere_cd_re100(self):
        """Cd for a sphere at Re=100 should be near 1.09 (Clift 1978).

        This uses a 64x32x32 grid (smaller than reference for test speed)
        with 1500 steps. On a coarser grid, we accept wider tolerance --
        the Cd should be positive and within a factor of 2 of the target.
        A properly converged run on 128x64x64 with 4000 steps should hit
        within 10%.
        """
        cd, fx = _run_sphere_drag(64, 32, 32, n_steps=1500, u_inlet=0.05, re=100.0)

        # Cd must be positive (drag opposes flow).
        assert cd > 0.0, f"Cd should be positive, got {cd:.4f}"

        # On this coarse grid, we just check the right order of magnitude.
        # Target: 1.09. Accept 0.3 to 3.0 for a coarse quick test.
        assert 0.3 < cd < 3.0, f"Cd={cd:.4f} is outside plausible range for Re=100 sphere"

    @pytest.mark.slow
    def test_sphere_force_nonzero(self):
        """A sphere in a flow should experience nonzero drag force."""
        cd, fx = _run_sphere_drag(64, 32, 32, n_steps=500, u_inlet=0.05, re=100.0)
        assert abs(fx) > 1e-6, f"Force should be nonzero, got {fx:.6e}"

    @pytest.mark.slow
    def test_sphere_cd_converged(self):
        """Full resolution sphere Cd should be within 20% of Clift value.

        128x64x64 grid, 3000 steps. This is the real validation.
        Skip in CI unless explicitly requested (takes minutes).
        """
        cd, fx = _run_sphere_drag(128, 64, 64, n_steps=3000, u_inlet=0.05, re=100.0)

        clift_cd = 1.09
        rel_error = abs(cd - clift_cd) / clift_cd

        assert rel_error < 0.20, (
            f"Cd={cd:.4f}, Clift reference=1.09, "
            f"relative error={rel_error:.1%} exceeds 20%"
        )
