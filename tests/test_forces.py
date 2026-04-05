"""Tests for momentum exchange force computation and drag coefficient.

Validates that the force computation returns zero when there are no walls,
produces nonzero forces when walls exist, and that the drag coefficient
formula is correct.
"""

import jax.numpy as jnp
import pytest

from jax_lbm.core.equilibrium import equilibrium
from jax_lbm.core.forces import (
    momentum_exchange,
    momentum_exchange_decomposed,
    projected_area,
    drag_coefficient,
)


class TestMomentumExchange:
    """Momentum exchange force computation."""

    def test_no_walls_zero_force(self):
        """With no boundary links (q = -1 everywhere), force should be zero."""
        nx, ny, nz = 16, 8, 8
        rho = jnp.ones((nx, ny, nz))
        u = jnp.zeros((3, nx, ny, nz)).at[0].set(0.05)
        f = equilibrium(rho, u)
        q = -jnp.ones((19, nx, ny, nz))
        force = momentum_exchange(f, f, q)
        assert jnp.allclose(force, 0.0, atol=1e-10)

    def test_with_walls_nonzero_force(self):
        """With boundary links, force should be nonzero."""
        nx, ny, nz = 16, 8, 8
        rho = jnp.ones((nx, ny, nz))
        u = jnp.zeros((3, nx, ny, nz)).at[0].set(0.05)
        f = equilibrium(rho, u)
        q = -jnp.ones((19, nx, ny, nz))
        # Create a wall link: direction 1 from (7,:,:) hits a solid.
        q = q.at[1, 7, :, :].set(0.5)
        force = momentum_exchange(f, f, q)
        assert jnp.sum(jnp.abs(force)) > 0.0

    def test_decomposition_sums(self):
        """total = pressure + friction (pressure from equilibrium splitting)."""
        nx, ny, nz = 16, 8, 8
        rho = jnp.ones((nx, ny, nz))
        u = jnp.zeros((3, nx, ny, nz)).at[0].set(0.05)
        f = equilibrium(rho, u)
        q = -jnp.ones((19, nx, ny, nz))
        q = q.at[1, 7, :, :].set(0.5)
        total, pressure = momentum_exchange_decomposed(f, f, q)
        friction = total - pressure
        # At equilibrium, friction should be near zero (f_neq = 0).
        assert jnp.allclose(friction, 0.0, atol=1e-5)


class TestProjectedArea:
    """Projected area computation."""

    def test_box_projection(self):
        """A 3x4x4 box projected onto Y-Z should give 4*4 = 16."""
        solid = jnp.zeros((16, 8, 8), dtype=jnp.int32)
        solid = solid.at[5:8, 2:6, 2:6].set(1)
        area = projected_area(solid, axis=0)
        assert jnp.allclose(area, 16.0, atol=0.5)

    def test_empty_domain(self):
        """No solid cells means zero projected area."""
        solid = jnp.zeros((16, 8, 8), dtype=jnp.int32)
        area = projected_area(solid, axis=0)
        assert jnp.allclose(area, 0.0)

    def test_ground_excluded(self):
        """Solid value 2 (ground) should not count toward projected area."""
        solid = jnp.zeros((16, 8, 8), dtype=jnp.int32)
        solid = solid.at[0:16, 0, :].set(2)  # ground plane
        area = projected_area(solid, axis=0)
        assert jnp.allclose(area, 0.0)


class TestDragCoefficient:
    """Drag coefficient formula."""

    def test_known_values(self):
        """Verify Cd = |Fx| / (0.5 * rho * U^2 * A) with known numbers."""
        force = jnp.array([0.01, 0.0, 0.0])
        cd = drag_coefficient(force, inlet_velocity=0.05, area=100.0)
        # Cd = 0.01 / (0.5 * 1.0 * 0.0025 * 100) = 0.01 / 0.125 = 0.08
        assert jnp.allclose(cd, 0.08, atol=1e-5)
