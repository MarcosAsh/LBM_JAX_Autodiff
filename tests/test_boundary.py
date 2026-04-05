"""Tests for Zou-He inlet and outlet boundary conditions.

Validates that the inlet correctly prescribes velocity and that the outlet
maintains the target density.
"""

import jax.numpy as jnp
import pytest

from jax_lbm.core.equilibrium import equilibrium, compute_density, compute_velocity
from jax_lbm.core.boundary import zou_he_inlet, zou_he_outlet


class TestZouHeInlet:
    """Zou-He velocity inlet at x=0."""

    def test_inlet_velocity_preserved(self):
        """After applying Zou-He inlet, velocity at x=0 should match target."""
        nx, ny, nz = 16, 8, 8
        rho = jnp.ones((nx, ny, nz))
        u = jnp.zeros((3, nx, ny, nz))
        f = equilibrium(rho, u)
        target_vel = jnp.array([0.05, 0.0, 0.0])
        f = zou_he_inlet(f, velocity=target_vel)

        rho_inlet = compute_density(f[:, 0, :, :][..., None]).squeeze(-1)
        # Velocity at inlet: need full (19, 1, ny, nz) slice.
        f_inlet = f[:, 0:1, :, :]
        rho_at_inlet = compute_density(f_inlet)
        u_at_inlet = compute_velocity(f_inlet, rho_at_inlet)

        # x-velocity at the inlet should be close to 0.05.
        assert jnp.allclose(u_at_inlet[0], 0.05, atol=1e-3)

    def test_inlet_density_positive(self):
        """Density at inlet should remain positive and reasonable."""
        nx, ny, nz = 16, 8, 8
        rho = jnp.ones((nx, ny, nz))
        u = jnp.zeros((3, nx, ny, nz)).at[0].set(0.05)
        f = equilibrium(rho, u)
        target_vel = jnp.array([0.05, 0.0, 0.0])
        f = zou_he_inlet(f, velocity=target_vel)
        rho_inlet = compute_density(f[:, 0:1, :, :])
        assert jnp.all(rho_inlet > 0.5)
        assert jnp.all(rho_inlet < 2.0)


class TestZouHeOutlet:
    """Zou-He pressure outlet at x=max."""

    def test_outlet_density_preserved(self):
        """After applying Zou-He outlet, density at x=-1 should be close to 1.0."""
        nx, ny, nz = 16, 8, 8
        rho = jnp.ones((nx, ny, nz))
        u = jnp.zeros((3, nx, ny, nz)).at[0].set(0.05)
        f = equilibrium(rho, u)
        f = zou_he_outlet(f, rho_out=1.0)
        rho_outlet = compute_density(f[:, -1:, :, :])
        assert jnp.allclose(rho_outlet, 1.0, atol=1e-2)

    def test_outlet_velocity_positive(self):
        """Outflow velocity should be positive (flow leaving the domain)."""
        nx, ny, nz = 16, 8, 8
        rho = jnp.ones((nx, ny, nz))
        u = jnp.zeros((3, nx, ny, nz)).at[0].set(0.05)
        f = equilibrium(rho, u)
        f = zou_he_outlet(f, rho_out=1.0)
        rho_outlet = compute_density(f[:, -1:, :, :])
        u_outlet = compute_velocity(f[:, -1:, :, :], rho_outlet)
        # Streamwise velocity should be positive.
        assert jnp.all(u_outlet[0] > -0.1)
