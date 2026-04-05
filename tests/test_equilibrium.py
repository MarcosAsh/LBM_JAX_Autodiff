"""Tests for equilibrium distribution, density, and velocity computation.

Validates that the equilibrium conserves mass and momentum, that density
and velocity extraction round-trip correctly, and that the weights are
self-consistent.
"""

import jax.numpy as jnp
import pytest

from jax_lbm.core.lattice import D3Q19_WEIGHTS, D3Q19_VELOCITIES
from jax_lbm.core.equilibrium import equilibrium, compute_density, compute_velocity


class TestWeights:
    """D3Q19 weight sanity checks."""

    def test_weights_sum_to_one(self):
        """Weights must sum to exactly 1 (lattice normalization)."""
        assert jnp.allclose(jnp.sum(D3Q19_WEIGHTS), 1.0, atol=1e-7)

    def test_weight_count(self):
        """Must have exactly 19 weights."""
        assert D3Q19_WEIGHTS.shape == (19,)

    def test_rest_weight(self):
        """Rest direction (index 0) has weight 1/3."""
        assert jnp.allclose(D3Q19_WEIGHTS[0], 1.0 / 3.0, atol=1e-7)

    def test_face_weights(self):
        """Face neighbors (indices 1-6) each have weight 1/18."""
        for i in range(1, 7):
            assert jnp.allclose(D3Q19_WEIGHTS[i], 1.0 / 18.0, atol=1e-7)

    def test_edge_weights(self):
        """Edge neighbors (indices 7-18) each have weight 1/36."""
        for i in range(7, 19):
            assert jnp.allclose(D3Q19_WEIGHTS[i], 1.0 / 36.0, atol=1e-7)


class TestEquilibriumConservation:
    """Verify that the equilibrium conserves density and momentum."""

    @pytest.fixture
    def grid(self):
        nx, ny, nz = 8, 8, 8
        rho = jnp.ones((nx, ny, nz)) * 1.5
        u = jnp.zeros((3, nx, ny, nz))
        u = u.at[0].set(0.04)
        u = u.at[1].set(-0.01)
        u = u.at[2].set(0.02)
        return rho, u

    def test_density_conservation(self, grid):
        """sum(f_eq) must equal rho at every cell."""
        rho, u = grid
        f_eq = equilibrium(rho, u)
        rho_recovered = compute_density(f_eq)
        assert jnp.allclose(rho_recovered, rho, atol=1e-5)

    def test_momentum_conservation(self, grid):
        """sum(f_eq * e_i) must equal rho * u at every cell."""
        rho, u = grid
        f_eq = equilibrium(rho, u)
        u_recovered = compute_velocity(f_eq, rho)
        assert jnp.allclose(u_recovered, u, atol=1e-5)

    def test_zero_velocity_equals_weights(self):
        """At rest (u=0, rho=1), f_eq should equal the lattice weights."""
        nx, ny, nz = 4, 4, 4
        rho = jnp.ones((nx, ny, nz))
        u = jnp.zeros((3, nx, ny, nz))
        f_eq = equilibrium(rho, u)
        expected = D3Q19_WEIGHTS[:, None, None, None] * jnp.ones((1, nx, ny, nz))
        assert jnp.allclose(f_eq, expected, atol=1e-6)


class TestDensityVelocityRoundTrip:
    """Verify that density/velocity extraction inverts equilibrium."""

    def test_roundtrip_uniform(self):
        """Uniform rho and u should survive equilibrium -> extract."""
        nx, ny, nz = 6, 6, 6
        rho = jnp.full((nx, ny, nz), 1.2)
        u = jnp.zeros((3, nx, ny, nz))
        u = u.at[0].set(0.03)
        f_eq = equilibrium(rho, u)
        assert jnp.allclose(compute_density(f_eq), rho, atol=1e-5)
        assert jnp.allclose(compute_velocity(f_eq, rho), u, atol=1e-5)

    def test_roundtrip_varying(self):
        """Spatially varying rho and u should round-trip through equilibrium."""
        nx, ny, nz = 8, 8, 8
        x = jnp.linspace(0.9, 1.1, nx)
        rho = jnp.broadcast_to(x[:, None, None], (nx, ny, nz))
        u = jnp.zeros((3, nx, ny, nz))
        u = u.at[0].set(0.01 * x[:, None, None])
        f_eq = equilibrium(rho, u)
        assert jnp.allclose(compute_density(f_eq), rho, atol=1e-5)
        assert jnp.allclose(compute_velocity(f_eq, rho), u, atol=1e-5)
