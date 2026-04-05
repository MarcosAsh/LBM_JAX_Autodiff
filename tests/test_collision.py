"""Tests for collision operators: BGK, MRT, and Smagorinsky.

Verifies equilibrium fixed-point behavior (collision should be an identity
at equilibrium), MRT moment transform round-trip, and Smagorinsky tau
modification.
"""

import jax.numpy as jnp
import pytest

from jax_lbm.core.lattice import (
    D3Q19_WEIGHTS,
    Q,
    mrt_forward,
    mrt_inverse,
    mrt_equilibrium_moments,
)
from jax_lbm.core.equilibrium import equilibrium, compute_density, compute_velocity
from jax_lbm.core.collision import bgk, mrt, smagorinsky_tau


class TestMRTTransforms:
    """MRT forward/inverse transform correctness."""

    def test_roundtrip_weights(self):
        """Forward then inverse on the weight vector should recover it."""
        f_cell = D3Q19_WEIGHTS
        m = mrt_forward(f_cell)
        f_recovered = mrt_inverse(m)
        assert jnp.allclose(f_cell, f_recovered, atol=1e-5)

    def test_roundtrip_random(self):
        """Forward then inverse on a random vector should recover it."""
        key = jax.random.PRNGKey(42)
        f_cell = jax.random.uniform(key, (19,), minval=0.0, maxval=0.1)
        m = mrt_forward(f_cell)
        f_recovered = mrt_inverse(m)
        assert jnp.allclose(f_cell, f_recovered, atol=1e-5)

    def test_conserved_moments_match_macroscopic(self):
        """m[0] should be rho, m[3]/m[5]/m[7] should be rho*ux/uy/uz."""
        rho_val = 1.3
        u_val = jnp.array([0.04, -0.01, 0.02])
        rho = jnp.full((4, 4, 4), rho_val)
        u = jnp.zeros((3, 4, 4, 4))
        u = u.at[0].set(u_val[0])
        u = u.at[1].set(u_val[1])
        u = u.at[2].set(u_val[2])
        f_eq = equilibrium(rho, u)
        f_cell = f_eq[:, 0, 0, 0]
        m = mrt_forward(f_cell)
        assert jnp.allclose(m[0], rho_val, atol=1e-5)
        assert jnp.allclose(m[3], rho_val * u_val[0], atol=1e-5)
        assert jnp.allclose(m[5], rho_val * u_val[1], atol=1e-5)
        assert jnp.allclose(m[7], rho_val * u_val[2], atol=1e-5)

    def test_equilibrium_moments(self):
        """mrt_equilibrium_moments should match forward transform of f_eq."""
        rho_val = 1.1
        u_val = jnp.array([0.03, 0.01, -0.02])
        rho = jnp.full((4, 4, 4), rho_val)
        u = jnp.zeros((3, 4, 4, 4))
        u = u.at[0].set(u_val[0])
        u = u.at[1].set(u_val[1])
        u = u.at[2].set(u_val[2])
        f_eq = equilibrium(rho, u)
        f_cell = f_eq[:, 0, 0, 0]
        m_from_f = mrt_forward(f_cell)
        m_eq = mrt_equilibrium_moments(rho_val, u_val)
        assert jnp.allclose(m_from_f, m_eq, atol=1e-4)


class TestBGK:
    """BGK collision operator tests."""

    def test_equilibrium_fixed_point(self):
        """Collision at equilibrium should return the same distributions."""
        nx, ny, nz = 8, 8, 8
        rho = jnp.ones((nx, ny, nz))
        u = jnp.zeros((3, nx, ny, nz)).at[0].set(0.05)
        f = equilibrium(rho, u)
        f_post = bgk(f, tau=jnp.float32(0.8))
        assert jnp.allclose(f, f_post, atol=1e-5)

    def test_conserves_mass(self):
        """BGK must conserve mass (total density unchanged)."""
        nx, ny, nz = 8, 8, 8
        rho = jnp.ones((nx, ny, nz))
        u = jnp.zeros((3, nx, ny, nz)).at[0].set(0.05)
        f = equilibrium(rho, u)
        # Perturb away from equilibrium.
        key = jax.random.PRNGKey(0)
        perturbation = jax.random.normal(key, f.shape) * 0.001
        perturbation = perturbation - jnp.mean(perturbation, axis=0, keepdims=True)
        f_perturbed = f + perturbation
        f_post = bgk(f_perturbed, tau=jnp.float32(0.8))
        rho_before = compute_density(f_perturbed)
        rho_after = compute_density(f_post)
        assert jnp.allclose(rho_before, rho_after, atol=1e-5)

    def test_conserves_momentum(self):
        """BGK must conserve momentum."""
        nx, ny, nz = 8, 8, 8
        rho = jnp.ones((nx, ny, nz))
        u = jnp.zeros((3, nx, ny, nz)).at[0].set(0.05)
        f = equilibrium(rho, u)
        key = jax.random.PRNGKey(1)
        perturbation = jax.random.normal(key, f.shape) * 0.001
        perturbation = perturbation - jnp.mean(perturbation, axis=0, keepdims=True)
        f_perturbed = f + perturbation
        rho_before = compute_density(f_perturbed)
        u_before = compute_velocity(f_perturbed, rho_before)
        f_post = bgk(f_perturbed, tau=jnp.float32(0.8))
        rho_after = compute_density(f_post)
        u_after = compute_velocity(f_post, rho_after)
        # Momentum = rho * u should be conserved.
        assert jnp.allclose(rho_before * u_before, rho_after * u_after, atol=1e-5)


class TestMRT:
    """MRT collision operator tests."""

    def test_equilibrium_fixed_point(self):
        """MRT collision at equilibrium should be the identity."""
        nx, ny, nz = 4, 4, 4
        rho = jnp.ones((nx, ny, nz))
        u = jnp.zeros((3, nx, ny, nz)).at[0].set(0.03)
        f = equilibrium(rho, u)
        f_post = mrt(f, tau=jnp.float32(0.8))
        assert jnp.allclose(f, f_post, atol=1e-4)


class TestSmagorinsky:
    """Smagorinsky SGS model tests."""

    def test_at_equilibrium_tau_unchanged(self):
        """At equilibrium, f_neq = 0, so tau_eff should equal tau."""
        nx, ny, nz = 8, 8, 8
        rho = jnp.ones((nx, ny, nz))
        u = jnp.zeros((3, nx, ny, nz)).at[0].set(0.05)
        f = equilibrium(rho, u)
        tau_eff = smagorinsky_tau(f, tau=jnp.float32(0.8), cs=0.1)
        assert jnp.allclose(tau_eff, 0.8, atol=1e-5)

    def test_tau_eff_geq_tau(self):
        """Smagorinsky can only increase tau, never decrease it."""
        nx, ny, nz = 8, 8, 8
        rho = jnp.ones((nx, ny, nz))
        u = jnp.zeros((3, nx, ny, nz)).at[0].set(0.05)
        f = equilibrium(rho, u)
        key = jax.random.PRNGKey(2)
        f_perturbed = f + jax.random.normal(key, f.shape) * 0.01
        tau_eff = smagorinsky_tau(f_perturbed, tau=jnp.float32(0.6), cs=0.1)
        assert jnp.all(tau_eff >= 0.6 - 1e-6)


# Need jax.random for some tests.
import jax
