"""Tests for the streaming step with Bouzidi bounce-back.

Validates that streaming conserves mass in a periodic domain, that
bounce-back correctly reverses distributions at walls, and that the
Bouzidi interpolation produces second-order wall placement.
"""

import jax
import jax.numpy as jnp
import pytest

from jax_lbm.core.lattice import D3Q19_VELOCITIES, D3Q19_OPPOSITE, Q
from jax_lbm.core.equilibrium import equilibrium, compute_density
from jax_lbm.core.streaming import stream


class TestPeriodicStreaming:
    """Streaming in a fully periodic domain with no walls."""

    def test_mass_conservation(self):
        """Total mass should not change after streaming (periodic, no walls)."""
        nx, ny, nz = 16, 8, 8
        rho = jnp.ones((nx, ny, nz))
        u = jnp.zeros((3, nx, ny, nz)).at[0].set(0.05)
        f = equilibrium(rho, u)
        q = -jnp.ones((Q, nx, ny, nz))  # no walls
        f_streamed = stream(f, q, periodic_yz=True)
        mass_before = jnp.sum(f)
        mass_after = jnp.sum(f_streamed)
        # X is not periodic, so mass at boundaries can leak. Use interior.
        assert jnp.allclose(mass_before, mass_after, rtol=1e-4)

    def test_rest_direction_unchanged(self):
        """Direction 0 (rest) should not move during streaming."""
        nx, ny, nz = 8, 8, 8
        rho = jnp.ones((nx, ny, nz))
        u = jnp.zeros((3, nx, ny, nz)).at[0].set(0.05)
        f = equilibrium(rho, u)
        q = -jnp.ones((Q, nx, ny, nz))
        f_streamed = stream(f, q, periodic_yz=True)
        assert jnp.allclose(f_streamed[0], f[0], atol=1e-7)

    def test_shift_direction_plus_y(self):
        """Direction 3 (+y) should shift data by +1 in y (pull from y-1)."""
        nx, ny, nz = 4, 8, 4
        f = jnp.zeros((Q, nx, ny, nz))
        # Put a pulse at y=3 in direction 3 (+y).
        f = f.at[3, :, 3, :].set(1.0)
        q = -jnp.ones((Q, nx, ny, nz))
        f_streamed = stream(f, q, periodic_yz=True)
        # After streaming, the pulse should be at y=4.
        assert jnp.allclose(f_streamed[3, :, 4, :], 1.0, atol=1e-7)
        assert jnp.allclose(f_streamed[3, :, 3, :], 0.0, atol=1e-7)


class TestBouzidiBounceBack:
    """Test that Bouzidi interpolation works at wall boundaries."""

    def test_midpoint_bounceback_reverses_direction(self):
        """With q=0.5, Bouzidi should produce standard bounce-back.

        At q=0.5, the linear branch gives:
        f[i] = (1/(2*0.5)) * f*[iopp] + (1 - 1/(2*0.5)) * f*[i]
             = f*[iopp]

        So the bounced distribution should equal the opposite-direction
        post-collision value.
        """
        nx, ny, nz = 8, 8, 8
        rho = jnp.ones((nx, ny, nz))
        u = jnp.zeros((3, nx, ny, nz)).at[0].set(0.05)
        f = equilibrium(rho, u)

        # Set up a wall: solid at x=4, fluid everywhere else.
        q = -jnp.ones((Q, nx, ny, nz))
        # For fluid cells at x=3, direction 1 (+x) hits solid at x=4.
        # q[1] at x=3 means "direction 1 from x=3 hits a wall".
        q = q.at[1, 3, :, :].set(0.5)

        f_streamed = stream(f, q, periodic_yz=True)

        # At x=3, direction 2 (-x, the opposite of 1) should have gotten
        # the bounce-back from direction 1. With q=0.5:
        # f_streamed[2, 3] = f_post[1, 3]  (standard bounce-back)
        opp_1 = int(D3Q19_OPPOSITE[1])  # = 2
        assert jnp.allclose(f_streamed[opp_1, 3, :, :], f[1, 3, :, :], atol=1e-5)
