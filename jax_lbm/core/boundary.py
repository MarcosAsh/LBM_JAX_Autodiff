"""Zou-He velocity inlet, pressure outlet, and periodic boundary conditions.

In an external flow simulation (wind tunnel setup), you need to prescribe
how fluid enters and exits the domain. The Zou-He method handles this by
computing the unknown distribution functions at boundary faces from the
known ones plus the prescribed macroscopic quantities.

At the inlet (x = 0), you prescribe the velocity. The method computes
the density from the known distributions and then fills in the five
distributions that point into the domain (+x directions) using
non-equilibrium bounce-back.

At the outlet (x = nx-1), you prescribe the pressure (density = 1.0,
atmospheric). The method computes the outflow velocity from the known
distributions and fills in the five distributions pointing back into the
domain (-x directions).

The Y and Z boundaries are typically periodic (handled in the streaming
step) or free-slip (also handled in streaming via clamping). This module
provides a convenience function for explicitly copying periodic
distributions if needed outside the streaming context.

References:
    Zou, Q. and He, X. (1997). "On pressure and velocity boundary
    conditions for the lattice Boltzmann BGK model." Physics of Fluids,
    9(6), pp.1591-1598.
"""

import jax
import jax.numpy as jnp

from jax_lbm.core.lattice import D3Q19_VELOCITIES, D3Q19_WEIGHTS


def zou_he_inlet(f: jax.Array, velocity: jax.Array, rho_target: float = None) -> jax.Array:
    """Apply Zou-He velocity boundary condition at the x=0 inlet.

    Computes the unknown distribution functions (those pointing into the
    domain) from the prescribed velocity using the non-equilibrium
    bounce-back method. Density at the inlet is derived from the known
    distributions and the target velocity.

    The five unknown distributions at x=0 are directions 1 (+x), 7 (+x+y),
    9 (+x-y), 11 (+x+z), and 13 (+x-z). These are the directions with
    a positive x-component, which stream into the domain from the left wall.

    This implementation follows Zou & He (1997), adapted for D3Q19. The
    boundary is applied to all cells at x=0.

    Args:
        f: Distribution functions, shape ``(19, nx, ny, nz)``.
        velocity: Inlet velocity vector ``(ux, uy, uz)`` as a 1D array
            of shape ``(3,)``. Applied uniformly across the entire inlet
            face.
        rho_target: If provided, use this density instead of computing it
            from the distributions. Useful for initialization.

    Returns:
        Updated distribution array with inlet distributions replaced.

    Example:

    ```python
    >>> import jax.numpy as jnp
    >>> from jax_lbm.core.equilibrium import equilibrium
    >>> from jax_lbm.core.boundary import zou_he_inlet
    >>> nx, ny, nz = 16, 8, 8
    >>> rho = jnp.ones((nx, ny, nz))
    >>> u = jnp.zeros((3, nx, ny, nz))
    >>> f = equilibrium(rho, u)
    >>> f = zou_he_inlet(f, velocity=jnp.array([0.05, 0.0, 0.0]))
    >>> f.shape
    (19, 16, 8, 8)
    ```

    References:
        Zou, Q. and He, X. (1997). "On pressure and velocity boundary
        conditions for the lattice Boltzmann BGK model." Physics of
        Fluids, 9(6), pp.1591-1598.
    """
    ux = velocity[0]
    uy = velocity[1]
    uz = velocity[2]

    # Read known distributions at x=0.
    # Directions with zero x-component (known from interior streaming or
    # periodic Y/Z): 0, 3, 4, 5, 6, 15, 16, 17, 18
    # Directions with negative x-component (known, streamed from interior):
    # 2 (-x), 8 (-x+y), 10 (-x-y), 12 (-x+z), 14 (-x-z)
    f0  = f[0,  0, :, :]
    f2  = f[2,  0, :, :]
    f3  = f[3,  0, :, :]
    f4  = f[4,  0, :, :]
    f5  = f[5,  0, :, :]
    f6  = f[6,  0, :, :]
    f8  = f[8,  0, :, :]
    f10 = f[10, 0, :, :]
    f12 = f[12, 0, :, :]
    f14 = f[14, 0, :, :]
    f15 = f[15, 0, :, :]
    f16 = f[16, 0, :, :]
    f17 = f[17, 0, :, :]
    f18 = f[18, 0, :, :]

    knowns_0   = f0 + f3 + f4 + f5 + f6 + f15 + f16 + f17 + f18
    knowns_neg = f2 + f8 + f10 + f12 + f14

    # Density from the constraint that all 19 distributions must sum to rho,
    # and the x-momentum must equal rho * ux.
    if rho_target is not None:
        rho_in = rho_target
    else:
        rho_in = (knowns_0 + 2.0 * knowns_neg) / (1.0 - ux)

    # Unknown distributions (positive x-component), from non-equilibrium
    # bounce-back. Each unknown is its opposite-direction known plus a
    # correction proportional to rho * velocity.
    f = f.at[1,  0, :, :].set(f2  + (1.0 / 3.0) * rho_in * ux)
    f = f.at[7,  0, :, :].set(f10 + (1.0 / 6.0) * rho_in * (ux + uy))
    f = f.at[9,  0, :, :].set(f8  + (1.0 / 6.0) * rho_in * (ux - uy))
    f = f.at[11, 0, :, :].set(f14 + (1.0 / 6.0) * rho_in * (ux + uz))
    f = f.at[13, 0, :, :].set(f12 + (1.0 / 6.0) * rho_in * (ux - uz))

    return f


def zou_he_outlet(f: jax.Array, rho_out: float = 1.0) -> jax.Array:
    """Apply Zou-He pressure boundary condition at the x=max outlet.

    Prescribes atmospheric pressure (rho = 1.0) at the right boundary.
    The outflow velocity is computed from the known distributions, and
    the five unknown distributions (those with negative x-component,
    pointing back into the domain) are filled in.

    This assumes fully developed flow at the outlet: uy = uz = 0. Only
    the streamwise component ux is computed from the distributions. This
    is standard practice for channel and wind-tunnel simulations where
    the outlet is far enough from the obstacle.

    Args:
        f: Distribution functions, shape ``(19, nx, ny, nz)``.
        rho_out: Target density at the outlet. Default 1.0 (atmospheric
            pressure in lattice units).

    Returns:
        Updated distribution array with outlet distributions replaced.

    Example:

    ```python
    >>> import jax.numpy as jnp
    >>> from jax_lbm.core.equilibrium import equilibrium
    >>> from jax_lbm.core.boundary import zou_he_outlet
    >>> nx, ny, nz = 16, 8, 8
    >>> rho = jnp.ones((nx, ny, nz))
    >>> u = jnp.zeros((3, nx, ny, nz)).at[0].set(0.05)
    >>> f = equilibrium(rho, u)
    >>> f = zou_he_outlet(f, rho_out=1.0)
    >>> f.shape
    (19, 16, 8, 8)
    ```

    References:
        Zou, Q. and He, X. (1997). "On pressure and velocity boundary
        conditions for the lattice Boltzmann BGK model." Physics of
        Fluids, 9(6), pp.1591-1598.
    """
    # Known distributions at x = nx-1.
    # Directions with zero x-component: 0, 3, 4, 5, 6, 15, 16, 17, 18
    # Directions with positive x-component (known, streamed from interior):
    # 1 (+x), 7 (+x+y), 9 (+x-y), 11 (+x+z), 13 (+x-z)
    f0  = f[0,  -1, :, :]
    f1  = f[1,  -1, :, :]
    f3  = f[3,  -1, :, :]
    f4  = f[4,  -1, :, :]
    f5  = f[5,  -1, :, :]
    f6  = f[6,  -1, :, :]
    f7  = f[7,  -1, :, :]
    f9  = f[9,  -1, :, :]
    f11 = f[11, -1, :, :]
    f13 = f[13, -1, :, :]
    f15 = f[15, -1, :, :]
    f16 = f[16, -1, :, :]
    f17 = f[17, -1, :, :]
    f18 = f[18, -1, :, :]

    knowns_0   = f0 + f3 + f4 + f5 + f6 + f15 + f16 + f17 + f18
    knowns_pos = f1 + f7 + f9 + f11 + f13

    # Outflow velocity from the constraint.
    ux = -1.0 + (knowns_0 + 2.0 * knowns_pos) / rho_out

    # Unknown distributions (negative x-component).
    f = f.at[2,  -1, :, :].set(f1  - (1.0 / 3.0) * rho_out * ux)
    f = f.at[10, -1, :, :].set(f7  - (1.0 / 6.0) * rho_out * ux)
    f = f.at[8,  -1, :, :].set(f9  - (1.0 / 6.0) * rho_out * ux)
    f = f.at[14, -1, :, :].set(f11 - (1.0 / 6.0) * rho_out * ux)
    f = f.at[12, -1, :, :].set(f13 - (1.0 / 6.0) * rho_out * ux)

    return f
