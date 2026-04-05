"""Equilibrium distribution, density, and velocity computation.

The equilibrium distribution is the cornerstone of the Lattice Boltzmann
Method. It encodes the Maxwell-Boltzmann distribution truncated to second
order in velocity, discretized onto the D3Q19 lattice. Every collision
operator relaxes the distribution toward this equilibrium.

The three functions here form a closed loop: given distributions ``f``,
you can extract macroscopic density and velocity, then reconstruct the
equilibrium that those macroscopic quantities imply.

All functions operate on full 3D fields (not single cells) for efficient
vectorized execution under ``jax.jit``.
"""

import jax
import jax.numpy as jnp

from jax_lbm.core.lattice import D3Q19_VELOCITIES, D3Q19_WEIGHTS, CS2


def compute_density(f: jax.Array) -> jax.Array:
    """Sum distributions over all 19 directions to get macroscopic density.

    Density is the zeroth moment of the distribution function. In a
    well-behaved simulation, rho should stay close to 1.0 everywhere.
    Large deviations (below 0.5 or above 1.5) indicate instability.

    Args:
        f: Distribution functions, shape ``(19, nx, ny, nz)``.

    Returns:
        Density field, shape ``(nx, ny, nz)``.

    Example:

    ```python
    >>> import jax.numpy as jnp
    >>> from jax_lbm.core.equilibrium import compute_density
    >>> from jax_lbm.core.lattice import D3Q19_WEIGHTS
    >>> nx, ny, nz = 4, 4, 4
    >>> f = D3Q19_WEIGHTS[:, None, None, None] * jnp.ones((1, nx, ny, nz))
    >>> rho = compute_density(f)
    >>> float(rho[0, 0, 0])  # weights sum to 1
    1.0
    ```
    """
    return jnp.sum(f, axis=0)


def compute_velocity(f: jax.Array, rho: jax.Array) -> jax.Array:
    """Compute macroscopic velocity from distributions and density.

    Velocity is the first moment of the distribution, divided by density:
    ``u_alpha = (1/rho) * sum_i(f_i * e_{i,alpha})`` for each spatial
    component alpha in {x, y, z}.

    Args:
        f: Distribution functions, shape ``(19, nx, ny, nz)``.
        rho: Density field, shape ``(nx, ny, nz)``. Must be nonzero
            everywhere; the function does not guard against division by
            zero since that would indicate a blown-up simulation.

    Returns:
        Velocity field, shape ``(3, nx, ny, nz)`` ordered as ``(ux, uy, uz)``.

    Example:

    ```python
    >>> import jax.numpy as jnp
    >>> from jax_lbm.core.equilibrium import compute_velocity, equilibrium
    >>> nx, ny, nz = 4, 4, 4
    >>> rho = jnp.ones((nx, ny, nz))
    >>> u = jnp.zeros((3, nx, ny, nz)).at[0].set(0.05)
    >>> f = equilibrium(rho, u)
    >>> u_recovered = compute_velocity(f, rho)
    >>> jnp.allclose(u, u_recovered, atol=1e-6)
    Array(True, dtype=bool)
    ```
    """
    # D3Q19_VELOCITIES is (19, 3). We need (19, 3, 1, 1, 1) to broadcast
    # against f which is (19, nx, ny, nz).
    e = D3Q19_VELOCITIES.astype(jnp.float32)  # (19, 3)
    # Einsum: sum over i (19 directions), keep alpha (3 components) and spatial dims.
    # f[i, x, y, z] * e[i, alpha] -> u[alpha, x, y, z]
    u = jnp.einsum("ixyz,ia->axyz", f, e)
    return u / rho[jnp.newaxis, :, :, :]


def equilibrium(rho: jax.Array, u: jax.Array) -> jax.Array:
    """Compute the equilibrium distribution from density and velocity.

    The equilibrium is the second-order truncation of the Maxwell-Boltzmann
    distribution on the D3Q19 lattice:

    ``f_eq[i] = w[i] * rho * (1 + (e_i . u)/cs2 + (e_i . u)^2/(2*cs2^2) - u.u/(2*cs2))``

    where ``cs2 = 1/3`` is the speed of sound squared, ``w[i]`` are the
    lattice weights, and ``e_i`` are the lattice velocity vectors.

    This function is the single most important building block. It appears
    in every collision operator, in boundary conditions, and in the force
    computation (equilibrium splitting for pressure drag). It conserves
    mass and momentum by construction.

    Args:
        rho: Density field, shape ``(nx, ny, nz)``.
        u: Velocity field, shape ``(3, nx, ny, nz)`` as ``(ux, uy, uz)``.

    Returns:
        Equilibrium distributions, shape ``(19, nx, ny, nz)``.

    Example:

    ```python
    >>> import jax.numpy as jnp
    >>> from jax_lbm.core.equilibrium import equilibrium, compute_density
    >>> nx, ny, nz = 8, 8, 8
    >>> rho = jnp.ones((nx, ny, nz))
    >>> u = jnp.zeros((3, nx, ny, nz))
    >>> u = u.at[0].set(0.05)  # uniform flow in x
    >>> f_eq = equilibrium(rho, u)
    >>> jnp.allclose(compute_density(f_eq), rho, atol=1e-6)
    Array(True, dtype=bool)
    ```

    References:
        Qian, Y.H., d'Humieres, D. and Lallemand, P. (1992). "Lattice BGK
        models for Navier-Stokes equation." Europhysics Letters, 17(6),
        pp.479-484.
    """
    # e_i dot u for each direction i. D3Q19_VELOCITIES is (19, 3),
    # u is (3, nx, ny, nz). Result is (19, nx, ny, nz).
    e = D3Q19_VELOCITIES.astype(jnp.float32)  # (19, 3)
    eu = jnp.einsum("ia,axyz->ixyz", e, u)     # (19, nx, ny, nz)

    # u dot u, broadcast to all directions.
    u_sq = jnp.sum(u * u, axis=0)  # (nx, ny, nz)

    # Weights reshaped for broadcasting: (19, 1, 1, 1).
    w = D3Q19_WEIGHTS[:, jnp.newaxis, jnp.newaxis, jnp.newaxis]

    # f_eq = w * rho * (1 + eu/cs2 + eu^2/(2*cs2^2) - u^2/(2*cs2))
    f_eq = w * rho[jnp.newaxis, :, :, :] * (
        1.0
        + eu / CS2
        + (eu * eu) / (2.0 * CS2 * CS2)
        - u_sq[jnp.newaxis, :, :, :] / (2.0 * CS2)
    )

    return f_eq
