"""Bouzidi fractional wall distance computation.

The Bouzidi interpolated bounce-back scheme needs to know, for each
boundary link, exactly how far along the lattice direction the wall sits.
This fraction ``q`` ranges from 0 (wall right at the fluid node) to 1
(wall right at the solid node). The streaming step uses q to interpolate
the bounced distribution, giving second-order accurate wall placement on
curved surfaces.

Computing q depends on the geometry representation:

- For analytic shapes (spheres), q comes from ray-shape intersection.
  See ``geometry.primitives.create_sphere``.

- For mesh-based geometry (OBJ files), q comes from ray-triangle
  intersection along each lattice direction. That is handled here.

- For axis-aligned boxes, q = 0.5 everywhere (standard bounce-back).

References:
    Bouzidi, M., Firdaouss, M. and Lallemand, P. (2001). "Momentum
    transfer of a Boltzmann-lattice fluid with boundaries." Physics of
    Fluids, 13(11), pp.3452-3459.
"""

import jax
import jax.numpy as jnp

from jax_lbm.core.lattice import D3Q19_VELOCITIES, Q


def compute_q_from_solid(
    solid: jax.Array,
    nx: int, ny: int, nz: int,
    default_q: float = 0.5,
) -> jax.Array:
    """Compute Bouzidi q-buffer from a binary solid mask using midpoint distances.

    This is the simplest q computation: for every fluid cell that has a
    solid neighbor along a D3Q19 link, set q = ``default_q`` (typically
    0.5). This gives standard halfway bounce-back, which is first-order
    accurate. Use this when you don't have an analytic shape or a
    triangle mesh to intersect.

    For second-order accuracy, use ``create_sphere`` (analytic) or
    ``compute_q_from_mesh`` (triangle intersection) instead.

    Args:
        solid: Solid mask, shape ``(nx, ny, nz)``. 1 = solid, 0 = fluid.
        nx: Grid size in x.
        ny: Grid size in y.
        nz: Grid size in z.
        default_q: Fractional distance to assign. Default 0.5.

    Returns:
        Bouzidi q-buffer, shape ``(19, nx, ny, nz)``. Non-negative at
        boundary links, -1 elsewhere.

    Example:

    ```python
    >>> import jax.numpy as jnp
    >>> from jax_lbm.geometry.bouzidi import compute_q_from_solid
    >>> solid = jnp.zeros((16, 8, 8), dtype=jnp.int32)
    >>> solid = solid.at[8, 3:5, 3:5].set(1)
    >>> q = compute_q_from_solid(solid, 16, 8, 8)
    >>> q.shape
    (19, 16, 8, 8)
    >>> bool(jnp.any(q >= 0))  # should have some boundary links
    True
    ```
    """
    e = D3Q19_VELOCITIES

    gx = jnp.arange(nx, dtype=jnp.int32)
    gy = jnp.arange(ny, dtype=jnp.int32)
    gz = jnp.arange(nz, dtype=jnp.int32)
    gx3d, gy3d, gz3d = jnp.meshgrid(gx, gy, gz, indexing="ij")

    q = -jnp.ones((Q, nx, ny, nz), dtype=jnp.float32)

    for i in range(1, Q):
        ex_i = int(e[i, 0])
        ey_i = int(e[i, 1])
        ez_i = int(e[i, 2])

        nbx = gx3d + ex_i
        nby = gy3d + ey_i
        nbz = gz3d + ez_i

        in_bounds = (
            (nbx >= 0) & (nbx < nx)
            & (nby >= 0) & (nby < ny)
            & (nbz >= 0) & (nbz < nz)
        )

        nbx_safe = jnp.clip(nbx, 0, nx - 1)
        nby_safe = jnp.clip(nby, 0, ny - 1)
        nbz_safe = jnp.clip(nbz, 0, nz - 1)

        nb_solid = solid[nbx_safe, nby_safe, nbz_safe]
        is_boundary = (solid == 0) & (nb_solid == 1) & in_bounds

        q = q.at[i].set(jnp.where(is_boundary, default_q, -1.0))

    return q
