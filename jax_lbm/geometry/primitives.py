"""Solid geometry primitives: spheres, boxes, and AABB marking.

These functions create the solid mask and Bouzidi q-buffer for simple
geometric shapes. The solid mask is a 3D integer array where 0 means
fluid and 1 means solid. The q-buffer stores the fractional wall distance
for each lattice link at each cell, used by the Bouzidi interpolated
bounce-back scheme in the streaming step.

For spheres, the Bouzidi q values are computed analytically via
ray-sphere intersection, giving exact fractional distances. For boxes,
q is set to 0.5 (midpoint), which reduces Bouzidi to standard
bounce-back. This matches the reference solver's approach.

The coordinate system maps lattice indices to a world space spanning
x in [-4, 4], y in [-2, 2], z in [-2, 2], with the cell center at
index (gx, gy, gz) located at world position
``((gx + 0.5) / scaleX - 4, (gy + 0.5) / scaleY - 2, (gz + 0.5) / scaleZ - 2)``.
"""

import jax
import jax.numpy as jnp

from jax_lbm.core.lattice import D3Q19_VELOCITIES, Q


def _grid_to_world(
    gx: jax.Array, gy: jax.Array, gz: jax.Array,
    nx: int, ny: int, nz: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Convert lattice indices to world coordinates.

    Places the cell center at the midpoint of each lattice cell. The
    world domain spans x in [-4, 4], y in [-2, 2], z in [-2, 2],
    matching the reference solver's coordinate convention.

    Args:
        gx: Lattice x-indices, any shape.
        gy: Lattice y-indices, same shape as gx.
        gz: Lattice z-indices, same shape as gx.
        nx: Grid size in x.
        ny: Grid size in y.
        nz: Grid size in z.

    Returns:
        Tuple of (wx, wy, wz) world coordinates, same shape as inputs.
    """
    scale_x = nx / 8.0
    scale_y = ny / 4.0
    scale_z = nz / 4.0
    wx = (gx + 0.5) / scale_x - 4.0
    wy = (gy + 0.5) / scale_y - 2.0
    wz = (gz + 0.5) / scale_z - 2.0
    return wx, wy, wz


def create_sphere(
    nx: int, ny: int, nz: int,
    center: tuple[float, float, float],
    radius: float,
) -> tuple[jax.Array, jax.Array]:
    """Create a solid sphere with analytic Bouzidi wall distances.

    Marks all lattice cells whose centers fall inside the sphere as solid.
    For each fluid cell adjacent to a solid cell along a D3Q19 link,
    computes the exact fractional wall distance q via ray-sphere
    intersection. The q values are clamped to [0.01, 0.99].

    This matches the reference solver's ``LBM_SetSolidSphere`` function
    in ``simulation/src/lbm.c``.

    Args:
        nx: Grid size in x.
        ny: Grid size in y.
        nz: Grid size in z.
        center: Sphere center in world coordinates ``(cx, cy, cz)``.
        radius: Sphere radius in world coordinates.

    Returns:
        A tuple of ``(solid, q)``:
        - ``solid``: Integer mask, shape ``(nx, ny, nz)``. 1 inside the
          sphere, 0 outside.
        - ``q``: Bouzidi distances, shape ``(19, nx, ny, nz)``. Values
          in [0.01, 0.99] at boundary links, -1.0 elsewhere.

    Example:

    ```python
    >>> import jax.numpy as jnp
    >>> from jax_lbm.geometry.primitives import create_sphere
    >>> solid, q = create_sphere(128, 64, 64, center=(0.0, 0.0, 0.0), radius=0.5)
    >>> solid.shape
    (128, 64, 64)
    >>> int(jnp.sum(solid > 0)) > 0
    True
    ```
    """
    cx, cy, cz = center

    # Build coordinate grids.
    gx = jnp.arange(nx, dtype=jnp.float32)
    gy = jnp.arange(ny, dtype=jnp.float32)
    gz = jnp.arange(nz, dtype=jnp.float32)
    gx3d, gy3d, gz3d = jnp.meshgrid(gx, gy, gz, indexing="ij")

    wx, wy, wz = _grid_to_world(gx3d, gy3d, gz3d, nx, ny, nz)

    # Mark cells whose centers are inside the sphere.
    dx = wx - cx
    dy = wy - cy
    dz = wz - cz
    dist_sq = dx * dx + dy * dy + dz * dz
    solid = jnp.where(dist_sq <= radius * radius, 1, 0).astype(jnp.int32)

    # Compute Bouzidi q for each fluid cell adjacent to a solid cell.
    q = -jnp.ones((Q, nx, ny, nz), dtype=jnp.float32)

    e = D3Q19_VELOCITIES  # (19, 3)
    scale_x = nx / 8.0
    scale_y = ny / 4.0
    scale_z = nz / 4.0

    for i in range(1, Q):
        ex_i, ey_i, ez_i = int(e[i, 0]), int(e[i, 1]), int(e[i, 2])

        # Neighbor indices along direction i.
        nbx = (gx3d + ex_i).astype(jnp.int32)
        nby = (gy3d + ey_i).astype(jnp.int32)
        nbz = (gz3d + ez_i).astype(jnp.int32)

        # Check if neighbor is in bounds and solid.
        in_bounds = (
            (nbx >= 0) & (nbx < nx)
            & (nby >= 0) & (nby < ny)
            & (nbz >= 0) & (nbz < nz)
        )

        # Clamp for safe indexing (out-of-bounds will be masked anyway).
        nbx_safe = jnp.clip(nbx, 0, nx - 1)
        nby_safe = jnp.clip(nby, 0, ny - 1)
        nbz_safe = jnp.clip(nbz, 0, nz - 1)

        nb_solid = solid[nbx_safe, nby_safe, nbz_safe]
        is_boundary_link = (solid == 0) & (nb_solid == 1) & in_bounds

        # Ray-sphere intersection for exact q.
        # Ray origin: world position of the fluid cell.
        # Ray direction: lattice velocity in world coordinates.
        ray_dx = jnp.float32(ex_i) / scale_x
        ray_dy = jnp.float32(ey_i) / scale_y
        ray_dz = jnp.float32(ez_i) / scale_z

        ox = wx - cx
        oy = wy - cy
        oz = wz - cz

        a = ray_dx * ray_dx + ray_dy * ray_dy + ray_dz * ray_dz
        b = 2.0 * (ox * ray_dx + oy * ray_dy + oz * ray_dz)
        c = ox * ox + oy * oy + oz * oz - radius * radius
        disc = b * b - 4.0 * a * c

        # Solve quadratic. Take the smallest positive root.
        sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))
        t = (-b - sqrt_disc) / (2.0 * a + 1e-20)

        # Valid if discriminant >= 0 and t in (0, 1].
        valid = (disc >= 0.0) & (t > 0.0) & (t <= 1.0)
        q_val = jnp.where(valid, t, 0.5)  # fallback to 0.5
        q_val = jnp.clip(q_val, 0.01, 0.99)

        # Only set q at actual boundary links.
        q_dir = jnp.where(is_boundary_link, q_val, -1.0)
        q = q.at[i].set(q_dir)

    return solid, q


def create_box(
    nx: int, ny: int, nz: int,
    corner_min: tuple[float, float, float],
    corner_max: tuple[float, float, float],
) -> tuple[jax.Array, jax.Array]:
    """Create a solid axis-aligned box with standard bounce-back (q=0.5).

    Marks all lattice cells whose centers fall inside the AABB as solid.
    Boundary links get q = 0.5, which makes Bouzidi equivalent to
    standard halfway bounce-back. This is first-order accurate but
    simple and stable.

    Args:
        nx: Grid size in x.
        ny: Grid size in y.
        nz: Grid size in z.
        corner_min: Minimum corner in world coordinates ``(xmin, ymin, zmin)``.
        corner_max: Maximum corner in world coordinates ``(xmax, ymax, zmax)``.

    Returns:
        A tuple of ``(solid, q)``:
        - ``solid``: Integer mask, shape ``(nx, ny, nz)``.
        - ``q``: Bouzidi distances, shape ``(19, nx, ny, nz)``.

    Example:

    ```python
    >>> import jax.numpy as jnp
    >>> from jax_lbm.geometry.primitives import create_box
    >>> solid, q = create_box(64, 32, 32,
    ...     corner_min=(-0.5, -0.5, -0.5), corner_max=(0.5, 0.5, 0.5))
    >>> int(jnp.sum(solid > 0)) > 0
    True
    ```
    """
    gx = jnp.arange(nx, dtype=jnp.float32)
    gy = jnp.arange(ny, dtype=jnp.float32)
    gz = jnp.arange(nz, dtype=jnp.float32)
    gx3d, gy3d, gz3d = jnp.meshgrid(gx, gy, gz, indexing="ij")

    wx, wy, wz = _grid_to_world(gx3d, gy3d, gz3d, nx, ny, nz)

    xmin, ymin, zmin = corner_min
    xmax, ymax, zmax = corner_max

    inside = (
        (wx >= xmin) & (wx <= xmax)
        & (wy >= ymin) & (wy <= ymax)
        & (wz >= zmin) & (wz <= zmax)
    )
    solid = jnp.where(inside, 1, 0).astype(jnp.int32)

    # Standard bounce-back: q = 0.5 at all boundary links.
    q = -jnp.ones((Q, nx, ny, nz), dtype=jnp.float32)
    e = D3Q19_VELOCITIES

    for i in range(1, Q):
        ex_i, ey_i, ez_i = int(e[i, 0]), int(e[i, 1]), int(e[i, 2])
        nbx = (gx3d + ex_i).astype(jnp.int32)
        nby = (gy3d + ey_i).astype(jnp.int32)
        nbz = (gz3d + ez_i).astype(jnp.int32)

        in_bounds = (
            (nbx >= 0) & (nbx < nx)
            & (nby >= 0) & (nby < ny)
            & (nbz >= 0) & (nbz < nz)
        )

        nbx_safe = jnp.clip(nbx, 0, nx - 1)
        nby_safe = jnp.clip(nby, 0, ny - 1)
        nbz_safe = jnp.clip(nbz, 0, nz - 1)

        nb_solid = solid[nbx_safe, nby_safe, nbz_safe]
        is_boundary_link = (solid == 0) & (nb_solid == 1) & in_bounds

        q_dir = jnp.where(is_boundary_link, 0.5, -1.0)
        q = q.at[i].set(q_dir)

    return solid, q
