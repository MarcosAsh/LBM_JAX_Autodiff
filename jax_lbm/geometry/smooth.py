"""Sigmoid-smoothed solid representation for differentiable shape optimization.

Binary solid masks break autodiff because the solid/fluid boundary is a
step function with zero gradient almost everywhere. To get useful gradients
for shape optimization, we replace the hard mask with a smooth porosity
field that transitions continuously between 0 (fluid) and 1 (solid).

The approach follows List et al. (2022): a sigmoid function is applied to
the signed distance from the obstacle surface, producing a porosity field
in [0, 1]. The collision step then blends between fluid behavior
(collision) and solid behavior (bounce-back) according to this porosity:

    f_out = (1 - porosity) * f_collided + porosity * f_bounced

This gives a well-defined gradient of the flow field with respect to the
obstacle shape, at the cost of a thin "mushy zone" near the surface where
the physics is approximate. The sharpness parameter controls the width
of this zone: higher sharpness means a thinner transition but noisier
gradients.

References:
    List, B. et al. (2022). "Learned Turbulence Modelling with
    Differentiable Fluid Solvers." NeurIPS 2022 Workshop on Machine
    Learning and the Physical Sciences.
"""

import jax
import jax.numpy as jnp
import numpy as np

from jax_lbm.core.lattice import D3Q19_VELOCITIES, D3Q19_OPPOSITE, Q


def signed_distance_sphere(
    nx: int, ny: int, nz: int,
    center: tuple[float, float, float],
    radius: jax.Array,
) -> jax.Array:
    """Compute the signed distance field for a sphere.

    Positive inside the sphere, negative outside. The zero level set is
    the sphere surface. This representation is differentiable with respect
    to the radius (and center, if you make those parameters too).

    Args:
        nx, ny, nz: Grid dimensions.
        center: Sphere center in world coordinates ``(cx, cy, cz)``.
        radius: Sphere radius in world coordinates. Can be a JAX scalar
            for differentiation.

    Returns:
        Signed distance field, shape ``(nx, ny, nz)``. Positive inside,
        negative outside.

    Example:

    ```python
    >>> import jax.numpy as jnp
    >>> from jax_lbm.geometry.smooth import signed_distance_sphere
    >>> sdf = signed_distance_sphere(32, 16, 16, (0.0, 0.0, 0.0), jnp.float32(0.5))
    >>> sdf.shape
    (32, 16, 16)
    ```
    """
    cx, cy, cz = center
    scale_x, scale_y, scale_z = nx / 8.0, ny / 4.0, nz / 4.0

    gx = jnp.arange(nx, dtype=jnp.float32)
    gy = jnp.arange(ny, dtype=jnp.float32)
    gz = jnp.arange(nz, dtype=jnp.float32)
    gx3d, gy3d, gz3d = jnp.meshgrid(gx, gy, gz, indexing="ij")

    wx = (gx3d + 0.5) / scale_x - 4.0
    wy = (gy3d + 0.5) / scale_y - 2.0
    wz = (gz3d + 0.5) / scale_z - 2.0

    dist = jnp.sqrt((wx - cx)**2 + (wy - cy)**2 + (wz - cz)**2)
    return radius - dist  # positive inside


def porosity_from_sdf(
    sdf: jax.Array,
    sharpness: float = 20.0,
) -> jax.Array:
    """Convert a signed distance field to a smooth porosity field.

    Applies a sigmoid to the SDF: cells deep inside the obstacle get
    porosity near 1, cells far outside get porosity near 0, and the
    surface has a smooth transition. The sharpness parameter controls
    how steep the transition is.

    Args:
        sdf: Signed distance field, shape ``(nx, ny, nz)``. Positive
            inside the obstacle.
        sharpness: Steepness of the sigmoid transition. Default 20.0.
            Higher values give a sharper boundary (closer to binary)
            but potentially noisier gradients.

    Returns:
        Porosity field in [0, 1], shape ``(nx, ny, nz)``.

    Example:

    ```python
    >>> import jax.numpy as jnp
    >>> from jax_lbm.geometry.smooth import signed_distance_sphere, porosity_from_sdf
    >>> sdf = signed_distance_sphere(32, 16, 16, (0.0, 0.0, 0.0), jnp.float32(0.5))
    >>> porosity = porosity_from_sdf(sdf, sharpness=20.0)
    >>> float(porosity.min()) < 0.01  # far from sphere
    True
    >>> float(porosity.max()) > 0.99  # inside sphere
    True
    ```
    """
    return jax.nn.sigmoid(sharpness * sdf)


def soft_bounce_back(
    f_collided: jax.Array,
    porosity: jax.Array,
) -> jax.Array:
    """Blend between collision and bounce-back based on porosity.

    In fluid regions (porosity near 0), the collided distributions pass
    through unchanged. In solid regions (porosity near 1), the
    distributions are reversed (bounce-back). In the transition zone,
    it's a smooth blend.

    This replaces the hard if/else solid check in the step function
    with a differentiable alternative.

    Args:
        f_collided: Post-collision distributions, shape ``(19, nx, ny, nz)``.
        porosity: Porosity field in [0, 1], shape ``(nx, ny, nz)``.

    Returns:
        Blended distributions, shape ``(19, nx, ny, nz)``.

    Example:

    ```python
    >>> import jax.numpy as jnp
    >>> from jax_lbm.geometry.smooth import soft_bounce_back
    >>> f = jnp.ones((19, 8, 8, 8)) / 19.0
    >>> porosity = jnp.zeros((8, 8, 8))  # all fluid
    >>> f_out = soft_bounce_back(f, porosity)
    >>> jnp.allclose(f, f_out, atol=1e-7)
    Array(True, dtype=bool)
    ```
    """
    opp = jnp.array([0, 2, 1, 4, 3, 6, 5, 10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15])
    f_bounced = f_collided[opp]

    p = porosity[jnp.newaxis, :, :, :]
    return (1.0 - p) * f_collided + p * f_bounced


def smooth_sphere_geometry(
    nx: int, ny: int, nz: int,
    center: tuple[float, float, float],
    radius: jax.Array,
    sharpness: float = 20.0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Create a differentiable sphere geometry: porosity, hard mask, and q-buffer.

    This is the all-in-one function for setting up a differentiable sphere
    obstacle. It returns:
    1. A smooth porosity field for differentiable solid/fluid blending.
    2. A hard solid mask (thresholded porosity) for the q computation.
    3. A Bouzidi q-buffer with differentiable fractional wall distances
       from ray-sphere intersection.

    The gradient of the drag coefficient with respect to ``radius`` flows
    through both the porosity field (soft bounce-back) and the q-values
    (Bouzidi interpolation), giving a useful signal for shape optimization.

    Args:
        nx, ny, nz: Grid dimensions.
        center: Sphere center in world coordinates.
        radius: Sphere radius (JAX scalar, differentiable).
        sharpness: Sigmoid steepness for the porosity field.

    Returns:
        Tuple of ``(porosity, solid, q)``:
        - ``porosity``: Smooth field in [0, 1], shape ``(nx, ny, nz)``.
        - ``solid``: Hard mask (int32), shape ``(nx, ny, nz)``.
        - ``q``: Bouzidi distances, shape ``(19, nx, ny, nz)``.
    """
    sdf = signed_distance_sphere(nx, ny, nz, center, radius)
    porosity = porosity_from_sdf(sdf, sharpness)
    solid = (porosity > 0.5).astype(jnp.int32)

    # Differentiable q via ray-sphere intersection.
    cx, cy, cz = center
    scale_x, scale_y, scale_z = nx / 8.0, ny / 4.0, nz / 4.0

    gx = jnp.arange(nx, dtype=jnp.float32)
    gy = jnp.arange(ny, dtype=jnp.float32)
    gz = jnp.arange(nz, dtype=jnp.float32)
    gx3d, gy3d, gz3d = jnp.meshgrid(gx, gy, gz, indexing="ij")

    wx = (gx3d + 0.5) / scale_x - 4.0
    wy = (gy3d + 0.5) / scale_y - 2.0
    wz = (gz3d + 0.5) / scale_z - 2.0

    e_np = np.asarray(D3Q19_VELOCITIES)
    q = -jnp.ones((Q, nx, ny, nz))

    for i in range(1, Q):
        ex_i, ey_i, ez_i = int(e_np[i, 0]), int(e_np[i, 1]), int(e_np[i, 2])

        nbx = (gx3d + ex_i).astype(jnp.int32)
        nby = (gy3d + ey_i).astype(jnp.int32)
        nbz = (gz3d + ez_i).astype(jnp.int32)

        in_bounds = (
            (nbx >= 0) & (nbx < nx) & (nby >= 0) & (nby < ny)
            & (nbz >= 0) & (nbz < nz)
        )

        nbx_safe = jnp.clip(nbx, 0, nx - 1)
        nby_safe = jnp.clip(nby, 0, ny - 1)
        nbz_safe = jnp.clip(nbz, 0, nz - 1)

        nb_solid = solid[nbx_safe, nby_safe, nbz_safe]
        is_boundary = (solid == 0) & (nb_solid == 1) & in_bounds

        # Ray-sphere intersection (differentiable w.r.t. radius).
        ray_dx = jnp.float32(ex_i) / scale_x
        ray_dy = jnp.float32(ey_i) / scale_y
        ray_dz = jnp.float32(ez_i) / scale_z

        ox, oy, oz = wx - cx, wy - cy, wz - cz
        a = ray_dx**2 + ray_dy**2 + ray_dz**2
        b = 2.0 * (ox * ray_dx + oy * ray_dy + oz * ray_dz)
        c = ox**2 + oy**2 + oz**2 - radius**2
        disc = b * b - 4.0 * a * c

        sqrt_disc = jnp.sqrt(jnp.maximum(disc, 1e-10))
        t = (-b - sqrt_disc) / (2.0 * a + 1e-20)

        valid = (disc >= 0.0) & (t > 0.0) & (t <= 1.0)
        q_val = jnp.where(valid, t, 0.5)
        q_val = jnp.clip(q_val, 0.01, 0.99)

        q = q.at[i].set(jnp.where(is_boundary, q_val, -1.0))

    return porosity, solid, q
