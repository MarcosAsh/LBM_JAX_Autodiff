"""Loss functions for inverse fluid design.

These functions define what "good" means for an optimization problem.
Each loss takes the simulation output (drag coefficient, velocity field,
pressure field) and returns a scalar that you minimize with gradient
descent.

Three categories:

1. **Target matching** -- drive a quantity toward a specific value.
   "Make Cd equal to 1.5" or "make the velocity field match this
   reference."

2. **Extremal objectives** -- minimize or maximize something directly.
   "Minimize drag" is the most common.

3. **Regularizers** -- penalize unphysical or undesirable solutions.
   Smoothness penalties on the geometry, total variation on the
   porosity field, constraints on the solid volume fraction.

All losses are pure functions that return a scalar. Combine them by
weighted addition:

    total_loss = cd_loss + 0.01 * smoothness_penalty
"""

import jax
import jax.numpy as jnp

from jax_lbm.core.equilibrium import compute_density, compute_velocity
from jax_lbm.core.forces import momentum_exchange, drag_coefficient, projected_area


def target_cd_loss(
    force: jax.Array,
    inlet_velocity: float,
    area: jax.Array,
    target_cd: float,
) -> jax.Array:
    """Squared error between computed and target drag coefficient.

    This is the simplest optimization objective: drive Cd toward a
    specific value. Useful for validation (can the optimizer find the
    inlet velocity that gives Cd = 1.09?) and for constrained design
    (find the shape that achieves a specified drag).

    Args:
        force: Force vector from ``momentum_exchange``, shape ``(3,)``.
        inlet_velocity: Freestream velocity magnitude (scalar).
        area: Projected frontal area (scalar).
        target_cd: Desired drag coefficient.

    Returns:
        Scalar loss: ``(Cd - target_cd)^2``.

    Example:

    ```python
    >>> import jax.numpy as jnp
    >>> from jax_lbm.optim.losses import target_cd_loss
    >>> force = jnp.array([0.01, 0.0, 0.0])
    >>> loss = target_cd_loss(force, inlet_velocity=0.05, area=100.0, target_cd=1.0)
    >>> loss.shape
    ()
    ```
    """
    cd = drag_coefficient(force, inlet_velocity, area)
    return (cd - target_cd) ** 2


def minimize_cd_loss(
    force: jax.Array,
    inlet_velocity: float,
    area: jax.Array,
) -> jax.Array:
    """Loss that minimizes drag coefficient directly.

    Returns Cd itself as the loss. Gradient descent on this pushes the
    design toward lower drag. For shape optimization, this is the
    natural objective: make the obstacle as streamlined as possible.

    Args:
        force: Force vector from ``momentum_exchange``, shape ``(3,)``.
        inlet_velocity: Freestream velocity magnitude.
        area: Projected frontal area.

    Returns:
        Scalar loss equal to Cd.

    Example:

    ```python
    >>> import jax.numpy as jnp
    >>> from jax_lbm.optim.losses import minimize_cd_loss
    >>> force = jnp.array([0.01, 0.0, 0.0])
    >>> loss = minimize_cd_loss(force, inlet_velocity=0.05, area=100.0)
    ```
    """
    return drag_coefficient(force, inlet_velocity, area)


def target_velocity_loss(
    f: jax.Array,
    target_u: jax.Array,
    mask: jax.Array = None,
) -> jax.Array:
    """Mean squared error between simulated and target velocity fields.

    Computes the velocity from the distribution functions, then measures
    the L2 distance to a reference velocity field. An optional mask
    restricts the comparison to a region of interest (e.g., the wake
    behind an obstacle, or a specific cross-section).

    Args:
        f: Distribution functions, shape ``(19, nx, ny, nz)``.
        target_u: Target velocity field, shape ``(3, nx, ny, nz)``.
        mask: Optional boolean mask, shape ``(nx, ny, nz)``. If provided,
            only cells where ``mask == True`` contribute to the loss.

    Returns:
        Scalar MSE loss.

    Example:

    ```python
    >>> import jax.numpy as jnp
    >>> from jax_lbm.core.equilibrium import equilibrium
    >>> from jax_lbm.optim.losses import target_velocity_loss
    >>> nx, ny, nz = 8, 8, 8
    >>> rho = jnp.ones((nx, ny, nz))
    >>> u = jnp.zeros((3, nx, ny, nz)).at[0].set(0.05)
    >>> f = equilibrium(rho, u)
    >>> target = jnp.zeros((3, nx, ny, nz)).at[0].set(0.05)
    >>> float(target_velocity_loss(f, target))  # should be ~0
    0.0
    ```
    """
    rho = compute_density(f)
    u = compute_velocity(f, rho)
    diff_sq = jnp.sum((u - target_u) ** 2, axis=0)  # (nx, ny, nz)

    if mask is not None:
        diff_sq = jnp.where(mask, diff_sq, 0.0)
        n_cells = jnp.sum(mask.astype(jnp.float32)) + 1e-10
    else:
        n_cells = jnp.float32(diff_sq.size)

    return jnp.sum(diff_sq) / n_cells


def total_variation_penalty(porosity: jax.Array) -> jax.Array:
    """Total variation of the porosity field (smoothness regularizer).

    Penalizes rapid spatial changes in the porosity, encouraging smooth
    obstacle boundaries. Without this, gradient-based shape optimization
    can produce jagged, checkerboard-like geometries that are physically
    meaningless.

    Computed as the sum of absolute differences between neighboring cells
    along each axis.

    Args:
        porosity: Porosity field in [0, 1], shape ``(nx, ny, nz)``.

    Returns:
        Scalar TV penalty.

    Example:

    ```python
    >>> import jax.numpy as jnp
    >>> from jax_lbm.optim.losses import total_variation_penalty
    >>> smooth = jnp.ones((8, 8, 8)) * 0.5  # uniform -> TV = 0
    >>> float(total_variation_penalty(smooth))
    0.0
    ```
    """
    dx = jnp.abs(porosity[1:, :, :] - porosity[:-1, :, :])
    dy = jnp.abs(porosity[:, 1:, :] - porosity[:, :-1, :])
    dz = jnp.abs(porosity[:, :, 1:] - porosity[:, :, :-1])
    return jnp.sum(dx) + jnp.sum(dy) + jnp.sum(dz)


def volume_penalty(porosity: jax.Array, target_fraction: float = 0.1) -> jax.Array:
    """Penalize deviation from a target solid volume fraction.

    Without a volume constraint, drag minimization trivially produces
    the empty domain (no obstacle = no drag). This penalty keeps the
    total solid volume near a target fraction of the domain.

    Args:
        porosity: Porosity field in [0, 1], shape ``(nx, ny, nz)``.
        target_fraction: Desired fraction of cells that should be solid.
            Default 0.1 (10% of the domain).

    Returns:
        Scalar penalty: ``(actual_fraction - target_fraction)^2``.

    Example:

    ```python
    >>> import jax.numpy as jnp
    >>> from jax_lbm.optim.losses import volume_penalty
    >>> porosity = jnp.zeros((10, 10, 10))  # all fluid
    >>> float(volume_penalty(porosity, target_fraction=0.1))  # penalty for too little solid
    0.010000000707805157
    ```
    """
    actual_fraction = jnp.mean(porosity)
    return (actual_fraction - target_fraction) ** 2
