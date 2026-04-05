"""Momentum exchange force computation and drag coefficient.

To find the aerodynamic force on an obstacle in the flow, we use the
momentum exchange method (Mei, Luo, and Shyy 2002). For every fluid cell
that sits next to a solid cell, the distribution bouncing off the wall
transfers momentum. Summing these contributions over all boundary links
gives the total force vector.

The method also supports decomposing the total force into pressure drag
and friction drag via equilibrium splitting. The pressure contribution
comes from the equilibrium part of the distributions (hydrostatic
momentum transfer), while friction comes from the non-equilibrium part
(viscous shear at the wall).

The drag coefficient Cd normalizes the streamwise force by the dynamic
pressure and the obstacle's projected area:

    Cd = |Fx| / (0.5 * rho * U^2 * A)

All functions here are pure and differentiable. The force computation is
the quantity you will ultimately want gradients through for shape
optimization.

References:
    Mei, R., Luo, L.S. and Shyy, W. (2002). "Force evaluation in the
    lattice Boltzmann method involving curved geometry." Physical Review
    E, 65(4), 041203.
"""

import jax
import jax.numpy as jnp

from jax_lbm.core.lattice import D3Q19_VELOCITIES, D3Q19_OPPOSITE, Q
from jax_lbm.core.equilibrium import equilibrium, compute_density, compute_velocity


def momentum_exchange(
    f_post: jax.Array,
    f_streamed: jax.Array,
    q: jax.Array,
) -> jax.Array:
    """Compute aerodynamic force on solid boundaries via momentum exchange.

    For each fluid cell adjacent to a solid wall (identified by non-negative
    entries in the ``q`` buffer), the momentum transferred in each boundary
    link is:

        F_i = e_i * (f*_i + f_iopp)

    where ``f*_i`` is the post-collision distribution heading toward the
    wall and ``f_iopp`` is the distribution arriving from the wall after
    streaming (bounce-back). The total force is the sum over all boundary
    links across the entire domain.

    This matches the reference solver's ``lbm_force.comp`` shader.

    Args:
        f_post: Post-collision distributions (before streaming), shape
            ``(19, nx, ny, nz)``. These are the f* values heading toward
            walls.
        f_streamed: Post-streaming distributions, shape
            ``(19, nx, ny, nz)``. These contain the bounced-back values
            at boundary links.
        q: Bouzidi wall distances, shape ``(19, nx, ny, nz)``. Non-negative
            values mark boundary links. Negative values (-1) mean no wall.

    Returns:
        Total force vector, shape ``(3,)`` as ``(Fx, Fy, Fz)``. Fx is the
        drag force (streamwise), Fy is the side force, Fz is the lift.

    Example:

    ```python
    >>> import jax.numpy as jnp
    >>> from jax_lbm.core.forces import momentum_exchange
    >>> nx, ny, nz = 16, 8, 8
    >>> f_post = jnp.ones((19, nx, ny, nz)) / 19.0
    >>> f_streamed = jnp.ones((19, nx, ny, nz)) / 19.0
    >>> q = -jnp.ones((19, nx, ny, nz))  # no walls
    >>> force = momentum_exchange(f_post, f_streamed, q)
    >>> force.shape
    (3,)
    ```

    References:
        Mei, R., Luo, L.S. and Shyy, W. (2002). Physical Review E,
        65(4), 041203.
    """
    import numpy as np
    e_np = np.asarray(D3Q19_VELOCITIES).astype(float)
    opp_list = [int(x) for x in np.asarray(D3Q19_OPPOSITE)]

    force = jnp.zeros(3)

    for i in range(1, Q):  # skip rest direction (no momentum transfer)
        iopp = opp_list[i]

        # Identify boundary links: q[i] >= 0 at a cell means direction i
        # hits a wall from that cell.
        has_wall = q[i] >= 0.0  # (nx, ny, nz) boolean

        # f* heading toward wall (pre-streaming) and f coming back
        # (post-streaming, which contains the bounced distribution).
        fi_star = f_post[i]         # (nx, ny, nz)
        f_iopp  = f_streamed[iopp]  # (nx, ny, nz)

        # Momentum exchange: F_i = e_i * (f*_i + f_iopp)
        ftotal = jnp.where(has_wall, fi_star + f_iopp, 0.0)

        # Accumulate into 3D force vector.
        ex, ey, ez = float(e_np[i, 0]), float(e_np[i, 1]), float(e_np[i, 2])
        link_force_x = jnp.sum(ftotal * ex)
        link_force_y = jnp.sum(ftotal * ey)
        link_force_z = jnp.sum(ftotal * ez)

        force = force.at[0].add(link_force_x)
        force = force.at[1].add(link_force_y)
        force = force.at[2].add(link_force_z)

    return force


def momentum_exchange_decomposed(
    f_post: jax.Array,
    f_streamed: jax.Array,
    q: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Compute total force and its pressure/friction decomposition.

    Same as ``momentum_exchange``, but also computes the pressure
    (equilibrium) contribution separately. The friction contribution is
    then ``total - pressure``.

    Pressure drag comes from the equilibrium distributions. It represents
    the force due to the pressure field around the obstacle. Friction drag
    comes from the non-equilibrium part, representing viscous shear stress
    at the wall.

    Args:
        f_post: Post-collision distributions, shape ``(19, nx, ny, nz)``.
        f_streamed: Post-streaming distributions, shape ``(19, nx, ny, nz)``.
        q: Bouzidi wall distances, shape ``(19, nx, ny, nz)``.

    Returns:
        A tuple of two force vectors, each shape ``(3,)``:
        ``(total_force, pressure_force)``. Friction force is
        ``total_force - pressure_force``.

    Example:

    ```python
    >>> import jax.numpy as jnp
    >>> from jax_lbm.core.forces import momentum_exchange_decomposed
    >>> nx, ny, nz = 16, 8, 8
    >>> f_post = jnp.ones((19, nx, ny, nz)) / 19.0
    >>> q = -jnp.ones((19, nx, ny, nz))
    >>> total, pressure = momentum_exchange_decomposed(f_post, f_post, q)
    >>> friction = total - pressure
    ```
    """
    import numpy as np
    e_np = np.asarray(D3Q19_VELOCITIES).astype(float)
    opp_list = [int(x) for x in np.asarray(D3Q19_OPPOSITE)]

    # Compute equilibrium at post-collision state for pressure splitting.
    rho = compute_density(f_post)
    u = compute_velocity(f_post, rho)
    f_eq = equilibrium(rho, u)

    total_force = jnp.zeros(3)
    pressure_force = jnp.zeros(3)

    for i in range(1, Q):
        iopp = opp_list[i]
        has_wall = q[i] >= 0.0

        fi_star = f_post[i]
        f_iopp  = f_streamed[iopp]
        ftotal = jnp.where(has_wall, fi_star + f_iopp, 0.0)

        # Pressure contribution: use equilibrium distributions instead.
        feq_i    = f_eq[i]
        feq_iopp = f_eq[iopp]
        fpressure = jnp.where(has_wall, feq_i + feq_iopp, 0.0)

        ex, ey, ez = float(e_np[i, 0]), float(e_np[i, 1]), float(e_np[i, 2])

        total_force = total_force.at[0].add(jnp.sum(ftotal * ex))
        total_force = total_force.at[1].add(jnp.sum(ftotal * ey))
        total_force = total_force.at[2].add(jnp.sum(ftotal * ez))

        pressure_force = pressure_force.at[0].add(jnp.sum(fpressure * ex))
        pressure_force = pressure_force.at[1].add(jnp.sum(fpressure * ey))
        pressure_force = pressure_force.at[2].add(jnp.sum(fpressure * ez))

    return total_force, pressure_force


def projected_area(solid: jax.Array, axis: int = 0) -> jax.Array:
    """Compute the projected area of solid cells onto a plane.

    Projects the solid mask along the given axis and counts the number of
    columns that contain at least one solid cell (with value 1, not 2
    which is ground plane in the reference solver). The result is in
    lattice units squared (number of lattice cells in the shadow).

    For drag coefficient computation, you want axis=0 (project onto the
    Y-Z plane perpendicular to the flow direction).

    Args:
        solid: Solid mask, shape ``(nx, ny, nz)``. Values: 0 = fluid,
            1 = body solid, 2 = ground (excluded from projection).
        axis: Axis to project along. 0 = X (default, for drag), 1 = Y
            (for side force), 2 = Z (for lift).

    Returns:
        Projected area as a scalar (number of cells in the shadow).

    Example:

    ```python
    >>> import jax.numpy as jnp
    >>> from jax_lbm.core.forces import projected_area
    >>> solid = jnp.zeros((16, 8, 8), dtype=jnp.int32)
    >>> solid = solid.at[5:8, 2:6, 2:6].set(1)  # 3x4x4 box
    >>> float(projected_area(solid, axis=0))  # shadow on Y-Z: 4*4 = 16
    16.0
    ```
    """
    # Only count body solids (value 1), not ground (value 2).
    body = (solid == 1).astype(jnp.float32)
    # Project: any() along the given axis, then count.
    has_solid = jnp.any(body > 0, axis=axis).astype(jnp.float32)
    return jnp.sum(has_solid)


def drag_coefficient(
    force: jax.Array,
    inlet_velocity: float,
    area: jax.Array,
    rho: float = 1.0,
) -> jax.Array:
    """Compute the drag coefficient from force, velocity, and area.

    The drag coefficient is a dimensionless measure of aerodynamic
    resistance:

        Cd = |Fx| / (0.5 * rho * U^2 * A)

    where Fx is the streamwise force, rho is the reference density, U is
    the freestream velocity, and A is the projected frontal area.

    Args:
        force: Force vector from ``momentum_exchange``, shape ``(3,)``.
        inlet_velocity: Freestream velocity magnitude (scalar, lattice
            units). Typically 0.05 for stability.
        area: Projected frontal area (scalar, lattice units squared).
        rho: Reference density. Default 1.0 (standard LBM convention).

    Returns:
        Drag coefficient as a scalar.

    Example:

    ```python
    >>> import jax.numpy as jnp
    >>> from jax_lbm.core.forces import drag_coefficient
    >>> force = jnp.array([0.001, 0.0, 0.0])
    >>> cd = drag_coefficient(force, inlet_velocity=0.05, area=16.0)
    >>> cd.shape
    ()
    ```
    """
    drag_force = jnp.abs(force[0])
    dynamic_pressure = 0.5 * rho * inlet_velocity * inlet_velocity
    return drag_force / (dynamic_pressure * area + 1e-10)
