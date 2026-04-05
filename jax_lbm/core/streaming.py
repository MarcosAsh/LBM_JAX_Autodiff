"""Streaming step with Bouzidi interpolated bounce-back.

Streaming propagates distributions between neighboring cells along the
lattice velocity directions. Each cell pulls the incoming distribution
from its upstream neighbor: ``f[x, i] = f_post[x - e_i, i]``.

At solid boundaries, distributions do not stream from inside the wall.
Instead, they bounce back. The simplest version (standard bounce-back)
reverses the direction: the distribution heading into the wall comes back
along the opposite direction. This gives first-order accuracy and places
the effective wall halfway between the fluid and solid nodes.

Bouzidi interpolated bounce-back improves this to second order. It uses
the fractional wall distance ``q`` (how far along the link the wall
actually sits, between 0 and 1) to interpolate the bounced distribution.
Two regimes:

- ``q >= 0.5``: linear interpolation using only the current cell.
  ``f[i] = (1/(2q)) * f*[iopp] + (1 - 1/(2q)) * f*[i]``

- ``q < 0.5``: quadratic interpolation reaching one cell further into the
  fluid. ``f[i] = 2q * f*[iopp] + (1 - 2q) * f*[iopp at x+e_i]``

The ``q`` buffer has the same shape as ``f``: ``(19, nx, ny, nz)``.
A value of -1 means "no wall in this direction" (normal streaming).
Non-negative values in [0, 1] trigger Bouzidi bounce-back.

References:
    Bouzidi, M., Firdaouss, M. and Lallemand, P. (2001). "Momentum
    transfer of a Boltzmann-lattice fluid with boundaries." Physics of
    Fluids, 13(11), pp.3452-3459.
"""

import jax
import jax.numpy as jnp

from jax_lbm.core.lattice import D3Q19_VELOCITIES, D3Q19_OPPOSITE, Q


def stream(
    f: jax.Array,
    q: jax.Array,
    periodic_yz: bool = True,
) -> jax.Array:
    """Stream distributions and apply Bouzidi bounce-back at walls.

    This is a pull-based streaming step. For each cell and each direction,
    it decides whether to pull from the upstream fluid neighbor (normal
    streaming) or to apply Bouzidi interpolated bounce-back (wall link).

    The X boundaries are not periodic: distributions that would stream
    from outside the X domain are left unchanged (zero-gradient outflow
    fallback). The Y and Z boundaries can be periodic or clamped
    (free-slip), controlled by ``periodic_yz``.

    Solid cells are not explicitly skipped here. The Zou-He boundary
    conditions (applied after streaming) overwrite the inlet and outlet
    slabs. Interior solid cells receive bounced distributions from the
    Bouzidi interpolation, which is correct: the force computation reads
    both pre- and post-streaming values at boundary links.

    Args:
        f: Post-collision distributions, shape ``(19, nx, ny, nz)``.
        q: Bouzidi fractional wall distances, shape ``(19, nx, ny, nz)``.
            Values in [0, 1] indicate a wall at that fractional distance
            along the link. Values < 0 (typically -1) mean no wall.
        periodic_yz: If True, Y and Z boundaries wrap periodically.
            If False, distributions clamp at the edges (free-slip).

    Returns:
        Post-streaming distributions, shape ``(19, nx, ny, nz)``.

    Example:

    ```python
    >>> import jax.numpy as jnp
    >>> from jax_lbm.core.equilibrium import equilibrium
    >>> from jax_lbm.core.streaming import stream
    >>> nx, ny, nz = 16, 8, 8
    >>> rho = jnp.ones((nx, ny, nz))
    >>> u = jnp.zeros((3, nx, ny, nz)).at[0].set(0.05)
    >>> f = equilibrium(rho, u)
    >>> q = -jnp.ones((19, nx, ny, nz))  # no walls
    >>> f_streamed = stream(f, q, periodic_yz=True)
    >>> f_streamed.shape
    (19, 16, 8, 8)
    ```
    """
    # Convert lattice constants to Python lists so indexing never touches
    # JAX tracers. This is critical for compatibility with jax.lax.scan
    # and jax.grad -- you cannot call int() on a traced JAX array.
    import numpy as np
    e_np = np.asarray(D3Q19_VELOCITIES)
    opp_list = [int(x) for x in np.asarray(D3Q19_OPPOSITE)]

    nx, ny, nz = f.shape[1], f.shape[2], f.shape[3]
    f_out = jnp.zeros_like(f)

    for i in range(Q):
        iopp = opp_list[i]
        ex, ey, ez = int(e_np[i, 0]), int(e_np[i, 1]), int(e_np[i, 2])

        # Where does the upstream neighbor live for direction i?
        # Pull-based: source is at (x - ex, y - ey, z - ez).

        # Normal streaming: pull from upstream neighbor at (x - e_i).
        # roll(arr, shift=+k, axis) gives new[x] = old[x - k], so
        # we roll by +e_i to pull from x - e_i.
        if i == 0:
            # Rest direction: no movement.
            f_normal = f[0]
        else:
            f_normal = f[i]
            if ex != 0:
                f_normal = jnp.roll(f_normal, ex, axis=0)
            if ey != 0:
                f_normal = jnp.roll(f_normal, ey, axis=1)
            if ez != 0:
                f_normal = jnp.roll(f_normal, ez, axis=2)

            # Handle non-periodic X boundaries: cells that would pull from
            # outside the domain keep their own post-collision value.
            if ex == 1:
                # Direction points in +x, so pulling from x-1. x=0 would
                # pull from x=-1 which wrapped to x=nx-1. Replace with
                # the original (no-stream fallback).
                f_normal = f_normal.at[0, :, :].set(f[i, 0, :, :])
            elif ex == -1:
                # Direction points in -x, pulling from x+1. x=nx-1 would
                # pull from x=nx, wrapped to 0. Replace.
                f_normal = f_normal.at[nx - 1, :, :].set(f[i, nx - 1, :, :])

            # Handle non-periodic Y/Z if requested.
            if not periodic_yz:
                if ey == 1:
                    f_normal = f_normal.at[:, 0, :].set(f[i, :, 0, :])
                elif ey == -1:
                    f_normal = f_normal.at[:, ny - 1, :].set(f[i, :, ny - 1, :])
                if ez == 1:
                    f_normal = f_normal.at[:, :, 0].set(f[i, :, :, 0])
                elif ez == -1:
                    f_normal = f_normal.at[:, :, nz - 1].set(f[i, :, :, nz - 1])

        if i == 0:
            f_out = f_out.at[0].set(f_normal)
            continue

        # Bouzidi bounce-back for wall links.
        # q values for the opposite direction: q[iopp] >= 0 means there
        # is a wall along direction iopp from this cell. The bounced
        # distribution arrives in direction i.
        q_iopp = q[iopp]  # (nx, ny, nz)
        has_wall = q_iopp >= 0.0

        # Post-collision distributions needed for Bouzidi.
        f_iopp_star = f[iopp]  # f*[iopp] at this cell
        f_i_star = f[i]        # f*[i] at this cell

        # q >= 0.5 branch: linear interpolation from this cell only.
        inv2q = 1.0 / (2.0 * jnp.clip(q_iopp, 0.5, 1.0))
        f_bouzidi_high = inv2q * f_iopp_star + (1.0 - inv2q) * f_i_star

        # q < 0.5 branch: quadratic using the next fluid neighbor (x + e_i).
        # To access values at x + e_i, we roll by -e_i:
        # roll(arr, shift=-k, axis) gives new[x] = old[x + k].
        f_iopp_ff = f[iopp]
        if ex != 0:
            f_iopp_ff = jnp.roll(f_iopp_ff, -ex, axis=0)
        if ey != 0:
            f_iopp_ff = jnp.roll(f_iopp_ff, -ey, axis=1)
        if ez != 0:
            f_iopp_ff = jnp.roll(f_iopp_ff, -ez, axis=2)

        twoq = 2.0 * jnp.clip(q_iopp, 0.0, 0.5)
        f_bouzidi_low = twoq * f_iopp_star + (1.0 - twoq) * f_iopp_ff

        # Pick the right branch based on q value.
        f_bouzidi = jnp.where(q_iopp >= 0.5, f_bouzidi_high, f_bouzidi_low)

        # Use Bouzidi where there's a wall, normal streaming otherwise.
        f_dir = jnp.where(has_wall, f_bouzidi, f_normal)

        f_out = f_out.at[i].set(f_dir)

    return f_out
