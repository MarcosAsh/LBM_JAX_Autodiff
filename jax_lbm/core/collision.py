"""Collision operators: BGK, MRT, and Smagorinsky subgrid-scale model.

Collision is where the physics happens. After streaming moves distributions
between cells, collision relaxes each cell toward local thermodynamic
equilibrium. The choice of collision operator controls accuracy, stability,
and the effective viscosity.

Three operators are provided, in order of complexity:

1. **BGK** (Bhatnagar-Gross-Krook): single relaxation time. Simple,
   differentiable, and the right starting point. All directions relax at
   the same rate omega = 1/tau.

2. **MRT** (multiple relaxation time): transforms to moment space,
   relaxes each moment independently, transforms back. More stable than
   BGK at the same Reynolds number because ghost modes can be damped
   aggressively without affecting physical viscosity.

3. **Smagorinsky SGS**: not a collision operator itself, but a modifier.
   It computes a turbulent eddy viscosity from the local non-equilibrium
   stress tensor, increasing tau in regions of strong shear. This lets
   you run at higher Reynolds numbers without blowing up. Combine it
   with either BGK or MRT.

All functions are pure, JIT-compilable, and differentiable.
"""

import jax
import jax.numpy as jnp

from jax_lbm.core.lattice import (
    D3Q19_VELOCITIES,
    D3Q19_WEIGHTS,
    D3Q19_OPPOSITE,
    CS2,
    Q,
    MRT_RELAXATION_RATES,
    MRT_STRESS_INDICES,
    mrt_forward,
    mrt_inverse,
    mrt_equilibrium_moments,
)
from jax_lbm.core.equilibrium import equilibrium, compute_density, compute_velocity


def bgk(f: jax.Array, tau: jax.Array) -> jax.Array:
    """BGK (single relaxation time) collision operator.

    The simplest and most widely used LBM collision. Each distribution
    relaxes toward equilibrium at a uniform rate:

    ``f_out[i] = f[i] - (f[i] - f_eq[i]) / tau``

    where tau is the dimensionless relaxation time, related to kinematic
    viscosity by ``nu = cs2 * (tau - 0.5)`` with ``cs2 = 1/3``.

    Stability requires ``tau > 0.5``. In practice, keep ``tau > 0.52``
    to avoid ringing near boundaries. Higher tau means higher viscosity
    (lower Reynolds number), which is more stable but more diffusive.

    Args:
        f: Distribution functions, shape ``(19, nx, ny, nz)``.
        tau: Relaxation time. Either a scalar (uniform viscosity) or an
            array of shape ``(nx, ny, nz)`` for spatially varying
            viscosity (e.g. from Smagorinsky). Must be > 0.5 everywhere.

    Returns:
        Post-collision distributions, shape ``(19, nx, ny, nz)``.

    Example:

    ```python
    >>> import jax.numpy as jnp
    >>> from jax_lbm.core.equilibrium import equilibrium
    >>> from jax_lbm.core.collision import bgk
    >>> nx, ny, nz = 8, 8, 8
    >>> rho = jnp.ones((nx, ny, nz))
    >>> u = jnp.zeros((3, nx, ny, nz)).at[0].set(0.05)
    >>> f = equilibrium(rho, u)
    >>> f_post = bgk(f, tau=0.8)
    >>> jnp.allclose(f, f_post, atol=1e-6)  # at equilibrium, collision is identity
    Array(True, dtype=bool)
    ```

    References:
        Bhatnagar, P.L., Gross, E.P. and Krook, M. (1954). "A Model for
        Collision Processes in Gases." Physical Review, 94(3), pp.511-525.
    """
    rho = compute_density(f)
    u = compute_velocity(f, rho)
    f_eq = equilibrium(rho, u)

    # tau can be scalar or (nx, ny, nz). Reshape for broadcasting against
    # the leading direction axis of f.
    if jnp.ndim(tau) > 0:
        omega = 1.0 / tau[jnp.newaxis, :, :, :]
    else:
        omega = 1.0 / tau

    return f - omega * (f - f_eq)


def mrt(f: jax.Array, tau: jax.Array) -> jax.Array:
    """MRT (multiple relaxation time) collision operator.

    Transforms each cell's distributions to moment space, relaxes each
    moment independently toward its equilibrium value, and transforms
    back. The stress moments relax at the physical viscosity rate
    (omega = 1/tau), while ghost and higher-order moments relax at
    rates tuned for stability.

    MRT is more stable than BGK at the same Reynolds number because you
    can damp non-physical modes aggressively. The downside is roughly 2x
    the arithmetic per cell. For differentiable simulation, MRT also
    produces smoother loss landscapes because the over-damped ghost modes
    reduce high-frequency noise in the gradient signal.

    The moment basis and relaxation rates match the reference solver's
    ``lbm_collide.comp`` and follow d'Humieres et al. (2002).

    Args:
        f: Distribution functions, shape ``(19, nx, ny, nz)``.
        tau: Relaxation time (scalar or ``(nx, ny, nz)``). The stress
            moments use ``omega = 1/tau``; other rates are fixed.

    Returns:
        Post-collision distributions, shape ``(19, nx, ny, nz)``.

    Example:

    ```python
    >>> import jax.numpy as jnp
    >>> from jax_lbm.core.equilibrium import equilibrium
    >>> from jax_lbm.core.collision import mrt
    >>> nx, ny, nz = 8, 8, 8
    >>> rho = jnp.ones((nx, ny, nz))
    >>> u = jnp.zeros((3, nx, ny, nz)).at[0].set(0.05)
    >>> f = equilibrium(rho, u)
    >>> f_post = mrt(f, tau=0.8)
    >>> jnp.allclose(f, f_post, atol=1e-6)  # equilibrium is a fixed point
    Array(True, dtype=bool)
    ```

    References:
        d'Humieres, D. et al. (2002). "Multiple-relaxation-time lattice
        Boltzmann models in three dimensions." Phil. Trans. R. Soc. A,
        360(1792), pp.437-451.
    """
    rho = compute_density(f)
    u = compute_velocity(f, rho)

    # Build relaxation rate vector. Start from the fixed rates, then
    # fill in stress moment slots with the actual omega.
    omega = 1.0 / tau  # scalar or (nx, ny, nz)

    def _collide_cell(f_cell, rho_cell, u_cell, omega_cell):
        """MRT collision for a single cell."""
        m = mrt_forward(f_cell)
        meq = mrt_equilibrium_moments(rho_cell, u_cell)

        # Set up per-moment relaxation rates.
        s = MRT_RELAXATION_RATES.at[MRT_STRESS_INDICES].set(omega_cell)

        # Relax: m_star = m - s * (m - meq)
        m_star = m - s * (m - meq)

        return mrt_inverse(m_star)

    # Vectorize over all spatial cells. f is (19, nx, ny, nz), so we
    # transpose to (nx, ny, nz, 19) for vmapping over the spatial dims,
    # then transpose back.
    nx, ny, nz = f.shape[1], f.shape[2], f.shape[3]
    f_flat = f.transpose(1, 2, 3, 0).reshape(-1, Q)        # (N, 19)
    rho_flat = rho.reshape(-1)                               # (N,)
    u_flat = u.transpose(1, 2, 3, 0).reshape(-1, 3)         # (N, 3)

    if jnp.ndim(tau) > 0:
        omega_flat = (1.0 / tau).reshape(-1)                 # (N,)
    else:
        omega_flat = jnp.broadcast_to(1.0 / tau, (rho_flat.shape[0],))

    f_out_flat = jax.vmap(_collide_cell)(f_flat, rho_flat, u_flat, omega_flat)

    return f_out_flat.reshape(nx, ny, nz, Q).transpose(3, 0, 1, 2)


def smagorinsky_tau(
    f: jax.Array,
    tau: jax.Array,
    cs: float = 0.1,
) -> jax.Array:
    """Compute effective relaxation time with Smagorinsky subgrid-scale model.

    In under-resolved simulations (coarse grids, high Reynolds numbers),
    the smallest turbulent eddies fall below the grid scale. The
    Smagorinsky model adds an eddy viscosity that increases in regions of
    strong velocity gradients, effectively widening the stable operating
    range.

    The effective tau is:

    ``tau_eff = 0.5 * (tau + sqrt(tau^2 + 18 * Cs^2 * |Pi_neq| / rho))``

    where ``|Pi_neq|`` is the Frobenius norm of the non-equilibrium stress
    tensor, computed directly from the distribution functions.

    This function returns the modified tau field. Pass it to ``bgk`` or
    ``mrt`` in place of the bare tau.

    Args:
        f: Distribution functions, shape ``(19, nx, ny, nz)``.
        tau: Base relaxation time (scalar). The molecular viscosity
            before turbulent enhancement.
        cs: Smagorinsky constant. Default 0.1, typical range 0.05-0.2.
            Larger values add more diffusion. The "correct" value depends
            on the flow and the grid resolution; 0.1 is the standard
            starting point.

    Returns:
        Effective relaxation time, shape ``(nx, ny, nz)``. Always >= tau,
        with equality in regions of zero velocity gradient.

    Example:

    ```python
    >>> import jax.numpy as jnp
    >>> from jax_lbm.core.equilibrium import equilibrium
    >>> from jax_lbm.core.collision import smagorinsky_tau, bgk
    >>> nx, ny, nz = 8, 8, 8
    >>> rho = jnp.ones((nx, ny, nz))
    >>> u = jnp.zeros((3, nx, ny, nz)).at[0].set(0.05)
    >>> f = equilibrium(rho, u)
    >>> tau_eff = smagorinsky_tau(f, tau=0.6, cs=0.1)
    >>> f_post = bgk(f, tau_eff)
    ```

    References:
        Hou, S. et al. (1996). "Simulation of Cavity Flow by the Lattice
        Boltzmann Method." J. Comp. Physics, 118(2), pp.329-347.
    """
    rho = compute_density(f)
    u = compute_velocity(f, rho)
    f_eq = equilibrium(rho, u)
    f_neq = f - f_eq

    e = D3Q19_VELOCITIES.astype(jnp.float32)  # (19, 3)

    # Non-equilibrium stress tensor components.
    # S_ab = sum_i(e_ia * e_ib * f_neq_i) for each cell.
    # We compute the 6 independent components: xx, yy, zz, xy, xz, yz.
    sxx = jnp.einsum("ixyz,i->xyz", f_neq, e[:, 0] * e[:, 0])
    syy = jnp.einsum("ixyz,i->xyz", f_neq, e[:, 1] * e[:, 1])
    szz = jnp.einsum("ixyz,i->xyz", f_neq, e[:, 2] * e[:, 2])
    sxy = jnp.einsum("ixyz,i->xyz", f_neq, e[:, 0] * e[:, 1])
    sxz = jnp.einsum("ixyz,i->xyz", f_neq, e[:, 0] * e[:, 2])
    syz = jnp.einsum("ixyz,i->xyz", f_neq, e[:, 1] * e[:, 2])

    # Frobenius norm of the symmetric stress tensor.
    pi_norm = jnp.sqrt(
        sxx * sxx + syy * syy + szz * szz
        + 2.0 * (sxy * sxy + sxz * sxz + syz * syz)
    )

    # tau_eff = 0.5 * (tau + sqrt(tau^2 + 18 * Cs^2 * |Pi| / rho))
    cs2 = cs * cs
    tau_eff = 0.5 * (
        tau + jnp.sqrt(tau * tau + 18.0 * cs2 * pi_norm / (rho + 1e-10))
    )

    return tau_eff
