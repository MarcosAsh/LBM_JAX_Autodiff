"""D3Q19 lattice constants and MRT moment-space transforms.

The D3Q19 lattice has 19 discrete velocity directions in 3D: one rest
velocity, six axis-aligned (face) neighbors, and twelve diagonal (edge)
neighbors. This module defines the velocity vectors, quadrature weights,
opposite-direction map, and the MRT (multiple relaxation time) transforms
needed for collision in moment space.

All arrays are immutable JAX arrays created at import time. They live on
the default device and can be used directly inside ``jit``-compiled
functions.

The velocity ordering and MRT basis follow d'Humieres et al. (2002) and
match the reference C/GLSL solver at
``~/coding_projects/personal/3d-fluid-dynamics/simulation/shaders/lbm_collide.comp``.

References:
    d'Humieres, D. et al. (2002). "Multiple-relaxation-time lattice
    Boltzmann models in three dimensions." Philosophical Transactions of
    the Royal Society A, 360(1792), pp.437-451.
"""

import jax.numpy as jnp


# Velocity vectors: 19 directions, each a 3-component integer vector.
# Index 0 is rest, 1-6 are face neighbors (+x, -x, +y, -y, +z, -z),
# 7-18 are edge (diagonal) neighbors.
D3Q19_VELOCITIES: jnp.ndarray = jnp.array([
    [0,  0,  0],   # 0: rest
    [1,  0,  0],   # 1: +x
    [-1, 0,  0],   # 2: -x
    [0,  1,  0],   # 3: +y
    [0, -1,  0],   # 4: -y
    [0,  0,  1],   # 5: +z
    [0,  0, -1],   # 6: -z
    [1,  1,  0],   # 7: +x+y
    [-1, 1,  0],   # 8: -x+y
    [1, -1,  0],   # 9: +x-y
    [-1, -1, 0],   # 10: -x-y
    [1,  0,  1],   # 11: +x+z
    [-1, 0,  1],   # 12: -x+z
    [1,  0, -1],   # 13: +x-z
    [-1, 0, -1],   # 14: -x-z
    [0,  1,  1],   # 15: +y+z
    [0, -1,  1],   # 16: -y+z
    [0,  1, -1],   # 17: +y-z
    [0, -1, -1],   # 18: -y-z
], dtype=jnp.int32)

# Quadrature weights. Rest gets 1/3, face neighbors 1/18, edge neighbors 1/36.
# These must sum to exactly 1.
D3Q19_WEIGHTS: jnp.ndarray = jnp.array([
    1.0 / 3.0,                                         # rest
    1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,               # +x, -x, +y
    1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,               # -y, +z, -z
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,   # edges xy
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,   # edges xz
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,   # edges yz
], dtype=jnp.float32)

# Opposite direction for each lattice velocity. If direction i points in
# some direction, opposite[i] points the other way. Used for bounce-back
# boundary conditions and momentum exchange force computation.
#
# Rest (0) maps to itself. The pairs are:
#   1<->2, 3<->4, 5<->6, 7<->10, 8<->9, 11<->14, 12<->13, 15<->18, 16<->17
D3Q19_OPPOSITE: jnp.ndarray = jnp.array(
    [0, 2, 1, 4, 3, 6, 5, 10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15],
    dtype=jnp.int32,
)

# Speed of sound squared in lattice units. For D3Q19 (and all standard
# lattices on a unit grid with unit timestep), cs^2 = 1/3.
CS2: float = 1.0 / 3.0

# Inverse of cs^2 and cs^4, used repeatedly in equilibrium and collision.
INV_CS2: float = 3.0
INV_CS4: float = 9.0

# Number of discrete velocities.
Q: int = 19


# MRT row norms: ||M_k||^2 for each moment row k. The inverse transform
# divides each scaled moment by these norms. Copied from the reference
# solver's inverseMRT function.
MRT_ROW_NORMS: jnp.ndarray = jnp.array([
    19.0, 2394.0, 252.0,
    10.0, 40.0, 10.0, 40.0, 10.0, 40.0,
    36.0, 72.0, 12.0, 24.0,
    4.0, 4.0, 4.0,
    8.0, 8.0, 8.0,
], dtype=jnp.float32)

# Default MRT relaxation rates. Conserved moments (rho, jx, jy, jz) have
# rate 0 -- they are not relaxed. Stress moments relax at the physical
# viscosity rate (omega = 1/tau). Ghost/higher-order moments are damped
# aggressively. The stress rate slots (indices 9, 11, 13, 14, 15) are
# filled in at runtime with the actual omega.
#
# Index:  0     1     2     3    4    5    6    7    8
#         rho   e     eps   jx   qx   jy   qy   jz   qz
# Index:  9     10    11    12   13   14   15   16   17   18
#         pxx  pixx   pww  piww  pxy  pyz  pxz  mx   my   mz
MRT_RELAXATION_RATES: jnp.ndarray = jnp.array([
    0.0,    # 0:  rho (conserved)
    1.19,   # 1:  energy
    1.4,    # 2:  energy square
    0.0,    # 3:  jx (conserved)
    1.2,    # 4:  qx (energy flux)
    0.0,    # 5:  jy (conserved)
    1.2,    # 6:  qy
    0.0,    # 7:  jz (conserved)
    1.2,    # 8:  qz
    0.0,    # 9:  pxx (stress -- replaced by omega at runtime)
    1.4,    # 10: pixx (ghost)
    0.0,    # 11: pww (stress -- replaced by omega at runtime)
    1.4,    # 12: piww (ghost)
    0.0,    # 13: pxy (stress -- replaced by omega at runtime)
    0.0,    # 14: pyz (stress -- replaced by omega at runtime)
    0.0,    # 15: pxz (stress -- replaced by omega at runtime)
    1.98,   # 16: mx (ghost)
    1.98,   # 17: my (ghost)
    1.98,   # 18: mz (ghost)
], dtype=jnp.float32)

# Indices of the stress moments whose relaxation rate equals omega.
MRT_STRESS_INDICES: jnp.ndarray = jnp.array([9, 11, 13, 14, 15], dtype=jnp.int32)


def mrt_forward(f_cell: jnp.ndarray) -> jnp.ndarray:
    """Transform distribution functions to moment space.

    Computes the 19 moments from the 19 distribution values at a single
    lattice cell. The transform exploits the sparsity of the D3Q19 moment
    matrix -- no full 19x19 matrix multiply. This matches the reference
    solver's ``forwardMRT`` function exactly.

    The moment ordering is:
        0: rho (density), 1: e (energy), 2: eps (energy square),
        3: jx, 4: qx, 5: jy, 6: qy, 7: jz, 8: qz (momentum and
        energy flux), 9: pxx, 10: pixx, 11: pww, 12: piww (normal
        stress and their ghosts), 13: pxy, 14: pyz, 15: pxz (shear
        stress), 16-18: mx, my, mz (ghost/kinetic modes).

    Args:
        f_cell: Distribution function values at one cell, shape ``(19,)``.

    Returns:
        Moment vector, shape ``(19,)``.

    Example:

    ```python
    >>> import jax.numpy as jnp
    >>> from jax_lbm.core.lattice import mrt_forward, D3Q19_WEIGHTS
    >>> f_cell = D3Q19_WEIGHTS  # equilibrium at rest (rho=1, u=0)
    >>> m = mrt_forward(f_cell)
    >>> float(m[0])  # rho = 1
    1.0
    ```

    References:
        d'Humieres, D. et al. (2002), Table II for the moment matrix.
    """
    # Unpack for readability. The compiler will optimize this.
    f0 = f_cell[0]
    f1, f2, f3, f4, f5, f6 = f_cell[1], f_cell[2], f_cell[3], f_cell[4], f_cell[5], f_cell[6]
    f7, f8, f9, f10 = f_cell[7], f_cell[8], f_cell[9], f_cell[10]
    f11, f12, f13, f14 = f_cell[11], f_cell[12], f_cell[13], f_cell[14]
    f15, f16, f17, f18 = f_cell[15], f_cell[16], f_cell[17], f_cell[18]

    # Partial sums used in multiple moments.
    sf = f1 + f2 + f3 + f4 + f5 + f6          # sum of face neighbors
    se = f7 + f8 + f9 + f10 + f11 + f12 + f13 + f14 + f15 + f16 + f17 + f18

    sXY = f7 + f8 + f9 + f10
    sXZ = f11 + f12 + f13 + f14
    sYZ = f15 + f16 + f17 + f18

    m = jnp.zeros(19)

    m = m.at[0].set(f0 + sf + se)                                # rho
    m = m.at[1].set(-30.0 * f0 - 11.0 * sf + 8.0 * se)          # energy
    m = m.at[2].set(12.0 * f0 - 4.0 * sf + se)                  # energy square

    m = m.at[3].set(f1 - f2 + f7 - f8 + f9 - f10 + f11 - f12 + f13 - f14)   # jx
    m = m.at[4].set(-4.0 * (f1 - f2) + f7 - f8 + f9 - f10 + f11 - f12 + f13 - f14)  # qx

    m = m.at[5].set(f3 - f4 + f7 + f8 - f9 - f10 + f15 - f16 + f17 - f18)   # jy
    m = m.at[6].set(-4.0 * (f3 - f4) + f7 + f8 - f9 - f10 + f15 - f16 + f17 - f18)  # qy

    m = m.at[7].set(f5 - f6 + f11 + f12 - f13 - f14 + f15 + f16 - f17 - f18)  # jz
    m = m.at[8].set(-4.0 * (f5 - f6) + f11 + f12 - f13 - f14 + f15 + f16 - f17 - f18)  # qz

    m = m.at[9].set(2.0 * (f1 + f2) - (f3 + f4) - (f5 + f6) + sXY + sXZ - 2.0 * sYZ)  # 3pxx
    m = m.at[10].set(-4.0 * (f1 + f2) + 2.0 * (f3 + f4) + 2.0 * (f5 + f6) + sXY + sXZ - 2.0 * sYZ)  # 3pixx

    m = m.at[11].set((f3 + f4) - (f5 + f6) + sXY - sXZ)         # pww
    m = m.at[12].set(-2.0 * (f3 + f4) + 2.0 * (f5 + f6) + sXY - sXZ)  # piww

    m = m.at[13].set(f7 - f8 - f9 + f10)                         # pxy
    m = m.at[14].set(f15 - f16 - f17 + f18)                      # pyz
    m = m.at[15].set(f11 - f12 - f13 + f14)                      # pxz

    m = m.at[16].set(f7 - f8 + f9 - f10 - f11 + f12 - f13 + f14)  # mx
    m = m.at[17].set(-f7 - f8 + f9 + f10 + f15 - f16 + f17 - f18)  # my
    m = m.at[18].set(f11 + f12 - f13 - f14 - f15 - f16 + f17 + f18)  # mz

    return m


def mrt_inverse(m: jnp.ndarray) -> jnp.ndarray:
    """Transform moment-space values back to distribution functions.

    This is the inverse of ``mrt_forward``. It uses the transpose of the
    moment matrix scaled by the row norms: M^{-1}_{ij} = M_{ji} / ||row_j||^2.
    The implementation is fully unrolled (no matrix multiply) for efficiency
    and to match the reference solver's ``inverseMRT`` line by line.

    Args:
        m: Moment vector, shape ``(19,)``.

    Returns:
        Distribution function values, shape ``(19,)``.

    Example:

    ```python
    >>> import jax.numpy as jnp
    >>> from jax_lbm.core.lattice import mrt_forward, mrt_inverse, D3Q19_WEIGHTS
    >>> f_cell = D3Q19_WEIGHTS
    >>> m = mrt_forward(f_cell)
    >>> f_recovered = mrt_inverse(m)
    >>> jnp.allclose(f_cell, f_recovered, atol=1e-6)
    Array(True, dtype=bool)
    ```

    References:
        d'Humieres, D. et al. (2002), Eq. (22) for the inverse transform.
    """
    # Scale each moment by the inverse row norm.
    ms = m / MRT_ROW_NORMS

    ms0, ms1, ms2 = ms[0], ms[1], ms[2]
    ms3, ms4, ms5, ms6 = ms[3], ms[4], ms[5], ms[6]
    ms7, ms8 = ms[7], ms[8]
    ms9, ms10, ms11, ms12 = ms[9], ms[10], ms[11], ms[12]
    ms13, ms14, ms15 = ms[13], ms[14], ms[15]
    ms16, ms17, ms18 = ms[16], ms[17], ms[18]

    f = jnp.zeros(19)

    f = f.at[0].set(ms0 - 30.0 * ms1 + 12.0 * ms2)

    f = f.at[1].set(ms0 - 11.0 * ms1 - 4.0 * ms2
                    + ms3 - 4.0 * ms4 + 2.0 * ms9 - 4.0 * ms10)

    f = f.at[2].set(ms0 - 11.0 * ms1 - 4.0 * ms2
                    - ms3 + 4.0 * ms4 + 2.0 * ms9 - 4.0 * ms10)

    f = f.at[3].set(ms0 - 11.0 * ms1 - 4.0 * ms2
                    + ms5 - 4.0 * ms6 - ms9 + 2.0 * ms10
                    + ms11 - 2.0 * ms12)

    f = f.at[4].set(ms0 - 11.0 * ms1 - 4.0 * ms2
                    - ms5 + 4.0 * ms6 - ms9 + 2.0 * ms10
                    + ms11 - 2.0 * ms12)

    f = f.at[5].set(ms0 - 11.0 * ms1 - 4.0 * ms2
                    + ms7 - 4.0 * ms8 - ms9 + 2.0 * ms10
                    - ms11 + 2.0 * ms12)

    f = f.at[6].set(ms0 - 11.0 * ms1 - 4.0 * ms2
                    - ms7 + 4.0 * ms8 - ms9 + 2.0 * ms10
                    - ms11 + 2.0 * ms12)

    f = f.at[7].set(ms0 + 8.0 * ms1 + ms2
                    + ms3 + ms4 + ms5 + ms6
                    + ms9 + ms10 + ms11 + ms12
                    + ms13 + ms16 - ms17)

    f = f.at[8].set(ms0 + 8.0 * ms1 + ms2
                    - ms3 - ms4 + ms5 + ms6
                    + ms9 + ms10 + ms11 + ms12
                    - ms13 - ms16 - ms17)

    f = f.at[9].set(ms0 + 8.0 * ms1 + ms2
                    + ms3 + ms4 - ms5 - ms6
                    + ms9 + ms10 + ms11 + ms12
                    - ms13 + ms16 + ms17)

    f = f.at[10].set(ms0 + 8.0 * ms1 + ms2
                     - ms3 - ms4 - ms5 - ms6
                     + ms9 + ms10 + ms11 + ms12
                     + ms13 - ms16 + ms17)

    f = f.at[11].set(ms0 + 8.0 * ms1 + ms2
                     + ms3 + ms4 + ms7 + ms8
                     + ms9 + ms10 - ms11 - ms12
                     + ms15 - ms16 + ms18)

    f = f.at[12].set(ms0 + 8.0 * ms1 + ms2
                     - ms3 - ms4 + ms7 + ms8
                     + ms9 + ms10 - ms11 - ms12
                     - ms15 + ms16 + ms18)

    f = f.at[13].set(ms0 + 8.0 * ms1 + ms2
                     + ms3 + ms4 - ms7 - ms8
                     + ms9 + ms10 - ms11 - ms12
                     - ms15 - ms16 - ms18)

    f = f.at[14].set(ms0 + 8.0 * ms1 + ms2
                     - ms3 - ms4 - ms7 - ms8
                     + ms9 + ms10 - ms11 - ms12
                     + ms15 + ms16 - ms18)

    f = f.at[15].set(ms0 + 8.0 * ms1 + ms2
                     + ms5 + ms6 + ms7 + ms8
                     - 2.0 * ms9 - 2.0 * ms10
                     + ms14 + ms17 - ms18)

    f = f.at[16].set(ms0 + 8.0 * ms1 + ms2
                     - ms5 - ms6 + ms7 + ms8
                     - 2.0 * ms9 - 2.0 * ms10
                     - ms14 - ms17 - ms18)

    f = f.at[17].set(ms0 + 8.0 * ms1 + ms2
                     + ms5 + ms6 - ms7 - ms8
                     - 2.0 * ms9 - 2.0 * ms10
                     - ms14 + ms17 + ms18)

    f = f.at[18].set(ms0 + 8.0 * ms1 + ms2
                     - ms5 - ms6 - ms7 - ms8
                     - 2.0 * ms9 - 2.0 * ms10
                     + ms14 - ms17 + ms18)

    return f


def mrt_equilibrium_moments(rho: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
    """Compute the equilibrium values of all 19 moments.

    In MRT collision, you relax each moment toward its equilibrium value
    independently. This function gives you those targets. The conserved
    moments (rho, jx, jy, jz) are trivially rho and rho*u. The stress
    moments depend on rho and u^2. Ghost moments are zero at equilibrium.

    Args:
        rho: Density at one cell, scalar.
        u: Velocity at one cell, shape ``(3,)`` as ``(ux, uy, uz)``.

    Returns:
        Equilibrium moment vector, shape ``(19,)``.

    Example:

    ```python
    >>> import jax.numpy as jnp
    >>> from jax_lbm.core.lattice import mrt_equilibrium_moments
    >>> meq = mrt_equilibrium_moments(1.0, jnp.array([0.05, 0.0, 0.0]))
    >>> float(meq[0])  # rho
    1.0
    >>> float(meq[3])  # jx = rho * ux
    0.05
    ```
    """
    ux, uy, uz = u[0], u[1], u[2]
    u2 = ux * ux + uy * uy + uz * uz

    meq = jnp.zeros(19)
    meq = meq.at[0].set(rho)
    meq = meq.at[1].set(-11.0 * rho + 19.0 * rho * u2)
    meq = meq.at[2].set(3.0 * rho - 5.5 * rho * u2)
    meq = meq.at[3].set(rho * ux)
    meq = meq.at[4].set(-2.0 / 3.0 * rho * ux)
    meq = meq.at[5].set(rho * uy)
    meq = meq.at[6].set(-2.0 / 3.0 * rho * uy)
    meq = meq.at[7].set(rho * uz)
    meq = meq.at[8].set(-2.0 / 3.0 * rho * uz)
    meq = meq.at[9].set(rho * (2.0 * ux * ux - uy * uy - uz * uz))
    meq = meq.at[10].set(-0.5 * rho * (2.0 * ux * ux - uy * uy - uz * uz))
    meq = meq.at[11].set(rho * (uy * uy - uz * uz))
    meq = meq.at[12].set(-0.5 * rho * (uy * uy - uz * uz))
    meq = meq.at[13].set(rho * ux * uy)
    meq = meq.at[14].set(rho * uy * uz)
    meq = meq.at[15].set(rho * ux * uz)
    # Ghost moments 16, 17, 18 are zero at equilibrium.

    return meq
