# The Lattice Boltzmann Method

This page explains the physics for readers who know PDEs but not LBM.
If you already know LBM and just want the API, skip to the
[API reference](../api/core/lattice.md).

## The idea in one paragraph

Instead of discretizing the Navier-Stokes equations directly (finite
volume, finite element), the Lattice Boltzmann Method works one level
deeper. It simulates the statistical distribution of fictitious gas
particles on a regular lattice. Particles collide at each node (exchange
momentum locally) and then stream to neighboring nodes along fixed
velocity directions. Macroscopic quantities -- density, velocity,
pressure -- emerge as moments of the particle distribution. The method
recovers the incompressible Navier-Stokes equations in the low-Mach limit
through a Chapman-Enskog expansion.

## D3Q19: the lattice

"D3Q19" means 3 dimensions, 19 discrete velocities. Each lattice node
connects to 18 neighbors plus itself (the rest direction):

- **1 rest** direction: the particle stays put. Weight 1/3.
- **6 face** neighbors: along +x, -x, +y, -y, +z, -z. Weight 1/18 each.
- **12 edge** neighbors: diagonal pairs like +x+y, -x+z, etc. Weight 1/36 each.

The weights encode the Maxwell-Boltzmann distribution discretized onto this
stencil. They sum to 1.

In code, the velocities and weights are constants defined in
[`lattice.py`](../api/core/lattice.md):

```python
from jax_lbm.core.lattice import D3Q19_VELOCITIES, D3Q19_WEIGHTS

print(D3Q19_VELOCITIES.shape)  # (19, 3)
print(float(D3Q19_WEIGHTS.sum()))  # 1.0
```

## The distribution function

At each lattice node, we store 19 floating-point values: `f[0]` through
`f[18]`. Each `f[i]` represents the density of particles moving in
direction `i`. The full state of the simulation is a 4D array of shape
`(19, nx, ny, nz)`.

Macroscopic density and velocity are the zeroth and first moments:

$$\rho = \sum_{i=0}^{18} f_i, \qquad \rho \mathbf{u} = \sum_{i=0}^{18} f_i \mathbf{e}_i$$

See [`compute_density`](../api/core/equilibrium.md) and
[`compute_velocity`](../api/core/equilibrium.md).

## Equilibrium

The equilibrium distribution is what the particles would look like if the
fluid were in local thermodynamic equilibrium at the given density and
velocity:

$$f_i^{eq} = w_i \rho \left(1 + \frac{\mathbf{e}_i \cdot \mathbf{u}}{c_s^2} + \frac{(\mathbf{e}_i \cdot \mathbf{u})^2}{2 c_s^4} - \frac{\mathbf{u} \cdot \mathbf{u}}{2 c_s^2}\right)$$

where $c_s^2 = 1/3$ is the speed of sound squared on the lattice.

This is a second-order truncation of the Maxwell-Boltzmann distribution.
It conserves mass and momentum by construction: $\sum f_i^{eq} = \rho$
and $\sum f_i^{eq} \mathbf{e}_i = \rho \mathbf{u}$.

See [`equilibrium`](../api/core/equilibrium.md).

## Collision

Collision models how particles at each node scatter off each other. The
simplest model is BGK (single relaxation time):

$$f_i^{out} = f_i - \frac{1}{\tau}(f_i - f_i^{eq})$$

The relaxation time $\tau$ controls how fast the distribution relaxes
toward equilibrium. It determines the kinematic viscosity:

$$\nu = c_s^2 \left(\tau - \frac{1}{2}\right) = \frac{1}{3}\left(\tau - \frac{1}{2}\right)$$

Stability requires $\tau > 0.5$. Lower tau means lower viscosity (higher
Reynolds number), which is less stable.

MRT (multiple relaxation time) improves on BGK by transforming to moment
space and relaxing each moment independently. The physical viscosity is
still controlled by the stress moments, but non-physical "ghost" modes
can be damped aggressively for better stability at the same Re.

See [`bgk`](../api/core/collision.md) and [`mrt`](../api/core/collision.md).

## Streaming

After collision, each distribution moves one lattice step in its
direction:

$$f_i(\mathbf{x} + \mathbf{e}_i, t+1) = f_i^{out}(\mathbf{x}, t)$$

In our pull-based implementation, each cell pulls from its upstream
neighbor: `f[x, i] = f_post[x - e_i, i]`.

At solid walls, distributions bounce back instead of streaming through.
The [Bouzidi scheme](../api/core/streaming.md) interpolates the bounce
using the fractional wall distance for 2nd-order accuracy on curved
surfaces.

See [`stream`](../api/core/streaming.md).

## Boundary conditions

- **Zou-He inlet**: prescribes velocity at the domain entrance.
  Computes unknown distributions from the target velocity using
  non-equilibrium bounce-back.
- **Zou-He outlet**: prescribes pressure (density = 1.0) at the exit.
- **Periodic**: wraps distributions at Y/Z boundaries.
- **Bounce-back**: reverses distributions at solid walls.

See [`zou_he_inlet`](../api/core/boundary.md) and
[`zou_he_outlet`](../api/core/boundary.md).

## Dimensionless numbers

Everything in LBM is in lattice units (grid spacing = 1, timestep = 1).
The Reynolds number connects to physical parameters:

$$Re = \frac{U \cdot L}{\nu}$$

where $U$ is the characteristic velocity (inlet), $L$ is the
characteristic length (obstacle diameter in lattice cells), and $\nu$ is
the kinematic viscosity from tau.

Typical safe values: $U = 0.05$ (well below the lattice speed of sound
$c_s = 1/\sqrt{3} \approx 0.577$), giving a Mach number of about 0.087.

## Further reading

- Kruger, T. et al. (2017). *The Lattice Boltzmann Method: Principles
  and Practice*. Springer. The standard textbook.
- d'Humieres, D. et al. (2002). "Multiple-relaxation-time lattice
  Boltzmann models in three dimensions." Phil. Trans. R. Soc. A.
  The MRT reference.
- Bouzidi, M. et al. (2001). "Momentum transfer of a Boltzmann-lattice
  fluid with boundaries." Physics of Fluids. The Bouzidi scheme.
