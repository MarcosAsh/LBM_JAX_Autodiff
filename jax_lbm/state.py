"""Simulation state, parameters, and the core step/simulate functions.

The LBM simulation state is a flat pytree (NamedTuple) containing all the
arrays that evolve over time: the distribution functions, solid mask, and
Bouzidi wall distances. The simulation parameters are a separate pytree
containing the physical constants and configuration that stay fixed during
a simulation run (but that you might want gradients with respect to).

The ``step`` function is the heart of the solver. It performs one LBM
timestep: collide, stream, apply boundary conditions. It is pure,
JIT-compilable, and differentiable.

The ``simulate`` function runs multiple steps using ``jax.lax.fori_loop``
for efficient compilation and optional ``jax.checkpoint`` for memory-bounded
backpropagation.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

import numpy as np

from jax_lbm.core.collision import bgk, mrt, smagorinsky_tau
from jax_lbm.core.streaming import stream
from jax_lbm.core.boundary import zou_he_inlet, zou_he_outlet
from jax_lbm.core.equilibrium import equilibrium, compute_density, compute_velocity
from jax_lbm.core.forces import momentum_exchange, drag_coefficient, projected_area
from jax_lbm.core.lattice import D3Q19_OPPOSITE, Q


class LBMState(NamedTuple):
    """Complete simulation state at one point in time.

    This is a JAX pytree, so it can be passed through ``jax.jit``,
    ``jax.grad``, ``jax.lax.fori_loop``, and ``jax.checkpoint``
    transparently. All fields are immutable JAX arrays.

    Fields:
        f: Distribution functions, shape ``(19, nx, ny, nz)``. The 19
            discrete velocity populations at every lattice cell. This is
            the primary state variable; density and velocity are derived
            from it.
        solid: Solid mask, shape ``(nx, ny, nz)``, dtype int32. 0 = fluid,
            1 = body solid, 2 = ground (optional). Does not change during
            simulation.
        q: Bouzidi fractional wall distances, shape ``(19, nx, ny, nz)``.
            Values in [0, 1] at boundary links, -1 elsewhere. Fixed during
            simulation unless you are doing shape optimization.
    """
    f: jax.Array
    solid: jax.Array
    q: jax.Array


class LBMParams(NamedTuple):
    """Physical parameters and solver configuration.

    These values are fixed during a forward simulation but may be
    differentiated with respect to for inverse design. For example,
    ``jax.grad`` of the drag coefficient with respect to ``inlet_velocity``
    or ``tau`` tells you how sensitive the flow is to those inputs.

    Fields:
        tau: Relaxation time (scalar). Related to kinematic viscosity by
            ``nu = (1/3) * (tau - 0.5)``. Must be > 0.5 for stability.
        inlet_velocity: Velocity vector at the inlet, shape ``(3,)`` as
            ``(ux, uy, uz)``. Typically ``(0.05, 0, 0)`` for a wind
            tunnel in the +x direction.
        collision_mode: Which collision operator to use. 0 = BGK (default),
            1 = MRT.
        use_smagorinsky: Whether to apply the Smagorinsky SGS model.
            0 = off (default), 1 = on.
        smagorinsky_cs: Smagorinsky constant. Default 0.1.
        periodic_yz: Whether Y/Z boundaries are periodic. True (default)
            or False (free-slip clamping).
    """
    tau: jax.Array
    inlet_velocity: jax.Array
    collision_mode: int = 0
    use_smagorinsky: int = 0
    smagorinsky_cs: float = 0.1
    periodic_yz: bool = True


def step(state: LBMState, params: LBMParams) -> LBMState:
    """Perform one LBM timestep: collide, stream, apply boundaries.

    This is the function you will loop over in a simulation, and the
    function you will differentiate through for optimization. It takes
    the current state, applies the collision operator, streams the
    distributions to neighboring cells (with Bouzidi bounce-back at
    walls), and then enforces the inlet/outlet boundary conditions.

    The function is pure (no side effects), JIT-compilable, and
    differentiable via ``jax.grad``.

    Args:
        state: Current simulation state (distributions, solid mask, q).
        params: Physical parameters (tau, inlet velocity, collision mode).

    Returns:
        Updated simulation state after one timestep.

    Example:

    ```python
    >>> import jax.numpy as jnp
    >>> from jax_lbm import LBMState, LBMParams, step, equilibrium
    >>> nx, ny, nz = 32, 16, 16
    >>> rho = jnp.ones((nx, ny, nz))
    >>> u = jnp.zeros((3, nx, ny, nz))
    >>> f = equilibrium(rho, u)
    >>> state = LBMState(
    ...     f=f,
    ...     solid=jnp.zeros((nx, ny, nz), dtype=jnp.int32),
    ...     q=-jnp.ones((19, nx, ny, nz)),
    ... )
    >>> params = LBMParams(
    ...     tau=jnp.float32(0.8),
    ...     inlet_velocity=jnp.array([0.05, 0.0, 0.0]),
    ... )
    >>> new_state = step(state, params)
    >>> new_state.f.shape
    (19, 32, 16, 16)
    ```
    """
    f = state.f
    solid = state.solid
    tau = params.tau

    # Optionally modify tau with Smagorinsky SGS.
    if params.use_smagorinsky:
        tau = smagorinsky_tau(f, tau, cs=params.smagorinsky_cs)

    # Collision: relax distributions toward equilibrium.
    if params.collision_mode == 1:
        f = mrt(f, tau)
    else:
        f = bgk(f, tau)

    # Solid cells: simple bounce-back. Reverse all distributions so that
    # any population heading into the wall bounces back the way it came.
    # This matches the reference solver's solid handling in lbm_collide.comp.
    opp_list = [int(x) for x in np.asarray(D3Q19_OPPOSITE)]
    f_bounced = f[jnp.array(opp_list)]  # (19, nx, ny, nz) with directions reversed
    is_solid = (solid >= 1)[jnp.newaxis, :, :, :]  # (1, nx, ny, nz)
    f = jnp.where(is_solid, f_bounced, f)

    # Streaming: propagate distributions and apply Bouzidi bounce-back.
    f = stream(f, state.q, periodic_yz=params.periodic_yz)

    # Boundary conditions: Zou-He inlet and outlet.
    f = zou_he_inlet(f, params.inlet_velocity)
    f = zou_he_outlet(f, rho_out=1.0)

    # Stability guard: reset diverged cells to equilibrium. Matches the
    # reference solver's guard in lbm_collide.comp. Without this, a single
    # bad cell can cascade into domain-wide NaN.
    rho = compute_density(f)
    u = compute_velocity(f, rho)
    u_mag = jnp.sqrt(jnp.sum(u * u, axis=0))
    unstable = (rho < 0.3) | (rho > 2.0) | (u_mag > 0.4) | jnp.isnan(rho)
    f_eq_reset = equilibrium(
        jnp.where(unstable, 1.0, rho),
        jnp.where(unstable[jnp.newaxis, :, :, :], 0.0, u),
    )
    f = jnp.where(unstable[jnp.newaxis, :, :, :], f_eq_reset, f)

    return LBMState(f=f, solid=state.solid, q=state.q)


def simulate(
    state: LBMState,
    params: LBMParams,
    n_steps: int,
    checkpoint: bool = True,
) -> LBMState:
    """Run multiple LBM timesteps using ``jax.lax.fori_loop``.

    This compiles the entire simulation loop into a single XLA program,
    avoiding Python loop overhead. When ``checkpoint=True`` (the default),
    each step is wrapped with ``jax.checkpoint`` so that backward passes
    recompute intermediates instead of storing them, keeping memory usage
    constant regardless of the number of steps.

    Args:
        state: Initial simulation state.
        params: Physical parameters (fixed during simulation).
        n_steps: Number of LBM timesteps to run.
        checkpoint: If True (default), wrap each step with
            ``jax.checkpoint`` for memory-bounded backpropagation. Set
            to False for short simulations where memory is not a concern,
            or when you want maximum forward-pass speed.

    Returns:
        Final simulation state after ``n_steps`` timesteps.

    Example:

    ```python
    >>> import jax.numpy as jnp
    >>> from jax_lbm import LBMState, LBMParams, simulate, equilibrium
    >>> nx, ny, nz = 32, 16, 16
    >>> rho = jnp.ones((nx, ny, nz))
    >>> u = jnp.zeros((3, nx, ny, nz))
    >>> f = equilibrium(rho, u)
    >>> state = LBMState(
    ...     f=f,
    ...     solid=jnp.zeros((nx, ny, nz), dtype=jnp.int32),
    ...     q=-jnp.ones((19, nx, ny, nz)),
    ... )
    >>> params = LBMParams(
    ...     tau=jnp.float32(0.8),
    ...     inlet_velocity=jnp.array([0.05, 0.0, 0.0]),
    ... )
    >>> final_state = simulate(state, params, n_steps=100)
    >>> final_state.f.shape
    (19, 32, 16, 16)
    ```
    """
    step_fn = step
    if checkpoint:
        step_fn = jax.checkpoint(step)

    def body(i, s):
        return step_fn(s, params)

    return jax.lax.fori_loop(0, n_steps, body, state)
