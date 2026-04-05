"""Shape optimization with Adam optimizer -- clean convergence for the paper.

Same problem as 04_shape_optimization.py (minimize drag on a sphere by
adjusting its radius), but uses Adam from optax instead of plain SGD.
Adam's momentum and adaptive learning rate smooth out the noisy LBM
gradients, producing a monotonic convergence curve suitable for
publication.

Also runs more simulation steps per evaluation (200) to let the flow
develop further toward steady state before measuring drag.

Usage:
    python examples/06_shape_optimization_adam.py
"""

import jax
import jax.numpy as jnp
import optax

from jax_lbm.core.equilibrium import equilibrium, compute_density, compute_velocity
from jax_lbm.core.collision import bgk
from jax_lbm.core.streaming import stream
from jax_lbm.core.boundary import zou_he_inlet, zou_he_outlet
from jax_lbm.core.forces import momentum_exchange, drag_coefficient, projected_area
from jax_lbm.geometry.smooth import smooth_sphere_geometry, soft_bounce_back


def simulate_cd(radius, nx, ny, nz, n_steps, u_inlet, tau):
    """Forward simulation: Cd as a differentiable function of radius."""
    porosity, solid, q = smooth_sphere_geometry(
        nx, ny, nz, center=(0.0, 0.0, 0.0), radius=radius, sharpness=20.0,
    )

    rho = jnp.ones((nx, ny, nz))
    u = jnp.zeros((3, nx, ny, nz)).at[0].set(u_inlet)
    f = equilibrium(rho, u)
    vel = jnp.array([u_inlet, 0.0, 0.0])

    def step_fn(carry, _):
        f_in, _ = carry
        f_c = bgk(f_in, tau)
        f_c = soft_bounce_back(f_c, porosity)
        f_post = f_c
        f_s = stream(f_c, q, periodic_yz=True)
        f_s = zou_he_inlet(f_s, vel)
        f_s = zou_he_outlet(f_s, rho_out=1.0)
        # Stability guard.
        rho_s = compute_density(f_s)
        u_s = compute_velocity(f_s, rho_s)
        u_mag = jnp.sqrt(jnp.sum(u_s * u_s, axis=0))
        bad = (rho_s < 0.3) | (rho_s > 2.0) | (u_mag > 0.4) | jnp.isnan(rho_s)
        f_eq_r = equilibrium(jnp.where(bad, 1.0, rho_s),
                             jnp.where(bad[None], 0.0, u_s))
        f_s = jnp.where(bad[None], f_eq_r, f_s)
        return (f_s, f_post), None

    (f_final, f_post), _ = jax.lax.scan(step_fn, (f, f), None, length=n_steps)

    force = momentum_exchange(f_post, f_final, q)
    area = jnp.maximum(projected_area(solid, axis=0), 1.0)
    return drag_coefficient(force, u_inlet, area)


def main():
    nx, ny, nz = 48, 24, 24
    u_inlet = 0.05
    n_steps = 200

    D_lattice = 48 * 0.5 / 4.0
    re = 50.0
    nu = u_inlet * D_lattice / re
    tau = jnp.float32(3.0 * nu + 0.5)

    print("Shape Optimization with Adam Optimizer")
    print(f"Grid: {nx}x{ny}x{nz}, Re={re:.0f}, tau={float(tau):.4f}")
    print(f"Steps per evaluation: {n_steps}")
    print()

    # JIT the forward + backward pass.
    loss_and_grad = jax.jit(jax.value_and_grad(
        lambda r: simulate_cd(r, nx, ny, nz, n_steps, u_inlet, tau)
    ))

    # Warm up.
    print("Compiling...")
    radius = jnp.float32(0.5)
    cd_init, _ = loss_and_grad(radius)
    print(f"Initial Cd at radius=0.5: {float(cd_init):.6f}")
    print()

    # Adam optimizer.
    optimizer = optax.adam(learning_rate=0.003)
    opt_state = optimizer.init(radius)

    n_iters = 30
    best_cd = float(cd_init)
    best_radius = float(radius)

    print(f"{'Iter':>4s}  {'Radius':>8s}  {'Cd':>10s}  {'dCd/dr':>10s}  {'Best Cd':>10s}")
    print("-" * 52)

    cd_history = []

    for i in range(n_iters):
        cd, grad = loss_and_grad(radius)
        cd_val = float(cd)
        cd_history.append(cd_val)

        if cd_val < best_cd:
            best_cd = cd_val
            best_radius = float(radius)

        print(f"{i:4d}  {float(radius):8.4f}  {cd_val:10.6f}  {float(grad):10.4f}  {best_cd:10.6f}")

        # Adam update.
        updates, opt_state = optimizer.update(grad, opt_state)
        radius = optax.apply_updates(radius, updates)
        radius = jnp.clip(radius, 0.2, 0.8)

    print()
    print(f"Initial:  Cd={float(cd_init):.6f}  radius=0.5000")
    print(f"Best:     Cd={best_cd:.6f}  radius={best_radius:.4f}")

    reduction = (1.0 - best_cd / float(cd_init)) * 100
    if reduction > 0:
        print(f"Drag reduced by {reduction:.1f}%")
    else:
        print("Cd did not decrease.")

    # Convergence history.
    print()
    print("Convergence:")
    for i, cd_val in enumerate(cd_history):
        bar_len = max(1, int(cd_val / float(cd_init) * 40))
        bar = "#" * bar_len
        print(f"  {i:3d}  {cd_val:8.5f}  {bar}")


if __name__ == "__main__":
    main()
