"""Shape optimisation with a physical baseline and a rigorous BB comparison.

Extends the paper's Bouzidi-vs-standard optimisation experiment
(examples/07) along three axes:

1. Larger grid (96x48x48, D=12) and longer per-iteration simulation
   (2000 MRT steps, ~2 flow-through times). This makes the baseline
   Cd physically meaningful instead of a transient startup artefact.

2. 60 Adam iterations instead of 25, so we can show the trajectory has
   plateaued rather than cutting it off early.

3. Four boundary treatments, not two, to rigorously isolate the
   contribution of each differentiable component:

     - bouzidi_soft : ray-sphere q + sigmoid porosity (paper's main).
     - halfway_soft : q=0.5 + sigmoid porosity (paper's comparison).
     - bouzidi_hard : ray-sphere q + hard solid mask (isolate q alone).
     - halfway_hard : q=0.5 + hard mask (truly classical LBM, grad=0).

   The fourth case is the one the paper is missing. Production LBM
   codes use hard masks and halfway bounce-back; there is no
   differentiable path to the geometry, and the gradient is exactly
   zero. That is the real baseline the Bouzidi+AD pipeline displaces.

MRT is used instead of BGK because two separate gradient-convergence
runs showed BGK at tau=0.518 going unstable beyond a few hundred
steps, contaminating the comparison.

Usage:
    python examples/11_optimisation_v2.py
    modal run modal_worker.py --example optimisation-v2
"""

import math
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_lbm.core.lattice import D3Q19_VELOCITIES, Q
from jax_lbm.core.equilibrium import equilibrium, compute_density, compute_velocity
from jax_lbm.core.collision import mrt
from jax_lbm.core.streaming import stream
from jax_lbm.core.boundary import zou_he_inlet, zou_he_outlet
from jax_lbm.core.forces import momentum_exchange, drag_coefficient, projected_area
from jax_lbm.geometry.smooth import (
    signed_distance_sphere,
    porosity_from_sdf,
    soft_bounce_back,
)


def _build_geometry(radius, nx, ny, nz, q_mode, porosity_mode, sharpness=20.0):
    """Build geometry. q_mode in {bouzidi, halfway}, porosity_mode in {soft, hard}."""
    sdf = signed_distance_sphere(nx, ny, nz, (0.0, 0.0, 0.0), radius)
    porosity_soft = porosity_from_sdf(sdf, sharpness=sharpness)
    solid = (porosity_soft > 0.5).astype(jnp.int32)

    if porosity_mode == "hard":
        porosity = solid.astype(jnp.float32)  # zero gradient w.r.t. radius
    else:
        porosity = porosity_soft

    scale_x, scale_y, scale_z = nx / 8.0, ny / 4.0, nz / 4.0
    cx, cy, cz = 0.0, 0.0, 0.0

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

        if q_mode == "bouzidi":
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
        else:
            q_val = 0.5

        q = q.at[i].set(jnp.where(is_boundary, q_val, -1.0))

    return porosity, solid, q


def _chunked_scan(step_fn, init_carry, n_steps):
    chunk_size = max(1, int(math.sqrt(n_steps)))
    n_chunks = n_steps // chunk_size
    remainder = n_steps - chunk_size * n_chunks

    def chunk_body(carry, _):
        carry, _ = jax.lax.scan(step_fn, carry, None, length=chunk_size)
        return carry, None

    carry = init_carry
    if n_chunks > 0:
        carry, _ = jax.lax.scan(
            jax.checkpoint(chunk_body), carry, None, length=n_chunks,
        )
    if remainder > 0:
        carry, _ = jax.lax.scan(step_fn, carry, None, length=remainder)
    return carry


def _make_simulate_cd(nx, ny, nz, n_steps, u_inlet, tau, q_mode, porosity_mode):
    def simulate_cd(radius):
        porosity, solid, q = _build_geometry(
            radius, nx, ny, nz, q_mode, porosity_mode,
        )

        rho = jnp.ones((nx, ny, nz))
        u = jnp.zeros((3, nx, ny, nz)).at[0].set(u_inlet)
        f = equilibrium(rho, u)
        vel = jnp.array([u_inlet, 0.0, 0.0])

        def step_fn(carry, _):
            f_in, _ = carry
            f_c = mrt(f_in, tau)
            f_c = soft_bounce_back(f_c, porosity)
            f_post = f_c
            f_s = stream(f_c, q, periodic_yz=True)
            f_s = zou_he_inlet(f_s, vel)
            f_s = zou_he_outlet(f_s, rho_out=1.0)
            return (f_s, f_post), None

        f_final, f_post = _chunked_scan(step_fn, (f, f), n_steps)
        force = momentum_exchange(f_post, f_final, q)
        area = jnp.maximum(projected_area(solid, axis=0), 1.0)
        return drag_coefficient(force, u_inlet, area)

    return simulate_cd


def run_optimisation(q_mode, porosity_mode, nx, ny, nz, n_steps,
                     n_iters, lr, u_inlet, tau, r_init,
                     r_min=0.3, r_max=0.8):
    simulate_cd = _make_simulate_cd(
        nx, ny, nz, n_steps, u_inlet, tau, q_mode, porosity_mode,
    )
    loss_and_grad = jax.jit(jax.value_and_grad(simulate_cd))

    # Warm up.
    r = jnp.float32(r_init)
    loss_and_grad(r)

    opt = optax.adam(learning_rate=lr)
    opt_state = opt.init(r)

    cd_hist, grad_hist, r_hist = [], [], []
    t0 = time.time()
    for i in range(n_iters):
        cd, g = loss_and_grad(r)
        cd_hist.append(float(cd))
        grad_hist.append(float(g))
        r_hist.append(float(r))
        updates, opt_state = opt.update(g, opt_state)
        r = optax.apply_updates(r, updates)
        r = jnp.clip(r, r_min, r_max)
    dt = time.time() - t0

    return {
        "q_mode": q_mode,
        "porosity_mode": porosity_mode,
        "cd": cd_hist,
        "grad": grad_hist,
        "radius": r_hist,
        "wall_s": dt,
    }


def single_gradient(q_mode, porosity_mode, nx, ny, nz, n_steps,
                    u_inlet, tau, r):
    """One forward + grad evaluation, for the grad-is-zero demonstration."""
    simulate_cd = _make_simulate_cd(
        nx, ny, nz, n_steps, u_inlet, tau, q_mode, porosity_mode,
    )
    loss_and_grad = jax.jit(jax.value_and_grad(simulate_cd))
    cd, g = loss_and_grad(jnp.float32(r))
    return float(cd), float(g)


def main():
    # Larger than the paper's 48x24x24 so the baseline Cd is physical.
    nx, ny, nz = 96, 48, 48
    u_inlet = 0.05
    re = 50.0
    n_steps = 2000
    n_iters = 60
    lr = 3e-3
    r_init = 0.5

    D_lattice = nx * 0.5 / 4.0
    nu = u_inlet * D_lattice / re
    tau = jnp.float32(3.0 * nu + 0.5)

    print("Shape Optimisation v2 (MRT, larger grid, longer runs)")
    print("=" * 80)
    print(f"Grid: {nx}x{ny}x{nz}  D={D_lattice:.0f}  Re={re}  tau={float(tau):.4f}")
    print(f"Steps/iter: {n_steps}  Iters: {n_iters}  lr: {lr}  r_init: {r_init}")
    print()

    results = {}

    for q_mode, porosity_mode, label in [
        ("bouzidi", "soft", "bouzidi_soft"),
        ("halfway", "soft", "halfway_soft"),
        ("bouzidi", "hard", "bouzidi_hard"),
    ]:
        print(f"Running mode: {label} (q={q_mode}, porosity={porosity_mode})")
        r = run_optimisation(
            q_mode, porosity_mode, nx, ny, nz, n_steps, n_iters, lr,
            u_inlet, tau, r_init,
        )
        results[label] = r
        print(f"  baseline Cd: {r['cd'][0]:.4f}   "
              f"best Cd: {min(r['cd']):.4f}   "
              f"final radius: {r['radius'][-1]:.4f}   "
              f"wall: {r['wall_s']:.1f}s")
        print()

    # halfway_hard gets a single evaluation because the gradient is
    # exactly zero and Adam would do nothing. We show that directly.
    print("Evaluating mode: halfway_hard (q=0.5, hard mask) at r=0.5 ...")
    cd_hh, g_hh = single_gradient(
        "halfway", "hard", nx, ny, nz, n_steps, u_inlet, tau, r_init,
    )
    print(f"  Cd = {cd_hh:.4f}   dCd/dr = {g_hh:.6e}")
    print()

    print("Summary")
    print("-" * 80)
    print(f"{'Mode':>14s}  {'base Cd':>8s}  {'best Cd':>8s}  "
          f"{'avg |grad|':>10s}  {'reduction':>10s}")
    for label, r in results.items():
        avg_g = sum(abs(v) for v in r['grad']) / len(r['grad'])
        red = (1.0 - min(r['cd']) / r['cd'][0]) * 100.0
        print(f"{label:>14s}  {r['cd'][0]:>8.4f}  {min(r['cd']):>8.4f}  "
              f"{avg_g:>10.4f}  {red:>9.2f}%")
    print(f"{'halfway_hard':>14s}  {cd_hh:>8.4f}  {'n/a':>8s}  "
          f"{abs(g_hh):>10.2e}  {'n/a':>10s}")

    print()
    print("Iteration history (bouzidi_soft vs halfway_soft):")
    print(f"{'Iter':>4s}  {'Cd (BS)':>9s}  {'Cd (HS)':>9s}  "
          f"{'|g| (BS)':>10s}  {'|g| (HS)':>10s}")
    for i in range(n_iters):
        bs = results['bouzidi_soft']
        hs = results['halfway_soft']
        print(f"{i:>4d}  {bs['cd'][i]:>9.4f}  {hs['cd'][i]:>9.4f}  "
              f"{abs(bs['grad'][i]):>10.4f}  {abs(hs['grad'][i]):>10.4f}")


if __name__ == "__main__":
    main()
