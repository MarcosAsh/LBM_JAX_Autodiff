"""Sensitivity of the Bouzidi gradient to the sigmoid sharpness alpha.

The paper uses a sigmoid-smoothed solid representation (List et al.\
2022) with sharpness alpha = 20. That choice governs the width of
the "mushy" transition zone between solid and fluid. Too low and the
physics is smeared; too high and the gradient is noisier.

A reviewer will ask whether alpha = 20 was cherry-picked. This
experiment sweeps alpha over a decade (5, 10, 20, 40, 80) and reports
the forward Cd and dCd/dr at a fixed grid, radius, and step count.
If Cd and the gradient sign are stable across alpha, the choice is
not cherry-picked; if any single alpha produces a wildly different
answer, the paper's main claims are alpha-dependent and must be
qualified.

Usage:
    python examples/12_sharpness_sensitivity.py
    modal run modal_worker.py --example sharpness
"""

import math
import time

import jax
import jax.numpy as jnp

from jax_lbm.core.equilibrium import equilibrium, compute_density, compute_velocity
from jax_lbm.core.collision import mrt
from jax_lbm.core.streaming import stream
from jax_lbm.core.boundary import zou_he_inlet, zou_he_outlet
from jax_lbm.core.forces import momentum_exchange, drag_coefficient, projected_area
from jax_lbm.geometry.smooth import smooth_sphere_geometry, soft_bounce_back


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


def _simulate_cd(radius, sharpness, nx, ny, nz, n_steps, u_inlet, tau):
    porosity, solid, q = smooth_sphere_geometry(
        nx, ny, nz, center=(0.0, 0.0, 0.0), radius=radius, sharpness=sharpness,
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


def run_case(sharpness, nx, ny, nz, n_steps, u_inlet, tau, radius, eps):
    cd_fn = lambda r: _simulate_cd(
        r, sharpness, nx, ny, nz, n_steps, u_inlet, tau,
    )
    cd_and_grad = jax.jit(jax.value_and_grad(cd_fn))
    cd_only = jax.jit(cd_fn)

    t0 = time.time()
    cd, g_ad = cd_and_grad(radius)
    cd = float(cd)
    g_ad = float(g_ad)
    t_ad = time.time() - t0

    cd_plus = float(cd_only(radius + eps))
    cd_minus = float(cd_only(radius - eps))
    g_fd = (cd_plus - cd_minus) / (2.0 * eps)

    return {
        "alpha": sharpness,
        "cd": cd,
        "grad_ad": g_ad,
        "grad_fd": g_fd,
        "rel_err": abs(g_ad - g_fd) / (abs(g_fd) + 1e-12),
        "t_ad": t_ad,
    }


def main():
    nx, ny, nz = 96, 48, 48
    u_inlet = 0.05
    re = 50.0
    radius = jnp.float32(0.5)
    eps = 1e-3
    n_steps = 2880

    D = nx * 0.5 / 4.0
    nu = u_inlet * D / re
    tau = jnp.float32(3.0 * nu + 0.5)

    sharpnesses = [5.0, 10.0, 20.0, 40.0, 80.0]

    print("Sharpness sensitivity (alpha sweep)")
    print("=" * 72)
    print(f"Grid: {nx}x{ny}x{nz}  D={D:.0f}  Re={re}  tau={float(tau):.4f}  "
          f"steps={n_steps}")
    print()
    print(f"{'alpha':>6s}  {'Cd':>8s}  {'grad AD':>10s}  {'grad FD':>10s}  "
          f"{'rel err':>8s}  {'t AD':>6s}")
    print("-" * 72)

    results = []
    for a in sharpnesses:
        r = run_case(a, nx, ny, nz, n_steps, u_inlet, tau, radius, eps)
        results.append(r)
        print(f"{r['alpha']:>6.1f}  {r['cd']:>8.4f}  "
              f"{r['grad_ad']:>+10.4f}  {r['grad_fd']:>+10.4f}  "
              f"{r['rel_err']*100:>7.2f}%  {r['t_ad']:>5.1f}s")

    print()
    cd_range = max(r['cd'] for r in results) - min(r['cd'] for r in results)
    ad_range = max(r['grad_ad'] for r in results) - min(r['grad_ad'] for r in results)
    print(f"Cd spread across alpha:       {cd_range:.4f}")
    print(f"grad_AD spread across alpha:  {ad_range:.4f}")
    print()
    print("If Cd and grad sign are stable across a decade in alpha, the")
    print("paper's alpha = 20 choice is representative rather than cherry-picked.")


if __name__ == "__main__":
    main()
