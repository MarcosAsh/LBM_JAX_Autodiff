"""Systematic grid convergence of dCd/dr across four resolutions.

Reports the autodiff gradient of drag coefficient with respect to
sphere radius at four grid resolutions (D = 4, 6, 8, 12), alongside
central finite differences. Uses the MRT collision operator and soft
Bouzidi at long-enough run lengths that the flow has developed (~3
flow-through times per grid).

Why MRT and not BGK? Two previous iterations of this script used BGK
to match the paper's shape-optimization example. They ran into trouble:

- BGK at tau close to 0.5 is marginally stable. At 200 steps the flow
  did not develop (gradient of transient noise). At 2880+ steps BGK
  hit instabilities that the step-function stability guard then
  papered over, corrupting the gradient (AD returned NaN, FD a huge
  spurious value).

- MRT damps ghost modes aggressively without touching physical
  viscosity, so it runs stably at low tau and produces smoother loss
  landscapes under reverse-mode AD. The paper's own methodology
  section already notes this; this experiment is what made the
  difference visible.

Memory: the backward pass chunks n_steps into sqrt(n_steps) blocks and
wraps each block in jax.checkpoint, giving O(sqrt(n_steps)) activation
memory. The previous per-step-checkpoint pattern did not achieve this
in practice on A100 (OOM at 128x64x64).

Usage:
    python examples/08_gradient_convergence.py
    modal run modal_worker.py --example grad-convergence
"""

import math
import time

import jax
import jax.numpy as jnp
import numpy as np

from jax_lbm.core.equilibrium import equilibrium, compute_density, compute_velocity
from jax_lbm.core.collision import mrt
from jax_lbm.core.streaming import stream
from jax_lbm.core.boundary import zou_he_inlet, zou_he_outlet
from jax_lbm.core.forces import momentum_exchange, drag_coefficient, projected_area
from jax_lbm.geometry.smooth import smooth_sphere_geometry, soft_bounce_back


def _chunked_scan(step_fn, init_carry, n_steps):
    """Run n_steps of step_fn with O(sqrt(n_steps)) backward memory."""
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


def _simulate_cd(radius, nx, ny, nz, n_steps, u_inlet, tau):
    porosity, solid, q = smooth_sphere_geometry(
        nx, ny, nz, center=(0.0, 0.0, 0.0), radius=radius, sharpness=20.0,
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


def run_case(nx, ny, nz, n_steps, u_inlet, re, radius, eps):
    D_lattice = nx * 0.5 / 4.0
    nu = u_inlet * D_lattice / re
    tau = jnp.float32(3.0 * nu + 0.5)

    cd_fn = lambda r: _simulate_cd(r, nx, ny, nz, n_steps, u_inlet, tau)
    cd_and_grad = jax.jit(jax.value_and_grad(cd_fn))
    cd_only = jax.jit(cd_fn)

    t0 = time.time()
    cd_val, grad_ad = cd_and_grad(radius)
    cd_val = float(cd_val)
    grad_ad = float(grad_ad)
    t_ad = time.time() - t0

    t0 = time.time()
    cd_plus = float(cd_only(radius + eps))
    cd_minus = float(cd_only(radius - eps))
    grad_fd = (cd_plus - cd_minus) / (2.0 * eps)
    t_fd = time.time() - t0

    return {
        "nx": nx, "ny": ny, "nz": nz,
        "D": float(D_lattice),
        "tau": float(tau),
        "steps": n_steps,
        "cd": cd_val,
        "grad_ad": grad_ad,
        "grad_fd": grad_fd,
        "rel_err": abs(grad_ad - grad_fd) / (abs(grad_fd) + 1e-12),
        "t_ad": t_ad,
        "t_fd": t_fd,
    }


def main():
    u_inlet = 0.05
    re = 50.0
    radius = jnp.float32(0.5)
    eps = 1e-3

    # (nx, ny, nz, n_steps). Step count targets ~3 flow-through times
    # (nx/u_inlet). Grids kept below 128x64x64 because chunked-remat
    # backward passes still OOM above that even for moderate n_steps.
    configs = [
        (32, 16, 16, 1920),
        (48, 24, 24, 2880),
        (64, 32, 32, 3840),
        (96, 48, 48, 5760),
    ]

    print("Grid Convergence of dCd/dr (MRT + soft Bouzidi, ~3 flow-through times)")
    print("=" * 90)
    print(f"Radius={float(radius)}, Re={re}, U={u_inlet}, eps={eps}")
    print()
    print(f"{'Grid':>14s}  {'D':>4s}  {'tau':>7s}  {'steps':>6s}  "
          f"{'Cd':>8s}  {'grad AD':>10s}  {'grad FD':>10s}  "
          f"{'rel err':>8s}  {'t AD':>6s}  {'t FD':>6s}")
    print("-" * 90)

    results = []
    for nx, ny, nz, n_steps in configs:
        try:
            r = run_case(nx, ny, nz, n_steps, u_inlet, re, radius, eps)
            results.append(r)
            grid = f"{nx}x{ny}x{nz}"
            print(f"{grid:>14s}  {r['D']:>4.0f}  {r['tau']:>7.4f}  {r['steps']:>6d}  "
                  f"{r['cd']:>8.4f}  {r['grad_ad']:>+10.4f}  {r['grad_fd']:>+10.4f}  "
                  f"{r['rel_err']*100:>7.2f}%  {r['t_ad']:>5.1f}s  {r['t_fd']:>5.1f}s")
        except Exception as e:
            print(f"  FAILED at {nx}x{ny}x{nz}: {e}")
            continue

    print()
    print("Gradient change between successive refinements:")
    for i in range(1, len(results)):
        delta = results[i]["grad_ad"] - results[i - 1]["grad_ad"]
        print(f"  D={results[i-1]['D']:.0f} -> D={results[i]['D']:.0f}:  "
              f"d(grad_AD) = {delta:+.4f}")

    print()
    print("If grad_AD stabilises across resolutions and matches grad_FD, the")
    print("differentiable Bouzidi gradient is trustworthy under this setup.")


if __name__ == "__main__":
    main()
