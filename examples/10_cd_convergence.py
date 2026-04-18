"""Systematic grid convergence of sphere Cd at Re=100.

Sweeps four resolutions -- D = 8, 16, 32, 48 lattice cells -- with the
MRT + Bouzidi + Zou-He setup used in the paper's sphere validation, and
reports time-averaged Cd against the Clift et al. (1978) reference of
1.09.

Two previous iterations of this script ran too few steps: at D=32 with
6000 steps the flow had reached only ~1.2 flow-through times, and the
snapshot Cd was 0.76 rather than anything close to Clift. The fix is
twofold:

1. Step counts now scale as 5x(nx/U) so every resolution sees at least
   5 flow-through times -- enough for the wake to mature.
2. Cd is time-averaged over the final 30% of the run, sampled every
   ``avg_stride`` steps. The reported mean and standard deviation make
   it visible when the wake is still unsteady rather than hiding it in
   a snapshot.

The D=48 case takes roughly 15 minutes on an A100.

Usage:
    python examples/10_cd_convergence.py
    modal run modal_worker.py --example cd-convergence
"""

import math
import time

import jax
import jax.numpy as jnp
import numpy as np

from jax_lbm.core.lattice import D3Q19_OPPOSITE
from jax_lbm.core.equilibrium import equilibrium, compute_density, compute_velocity
from jax_lbm.core.collision import mrt
from jax_lbm.core.streaming import stream
from jax_lbm.core.boundary import zou_he_inlet, zou_he_outlet
from jax_lbm.core.forces import momentum_exchange, drag_coefficient, projected_area
from jax_lbm.geometry.primitives import create_sphere


CLIFT_REF = 1.09


def run_case(nx, ny, nz, n_steps, u_inlet, re, avg_fraction=0.3, avg_stride=500):
    """Run one resolution, return time-averaged Cd plus history."""
    center = (0.0, 0.0, 0.0)
    radius = 0.5

    scale_x = nx / 8.0
    D_lattice = 2.0 * radius * scale_x
    nu = u_inlet * D_lattice / re
    tau = 3.0 * nu + 0.5

    solid, q = create_sphere(nx, ny, nz, center, radius)
    area = projected_area(solid, axis=0)

    rho = jnp.ones((nx, ny, nz))
    u = jnp.zeros((3, nx, ny, nz)).at[0].set(u_inlet)
    f = equilibrium(rho, u)
    vel = jnp.array([u_inlet, 0.0, 0.0])

    opp_arr = jnp.array([int(x) for x in np.asarray(D3Q19_OPPOSITE)])
    is_solid = (solid >= 1)[jnp.newaxis, :, :, :]
    tau_val = jnp.float32(tau)
    f_eq_ref = equilibrium(rho, u)

    def step_fn(carry, _):
        f_in, _ = carry
        rho_in = compute_density(f_in)
        u_in = compute_velocity(f_in, rho_in)
        speed = jnp.sqrt(jnp.sum(u_in ** 2, axis=0))
        bad = (rho_in < 0.3) | (rho_in > 2.0) | (speed > 0.4) | jnp.isnan(rho_in)
        f_in = jnp.where(bad[None], f_eq_ref, f_in)

        f_c = mrt(f_in, tau_val)
        f_c = jnp.where(is_solid, f_c[opp_arr], f_c)
        f_post = f_c
        f_s = stream(f_c, q, periodic_yz=True)
        f_s = zou_he_inlet(f_s, vel)
        f_s = zou_he_outlet(f_s, rho_out=1.0)
        return (f_s, f_post), None

    print(f"  {nx}x{ny}x{nz}: D={D_lattice:.0f}, tau={tau:.4f}, "
          f"steps={n_steps}, flow-through time={nx/u_inlet:.0f} steps")

    # Run in chunks of ``avg_stride`` steps so we get a Cd sample per chunk.
    chunk = avg_stride
    cd_history = []
    t0 = time.time()
    step = 0
    while step < n_steps:
        length = min(chunk, n_steps - step)
        (f, f_post), _ = jax.lax.scan(step_fn, (f, f), None, length=length)
        force = momentum_exchange(f_post, f, q)
        cd = float(drag_coefficient(force, u_inlet, area))
        step += length
        cd_history.append((step, cd))
    dt = time.time() - t0

    # Time-average over the last ``avg_fraction`` of the run.
    n_tail = max(3, int(len(cd_history) * avg_fraction))
    tail = [cd for _, cd in cd_history[-n_tail:]]
    cd_mean = float(np.mean(tail))
    cd_std = float(np.std(tail))

    cd_snapshot = cd_history[-1][1]

    return {
        "nx": nx, "ny": ny, "nz": nz,
        "D": float(D_lattice),
        "tau": float(tau),
        "steps": n_steps,
        "cd_snapshot": cd_snapshot,
        "cd_mean": cd_mean,
        "cd_std": cd_std,
        "cd_history": cd_history,
        "err": abs(cd_mean - CLIFT_REF) / CLIFT_REF,
        "wall_s": dt,
    }


def main():
    u_inlet = 0.05
    re = 100.0

    # Step count scales as 5 * nx / u_inlet so every resolution sees
    # at least 5 flow-through times.
    configs = [
        (64,  32,  32,   8000),
        (128, 64,  64,  16000),
        (256, 128, 128, 28000),
        (384, 192, 192, 40000),
    ]

    print("Grid Convergence of Sphere Cd at Re=100 (MRT + Bouzidi)")
    print("=" * 80)
    print()

    results = []
    for nx, ny, nz, n_steps in configs:
        r = run_case(nx, ny, nz, n_steps, u_inlet, re)
        results.append(r)
        print(f"    snapshot Cd = {r['cd_snapshot']:.4f}   "
              f"time-avg Cd = {r['cd_mean']:.4f} +/- {r['cd_std']:.4f}   "
              f"|err| = {r['err']*100:5.2f}%   ({r['wall_s']:.1f}s)")

        # Show the Cd-vs-step trajectory so unsteadiness is visible.
        history = r['cd_history']
        sample_every = max(1, len(history) // 10)
        samples = history[::sample_every][-10:]
        trail = "  ".join(f"{step}:{cd:.3f}" for step, cd in samples)
        print(f"    trajectory (step:Cd): {trail}")
        print()

    print("Summary (time-averaged Cd over final 30% of run)")
    print("-" * 80)
    print(f"{'Grid':>16s}  {'D':>5s}  {'steps':>6s}  "
          f"{'Cd (mean)':>10s}  {'Cd (std)':>9s}  {'rel err':>8s}")
    for r in results:
        grid = f"{r['nx']}x{r['ny']}x{r['nz']}"
        print(f"{grid:>16s}  {r['D']:>5.0f}  {r['steps']:>6d}  "
              f"{r['cd_mean']:>10.4f}  {r['cd_std']:>9.4f}  "
              f"{r['err']*100:>7.2f}%")
    print(f"Clift et al. (1978) reference: Cd = {CLIFT_REF}")
    print()

    # Richardson order estimates. The first three grids double (8->16->32)
    # so p = log2(|Cd0-Cd1| / |Cd1-Cd2|). The last step is 1.5x not 2x.
    if len(results) >= 3:
        cds = [r["cd_mean"] for r in results[:3]]
        num = abs(cds[0] - cds[1])
        den = abs(cds[1] - cds[2])
        if den > 1e-8:
            p = math.log2(num / den)
            print(f"Observed order (D=8 -> 16 -> 32, doubling):  p = {p:.2f}")

    if len(results) >= 4:
        r_ratio = results[3]["D"] / results[2]["D"]
        cds = [results[1]["cd_mean"], results[2]["cd_mean"], results[3]["cd_mean"]]
        num = abs(cds[0] - cds[1])
        den = abs(cds[1] - cds[2])
        if den > 1e-8 and r_ratio > 1.0:
            p = math.log(num / den) / math.log(r_ratio)
            print(f"Observed order (D=16 -> 32 -> 48, r={r_ratio:.2f}):  p = {p:.2f}")

    print()
    print("For first-to-second-order Bouzidi, p should fall in [1, 2].")
    print("Large Cd std indicates unsteady wake: report time-average, not snapshot.")


if __name__ == "__main__":
    main()
