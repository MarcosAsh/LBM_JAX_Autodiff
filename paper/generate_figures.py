"""Generate figures for the paper.

Runs simulations and produces publication-quality matplotlib figures:

1. Poiseuille velocity profile (LBM vs analytic parabola)
2. Bouzidi vs standard bounce-back: Cd and gradient magnitude over
   optimization iterations

Usage:
    python paper/generate_figures.py

Output:
    paper/figures/poiseuille_profile.pdf
    paper/figures/optimization_trajectory.pdf
"""

import os
import sys

# Add project root to path so jax_lbm is importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optax

from jax_lbm.core.lattice import D3Q19_OPPOSITE
from jax_lbm.core.equilibrium import equilibrium, compute_density, compute_velocity
from jax_lbm.core.collision import bgk
from jax_lbm.core.streaming import stream
from jax_lbm.core.boundary import zou_he_inlet, zou_he_outlet
from jax_lbm.core.forces import momentum_exchange, drag_coefficient, projected_area
from jax_lbm.geometry.bouzidi import compute_q_from_solid
from jax_lbm.geometry.smooth import (
    signed_distance_sphere,
    porosity_from_sdf,
    soft_bounce_back,
)
from jax_lbm.core.lattice import D3Q19_VELOCITIES, Q


FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")


# ------------------------------------------------------------------ #
#  Figure 1: Poiseuille velocity profile                             #
# ------------------------------------------------------------------ #

def generate_poiseuille_figure():
    """Run Poiseuille flow and plot velocity profile vs analytic solution."""
    print("Generating Poiseuille profile figure...")

    nx, ny, nz = 60, 20, 4
    u_inlet = 0.05
    tau = jnp.float32(0.8)
    n_steps = 3000

    # Walls at y=0 and y=ny-1.
    solid = jnp.zeros((nx, ny, nz), dtype=jnp.int32)
    solid = solid.at[:, 0, :].set(1)
    solid = solid.at[:, ny - 1, :].set(1)
    q = compute_q_from_solid(solid, nx, ny, nz, default_q=0.5)

    rho = jnp.ones((nx, ny, nz))
    u = jnp.zeros((3, nx, ny, nz))
    f = equilibrium(rho, u)
    vel = jnp.array([u_inlet, 0.0, 0.0])

    opp_arr = jnp.array([int(x) for x in np.asarray(D3Q19_OPPOSITE)])
    is_solid = (solid >= 1)[jnp.newaxis, :, :, :]

    def step_fn(f_in, _):
        f_c = bgk(f_in, tau)
        f_c = jnp.where(is_solid, f_c[opp_arr], f_c)
        f_c = stream(f_c, q, periodic_yz=True)
        f_c = zou_he_inlet(f_c, vel)
        f_c = zou_he_outlet(f_c, rho_out=1.0)
        return f_c, None

    f, _ = jax.lax.scan(step_fn, f, None, length=n_steps)

    rho_final = compute_density(f)
    u_final = compute_velocity(f, rho_final)
    x_mid = nx // 2
    ux_profile = np.array(jnp.mean(u_final[0, x_mid, :, :], axis=-1))

    # Analytic parabola.
    H = float(ny - 2)
    u_max = float(np.max(ux_profile[1:-1]))
    y_coords = np.arange(ny)
    ux_analytic = np.zeros(ny)
    for j in range(1, ny - 1):
        y = j - 0.5
        ux_analytic[j] = u_max * 4.0 * y * (H - y) / (H * H)

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.plot(y_coords[1:-1], ux_analytic[1:-1], "k-", linewidth=1.5,
            label="Analytic")
    ax.plot(y_coords[1:-1], ux_profile[1:-1], "o", color="#2171b5",
            markersize=5, markerfacecolor="white", markeredgewidth=1.2,
            label="LBM (BGK)")
    ax.set_xlabel("$y$ (lattice units)")
    ax.set_ylabel("$u_x$")
    ax.legend(frameon=False)
    ax.set_xlim(0, ny - 1)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    path = os.path.join(FIGURES_DIR, "poiseuille_profile.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ------------------------------------------------------------------ #
#  Figure 2: Bouzidi vs standard optimization trajectories           #
# ------------------------------------------------------------------ #

def _build_geometry(radius, nx, ny, nz, use_bouzidi):
    """Build smooth sphere geometry with Bouzidi or standard q."""
    sdf = signed_distance_sphere(nx, ny, nz, (0.0, 0.0, 0.0), radius)
    porosity = porosity_from_sdf(sdf, sharpness=20.0)
    solid = (porosity > 0.5).astype(jnp.int32)

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
    q_buf = -jnp.ones((Q, nx, ny, nz))

    for i in range(1, Q):
        ex_i = int(e_np[i, 0])
        ey_i = int(e_np[i, 1])
        ez_i = int(e_np[i, 2])

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

        if use_bouzidi:
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

        q_buf = q_buf.at[i].set(jnp.where(is_boundary, q_val, -1.0))

    return porosity, solid, q_buf


def _run_optimization(use_bouzidi, n_iters=25):
    """Run shape optimization, return per-iteration Cd and gradient."""
    nx, ny, nz = 48, 24, 24
    u_inlet = 0.05
    n_steps = 150

    D_lattice = 48 * 0.5 / 4.0
    re = 50.0
    nu = u_inlet * D_lattice / re
    tau = jnp.float32(3.0 * nu + 0.5)

    def simulate_cd(radius):
        porosity, solid, q = _build_geometry(radius, nx, ny, nz, use_bouzidi)

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
            rho_s = compute_density(f_s)
            u_s = compute_velocity(f_s, rho_s)
            u_mag = jnp.sqrt(jnp.sum(u_s * u_s, axis=0))
            bad = ((rho_s < 0.3) | (rho_s > 2.0)
                   | (u_mag > 0.4) | jnp.isnan(rho_s))
            f_eq_r = equilibrium(
                jnp.where(bad, 1.0, rho_s),
                jnp.where(bad[None], 0.0, u_s),
            )
            f_s = jnp.where(bad[None], f_eq_r, f_s)
            return (f_s, f_post), None

        (f_final, f_post), _ = jax.lax.scan(
            step_fn, (f, f), None, length=n_steps
        )

        force = momentum_exchange(f_post, f_final, q)
        area = jnp.maximum(projected_area(solid, axis=0), 1.0)
        return drag_coefficient(force, u_inlet, area)

    loss_and_grad = jax.jit(jax.value_and_grad(simulate_cd))

    radius = jnp.float32(0.5)
    loss_and_grad(radius)  # JIT warm-up

    optimizer = optax.adam(learning_rate=0.003)
    opt_state = optimizer.init(radius)

    cd_hist, grad_hist = [], []

    for i in range(n_iters):
        cd, grad = loss_and_grad(radius)
        cd_hist.append(float(cd))
        grad_hist.append(float(grad))
        updates, opt_state = optimizer.update(grad, opt_state)
        radius = optax.apply_updates(radius, updates)
        radius = jnp.clip(radius, 0.2, 0.8)
        print(f"    iter {i:2d}  Cd={cd_hist[-1]:.4f}  |grad|={abs(grad_hist[-1]):.4f}")

    return cd_hist, grad_hist


def generate_optimization_figure():
    """Run Bouzidi vs standard comparison, plot Cd and gradient trajectories."""
    print("Generating optimization trajectory figure...")

    print("  Running Bouzidi optimization...")
    cd_bouzidi, grad_bouzidi = _run_optimization(use_bouzidi=True)
    print("  Running standard bounce-back optimization...")
    cd_standard, grad_standard = _run_optimization(use_bouzidi=False)

    iters = np.arange(len(cd_bouzidi))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

    # Left panel: Cd trajectory.
    ax1.plot(iters, cd_bouzidi, "o-", color="#2171b5", markersize=3,
             linewidth=1.2, label="Bouzidi")
    ax1.plot(iters, cd_standard, "s-", color="#cb181d", markersize=3,
             linewidth=1.2, label="Standard")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("$C_d$")
    ax1.legend(frameon=False)
    ax1.set_title("(a) Drag coefficient", fontsize=9)

    # Right panel: gradient magnitude.
    ax2.plot(iters, [abs(g) for g in grad_bouzidi], "o-", color="#2171b5",
             markersize=3, linewidth=1.2, label="Bouzidi")
    ax2.plot(iters, [abs(g) for g in grad_standard], "s-", color="#cb181d",
             markersize=3, linewidth=1.2, label="Standard")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("$|\\partial C_d / \\partial r|$")
    ax2.set_yscale("log")
    ax2.legend(frameon=False)
    ax2.set_title("(b) Gradient magnitude", fontsize=9)

    fig.tight_layout()

    path = os.path.join(FIGURES_DIR, "optimization_trajectory.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ------------------------------------------------------------------ #
#  Main                                                              #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)
    generate_poiseuille_figure()
    generate_optimization_figure()
    print("Done.")
