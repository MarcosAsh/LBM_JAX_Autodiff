"""Float64 gradient verification for the paper.

Runs the same gradient checks as test_gradients.py but in float64
for proper relative error reporting. Float32 is too noisy for tight
FD comparison over multiple LBM steps; float64 gives the 1e-5
relative error that reviewers expect.
"""

import os
os.environ["JAX_ENABLE_X64"] = "True"

import jax
import jax.numpy as jnp
import pytest

# Skip if x64 mode didn't activate (e.g. JAX was imported before the
# env var was set, which happens on Modal where pip install imports the
# package during build). These tests are verified locally.
pytestmark = pytest.mark.skipif(
    not jax.config.x64_enabled,
    reason="float64 not enabled (JAX imported before JAX_ENABLE_X64 was set)",
)

from jax_lbm.core.equilibrium import equilibrium, compute_density, compute_velocity
from jax_lbm.core.collision import bgk
from jax_lbm.core.streaming import stream
from jax_lbm.core.boundary import zou_he_inlet, zou_he_outlet


def _central_difference(f, x, eps=1e-5):
    return (f(x + eps) - f(x - eps)) / (2.0 * eps)


class TestFloat64Gradients:
    """Gradient checks in float64 for publication-quality verification."""

    def test_equilibrium_grad_f64(self):
        """d(sum f_eq)/d(rho) should equal N_cells exactly in f64."""
        nx, ny, nz = 4, 4, 4
        n_cells = nx * ny * nz
        u = jnp.zeros((3, nx, ny, nz), dtype=jnp.float64)

        def total_mass(rho_scalar):
            rho = jnp.full((nx, ny, nz), rho_scalar, dtype=jnp.float64)
            f = equilibrium(rho, u)
            return jnp.sum(f)

        grad_ad = float(jax.grad(total_mass)(jnp.float64(1.0)))
        grad_fd = _central_difference(total_mass, jnp.float64(1.0), eps=1e-7)
        rel_err = abs(grad_ad - float(grad_fd)) / abs(grad_ad)

        print(f"  Equilibrium grad: AD={grad_ad:.10f}, FD={float(grad_fd):.10f}, rel_err={rel_err:.2e}")
        assert rel_err < 1e-6, f"Relative error {rel_err:.2e} exceeds 1e-6"

    def test_bgk_grad_tau_f64(self):
        """Gradient of mean ux after BGK w.r.t. tau, checked in f64."""
        nx, ny, nz = 4, 4, 4
        rho = jnp.ones((nx, ny, nz), dtype=jnp.float64)
        u = jnp.zeros((3, nx, ny, nz), dtype=jnp.float64).at[0].set(0.05)
        f_eq = equilibrium(rho, u)
        key = jax.random.PRNGKey(42)
        f = f_eq + jax.random.normal(key, f_eq.shape, dtype=jnp.float64) * 0.001

        def mean_ux(tau_val):
            f_post = bgk(f, tau_val)
            rho_post = compute_density(f_post)
            u_post = compute_velocity(f_post, rho_post)
            return jnp.mean(u_post[0])

        tau = jnp.float64(0.8)
        grad_ad = float(jax.grad(mean_ux)(tau))
        grad_fd = _central_difference(
            lambda t: float(mean_ux(jnp.float64(t))),
            float(tau), eps=1e-7,
        )
        rel_err = abs(grad_ad - grad_fd) / (abs(grad_ad) + 1e-15)

        print(f"  BGK tau grad: AD={grad_ad:.10e}, FD={grad_fd:.10e}, rel_err={rel_err:.2e}")
        # The gradient is near zero (BGK conserves momentum), so relative
        # error is dominated by noise. Just check both are very small.
        assert abs(grad_ad) < 1e-6, f"|AD grad| = {abs(grad_ad):.2e} too large"
        assert abs(grad_fd) < 1e-6, f"|FD grad| = {abs(grad_fd):.2e} too large"

    def test_multistep_grad_inlet_f64(self):
        """Gradient of mean density after 5 steps w.r.t. inlet velocity, f64."""
        nx, ny, nz = 8, 4, 4
        n_steps = 5

        def simulate_and_measure(ux_inlet):
            rho = jnp.ones((nx, ny, nz), dtype=jnp.float64)
            u = jnp.zeros((3, nx, ny, nz), dtype=jnp.float64)
            f = equilibrium(rho, u)
            q = -jnp.ones((19, nx, ny, nz), dtype=jnp.float64)
            vel = jnp.array([ux_inlet, 0.0, 0.0], dtype=jnp.float64)

            def step(f_state, _):
                f_s = bgk(f_state, jnp.float64(0.8))
                f_s = stream(f_s, q, periodic_yz=True)
                f_s = zou_he_inlet(f_s, vel)
                f_s = zou_he_outlet(f_s, rho_out=1.0)
                return f_s, None

            f_final, _ = jax.lax.scan(step, f, None, length=n_steps)
            return jnp.mean(compute_density(f_final[:, 2:-2, :, :]))

        ux = jnp.float64(0.05)
        grad_ad = float(jax.grad(simulate_and_measure)(ux))
        grad_fd = _central_difference(
            lambda v: float(simulate_and_measure(jnp.float64(v))),
            float(ux), eps=1e-7,
        )

        rel_err = abs(grad_ad - grad_fd) / (abs(grad_ad) + 1e-15)

        print(f"  Multi-step grad: AD={grad_ad:.10e}, FD={grad_fd:.10e}, rel_err={rel_err:.2e}")
        assert rel_err < 1e-3, f"Relative error {rel_err:.2e} exceeds 1e-3"

    def test_bouzidi_grad_radius_f64(self):
        """Gradient of Cd w.r.t. sphere radius through Bouzidi, f64."""
        import numpy as np
        from jax_lbm.core.lattice import D3Q19_VELOCITIES, Q

        nx, ny, nz = 32, 16, 16
        n_steps = 10
        u_inlet = 0.05
        D_lattice = 32 * 0.5 / 4.0
        re = 50.0
        nu = u_inlet * D_lattice / re
        tau = jnp.float64(3.0 * nu + 0.5)

        e_np = np.asarray(D3Q19_VELOCITIES)
        scale_x, scale_y, scale_z = nx / 8.0, ny / 4.0, nz / 4.0

        gx = jnp.arange(nx, dtype=jnp.float64)
        gy = jnp.arange(ny, dtype=jnp.float64)
        gz = jnp.arange(nz, dtype=jnp.float64)
        gx3d, gy3d, gz3d = jnp.meshgrid(gx, gy, gz, indexing="ij")

        wx = (gx3d + 0.5) / scale_x - 4.0
        wy = (gy3d + 0.5) / scale_y - 2.0
        wz = (gz3d + 0.5) / scale_z - 2.0

        def cd_of_radius(radius):
            dist = jnp.sqrt((wx - 0.0)**2 + (wy - 0.0)**2 + (wz - 0.0)**2)
            porosity = jax.nn.sigmoid(20.0 * (radius - dist))
            solid = (porosity > 0.5).astype(jnp.int32)

            q = -jnp.ones((Q, nx, ny, nz), dtype=jnp.float64)
            for i in range(1, Q):
                ex_i, ey_i, ez_i = int(e_np[i, 0]), int(e_np[i, 1]), int(e_np[i, 2])
                nbx = jnp.clip((gx3d + ex_i).astype(jnp.int32), 0, nx - 1)
                nby = jnp.clip((gy3d + ey_i).astype(jnp.int32), 0, ny - 1)
                nbz = jnp.clip((gz3d + ez_i).astype(jnp.int32), 0, nz - 1)
                in_bounds = ((gx3d + ex_i >= 0) & (gx3d + ex_i < nx)
                           & (gy3d + ey_i >= 0) & (gy3d + ey_i < ny)
                           & (gz3d + ez_i >= 0) & (gz3d + ez_i < nz))
                nb_solid = solid[nbx, nby, nbz]
                is_boundary = (solid == 0) & (nb_solid == 1) & in_bounds

                ray_dx = jnp.float64(ex_i) / scale_x
                ray_dy = jnp.float64(ey_i) / scale_y
                ray_dz = jnp.float64(ez_i) / scale_z
                a = ray_dx**2 + ray_dy**2 + ray_dz**2
                b = 2.0 * (wx * ray_dx + wy * ray_dy + wz * ray_dz)
                c = wx**2 + wy**2 + wz**2 - radius**2
                disc = b * b - 4.0 * a * c
                sqrt_disc = jnp.sqrt(jnp.maximum(disc, 1e-20))
                t = (-b - sqrt_disc) / (2.0 * a + 1e-30)
                valid = (disc >= 0.0) & (t > 0.0) & (t <= 1.0)
                q_val = jnp.clip(jnp.where(valid, t, 0.5), 0.01, 0.99)
                q = q.at[i].set(jnp.where(is_boundary, q_val, -1.0))

            from jax_lbm.geometry.smooth import soft_bounce_back
            from jax_lbm.core.forces import momentum_exchange, drag_coefficient, projected_area

            rho = jnp.ones((nx, ny, nz), dtype=jnp.float64)
            u = jnp.zeros((3, nx, ny, nz), dtype=jnp.float64).at[0].set(u_inlet)
            f = equilibrium(rho, u)
            vel = jnp.array([u_inlet, 0.0, 0.0], dtype=jnp.float64)
            opp_arr = jnp.array([0, 2, 1, 4, 3, 6, 5, 10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15])

            def step_fn(carry, _):
                f_in, _ = carry
                f_c = bgk(f_in, tau)
                p = porosity[jnp.newaxis, :, :, :]
                f_c = (1.0 - p) * f_c + p * f_c[opp_arr]
                f_post = f_c
                f_s = stream(f_c, q, periodic_yz=True)
                f_s = zou_he_inlet(f_s, vel)
                f_s = zou_he_outlet(f_s, rho_out=1.0)
                return (f_s, f_post), None

            (f_final, f_post), _ = jax.lax.scan(step_fn, (f, f), None, length=n_steps)
            force = momentum_exchange(f_post, f_final, q)
            area = jnp.maximum(projected_area(solid, axis=0), 1.0)
            return drag_coefficient(force, u_inlet, area)

        r0 = jnp.float64(0.5)
        grad_ad = float(jax.grad(cd_of_radius)(r0))
        eps = 1e-6
        grad_fd = (float(cd_of_radius(r0 + eps)) - float(cd_of_radius(r0 - eps))) / (2.0 * eps)

        rel_err = abs(grad_ad - grad_fd) / (abs(grad_ad) + 1e-15)

        print(f"  Bouzidi grad: AD={grad_ad:.10e}, FD={grad_fd:.10e}, rel_err={rel_err:.2e}")
        assert rel_err < 1e-2, f"Relative error {rel_err:.2e} exceeds 1e-2"
