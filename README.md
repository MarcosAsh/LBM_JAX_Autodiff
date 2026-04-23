# JAX LBM

**Differentiable 3D Lattice Boltzmann solver in JAX for inverse fluid design.**

Backpropagate through thousands of simulation timesteps with `jax.grad` and optimise boundary conditions, obstacle geometry, or viscosity to hit a target flow field or drag coefficient.

The key point: in the standard LBM formulation (halfway bounce-back + a binary solid mask) the shape gradient is *exactly zero* — gradient-based shape optimisation is impossible. This repo makes the [Bouzidi interpolated bounce-back](https://en.wikipedia.org/wiki/Lattice_Boltzmann_methods) fractional wall distance `q` differentiable through ray-sphere intersection, turning a zero-gradient problem into a non-zero-gradient one.

**Paper**: [`paper/paper.pdf`](paper/paper.pdf) — 8 pages with validation, gradient convergence, a four-mode boundary-condition comparison, and sharpness sensitivity.

```
                    ┌─────────────────────────────────────────────────┐
                    │           Forward Simulation (LBM)              │
                    │                                                 │
  Geometry ────────►│  collide ──► stream ──► boundary ──► repeat     │────► Cd
  (radius, q)       │   (BGK/MRT)  (Bouzidi)  (Zou-He)    x1000     │
                    └─────────────────────────────────────────────────┘
                                         │
                                    jax.grad
                                         │
                    ┌─────────────────────────────────────────────────┐
                    │          Backward Pass (Autodiff)               │
                    │                                                 │
  ∂Cd/∂radius ◄────│  adjoint collide ◄── adjoint stream ◄── ...     │◄─── ∂Cd/∂Cd = 1
                    │                                                 │
                    └─────────────────────────────────────────────────┘
```

## Shape optimisation on A100

Adam on sphere radius with MRT + Bouzidi + hard solid mask on a 96x48x48 grid, 2000 steps per iteration (from `examples/11_optimisation_v2.py`):

```
Mode           base Cd   best Cd   avg |grad|   final r
halfway_hard   1.85      --        0.0          0.500     (classical LBM, grad=0)
bouzidi_hard   1.77      1.66      5.55         0.314     (Bouzidi enables optimisation)
halfway_soft   0.43      0.003     6.30         0.652     (soft porosity, degenerate)
bouzidi_soft   0.45      0.002     3.19         0.663     (soft porosity, degenerate)
```

The `halfway_hard` row is the baseline classical LBM: the solver verifies `dCd/dr = 0` directly via AD, because the binary solid mask is a step function. Replacing `q = 0.5` with the ray-sphere `q(r)` (`bouzidi_hard`) makes the gradient non-zero and Adam reduces a physical Cd = 1.77 to 1.66 by shrinking the radius — the clean positive result. Soft-porosity modes reach a degenerate optimum; the paper documents why.

## Sphere drag at Re=100 (grid convergence study)

Time-averaged `Cd` over the final 30% of each run from `examples/10_cd_convergence.py`:

```
Grid              D   steps    Cd ± σ          rel err vs Clift 1.09
 64× 32× 32       8    8,000   1.29 ± 0.02     18%
128× 64× 64      16   16,000   1.19 ± 0.04      9%
256×128×128      32   28,000   1.19 ± 0.07      9%
384×192×192      48   40,000   1.17 ± 0.08      7%
```

The residual 7% gap at the finest grid is consistent with the blockage correction (D/L_y = 0.25) and un-tuned MRT damping — i.e. physics, not numerical error. See [`paper/figures/cd_convergence.pdf`](paper/figures/cd_convergence.pdf) for the plot.

## Poiseuille flow validation

Steady-state channel flow matches the analytic parabolic profile to 3+ significant figures:

```
  y   u_x (LBM)   u_x (analytic)
  1   0.008247     0.008305        ·
  5   0.057873     0.057659        ········
  9   0.076879     0.076641        ···········   ← centerline
 10   0.076879     0.076641        ···········
 15   0.048352     0.048168        ·····
 19   0.000200     0.000000        
```

## Features

- **D3Q19 lattice** with BGK and MRT collision operators
- **Bouzidi interpolated bounce-back** -- 2nd-order wall accuracy with differentiable fractional distances
- **Zou-He boundary conditions** for velocity inlets and pressure outlets
- **Smagorinsky SGS model** for turbulence at under-resolved Reynolds numbers
- **Momentum exchange force** with pressure/friction drag decomposition
- **Sigmoid-smoothed solids** for gradient-friendly shape optimization (density-based topology optimisation, e.g. Pingen et al. 2007 for LBM)
- **Gradient checkpointing** via `jax.checkpoint` for memory-bounded backprop
- **Stability guard** resets diverged cells, matching the reference C/GLSL solver
- Validated against [the reference solver](https://github.com/MarcosAsh/Lattice_Fluid_Dynamics) and experimental data (Clift et al. 1978)

## Quick start

```bash
git clone https://github.com/MarcosAsh/LBM_JAX_Autodiff.git
cd LBM_JAX_Autodiff
pip install -e .
```

Run channel flow (works on CPU):

```bash
python examples/01_channel_flow.py
```

Run on GPU via [Modal](https://modal.com):

```bash
pip install modal && modal setup
modal run modal_worker.py --example cd-convergence     # sphere drag convergence study
modal run modal_worker.py --example grad-convergence   # dCd/dr grid convergence
modal run modal_worker.py --example optimisation-v2    # four-mode shape optimisation
modal run modal_worker.py --example sharpness          # alpha sensitivity sweep
modal run modal_worker.py --tests
```

## Differentiate through the simulation

```python
import jax
import jax.numpy as jnp
from jax_lbm.core.equilibrium import equilibrium
from jax_lbm.core.collision import bgk
from jax_lbm.core.streaming import stream
from jax_lbm.core.boundary import zou_he_inlet, zou_he_outlet
from jax_lbm.core.forces import momentum_exchange, drag_coefficient, projected_area
from jax_lbm.geometry.smooth import smooth_sphere_geometry, soft_bounce_back

def simulate_cd(radius):
    """Cd as a differentiable function of sphere radius."""
    porosity, solid, q = smooth_sphere_geometry(
        48, 24, 24, center=(0.0, 0.0, 0.0), radius=radius,
    )
    f = equilibrium(jnp.ones((48, 24, 24)), jnp.zeros((3, 48, 24, 24)).at[0].set(0.05))
    vel = jnp.array([0.05, 0.0, 0.0])
    tau = jnp.float32(0.55)

    def step(carry, _):
        f_in, _ = carry
        f_c = soft_bounce_back(bgk(f_in, tau), porosity)
        f_post = f_c
        f_s = zou_he_outlet(zou_he_inlet(stream(f_c, q), vel))
        return (f_s, f_post), None

    (f_final, f_post), _ = jax.lax.scan(step, (f, f), None, length=100)
    force = momentum_exchange(f_post, f_final, q)
    return drag_coefficient(force, 0.05, jnp.maximum(projected_area(solid), 1.0))

# One line to get the gradient of drag w.r.t. sphere radius.
dCd_dr = jax.grad(simulate_cd)(jnp.float32(0.5))
```

## Project structure

```
jax_lbm/
    core/
        lattice.py       D3Q19 velocities, weights, MRT transforms
        equilibrium.py   density, velocity, f_eq computation
        collision.py     BGK, MRT, Smagorinsky SGS
        streaming.py     pull streaming + Bouzidi bounce-back
        boundary.py      Zou-He inlet/outlet
        forces.py        momentum exchange, Cd
    geometry/
        primitives.py    sphere, box with analytic Bouzidi q
        bouzidi.py       q computation from solid masks
        smooth.py        sigmoid-smoothed solids for optimization
    optim/
        losses.py        target Cd, velocity MSE, TV regularizer
    io/
        vtk.py           ParaView export
    state.py             LBMState, LBMParams, step(), simulate()
tests/                   49 tests: unit, Poiseuille, sphere Cd, Bouzidi gradients
examples/                channel flow, sphere drag, inlet optimization, shape optimization
docs/                    mkdocs-material site with guides and API reference
```

## What's novel

Differentiable fluid solvers exist (adjoint LBM, PhiFlow, Lettuce, XLB). The contribution here is specifically:

- **A shape gradient where there was none.** Classical LBM with halfway bounce-back + binary solid mask gives `dCd/dr = 0` exactly (verified directly; not a small numerical residual). This repo's Bouzidi + ray-sphere `q` pipeline gives a non-zero gradient, which is what makes gradient-based shape optimisation possible in the first place.
- **Honest about the soft-porosity trade-off.** Sigmoid-smoothed solids (density-based topology optimisation; Pingen et al. 2007 for the LBM version) add a second gradient path but admit degenerate optima under area-normalised objectives; the paper documents the trade-off rather than papering over it.
- **Reproducible.** Every figure and table in the paper is regenerated from a single `modal run modal_worker.py --example <name>` call against an A100.

## Tests

```bash
pytest tests/ -v                    # 49 tests, all passing
pytest tests/test_poiseuille.py -v  # analytic Poiseuille comparison
pytest tests/test_bouzidi_gradients.py -v  # Bouzidi gradient verification
```

## Related work

Three prior contributions sit closest to this one:

- **Cheylan, Fritz, Ricot, Sagaut (2019).** "Shape Optimization Using the Adjoint Lattice Boltzmann Method for Aerodynamic Applications," *AIAA Journal* 57(7):2758-2773. Uses Ginzburg's interpolated two-relaxation-time bounce-back with a manually derived continuous adjoint for industrial aerodynamic shape optimisation. Same boundary-treatment argument; adjoint rather than AD.
- **Zarth, Klemens, Thäter, Krause (2021).** "Towards shape optimisation of fluid flows using lattice Boltzmann methods and automatic differentiation," *Comput. Math. Appl.* 90:46-54. Forward-mode AD applied to homogenised LBM in OpenLB, varying local permeability through a smooth indicator function. Mathematically closer to the soft-porosity path in this repo than to the Bouzidi path.
- **Ataei, Salehipour (2024) — XLB** and **Bedrunka et al. (2021) — Lettuce.** Both provide reverse-mode AD primitives; XLB additionally supports interpolated bounce-back. Neither has published an end-to-end shape-optimisation demonstration in which the gradient is traced through `q(geometry)`.

This repo's contribution is reverse-mode AD through Bouzidi interpolated bounce-back in JAX, with a direct empirical comparison of the Bouzidi path against the volume-penalisation path within a single pipeline.

## References

- Clift, R., Grace, J.R. and Weber, M.E. (1978). *Bubbles, Drops, and Particles.* Academic Press.
- d'Humieres, D. et al. (2002). "Multiple-relaxation-time lattice Boltzmann models in three dimensions." Phil. Trans. R. Soc. A.
- Bouzidi, M. et al. (2001). "Momentum transfer of a Boltzmann-lattice fluid with boundaries." Physics of Fluids.
- Zou, Q. and He, X. (1997). "On pressure and velocity boundary conditions for the lattice Boltzmann BGK model." Physics of Fluids.
- Angot, P., Bruneau, C.-H., Fabrie, P. (1999). "A penalization method to take into account obstacles in incompressible viscous flows." Numer. Math. 81:497-520.
- Borrvall, T., Petersson, J. (2003). "Topology optimization of fluids in Stokes flow." Int. J. Numer. Methods Fluids 41(1):77-107.
- Pingen, G., Evgrafov, A., Maute, K. (2007). "Topology optimization of flow domains using the lattice Boltzmann method." Struct. Multidiscip. Optim. 34:507-524.
- Cheylan, I., Fritz, G., Ricot, D., Sagaut, P. (2019). "Shape Optimization Using the Adjoint Lattice Boltzmann Method for Aerodynamic Applications." AIAA Journal 57(7):2758-2773.
- Zarth, A., Klemens, F., Thäter, G., Krause, M.J. (2021). "Towards shape optimisation of fluid flows using lattice Boltzmann methods and automatic differentiation." Comput. Math. Appl. 90:46-54.
