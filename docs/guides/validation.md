# Validation Against Reference Data

Every physics solver needs validation. This page documents what we test
against and what accuracy to expect.

## Reference solver

The ground truth is a C + OpenGL compute shader D3Q19 solver at
[github.com/MarcosAsh/Lattice_Fluid_Dynamics](https://github.com/MarcosAsh/Lattice_Fluid_Dynamics).
It implements the same collision operators, boundary conditions, and force
computation, validated on A100 GPUs.

## Validation cases

### Poiseuille channel flow

Analytic solution exists. The steady-state velocity profile between
parallel walls is a parabola. We achieve 3+ significant figures of
agreement on a 60x20x4 grid with 3000 BGK steps at tau=0.8.

### Sphere drag at Re=100

Reference value: Cd = 1.09 (Clift, Grace, and Weber 1978, Table 5.2).
On a 128x64x64 grid with 16-cell sphere diameter and 4000 BGK steps,
the solver should produce Cd within 10% of the reference. Coarser grids
give wider error bars.

### Gradient verification

Every differentiable function is checked against central finite
differences with step size 1e-4. The relative error should be below
1e-3 for equilibrium and collision, and within a factor of 2 for the
full multi-step pipeline (where float32 noise accumulates).

## Running the validation suite

```bash
# Fast unit tests (seconds)
pytest tests/ -v --ignore=tests/test_poiseuille.py --ignore=tests/test_sphere_cd.py

# Poiseuille flow (minutes on CPU, seconds on GPU)
pytest tests/test_poiseuille.py -v

# Sphere drag (minutes on CPU, runs on Modal for GPU)
modal run modal_worker.py --tests

# Bouzidi gradient verification
pytest tests/test_bouzidi_gradients.py -v
```
