# Sphere Drag at Re=100

Places a sphere in a wind tunnel and measures the drag coefficient as
the simulation converges. Validates against Clift et al. (1978): Cd = 1.09
for a sphere at Re=100.

Grid: 128x64x64, sphere diameter 16 cells, 4000 BGK steps.

**Run it:**

```bash
python examples/02_sphere_drag.py        # local (slow on CPU)
modal run modal_worker.py --example sphere   # Modal A100
```

**Source:** [`examples/02_sphere_drag.py`](https://github.com/MarcosAsh/LBM_JAX_Autodiff/blob/main/examples/02_sphere_drag.py)
