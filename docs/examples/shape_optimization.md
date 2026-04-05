# Shape Optimization

The main demonstration of differentiable Bouzidi bounce-back. Optimizes
a sphere's radius to minimize drag by backpropagating through the entire
simulation pipeline: geometry -> Bouzidi q-values -> streaming -> force -> Cd.

Uses sigmoid-smoothed solids for differentiable solid/fluid blending and
analytic ray-sphere intersection for differentiable wall distances.

See the [differentiable LBM guide](../guides/differentiable_lbm.md)
for the theory behind this.

**Run it:**

```bash
python examples/04_shape_optimization.py    # local (slow)
modal run modal_worker.py --example optimize    # Modal A100
```

**Source:** [`examples/04_shape_optimization.py`](https://github.com/MarcosAsh/LBM_JAX_Autodiff/blob/main/examples/04_shape_optimization.py)
