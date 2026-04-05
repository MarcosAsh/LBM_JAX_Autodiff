# Inlet Velocity Optimization

The simplest differentiable optimization: find the inlet velocity that
produces a target drag coefficient. Uses `jax.value_and_grad` to compute
both Cd and its gradient with respect to inlet velocity in a single pass.

See the [first optimization tutorial](../getting_started/first_optimization.md)
for a detailed walkthrough of the gradient computation.

**Run it:**

```bash
python examples/03_optimize_inlet.py     # local
modal run modal_worker.py --example inlet    # Modal A100
```

**Source:** [`examples/03_optimize_inlet.py`](https://github.com/MarcosAsh/LBM_JAX_Autodiff/blob/main/examples/03_optimize_inlet.py)
