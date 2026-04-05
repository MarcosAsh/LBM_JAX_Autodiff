# Channel Flow (Poiseuille)

Poiseuille flow through a rectangular channel. Walls at the top and
bottom, Zou-He inlet on the left, pressure outlet on the right. The
steady-state profile is a parabola.

See the [first simulation tutorial](../getting_started/first_simulation.md)
for a step-by-step walkthrough.

**Run it:**

```bash
python examples/01_channel_flow.py       # local CPU
modal run modal_worker.py --example channel  # Modal A100
```

**Source:** [`examples/01_channel_flow.py`](https://github.com/MarcosAsh/LBM_JAX_Autodiff/blob/main/examples/01_channel_flow.py)
