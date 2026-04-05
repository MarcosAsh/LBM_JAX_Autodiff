# Checkpointing for Long Simulations

Backpropagation through $N$ timesteps stores $O(N)$ intermediate states.
This page explains how to use `jax.checkpoint` to trade compute for
memory, keeping GPU memory constant regardless of simulation length.

## The problem

Each LBM state for a 128^3 grid is about 150 MB (19 floats per cell,
float32). Backprop through 1000 steps would store all 1000 states: 150 GB.
That exceeds any GPU's memory.

## The solution: rematerialization

Wrap the step function with `jax.checkpoint`:

```python
import jax
from jax_lbm import step

checkpointed_step = jax.checkpoint(step)
```

During the backward pass, instead of reading stored intermediates, JAX
re-runs the forward computation to regenerate them. Memory drops to
$O(1)$ in the number of steps. The cost is roughly 2x wall-clock time
for the backward pass (one extra forward pass).

## Using it with `simulate`

The [`simulate`](../api/core/equilibrium.md) function has a
`checkpoint` argument that enables this automatically:

```python
from jax_lbm import simulate

final_state = simulate(state, params, n_steps=1000, checkpoint=True)
```

## Nested checkpointing

For very long simulations (10,000+ steps), even the 2x overhead of full
rematerialization is expensive. The nested scheme checkpoints every
$\sqrt{N}$ steps, then recomputes within each segment. This gives
$O(\sqrt{N})$ memory and $O(N \sqrt{N})$ compute, which is better than
both extremes.

JAX supports this through `jax.checkpoint` with a custom policy, but
for most practical cases (under 5000 steps), simple per-step
checkpointing is sufficient.
