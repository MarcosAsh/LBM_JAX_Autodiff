# Installation

## Requirements

- Python 3.10+
- A CUDA-capable GPU (A100, L4, T4) for serious work. CPU works for
  development and small grids but is 50-100x slower.

## Install from source

```bash
git clone https://github.com/MarcosAsh/LBM_JAX_Autodiff.git
cd LBM_JAX_Autodiff
pip install -e .
```

This installs JAX with CUDA 12 support, optax for optimizers, and the
`jax_lbm` package itself.

## Verify the installation

```bash
python -c "
import jax
import jax_lbm
print(f'JAX backend: {jax.default_backend()}')
print(f'JAX LBM loaded: {jax_lbm.__name__}')
"
```

You should see:

```
JAX backend: gpu
JAX LBM loaded: jax_lbm
```

If it says `cpu` instead of `gpu`, JAX did not find your CUDA installation.
Check the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html)
for GPU setup.

## Run the tests

```bash
pytest tests/ -v
```

All tests should pass. The slow integration tests (sphere drag, Poiseuille
flow) take a few minutes on CPU. On GPU they finish in seconds.

## Optional: Modal for cloud GPU

If you don't have a local GPU, you can run everything on an A100 via
[Modal](https://modal.com):

```bash
pip install modal
modal setup  # one-time auth
modal run modal_worker.py --tests
modal run modal_worker.py --example channel
```
