# Representing Geometry Differentiably

How to define obstacles that gradients can flow through.

## Analytic shapes

Spheres and boxes have closed-form signed distance functions and
ray-intersection formulas. The Bouzidi $q$ values are computed
analytically and are differentiable with respect to shape parameters
(radius, center position).

```python
from jax_lbm.geometry.primitives import create_sphere

solid, q = create_sphere(128, 64, 64, center=(0.0, 0.0, 0.0), radius=0.5)
```

See [`create_sphere`](../api/core/primitives.md) and
[`create_box`](../api/core/primitives.md).

## Smooth representations

For gradient-based shape optimization, use the sigmoid-smoothed approach:

```python
from jax_lbm.geometry.smooth import smooth_sphere_geometry

porosity, solid, q = smooth_sphere_geometry(
    64, 32, 32, center=(0.0, 0.0, 0.0), radius=jnp.float32(0.5), sharpness=20.0,
)
```

The porosity field is differentiable with respect to the radius. The
Bouzidi q-values are differentiable through the ray-sphere intersection.

See the [Smooth geometry API](../api/geometry/smooth.md) and the
[Differentiable LBM guide](differentiable_lbm.md) for the theory.

## From solid masks

If you have a binary solid mask from an external source (e.g., a
voxelized mesh), you can compute Bouzidi q-values with a default
distance of 0.5 (standard bounce-back):

```python
from jax_lbm.geometry.bouzidi import compute_q_from_solid

q = compute_q_from_solid(solid, nx, ny, nz, default_q=0.5)
```

This is first-order accurate but works for any geometry.
