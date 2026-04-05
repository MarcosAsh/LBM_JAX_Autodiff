# Differentiable LBM: Why and How

This is the core idea behind the project. If you can differentiate
through a fluid simulation, you can optimize the inputs (geometry,
boundary conditions, material properties) to achieve a desired output
(target drag, target flow field) using gradient descent instead of
trial and error.

## The gradient you want

Say you have a simulation that takes an inlet velocity and produces a
drag coefficient:

```
u_inlet  -->  [LBM: 1000 steps]  -->  Cd
```

You want $\partial Cd / \partial u_{inlet}$. With that gradient, you can
adjust the inlet velocity to push Cd toward a target value.

The same idea extends to more interesting parameters:

- $\partial Cd / \partial \tau$ -- how does viscosity affect drag?
- $\partial Cd / \partial q$ -- how do wall distances affect drag?
  (This is shape optimization.)
- $\partial Cd / \partial \mathbf{v}$ -- how do mesh vertex positions
  affect drag? (The hard research problem.)

## Why JAX

JAX traces your Python code into an XLA computation graph, then
differentiates that graph symbolically. You write the forward simulation
in normal Python/NumPy-like code, and `jax.grad` gives you the backward
pass for free.

The requirements for this to work:

1. **Pure functions.** No side effects, no mutable state. Every function
   takes arrays in and returns arrays out.
2. **JAX primitives only.** Use `jnp` instead of `np`, `jax.lax.fori_loop`
   instead of Python for-loops inside JIT.
3. **No data-dependent control flow.** You can't branch on a traced value
   (e.g., `if f[i] > 0`). Use `jnp.where` instead.

Every function in this library satisfies these constraints.

## What makes Bouzidi special

Standard bounce-back places the wall exactly halfway between a fluid
node and a solid node. The wall position is locked to the grid -- you
can't move it by a fraction of a cell. This means the gradient of drag
with respect to wall position is zero almost everywhere (it only changes
when a cell flips between solid and fluid).

Bouzidi interpolated bounce-back uses the fractional distance $q$ to the
wall along each lattice link. The streaming step interpolates using $q$:

For $q \ge 0.5$:
$$f_i = \frac{1}{2q} f^*_{i'} + \left(1 - \frac{1}{2q}\right) f^*_i$$

For $q < 0.5$:
$$f_i = 2q \, f^*_{i'} + (1 - 2q) \, f^*_{i''}$$

where $i'$ is the opposite direction and $i''$ is the far-field neighbor.

The critical point: $q$ is a continuous function of the wall position.
For a sphere of radius $r$, the ray-sphere intersection gives:

$$q = \frac{-b - \sqrt{b^2 - 4ac}}{2a}$$

where $a$, $b$, $c$ depend on the ray origin, direction, and $r$.
This is a smooth function of $r$, so $\partial q / \partial r$ exists
and is well-defined. JAX traces through the square root and the division
automatically.

The gradient chain is:

$$\frac{\partial Cd}{\partial r} = \sum_{\text{links}} \frac{\partial Cd}{\partial f_i} \cdot \frac{\partial f_i}{\partial q} \cdot \frac{\partial q}{\partial r}$$

Every term in this chain is computed by JAX's reverse-mode autodiff.

## The solid mask problem

The solid/fluid boundary is a step function: a cell is either inside the
obstacle or outside. Step functions have zero gradient everywhere except
at the discontinuity, where the gradient is undefined.

Two workarounds:

### 1. Sigmoid smoothing (what we use)

Replace the binary mask with a smooth porosity field:

$$\phi(\mathbf{x}) = \sigma\left(\alpha \cdot (r - |\mathbf{x} - \mathbf{c}|)\right)$$

where $\sigma$ is the sigmoid function and $\alpha$ controls sharpness.
The collision step blends between fluid and solid behavior:

$$f^{out} = (1 - \phi) \, f^{collided} + \phi \, f^{bounced}$$

This has a well-defined gradient everywhere. The downside is a thin
"mushy zone" near the surface where the physics is approximate. Higher
$\alpha$ shrinks the zone but makes gradients spikier.

See [`porosity_from_sdf`](../api/geometry/smooth.md) and
[`soft_bounce_back`](../api/geometry/smooth.md).

### 2. Level-set reparameterization

Represent the geometry as a signed distance function. The solid mask is
$\text{SDF} > 0$ (non-differentiable), but the Bouzidi $q$ values
computed from the SDF are differentiable. This separates the
"which cells are solid" question (discrete) from the "where exactly is
the wall" question (continuous).

In practice, we use both: sigmoid smoothing for the collision blending,
and differentiable ray-intersection for the Bouzidi q-values.

## Memory: checkpointing

Backpropagation through $N$ timesteps stores $O(N)$ intermediate states.
For a 128^3 grid with 19 distributions per cell, one state is about
150 MB. A thousand steps would need 150 GB of memory.

`jax.checkpoint` (rematerialization) fixes this by recomputing forward
values during the backward pass instead of storing them:

```python
@jax.checkpoint
def step(state, params):
    ...
```

Memory usage becomes $O(1)$ in the number of steps, at the cost of
roughly 2x wall-clock time for the backward pass. For very long
simulations, nested checkpointing (every $\sqrt{N}$ steps) reduces
the recomputation overhead further.

See the [Checkpointing guide](checkpointing.md) for details.

## What's novel here

Differentiable LBM is not new. Adjoint methods for LBM go back to
Tekitek et al. (2009), and JAX-based differentiable fluid solvers exist
(PhiFlow for Navier-Stokes).

What's new in this project is making **Bouzidi interpolated bounce-back
differentiable** -- gradients flowing through the fractional wall
distance $q$ into obstacle geometry, enabling shape optimization with
2nd-order accurate wall treatment. Most prior work either uses standard
bounce-back (1st order, no useful shape gradient) or treats $q$ as fixed
geometry.
