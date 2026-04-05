"""VTK export for visualization in ParaView.

Writes the velocity field, density, and solid mask to VTK structured
grid files that ParaView can load directly. Use this to produce
publication-quality flow visualizations.

Requires the ``meshio`` package (optional dependency).
"""

import jax.numpy as jnp
import numpy as np


def export_vtk(
    filename: str,
    f: jnp.ndarray,
    solid: jnp.ndarray = None,
    porosity: jnp.ndarray = None,
):
    """Export simulation state to a VTK structured grid file.

    Computes density and velocity from the distribution functions and
    writes them as point data. Optionally includes the solid mask and
    porosity field.

    Args:
        filename: Output path (should end in ``.vtk`` or ``.vtu``).
        f: Distribution functions, shape ``(19, nx, ny, nz)``.
        solid: Optional solid mask, shape ``(nx, ny, nz)``.
        porosity: Optional porosity field, shape ``(nx, ny, nz)``.

    Example:

    ```python
    >>> from jax_lbm.io.vtk import export_vtk
    >>> from jax_lbm.core.equilibrium import equilibrium
    >>> import jax.numpy as jnp
    >>> rho = jnp.ones((16, 8, 8))
    >>> u = jnp.zeros((3, 16, 8, 8)).at[0].set(0.05)
    >>> f = equilibrium(rho, u)
    >>> export_vtk("flow.vtk", f)
    ```
    """
    try:
        import meshio
    except ImportError:
        raise ImportError(
            "meshio is required for VTK export. Install it with: pip install meshio"
        )

    from jax_lbm.core.equilibrium import compute_density, compute_velocity

    rho = np.asarray(compute_density(f))
    u = np.asarray(compute_velocity(f, jnp.array(rho)))

    nx, ny, nz = rho.shape

    # Build structured grid coordinates.
    x = np.arange(nx + 1, dtype=np.float64)
    y = np.arange(ny + 1, dtype=np.float64)
    z = np.arange(nz + 1, dtype=np.float64)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)

    # Hexahedral cells for structured grid.
    cells = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                n0 = i * (ny + 1) * (nz + 1) + j * (nz + 1) + k
                n1 = n0 + 1
                n2 = n0 + (nz + 1)
                n3 = n2 + 1
                n4 = n0 + (ny + 1) * (nz + 1)
                n5 = n4 + 1
                n6 = n4 + (nz + 1)
                n7 = n6 + 1
                cells.append([n0, n1, n3, n2, n4, n5, n7, n6])

    cells_array = np.array(cells)

    # Cell data.
    cell_data = {
        "density": [rho.ravel()],
        "velocity_x": [u[0].ravel()],
        "velocity_y": [u[1].ravel()],
        "velocity_z": [u[2].ravel()],
        "velocity_magnitude": [np.sqrt(
            u[0].ravel()**2 + u[1].ravel()**2 + u[2].ravel()**2
        )],
    }

    if solid is not None:
        cell_data["solid"] = [np.asarray(solid).ravel().astype(np.float64)]

    if porosity is not None:
        cell_data["porosity"] = [np.asarray(porosity).ravel().astype(np.float64)]

    mesh = meshio.Mesh(
        points=points,
        cells=[("hexahedron", cells_array)],
        cell_data=cell_data,
    )
    mesh.write(filename)
