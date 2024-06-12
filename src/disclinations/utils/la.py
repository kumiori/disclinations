import dolfinx 
from mpi4py import MPI
import numpy as np
import ufl

def compute_cell_contribution_point(V, points):
    # Determine what process owns a point and what cells it lies within
    mesh = V.mesh
    _, _, owning_points, cells = dolfinx.cpp.geometry.determine_point_ownership(
        mesh._cpp_object, points, 1e-6)
    owning_points = np.asarray(owning_points).reshape(-1, 3)

    # Pull owning points back to reference cell
    mesh_nodes = mesh.geometry.x
    cmap = mesh.geometry.cmaps[0]
    ref_x = np.zeros((len(cells), mesh.geometry.dim),
                     dtype=mesh.geometry.x.dtype)
    for i, (point, cell) in enumerate(zip(owning_points, cells)):
        geom_dofs = mesh.geometry.dofmap[cell]
        ref_x[i] = cmap.pull_back(point.reshape(-1, 3), mesh_nodes[geom_dofs])

    # Create expression evaluating a trial function (i.e. just the basis function)
    u = ufl.TrialFunction(V)
    num_dofs = V.dofmap.dof_layout.num_dofs * V.dofmap.bs
    if len(cells) > 0:
        # NOTE: Expression lives on only this communicator rank
        expr = dolfinx.fem.Expression(u, ref_x, comm=MPI.COMM_SELF)
        values = expr.eval(mesh, np.asarray(cells, dtype=np.int32))

        # Strip out basis function values per cell
        basis_values = values[:num_dofs:num_dofs*len(cells)]
    else:
        basis_values = np.zeros(
            (0, num_dofs), dtype=dolfinx.default_scalar_type)
    return cells, basis_values


def compute_cell_contributions(V, points):
    # Initialize empty arrays to store cell indices and basis values
    all_cells = []
    all_basis_values = []

    for point in points:
        # Compute cell contributions for the current point
        cells, basis_values = compute_cell_contribution_point(V, point)

        # Append the results to the arrays
        all_cells.append(cells)
        all_basis_values.append(basis_values)

    # Concatenate the lists to create NumPy arrays
    all_cells = np.concatenate(all_cells)
    all_basis_values = np.concatenate(all_basis_values)

    return all_cells, all_basis_values

