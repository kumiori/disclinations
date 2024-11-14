import dolfinx
from mpi4py import MPI
import numpy as np
import ufl


def compute_cell_contribution_point(V, points):
    # Determine what process owns a point and what cells it lies within
    mesh = V.mesh
    _, _, owning_points, cells = dolfinx.cpp.geometry.determine_point_ownership(
        mesh._cpp_object, points, 1e-6
    )
    owning_points = np.asarray(owning_points).reshape(-1, 3)

    # Pull owning points back to reference cell
    mesh_nodes = mesh.geometry.x
    # cmap = mesh.geometry.cmaps[0]
    # Check for the existence of V.cmap or V.cmaps and select the appropriate one
    if hasattr(mesh.geometry, "cmap"):
        cmap = mesh.geometry.cmap
    elif (
        hasattr(mesh.geometry, "cmaps")
        and isinstance(mesh.geometry.cmaps, list)
        and len(mesh.geometry.cmaps) > 0
    ):
        cmap = mesh.geometry.cmaps[0]

    ref_x = np.zeros((len(cells), mesh.geometry.dim), dtype=mesh.geometry.x.dtype)
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
        basis_values = values[: num_dofs : num_dofs * len(cells)]
    else:
        basis_values = np.zeros((0, num_dofs), dtype=dolfinx.default_scalar_type)
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
    all_cells = [int(cell) for cell in all_cells]
    all_basis_values = np.concatenate(all_basis_values)

    return all_cells, all_basis_values


# def compute_disclination_loads(points, signs, V, b=None):
#     cells, basis_values = compute_cell_contributions(V, points)

#     b = dolfinx.fem.Function(V)
#     b.x.array[:] = 0
#     # Loop over the cells, basis values, and signs
#     V = b.function_space
#     for cell, basis_value, sign in zip(cells, basis_values, signs):
#         dofs = V.dofmap.cell_dofs(cell)
#         # Update the function values
#         b.x.array[dofs] += sign * basis_value

#     return b


def compute_disclination_loads(points, signs, V, V_sub_to_V_dofs=None, V_sub=None):
    b = dolfinx.fem.Function(V)
    b.x.array[:] = 0

    if V_sub_to_V_dofs is None:
        # Single space case
        cells, basis_values = compute_cell_contributions(V, points)
    else:
        # Mixed formulation case
        cells, basis_values = compute_cell_contributions(V_sub, points)

    # Loop over the cells, basis values, and signs
    for cell, basis_value, sign in zip(cells, basis_values, signs):
        if V_sub_to_V_dofs is None:
            dofs = V.dofmap.cell_dofs(cell)
        else:
            subspace_dofs = V_sub.dofmap.cell_dofs(cell)
            dofs = np.array(V_sub_to_V_dofs)[subspace_dofs]
        # Update the function values
        b.x.array[dofs] += sign * basis_value

    return b


from dolfinx import fem
from ufl import dx, grad, div, inner, TrialFunction, TestFunction


def compute_norms(v, w, mesh):
    norms = {}

    # Compute L2 norms for v and w
    v_L2 = fem.assemble_scalar(fem.form(inner(v, v) * dx))
    w_L2 = fem.assemble_scalar(fem.form(inner(w, w) * dx))
    norms["v_L2"] = mesh.comm.allreduce(np.sqrt(v_L2), op=MPI.SUM)
    norms["w_L2"] = mesh.comm.allreduce(np.sqrt(w_L2), op=MPI.SUM)

    # Compute H1 norms for v and w (L2 norm + gradient)
    v_H1 = fem.assemble_scalar(
        fem.form(inner(v, v) * dx + inner(grad(v), grad(v)) * dx)
    )
    w_H1 = fem.assemble_scalar(
        fem.form(inner(w, w) * dx + inner(grad(w), grad(w)) * dx)
    )
    norms["v_H1"] = mesh.comm.allreduce(np.sqrt(v_H1), op=MPI.SUM)
    norms["w_H1"] = mesh.comm.allreduce(np.sqrt(w_H1), op=MPI.SUM)

    # Compute H2 norms for v and w (L2 norm + gradient + Hessian)
    # Define the Hessian as grad(grad(u)) for second derivatives
    v_H2 = fem.assemble_scalar(
        fem.form(
            inner(v, v) * dx
            + inner(grad(v), grad(v)) * dx
            + inner(div(grad(v)), div(grad(v))) * dx
        )
    )
    w_H2 = fem.assemble_scalar(
        fem.form(
            inner(w, w) * dx
            + inner(grad(w), grad(w)) * dx
            + inner(div(grad(w)), div(grad(w))) * dx
        )
    )
    norms["v_H2"] = mesh.comm.allreduce(np.sqrt(v_H2), op=MPI.SUM)
    norms["w_H2"] = mesh.comm.allreduce(np.sqrt(w_H2), op=MPI.SUM)

    return norms
