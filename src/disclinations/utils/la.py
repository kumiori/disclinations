"""Collective, ownership-aware assembly of scalar Dirac loads (DOLFINx 0.9)."""
import dolfinx
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np


def compute_cell_contribution_point(V, points):
    """Return owning cells and scalar basis values for local input points.

    Collective on V.mesh.comm. Each global point must be supplied on exactly
    one rank. Points have shape (N, 3), including (0, 3) on other ranks.
    Outside points raise collectively instead of silently removing a load.
    """
    msh = V.mesh
    points = np.asarray(points, dtype=msh.geometry.x.dtype).reshape(-1, 3)
    data = dolfinx.cpp.geometry.determine_point_ownership(msh._cpp_object, points, 1e-8)
    missing = np.any(np.asarray(data.src_owner) < 0)
    if msh.comm.allreduce(bool(missing), op=MPI.LOR):
        raise ValueError("Defect point lies outside the mesh")
    cells = np.asarray(data.dest_cells, dtype=np.int32)
    owning_points = np.asarray(data.dest_points).reshape(-1, 3)
    if V.ufl_element().reference_value_shape or V.dofmap.bs != 1:
        raise ValueError("Point loads require a scalar collapsed space")
    num_dofs = V.dofmap.dof_layout.num_dofs
    basis = np.empty((len(cells), num_dofs), dtype=dolfinx.default_scalar_type)
    # Scalar H1 Lagrange basis: identity pullback, no Piola transform.
    if (
        V.element.basix_element.family.name != "P"
        or V.element.basix_element.discontinuous
    ):
        raise ValueError("Point loads support continuous scalar Lagrange elements")
    for i, (point, cell) in enumerate(zip(owning_points, cells)):
        geom = msh.geometry.x[msh.geometry.dofmap[cell]]
        ref = msh.geometry.cmap.pull_back(point.reshape(1, 3), geom)
        basis[i] = V.element.basix_element.tabulate(0, ref)[0, 0, :, 0]
    return cells, basis


def compute_cell_contributions(V, points):
    """Return cell/basis arrays; strengths must be paired before concatenation."""
    contributions = [compute_cell_contribution_point(V, point) for point in points]
    if not contributions:
        return np.empty(0, dtype=np.int32), np.empty((0, V.dofmap.dof_layout.num_dofs))
    return tuple(np.concatenate([item[i] for item in contributions]) for i in (0, 1))


def compute_disclination_loads(points, signs, V, V_sub_to_V_dofs=None, V_sub=None):
    """Assemble b_i = sum_k s_k phi_i(x_k), returned as a Function dual vector.

    points is a list of per-defect arrays of shape (1,3) on one source rank,
    (0,3) elsewhere. signs is the SAME global ordered strength list on every
    rank. V_sub and its collapse map select a scalar mixed-space component.
    The returned vector is reverse accumulated and forward synchronised.
    It is already assembled: solver callbacks must add owned entries only.
    This routine defines numerical source signs, not a Burgers convention.
    """
    comm = V.mesh.comm
    counts = comm.allgather((len(points), len(signs)))
    if any(p != s or (p, s) != counts[0] for p, s in counts):
        raise ValueError("points and signs must have equal global lengths on all ranks")
    if (V_sub_to_V_dofs is None) != (V_sub is None):
        raise ValueError("Provide both V_sub and V_sub_to_V_dofs")
    scalar_space = V if V_sub is None else V_sub
    b = dolfinx.fem.Function(V)
    b.x.array[:] = 0
    mapping = None if V_sub_to_V_dofs is None else np.asarray(V_sub_to_V_dofs)
    for point, sign in zip(points, signs):
        count = comm.allreduce(np.asarray(point).size // 3, op=MPI.SUM)
        if count != 1:
            raise ValueError("Each defect must be supplied on exactly one rank")
        cells, values = compute_cell_contribution_point(scalar_space, point)
        for cell, basis in zip(cells, values):
            dofs = scalar_space.dofmap.cell_dofs(cell)
            if mapping is not None:
                dofs = mapping[dofs]
            np.add.at(b.x.array, dofs, sign * basis)
    b.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    b.x.scatter_forward()
    return b
