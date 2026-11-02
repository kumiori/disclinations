"""Assembly regression; replaces an import-time debugger demonstration."""
import numpy as np
from dolfinx import fem, mesh
from mpi4py import MPI
import ufl


def test_scalar_assembly():
    msh = mesh.create_unit_square(MPI.COMM_WORLD, 3, 3)
    V = fem.functionspace(msh, ("Lagrange", 2))
    v = fem.Function(V)
    v.interpolate(lambda x: x[0] ** 2 + x[1] ** 2)
    value = MPI.COMM_WORLD.allreduce(
        fem.assemble_scalar(fem.form(ufl.div(ufl.grad(v)) ** 2 * ufl.dx)), op=MPI.SUM
    )
    assert np.isclose(value, 16.0)
