"""Scalar bilaplacian regression with a known exact polynomial load."""
from dolfinx import fem, mesh
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from disclinations.models import Biharmonic
from disclinations.solvers import SNESSolver


def test_bilaplacian_solution():
    msh = mesh.create_unit_square(
        MPI.COMM_WORLD, 8, 8, ghost_mode=mesh.GhostMode.shared_facet
    )
    V = fem.functionspace(msh, ("Lagrange", 3))
    u = fem.Function(V)
    x = ufl.SpatialCoordinate(msh)
    exact = x[0] ** 2 * (1 - x[0]) ** 2 * x[1] ** 2 * (1 - x[1]) ** 2
    f = ufl.div(ufl.grad(ufl.div(ufl.grad(exact))))
    model = Biharmonic(msh, {"alpha_penalty": 40.0})
    state = {"u": u}
    functional = model.energy(state) + model.penalisation(state) - f * u * ufl.dx
    msh.topology.create_connectivity(1, 2)
    dofs = fem.locate_dofs_topological(V, 1, mesh.exterior_facet_indices(msh.topology))
    bc = fem.dirichletbc(PETSc.ScalarType(0), dofs, V)
    solver = SNESSolver(
        ufl.derivative(functional, u, ufl.TestFunction(V)),
        u,
        bcs=[bc],
        monitor=None,
        petsc_options={"snes_atol": 1e-12},
    )
    solver.solve()
    error = MPI.COMM_WORLD.allreduce(
        fem.assemble_scalar(fem.form((u - exact) ** 2 * ufl.dx)), op=MPI.SUM
    )
    assert error < 1e-9
