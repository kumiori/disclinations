import numpy as np
import pytest
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI
from petsc4py import PETSc
from disclinations.utils.la import compute_disclination_loads
from disclinations.solvers import SNESSolver


def space():
    msh = mesh.create_unit_square(
        MPI.COMM_WORLD, 4, 4, ghost_mode=mesh.GhostMode.shared_facet
    )
    return fem.functionspace(msh, ("Lagrange", 2))


def test_point_load_strengths_and_pairing():
    V = space()
    coords = [[0.13, 0.19, 0.0], [0.71, 0.83, 0.0]]
    points = [
        np.array([p]) if V.mesh.comm.rank == 0 else np.empty((0, 3)) for p in coords
    ]
    b = compute_disclination_loads(points, [2.0, -3.0], V)
    z = fem.Function(V)
    z.interpolate(lambda x: 1 + 2 * x[0] - x[1])
    expected = sum(s * (1 + 2 * p[0] - p[1]) for s, p in zip([2.0, -3.0], coords))
    assert b.x.petsc_vec.dot(z.x.petsc_vec) == pytest.approx(expected, abs=1e-12)


def test_point_load_outside_fails():
    V = space()
    points = [
        np.array([[3.0, 3.0, 0.0]]) if V.mesh.comm.rank == 0 else np.empty((0, 3))
    ]
    with pytest.raises(ValueError, match="outside"):
        compute_disclination_loads(points, [1.0], V)


def test_solver_configuration_and_convergence():
    V = space()
    u = fem.Function(V)
    v = ufl.TestFunction(V)
    solver = SNESSolver(
        (u - 2) * v * ufl.dx,
        u,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "snes_rtol": 1e-9},
        monitor=None,
    )
    assert solver.solver.getKSP().getType() == "preonly"
    assert solver.solver.getKSP().getPC().getType() == "lu"
    assert solver.solver.getTolerances()[0] == pytest.approx(1e-9)
    _, reason = solver.solve()
    assert reason > 0
    assert np.max(abs(u.x.array - 2)) < 1e-10


def test_solver_nonconvergence_raises():
    V = space()
    u = fem.Function(V)
    solver = SNESSolver(
        (u - 2) * ufl.TestFunction(V) * ufl.dx,
        u,
        petsc_options={"snes_max_it": 0},
        monitor=None,
    )
    with pytest.raises(RuntimeError, match="converg"):
        solver.solve()


def test_solver_prefixes_are_isolated():
    V = space()
    u, v = fem.Function(V), fem.Function(V)
    first = SNESSolver(
        (u - 1) * ufl.TestFunction(V) * ufl.dx,
        u,
        petsc_options={"snes_rtol": 1e-7},
        monitor=None,
    )
    second = SNESSolver(
        (v - 1) * ufl.TestFunction(V) * ufl.dx,
        v,
        petsc_options={"snes_rtol": 1e-5},
        monitor=None,
    )
    assert first.prefix != second.prefix
    assert first.solver.getTolerances()[0] == pytest.approx(1e-7)
    assert second.solver.getTolerances()[0] == pytest.approx(1e-5)
    options = PETSc.Options().getAll()
    assert first.prefix + "snes_rtol" not in options
    assert second.prefix + "snes_rtol" not in options
