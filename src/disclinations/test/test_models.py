"""Nonlinear solver smoke tests; independent mathematical tests live in tests/.

The original plotting scripts are recoverable at the audit checkpoint.
These checks establish executable residual assembly and solver convergence;
they do not establish equivalence of the three nonlinear coupling schemes.
"""
import basix.ufl
import dolfinx
import numpy as np
import pytest
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI
from petsc4py import PETSc
from disclinations.models import (
    NonlinearPlateFVK,
    NonlinearPlateFVK_brenner,
    NonlinearPlateFVK_carstensen,
)
from disclinations.solvers import SNESSolver
from disclinations.utils.la import compute_disclination_loads


def run_model(variant, transverse=1e-3, defect=0.0):
    msh = mesh.create_unit_square(
        MPI.COMM_WORLD, 4, 4, ghost_mode=mesh.GhostMode.shared_facet
    )
    P = basix.ufl.element("P", msh.basix_cell(), 2)
    Q = fem.functionspace(msh, basix.ufl.mixed_element([P, P]))
    q = fem.Function(Q)
    v, w = ufl.split(q)
    model = [
        NonlinearPlateFVK,
        NonlinearPlateFVK_brenner,
        NonlinearPlateFVK_carstensen,
    ][variant](
        msh, {"order": 2, "E": 7.0, "nu": 0.23, "thickness": 0.4, "alpha_penalty": 40.0}
    )
    state = {"v": v, "w": w}
    functional = (
        model.energy(state)[0]
        + model.penalisation(state)
        - transverse * w * ufl.Measure("dx", domain=msh)
    )
    F = ufl.derivative(functional, q, ufl.TestFunction(Q))
    if variant:
        tv, tw = ufl.TestFunctions(Q)
        F += model.coupling_term(state, tv, tw)
    msh.topology.create_connectivity(1, 2)
    facets = mesh.exterior_facet_indices(msh.topology)
    bcs = [
        fem.dirichletbc(
            PETSc.ScalarType(0),
            fem.locate_dofs_topological(Q.sub(i), 1, facets),
            Q.sub(i),
        )
        for i in (0, 1)
    ]
    b0 = None
    if defect:
        V, mapping = Q.sub(0).collapse()
        points = [
            np.array([[0.43, 0.37, 0.0]]) if msh.comm.rank == 0 else np.empty((0, 3))
        ]
        load = compute_disclination_loads(points, [defect], Q, mapping, V)
        b0 = load.x.petsc_vec.copy()
    solver = SNESSolver(
        F,
        q,
        bcs=bcs,
        b0=b0,
        monitor=None,
        petsc_options={"snes_rtol": 1e-9, "snes_atol": 1e-12},
    )
    _, reason = solver.solve()
    solver.F(solver.solver, q.x.petsc_vec, solver.b)
    assert reason > 0 and solver.b.norm() < 1e-9
    assert np.all(np.isfinite(q.x.array))
    if b0 is not None:
        b0.destroy()


@pytest.mark.parametrize("variant", [0, 1, 2])
def test_model_computation(variant):
    run_model(variant)
