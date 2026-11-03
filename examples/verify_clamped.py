"""Reproducible plane-stress Airy manufactured problem, without plotting.

Run from repo root: PYTHONPATH=src python examples/verify_clamped.py
The exact Airy function is x²(1-x)² y²(1-y)². This is a regular bulk source,
not a core-regularised point defect or a crack-tip benchmark.
"""
import json
import numpy as np
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI
from petsc4py import PETSc
from disclinations.models import NonlinearPlateFVK
from disclinations.solvers import SNESSolver


def assembled(expr, comm):
    return float(
        np.real(comm.allreduce(fem.assemble_scalar(fem.form(expr)), op=MPI.SUM))
    )


def solve_case(n=8, degree=2, alpha=40.0, intensity=1.0, E=7.0, nu=0.23, thickness=0.4):
    """Solve a smooth clamped Airy cell and report physical vs discrete energy.

    Intensity multiplies a numerical incompatibility load; material parameters
    are SI. This unit-square manufactured amplitude is chosen numerically.
    Returns diagnostics plus the solution for superposition tests.
    """
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n, ghost_mode=mesh.GhostMode.shared_facet)
    V = fem.functionspace(msh, ("Lagrange", degree))
    u, zero = fem.Function(V), fem.Function(V)
    model = NonlinearPlateFVK(
        msh,
        {
            "order": degree,
            "E": E,
            "nu": nu,
            "thickness": thickness,
            "alpha_penalty": alpha,
        },
    )
    state = {"v": u, "w": zero}
    physical = model.energy(state)[2]
    correction = -model.penalisation(state)  # remove saddle sign on the Airy block
    x = ufl.SpatialCoordinate(msh)
    exact = intensity * x[0] ** 2 * (1 - x[0]) ** 2 * x[1] ** 2 * (1 - x[1]) ** 2
    load = model.material.airy_compliance * ufl.div(ufl.grad(ufl.div(ufl.grad(exact))))
    work = load * u * ufl.dx
    functional = physical + correction - work
    residual = ufl.derivative(functional, u, ufl.TestFunction(V))
    msh.topology.create_connectivity(1, 2)
    boundary = mesh.exterior_facet_indices(msh.topology)
    dofs = fem.locate_dofs_topological(V, 1, boundary)
    bc = fem.dirichletbc(PETSc.ScalarType(0), dofs, V)
    solver = SNESSolver(
        residual,
        u,
        bcs=[bc],
        monitor=None,
        petsc_options={
            "snes_rtol": 1e-10,
            "snes_atol": 1e-12,
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )
    iterations, reason = solver.solve()
    exact_energy = ufl.replace(physical, {u: exact})
    solver.F(solver.solver, u.x.petsc_vec, solver.b)
    report = {
        "n": n,
        "degree": degree,
        "alpha": alpha,
        "intensity": intensity,
        "dofs": V.dofmap.index_map.size_global,
        "physical_membrane_energy": assembled(physical, comm),
        "exact_membrane_energy": assembled(exact_energy, comm),
        "discrete_correction": assembled(correction, comm),
        "discrete_quadratic": assembled(physical + correction, comm),
        "load_work": assembled(work, comm),
        "functional": assembled(functional, comm),
        "l2_error": np.sqrt(assembled((u - exact) ** 2 * ufl.dx, comm)),
        "residual_norm": solver.b.norm(),
        "iterations": iterations,
        "reason": int(reason),
        "ksp_type": solver.solver.getKSP().getType(),
        "pc_type": solver.solver.getKSP().getPC().getType(),
    }
    return report, u


if __name__ == "__main__":
    reports = [solve_case(n, degree)[0] for degree in (2, 3) for n in (4, 8, 16)]
    if MPI.COMM_WORLD.rank == 0:
        print(json.dumps(reports, indent=2))
