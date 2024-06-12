
from mpi4py import MPI
from petsc4py import PETSc

import numpy as np

import ufl
from dolfinx import cpp as _cpp
from dolfinx import default_real_type
from dolfinx.fem import (Function, FunctionSpace, dirichletbc, form,
                         locate_dofs_geometrical)
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               create_matrix, create_vector, set_bc)
from dolfinx.la import create_petsc_vector
from dolfinx.mesh import create_unit_square
from ufl import TestFunction, TrialFunction, derivative, dx, grad, inner
import pyvista
from pyvista.plotting.utilities import xvfb
import logging

from disclinations.solvers import SNESProblem

mesh = create_unit_square(MPI.COMM_WORLD, 12, 15)
V = FunctionSpace(mesh, ("Lagrange", 1))
u = Function(V)
v = TestFunction(V)
F = inner(5.0, v) * dx - ufl.sqrt(u * u) * inner(grad(u), grad(v)) * dx - inner(u, v) * dx
J = derivative(F, u, TrialFunction(V))
J_form = form(J)


u_bc = Function(V)
u_bc.x.array[:] = 1.0
bc = dirichletbc(u_bc, locate_dofs_geometrical(V, lambda x: np.logical_or(np.isclose(x[0], 0.0),
                                                                            np.isclose(x[0], 1.0))))

def monitor(snes, it, norm):
    logging.info(f"Iteration {it}, residual {norm}")
    return PETSc.SNES.ConvergedReason.ITERATING


solver = SNESProblem(F, u, [bc], monitor = monitor)
u.x.array[:] = 0.9

solver.snes.solve(None, u.vector)
print(solver.snes.getConvergedReason())
