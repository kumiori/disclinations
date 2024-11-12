
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

from disclinations.solvers import SNESProblem, SNESSolver


mesh = create_unit_square(MPI.COMM_WORLD, 12, 15)
V = FunctionSpace(mesh, ("Lagrange", 1))
u = Function(V)
v = TestFunction(V)
F = inner(5.0, v) * dx - ufl.sqrt(u * u) * inner(grad(u), grad(v)) * dx - inner(u, v) * dx

u_bc = Function(V)
u_bc.x.array[:] = 1.0
bc = dirichletbc(u_bc, locate_dofs_geometrical(V, lambda x: np.logical_or(np.isclose(x[0], 0.0),
                                                                            np.isclose(x[0], 1.0))))

# Create nonlinear problem
problem = SNESProblem(F, u, bc)

u.x.array[:] = 0.9
b = create_petsc_vector(V.dofmap.index_map, V.dofmap.index_map_bs)
J = create_matrix(problem.a)

# Create Newton solver and solve
snes = PETSc.SNES().create()
snes.setFunction(problem.F, b)
snes.setJacobian(problem.J, J)

snes.setTolerances(rtol=1.0e-9, max_it=10)
snes.getKSP().setType("preonly")
snes.getKSP().setTolerances(rtol=1.0e-9)
snes.getKSP().getPC().setType("lu")

snes.solve(None, u.vector)
assert snes.getConvergedReason() > 0
assert snes.getIterationNumber() < 6

# Modify boundary condition and solve again
u_bc.x.array[:] = 0.6
snes.solve(None, u.vector)
assert snes.getConvergedReason() > 0
assert snes.getIterationNumber() < 6
# print(snes.getIterationNumber())
print(snes.getFunctionNorm())
print(snes.getConvergedReason())
print(f"u norm {u.vector.norm()}")

snes.destroy()
b.destroy()
J.destroy()


u = Function(V)
print("now solve with our SNESSolver class")
print(f"u norm {u.vector.norm()}")

solver = SNESSolver(
    F,
    u,
    [bc],
    bounds=None,
    petsc_options={},
    prefix='test_nonlinear',
)

res = solver.solve()
print("solver returns", res)

__import__('pdb').set_trace()