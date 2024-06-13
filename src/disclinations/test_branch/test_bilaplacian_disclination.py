# Solving a simple Poisson problem
# H^1_0 formulation
#

import json
import logging
import os
import pdb
import sys
from pathlib import Path

import dolfinx
import dolfinx.plot
import numpy as np
import petsc4py
import ufl
import yaml
from disclinations.meshes import mesh_bounding_box
from disclinations.meshes.primitives import mesh_circle_gmshapi

# from damage.utils import ColorPrint
from disclinations.solvers import SNESSolver, SNESProblem
from disclinations.utils.la import compute_cell_contributions
from disclinations.utils.viz import plot_mesh

from dolfinx import log
from dolfinx.io import XDMFFile
from mpi4py import MPI
from petsc4py import PETSc

logging.basicConfig(level=logging.INFO)


from dolfinx.fem.petsc import (assemble_matrix, create_vector, create_matrix, assemble_vector)

import sys

import dolfinx
import dolfinx.io
import dolfinx.mesh
import dolfinx.plot
import ufl
import yaml
from dolfinx.fem import Constant, dirichletbc
from dolfinx.io import XDMFFile, gmshio
from mpi4py import MPI
from ufl import (
    CellDiameter,
    FacetNormal,
    TestFunction,
    TrialFunction,
    avg,
    ds,
    dS,
    dx,
    div,
    grad,
    inner,
    dot,
    jump,
    outer,
    as_matrix,
    sym,
)

petsc4py.init(sys.argv)
log.set_log_level(log.LogLevel.WARNING)

comm = MPI.COMM_WORLD

def monitor(snes, it, norm):
    logging.info(f"Iteration {it}, residual {norm}")
    return PETSc.SNES.ConvergedReason.ITERATING


with open("parameters.yml") as f:
    parameters = yaml.load(f, Loader=yaml.FullLoader)

mesh_size = parameters["geometry"]["mesh_size"]
# mesh_size = .3
parameters["geometry"]["radius"] = 1
parameters["geometry"]["geom_type"] = "circle"
order = parameters["model"]["order"]

model_rank = 0
tdim = 2


gmsh_model, tdim = mesh_circle_gmshapi(
    parameters["geometry"]["geom_type"], 1, mesh_size, tdim
)
mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, model_rank, tdim)

import matplotlib.pyplot as plt

plt.figure()
ax = plot_mesh(mesh)
fig = ax.get_figure()
fig.savefig(f"output/coarse-mesh.png")

X = ufl.FiniteElement("CG", mesh.ufl_cell(), 1)

V = dolfinx.fem.functionspace(mesh, ("Lagrange", order))

mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
bndry_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)

dofs = dolfinx.fem.locate_dofs_topological(V=V, entity_dim=1, entities=bndry_facets)

bcs = [dolfinx.fem.dirichletbc(value=np.array(0, dtype=PETSc.ScalarType), dofs=dofs, V=V)]

u = dolfinx.fem.Function(V)
D = dolfinx.fem.Constant(mesh, 1.)
α = dolfinx.fem.Constant(mesh, 10.)
load = dolfinx.fem.Constant(mesh, 1.)
h = ufl.CellDiameter(mesh)
h_avg = (h('+') + h('-')) / 2.0

tdim = mesh.topology.dim
num_cells = mesh.topology.index_map(tdim).size_local
_h = dolfinx.cpp.mesh.h(mesh._cpp_object, tdim, np.arange(num_cells, dtype=np.int32))
# print(_h)

n = ufl.FacetNormal(mesh)

dx = ufl.Measure("dx")
dS = ufl.Measure("dS")

bending = (D/2 * (inner(div(grad(u)), div(grad(u))))) * dx 
W_ext = load * u * dx

dg1 = lambda u: 1/2 * dot(jump(grad(u)), avg(grad(grad(u)) * n)) * dS
dg2 = lambda u: 1/2 * α/avg(h) * inner(jump(grad(u)), jump(grad(u))) * dS
dg3 = lambda u: 1/2 * α/h * inner(grad(u), grad(u)) * ds


L = bending + dg1(u) + dg2(u) + dg3(u) - W_ext
F = ufl.derivative(L, u, ufl.TestFunction(V))

solver = SNESProblem(F, u, bcs, monitor = monitor)

solver.snes.solve(None, u.vector)
print(solver.snes.getConvergedReason())

pdb.set_trace()


# Point sources
b = dolfinx.fem.Function(Q)
b.x.array[:] = 0

if mesh.comm.rank == 0:
    point = np.array([[0.68, 0.36, 0]], dtype=mesh.geometry.x.dtype)
    
    points = [np.array([[-0.2, 0.1, 0]], dtype=mesh.geometry.x.dtype),
              np.array([[0.2, -0.1, 0]], dtype=mesh.geometry.x.dtype)]
    signs = [-1, 1]
else:
    point = np.zeros((0, 3), dtype=mesh.geometry.x.dtype)
    points = [np.zeros((0, 3), dtype=mesh.geometry.x.dtype),
              np.zeros((0, 3), dtype=mesh.geometry.x.dtype)]

# cells, basis_values = compute_cell_contribution_point(V, point)
_cells, _basis_values = compute_cell_contributions(Q, points)
Q_0, Q_0_to_Q_dofs = Q.sub(0).collapse()
# https://fenicsproject.discourse.group/t/meaning-of-collapse/10641/2
_cells, _basis_values = compute_cell_contributions(Q_0, points)
_cells, _basis_values = compute_cell_contributions(Q.sub(0), points)

for cell, basis_value, sign in zip(_cells, _basis_values, signs):
    subspace_dofs = Q_0.dofmap.cell_dofs(cell)
    # dofs = Q_0_to_Q_dofs[subspace_dofs]
    dofs = np.array(Q_0_to_Q_dofs)[subspace_dofs]
    b.x.array[dofs] += sign * basis_value

solver = SNESProblem(F, q, bcs, monitor = monitor)


dolfinx.fem.petsc.apply_lifting(b.vector, [solver.J_form], [bcs])
b.vector.ghostUpdate(addv=PETSc.InsertMode.ADD,
                      mode=PETSc.ScatterMode.REVERSE)
dolfinx.fem.petsc.set_bc(b.vector, bcs)
# dolfinx.fem.petsc.set_bc(b.vector, bcs, -1.)
# b.x.scatter_forward()
b.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                      mode=PETSc.ScatterMode.FORWARD)

# now b has to be added into the residual
__import__('pdb').set_trace()

solver.snes.solve(None, q.vector)
print(solver.snes.getConvergedReason())
