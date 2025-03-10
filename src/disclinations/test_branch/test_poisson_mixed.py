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
parameters["geometry"]["radius"] = 1
parameters["geometry"]["geom_type"] = "circle"

model_rank = 0
tdim = 2
# order = 3

gmsh_model, tdim = mesh_circle_gmshapi(
    parameters["geometry"]["geom_type"], 1, mesh_size, tdim
)
mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, model_rank, tdim)

outdir = "output"
prefix = os.path.join(outdir, "poisson")

V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
X = ufl.FiniteElement("CG", mesh.ufl_cell(), 1)

Q = dolfinx.fem.FunctionSpace(mesh, ufl.MixedElement([X, X]))

mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
bndry_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)

dofs_v = dolfinx.fem.locate_dofs_topological(V=Q.sub(0), entity_dim=1, entities=bndry_facets)
dofs_w = dolfinx.fem.locate_dofs_topological(V=Q.sub(1), entity_dim=1, entities=bndry_facets)

bcs_w = dirichletbc(
    np.array(0.0, dtype=PETSc.ScalarType),
    dofs_v, Q.sub(0)
)
bcs_v = dirichletbc(
    np.array(1.0, dtype=PETSc.ScalarType),
    dofs_w, Q.sub(1)
)

bcs = [bcs_w, bcs_v]

q = dolfinx.fem.Function(Q)
v, w = ufl.split(q)

x = ufl.SpatialCoordinate(mesh)
f = 10 * ufl.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.02)
g = ufl.sin(5 * x[0])
energy = 1./2. * (inner(grad(v), grad(v)) + inner(grad(w), grad(w))) * dx - f * v * dx - g * w * dx

F = ufl.derivative(energy, q, ufl.TestFunction(Q))
solver = SNESProblem(F, q, bcs, monitor = monitor)

solver.snes.solve(None, q.vector)
print(solver.snes.getConvergedReason())

pdb.set_trace()
# ------------------------------

from disclinations.utils.viz import plot_scalar

import pyvista
from pyvista.plotting.utilities import xvfb

xvfb.start_xvfb(wait=0.05)
pyvista.OFF_SCREEN = True

plotter = pyvista.Plotter(
        title="Displacement",
        window_size=[1600, 600],
        shape=(1, 1),
    )

v, w = q.split()

V_v, dofs_v = Q.sub(0).collapse()
V_w, dofs_w = Q.sub(1).collapse()

pdb.set_trace()

scalar_plot = plot_scalar(w, plotter, subplot=(0, 0), V_sub=V_w, dofs=dofs_w)
scalar_plot.screenshot("output/test_poisson-mixed-w.png")

plotter = pyvista.Plotter(
        title="Displacement",
        window_size=[1600, 600],
        shape=(1, 1),
    )

scalar_plot = plot_scalar(v, plotter, subplot=(0, 0), V_sub=V_v, dofs=dofs_v)
scalar_plot.screenshot("output/test_poisson-mixed-v.png")
print("plotted scalar")


# 0---------------------------0

solver = SNESSolver(
    F,
    u,
    bcs,
    bounds=None,
    petsc_options=parameters.get("solvers").get("elasticity").get("snes"),
    prefix='test_snes',
)

res = solver.solve()


scalar_plot = plot_scalar(u, plotter, subplot=(0, 0), lineproperties={"scalars": "u"})
scalar_plot.screenshot("output/test_poisson.png")
print("plotted scalar")
