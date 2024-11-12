# Solving a simple Bilaplacian problem
# H^2_0 formulation
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
from disclinations.solvers import SNESSolver
from disclinations.models import Biharmonic
from dolfinx import log
from dolfinx.io import XDMFFile
from mpi4py import MPI
from petsc4py import PETSc

logging.basicConfig(level=logging.INFO)

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


gmsh_model, tdim = mesh_circle_gmshapi(
    parameters["geometry"]["geom_type"], 1, mesh_size, tdim
)
mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, model_rank, tdim)

outdir = "output"
prefix = os.path.join(outdir, "biharmonic")

if comm.rank == 0:
    Path(prefix).mkdir(parents=True, exist_ok=True)

V = dolfinx.fem.functionspace(mesh, ("Lagrange", parameters["model"]["order"]))

mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
bndry_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)

dofs = dolfinx.fem.locate_dofs_topological(V=V, entity_dim=1, entities=bndry_facets)

bcs = [dolfinx.fem.dirichletbc(value=np.array(0, dtype=PETSc.ScalarType), dofs=dofs, V=V)]

u = dolfinx.fem.Function(V)
state = {"u": u}

model = Biharmonic(mesh, parameters["model"])
W_ext = u * ufl.dx
L = model.energy(state) + model.penalisation(state) - W_ext

F = ufl.derivative(L, u, ufl.TestFunction(V))

solver = SNESSolver(
    F_form=F,
    u=u,
    bcs=bcs,
    bounds=None,
    petsc_options=parameters.get("solvers").get("elasticity").get("snes"),
    prefix='biharmonic',
)

solver.solve()

from disclinations.utils.viz import plot_scalar, plot_profile
from dolfinx import plot

import pyvista
from pyvista.plotting.utilities import xvfb
xvfb.start_xvfb(wait=0.05)
pyvista.OFF_SCREEN = True

plotter = pyvista.Plotter(
        title="Displacement",
        window_size=[1600, 600],
        shape=(1, 2),
    )

scalar_plot = plot_scalar(u, plotter, subplot=(0, 0), lineproperties={"show_edges": True})

plotter.subplot(0, 1)
cells, types, x = plot.vtk_mesh(V)
grid = pyvista.UnstructuredGrid(cells, types, x)
grid.point_data["u"] = u.x.array.real
grid.set_active_scalars("u")
# plotter = pyvista.Plotter()
# plotter.add_mesh(grid, show_edges=True)
warped = grid.warp_by_scalar("u", scale_factor=100)
plotter.add_mesh(warped, show_edges=False)

# plotter.show_edges=True
scalar_plot.screenshot(f"{prefix}/test_bilaplacian.png")
print("plotted scalar")

