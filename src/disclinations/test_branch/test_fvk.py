# Solving a Foppl-von-Karman plate problem with a penalised formulation
#
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
from disclinations.solvers import SNESSolver as PlateSolver
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
from ufl import (
    CellDiameter,
    FacetNormal,
    TestFunction,
    TrialFunction,
    avg,
    ds,
    dS,
    dx,
    grad,
    inner,
    jump,
    outer,
    sym,
)

petsc4py.init(sys.argv)
log.set_log_level(log.LogLevel.WARNING)

comm = MPI.COMM_WORLD

with open("parameters.yml") as f:
    parameters = yaml.load(f, Loader=yaml.FullLoader)

mesh_size = parameters["geometry"]["mesh_size"]
parameters["geometry"]["radius"] = 1
parameters["geometry"]["geom_type"] = "circle"


model_rank = 0
tdim = 2
# order = 3

gmsh_model, tdim = mesh_circle_gmshapi(parameters["geometry"]["geom_type"], 1, mesh_size, tdim)
mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, model_rank, tdim)

outdir = "output"
if comm.rank == 0:
    Path(outdir).mkdir(parents=True, exist_ok=True)
prefix = os.path.join(outdir, "plate")

h = CellDiameter(mesh)
n = FacetNormal(mesh)

# Boundaries
facets = dolfinx.mesh.locate_entities_boundary(
    mesh,
    1,
    marker=lambda x: np.isclose(
        x[0] ** 2 + x[1] ** 2, parameters["geometry"]["radius"] ** 2, atol=1e-10
    ),
)
pdb.set_trace()


# Function spaces
X = ufl.FiniteElement("CG", mesh.ufl_cell(), parameters["model"]["order"])
Q = ufl.FunctionSpace(mesh, X * X)

# Define the variational problem
