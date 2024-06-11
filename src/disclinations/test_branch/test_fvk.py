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
if comm.rank == 0:
    Path(outdir).mkdir(parents=True, exist_ok=True)
prefix = os.path.join(outdir, "plate")

h = CellDiameter(mesh)
n = FacetNormal(mesh)

# Function spaces
X = ufl.FiniteElement("CG", mesh.ufl_cell(), parameters["model"]["order"])
# Q_el = 
Q = dolfinx.fem.FunctionSpace(mesh, ufl.MixedElement([X, X]))

# Boundaries
# facets = dolfinx.mesh.locate_entities_boundary(
#     mesh,
#     1,
#     marker=lambda x: np.isclose(
#         x[0] ** 2 + x[1] ** 2, parameters["geometry"]["radius"] ** 2, atol=1e-10
#     ),
# )


# Material parameters

# Graphene-like properties
nu = dolfinx.fem.Constant(mesh, 0.15)
D = dolfinx.fem.Constant(mesh, 1.)
Eh = dolfinx.fem.Constant(mesh, 1.)
α = dolfinx.fem.Constant(mesh, 10.)
k_g = -D*(1-nu)
n = ufl.FacetNormal(mesh)

# dofs = dolfinx.fem.locate_dofs_geometrical(
mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
bndry_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
dofs_0 = dolfinx.fem.locate_dofs_topological(V=Q.sub(0), entity_dim=1, entities=bndry_facets)
dofs_1 = dolfinx.fem.locate_dofs_topological(V=Q.sub(1), entity_dim=1, entities=bndry_facets)

# Define the variational problem

bcs_w = dirichletbc(
    np.array(0, dtype=PETSc.ScalarType),
    dofs_0, Q.sub(0)
)
bcs_v = dirichletbc(
    np.array(0, dtype=PETSc.ScalarType),
    dofs_1, Q.sub(1)
)

bcs = [bcs_w, bcs_v]



ds = ufl.Measure("ds")(subdomain_data=mts)
dx = ufl.Measure("dx")
dS = ufl.Measure("dS")
h = ufl.CellDiameter(mesh)

v, w = dolfinx.fem.Function(Q).split()
v.name = "airy"
w.name = "Deflection"
J = as_matrix([[0, -1], [1, 0]])
# v = dolfinx.fem.Function(Q.sub(1).collapse(), name="sigma")
# w = dolfinx.fem.Function(Q.sub(0).collapse(), name="Deflection")

state = {"v": v, "w": w}



# Define the variational problem

σ = lambda v: div(grad(v)) * ufl.Identity(2) - grad(grad(v))

# Stress-displacement operator (membrane)
N = lambda v: σ(v)
bracket = lambda f, g:  inner(grad(grad(f)), J.T * grad(grad(g)) * J)

# bracket(w, w) = det Hess(w)
# Moment-displacement operator
M = lambda w: grad(grad(w)) + nu*σ(w)

# Define the new functional. See the reference article for details.
# bending = 1/2 * inner(M(w), grad(grad(w))) * dx 
bending = (D/2 * (inner(div(grad(w)), div(grad(w)))) + k_g * bracket(w, w)) * dx 
membrane = (-1/(2*Eh) * inner(grad(grad(v)), grad(grad(v))) + nu/(2*Eh) * bracket(v, v)) *dx 
# membrane = 1/2 * inner(Ph(ph_), grad(grad(ph_)))*dx 
coupling = 1/2 * inner(σ(v), outer(grad(w), grad(w)))*dx # compatibility coupling term
energy = bending + membrane + coupling

# inner discontinuity penalisations
dg1 = lambda u: dot(jump(grad(u)), avg(grad(grad(u)) * n)) * dS
dg2 = lambda u: 1/2 * α/avg(h) * inner(jump(grad(u)), jump(grad(u))) * dS
pdb.set_trace()

# exterior boundary penalisations
bc1 = lambda u: inner(grad(u), grad(grad(u)) * n) * ds
bc2 = lambda u: 1/2 * α/h * inner(grad(u) * n, grad(u) * n) * ds