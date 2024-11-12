# Solving a Foppl-von-Karman plate problem with a penalised formulation
# H^0_2 formulation
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

import basix
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

import matplotlib.tri as tri

def plot_mesh(mesh, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.set_aspect("equal")
    points = mesh.geometry.x
    cells = mesh.geometry.dofmap.reshape((-1, mesh.topology.dim + 1))
    tria = tri.Triangulation(points[:, 0], points[:, 1], cells)
    ax.triplot(tria, color="k")
    return ax

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
prefix = os.path.join(outdir, "plate_fvk")

if comm.rank == 0:
    Path(prefix).mkdir(parents=True, exist_ok=True)

h = CellDiameter(mesh)
n = FacetNormal(mesh)

# Function spaces
# X = ufl.FiniteElement("CG", mesh.ufl_cell(), parameters["model"]["order"])
# Q_el = 

X = basix.ufl.element("P", str(mesh.ufl_cell()), parameters["model"]["order"]) 
Q = dolfinx.fem.functionspace(mesh, basix.ufl.mixed_element([X, X]))

# Material parameters
# Graphene-like properties
nu = dolfinx.fem.Constant(mesh, parameters["model"]["nu"])
D = dolfinx.fem.Constant(mesh, parameters["model"]["D"])
Eh = dolfinx.fem.Constant(mesh, parameters["model"]["E"])
α = dolfinx.fem.Constant(mesh, parameters["model"]["alpha_penalty"])
k_g = -D*(1-nu)
n = ufl.FacetNormal(mesh)

AIRY = 0
TRANSVERSE = 1

mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
bndry_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
dofs_v = dolfinx.fem.locate_dofs_topological(V=Q.sub(AIRY), entity_dim=1, entities=bndry_facets)
dofs_w = dolfinx.fem.locate_dofs_topological(V=Q.sub(TRANSVERSE), entity_dim=1, entities=bndry_facets)

# Define the variational problem

bcs_w = dirichletbc(
    np.array(0, dtype=PETSc.ScalarType),
    dofs_v, Q.sub(TRANSVERSE)
)
bcs_v = dirichletbc(
    np.array(0, dtype=PETSc.ScalarType),
    dofs_w, Q.sub(AIRY)
)
# keep track of the ordering of fields in boundary conditions

_bcs = {AIRY: bcs_v, TRANSVERSE: bcs_w}
bcs = list(_bcs.values())


ds = ufl.Measure("ds")
dx = ufl.Measure("dx")
dS = ufl.Measure("dS")
h = ufl.CellDiameter(mesh)

q = dolfinx.fem.Function(Q)
v, w = ufl.split(q)

J = as_matrix([[0, -1], [1, 0]])
# v = dolfinx.fem.Function(Q.sub(1).collapse(), name="sigma")
# w = dolfinx.fem.Function(Q.sub(0).collapse(), name="Deflection")

state = {"v": v, "w": w}

# Define the variational problem

# Stress-displacement operator (membrane)
# N = lambda v: σ(v)
c_nu = 1
hessian   = lambda f : grad(grad(f))
laplacian = lambda f : div(grad(f))
σ         = lambda f : laplacian(f) * ufl.Identity(2) - hessian(f)
M         = lambda f : hessian(f) + nu*σ(f)
Ph        = lambda f : (1.0/c_nu)*hessian(f) - nu*σ(f)
W         = lambda f : -0.5*J.T*(outer(grad(f), grad(f))) * J
bracket   = lambda f, g:  inner(grad(grad(f)), J.T * grad(grad(g)) * J)
# Moment-displacement operator

# Define the new functional. See the reference article for details.
bending = (D/2 * (inner(laplacian(w), laplacian(w))) + k_g * bracket(w, w)) * dx
membrane = (-1/(2*Eh) * inner(hessian(v), hessian(v)) + nu/(2*Eh) * bracket(v, v)) * dx
# membrane = 1/2 * inner(Ph(ph_), grad(grad(ph_)))*dx
coupling = 1/2 * inner(σ(v), outer(grad(w), grad(w))) * dx # compatibility coupling term
energy = bending + membrane + coupling

# inner discontinuity penalisations
dg1_w = lambda u: -1/2 * jump(grad(u), n)*avg(inner(M(w), outer(n, n))) * dS
dg1_v = lambda u: -1/2 * jump(grad(u), n)*avg(inner(Ph(w), outer(n, n))) * dS
dg2   = lambda u: 1/2 * α/avg(h) * inner(jump(grad(u)), jump(grad(u))) * dS
dgc   = lambda f, g: avg(inner(W(f), outer(n, n)))*jump(grad(g), n)*dS

# exterior boundary penalisations
#bc1 = lambda u: 1/2 * inner(grad(u), grad(grad(u)) * n) * ds
bc1_w = lambda u: -1/2 * inner(grad(v), n)*inner(M(w), outer(n, n)) * ds
bc1_v = lambda u: -1/2 * inner(grad(v), n)*inner(Ph(w), outer(n, n)) * ds
bc2   = lambda u: 1/2 * α/h * inner(dot(grad(u), n), dot(grad(u), n)) * ds

# Dead load (transverse)
W_ext = Constant(mesh, np.array(-1.0, dtype=PETSc.ScalarType)) * w * dx

# Define the functional
L = energy + dg1_w(w) + dg2(w) \
           + dg1_v(v) + dg2(v) \
           + bc1_w(w) + bc2(w) \
           + bc1_v(v) + bc2(v) \
    - W_ext
           
F = ufl.derivative(L, q, ufl.TestFunction(Q))
J = ufl.derivative(F, q, ufl.TrialFunction(Q))

# --------------------------------
solver = SNESProblem(F, q, bcs, monitor = monitor)

solver.snes.solve(None, q.vector)
# print(solver.snes.getConvergedReason())
# ---------------------------------

solver = SNESSolver(
    F_form=F,
    J_form=J,
    u=q,
    bcs=bcs,
    bounds=None,
    petsc_options=parameters.get("solvers").get("elasticity").get("snes"),
    prefix='plate_fvk',
)
solver.solve()

import matplotlib.pyplot as plt

plt.figure()
ax = plot_mesh(mesh)
fig = ax.get_figure()
fig.savefig(f"{prefix}/mesh.png")


elastic_energy = comm.allreduce(
    dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(energy)),
    op=MPI.SUM,
)


# ------------------------------

from disclinations.utils.viz import plot_scalar, plot_profile

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
v.name = "Airy"
w.name = "deflection"

V_v, dofs_v = Q.sub(0).collapse()
V_w, dofs_w = Q.sub(1).collapse()


scalar_plot = plot_scalar(w, plotter, subplot=(0, 0), V_sub=V_w, dofs=dofs_w)
scalar_plot.screenshot(f"{prefix}/test_fvk-w.png")

plotter = pyvista.Plotter(
        title="Displacement",
        window_size=[1600, 600],
        shape=(1, 1),
    )

scalar_plot = plot_scalar(v, plotter, subplot=(0, 0), V_sub=V_v, dofs=dofs_v)
scalar_plot.screenshot(f"{prefix}/test_fvk-v.png")
print("plotted scalar")



tol = 1e-3
xs = np.linspace(0 + tol, parameters["geometry"]["radius"] - tol, 101)
points = np.zeros((3, 101))
points[0] = xs

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

_plt, data = plot_profile(
    w,
    points,
    None,
    subplot=(1, 2),
    lineproperties={
        "c": "k",
        "label": f"$w(x)$"
    },
    fig=fig,
    subplotnumber=1
)

_plt, data = plot_profile(
    v,
    points,
    None,
    subplot=(1, 2),
    lineproperties={
        "c": "k",
        "label": f"$v(x)$"
    },
    fig=fig,
    subplotnumber=2
)

_plt.legend()

_plt.savefig(f"{prefix}/test_fvk-profiles.png")


# 0---------------------------0
