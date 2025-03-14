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

from disclinations.models import FVKAdimensional
from disclinations.utils.la import compute_disclination_loads
from disclinations.meshes.primitives import mesh_circle_gmshapi
from disclinations.utils.viz import plot_scalar, plot_profile, plot_mesh

# from damage.utils import ColorPrint
from disclinations.solvers import SNESSolver, SNESProblem
from dolfinx import log
from dolfinx.io import XDMFFile
from mpi4py import MPI
from petsc4py import PETSc

logging.basicConfig(level=logging.INFO)

import sys
import warnings

required_version = "0.8.0"

if dolfinx.__version__ != required_version:
    warnings.warn(f"We need dolfinx version {required_version}, but found version {dolfinx.__version__}. Exiting.")
    sys.exit(1)

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
    dx,
)

petsc4py.init(sys.argv)
log.set_log_level(log.LogLevel.WARNING)

comm = MPI.COMM_WORLD
AIRY = 0
TRANSVERSE = 1

def monitor(snes, it, norm):
    logging.info(f"Iteration {it}, residual {norm}")
    print(f"Iteration {it}, residual {norm}")
    return PETSc.SNES.ConvergedReason.ITERATING

import matplotlib.tri as tri

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
prefix = os.path.join(outdir, "plate_fvk_adim")


if comm.rank == 0:
    Path(prefix).mkdir(parents=True, exist_ok=True)

h = CellDiameter(mesh)
n = FacetNormal(mesh)

X = basix.ufl.element("P", str(mesh.ufl_cell()), parameters["model"]["order"]) 
Q = dolfinx.fem.functionspace(mesh, basix.ufl.mixed_element([X, X]))

mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
bndry_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
dofs_v = dolfinx.fem.locate_dofs_topological(V=Q.sub(AIRY), entity_dim=1, entities=bndry_facets)
dofs_w = dolfinx.fem.locate_dofs_topological(V=Q.sub(TRANSVERSE), entity_dim=1, entities=bndry_facets)

# Define the variational problem

bcs_w = dirichletbc(
    np.array(0, dtype=PETSc.ScalarType),
    dofs_w, Q.sub(TRANSVERSE)
)
bcs_v = dirichletbc(
    np.array(0, dtype=PETSc.ScalarType),
    dofs_v, Q.sub(AIRY)
)

# keep track of the ordering of fields in boundary conditions
_bcs = {AIRY: bcs_v, TRANSVERSE: bcs_w}
bcs = list(_bcs.values())

q = dolfinx.fem.Function(Q)
v, w = ufl.split(q)

state = {"v": v, "w": w}

# Define the variational problem

model = FVKAdimensional(mesh, nu = parameters["model"]["nu"])
energy = model.energy(state)[0]

_nu = parameters["model"]["nu"]
_h = parameters["model"]["thickness"]
_E = parameters["model"]["E"]

w_scale = _h * (6*(1-_nu**2)) ** 1./2.
v_scale = _E * _h**3 / (12 * (1 - _nu**2))
p_scale = 12.*np.sqrt(6) * (1 - _nu**2)**3./2. / (_E * _h**4)
p_scale = 40/3

# print numerical values of scalings
print(f"v_scale: {v_scale}")
print(f"w_scale: {w_scale}")    
print(f"p_scale: {p_scale}")

# Dead load (transverse)
# W_ext = Constant(mesh, np.array(-1.0, dtype=PETSc.ScalarType)) * w * dx

# Explicit dead load (transverse)
# Cf. thesis ref: 
f = dolfinx.fem.Function(Q.sub(TRANSVERSE).collapse()[0])

# def transverse_load(x):
#     _rsq = x[0]**2 + x[1]**2
#     _f = (40/3) * (1 - _rsq)**4 + (16/3) * (11 + _rsq)
#     return _f / p_scale

# f.interpolate(transverse_load)

f0 = parameters["model"]["E"] * (parameters["model"]["thickness"] / parameters["geometry"]["radius"])**4 
b0 = (parameters["geometry"]["radius"] / parameters["model"]["thickness"])**2

f = Constant(mesh, np.array(1.0, dtype=PETSc.ScalarType))

# Point sources
if mesh.comm.rank == 0:
    # point = np.array([[0.68, 0.36, 0]], dtype=mesh.geometry.x.dtype)
    points = [np.array([[-0.2, 0.0, 0]], dtype=mesh.geometry.x.dtype),
              np.array([[0.2, -0.0, 0]], dtype=mesh.geometry.x.dtype)]
    signs = [-1, 1]
else:
    # point = np.zeros((0, 3), dtype=mesh.geometry.x.dtype)
    points = [np.zeros((0, 3), dtype=mesh.geometry.x.dtype),
              np.zeros((0, 3), dtype=mesh.geometry.x.dtype)]
    
Q_v, Q_v_to_Q_dofs = Q.sub(AIRY).collapse()

b = compute_disclination_loads(points, signs, Q, V_sub_to_V_dofs=Q_v_to_Q_dofs, V_sub=Q_v)

W_transv_coeff = (parameters["geometry"]["radius"] / parameters["model"]["thickness"])**4 / parameters["model"]["E"] 
W_ext = W_transv_coeff * f0 * f * w * dx

penalisation = model.penalisation(state)

# Define the functional
L = energy - W_ext + penalisation

F = ufl.derivative(L, q, ufl.TestFunction(Q))
F_v = ufl.derivative(L, v, ufl.TestFunction(Q.sub(AIRY)))
F_w = ufl.derivative(L, w, ufl.TestFunction(Q.sub(TRANSVERSE)))

Ec_w = ufl.derivative(model.energy(state)[3], w, ufl.TestFunction(Q.sub(TRANSVERSE)))
Ec_v = ufl.derivative(model.energy(state)[3], v, ufl.TestFunction(Q.sub(AIRY)))
Em_v = ufl.derivative(model.energy(state)[2], w, ufl.TestFunction(Q.sub(AIRY)))

labels = ["F", "F_v", "F_w", "Ec_w","Ec_v", "Em_v"]


print(parameters.get("solvers").get("elasticity").get("snes"))

solver_parameters = {
    "snes_type": "newtonls",      # Solver type: NGMRES (Nonlinear GMRES)
    "snes_max_it": 100,           # Maximum number of iterations
    "snes_rtol": 1e-6,            # Relative tolerance for convergence
    "snes_atol": 1e-15,           # Absolute tolerance for convergence
    "snes_stol": 1e-5,           # Tolerance for the change in solution norm
    "snes_monitor": None,         # Function for monitoring convergence (optional)
    "snes_linesearch_type": "basic",  # Type of line search
}

solver = SNESSolver(
    F_form=F,
    u=q,
    bcs=bcs,
    bounds=None,
    petsc_options=solver_parameters,
    prefix='plate_fvk',
    b0=b.vector,
)
solver.solve()

v, w = q.split()
v.name = "Airy"
w.name = "deflection"

V_v, dofs_v = Q.sub(0).collapse()
V_w, dofs_w = Q.sub(1).collapse()

energy_components = {"bending": model.energy(state)[1],
                    "membrane": model.energy(state)[2],
                    "coupling": model.energy(state)[3]}

computed_energy_terms = {label: comm.allreduce(
    dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(energy_term)),
    op=MPI.SUM,
) for label, energy_term in energy_components.items()}

for label, energy_term in computed_energy_terms.items():
    print(f"{label}: {energy_term}")

# compute maximum of w and v
max_w = max(w.vector.array)
print(f"Max w: {max_w}")

print("")
# do the same for v
max_v = max(v.vector.array)
print(f"Max v: {max_v}")

pdb.set_trace()

for label, form in zip(labels, [F, F_v, F_w, Ec_w, Em_v]):
    _F = create_vector(dolfinx.fem.form(form))
    assemble_vector(_F, dolfinx.fem.form(form))
    print(f"Norm of {label}: {_F.norm()}")
    
    
import matplotlib.pyplot as plt

plt.figure()
ax = plot_mesh(mesh)
fig = ax.get_figure()
fig.savefig(f"{prefix}/mesh.png")

# ------------------------------

import pyvista
from pyvista.plotting.utilities import xvfb

xvfb.start_xvfb(wait=0.05)
pyvista.OFF_SCREEN = True

plotter = pyvista.Plotter(
        title="Displacement",
        window_size=[1200, 600],
        shape=(2, 2),
    )

scalar_plot = plot_scalar(v, plotter, subplot=(0, 0), V_sub=V_v, dofs=dofs_v)
scalar_plot = plot_scalar(w, plotter, subplot=(0, 1), V_sub=V_w, dofs=dofs_w)

scalar_plot.screenshot(f"{prefix}/test_fvk.png")
print("plotted scalar")

tol = 1e-3
xs = np.linspace(-parameters["geometry"]["radius"] + tol, parameters["geometry"]["radius"] - tol, 101)
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

ax = _plt.gca()
axw = ax.twinx()

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
