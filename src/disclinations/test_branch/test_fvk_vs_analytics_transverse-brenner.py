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
from disclinations.models import NonlinearPlateFVK, NonlinearPlateFVK_brenner
from disclinations.meshes import mesh_bounding_box
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

model_rank = 0
tdim = 2
# order = 3

gmsh_model, tdim = mesh_circle_gmshapi(
    parameters["geometry"]["geom_type"], 1, mesh_size, tdim
)
mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, model_rank, tdim)

outdir = "output"
prefix = os.path.join(outdir, "plate_fvk_analytic_transverse-brenner")

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

q_exact = dolfinx.fem.Function(Q)
v_exact, w_exact = q_exact.split()

f = dolfinx.fem.Function(Q.sub(TRANSVERSE).collapse()[0])

state = {"v": v, "w": w}

# Define the variational problem

model = NonlinearPlateFVK_brenner(mesh, parameters["model"])
energy = model.energy(state)[0]

# Dead load (transverse)
# W_ext = Constant(mesh, np.array(-1.0, dtype=PETSc.ScalarType)) * w * dx

# Explicit dead load (transverse)
# Cf. thesis ref: 
# f = dolfinx.fem.Function(Q.sub(TRANSVERSE).collapse()[0])

nu = parameters["model"]["nu"]
# _h = parameters["model"]["thickness"]
thickness = parameters["model"]["thickness"]
_E = parameters["model"]["E"]
_D = _E * thickness**3 / (12 * (1 - nu**2))

# w_scale = _h * (6*(1-_nu**2)) ** 1./2.
w_scale = np.sqrt(2*_D/(_E*thickness))
v_scale = _D
# p_scale = 12.*np.sqrt(6) * (1 - _nu**2)**3./2. / (_E * _h**4)
f_scale = np.sqrt(2 * _D**3 / (_E * thickness))

def _v_initial_guess(x):
    return np.cos(np.pi * np.sqrt(x[0]**2 + x[1]**2))

def _v_exact(x):
    v_scale = _D
    a1=-1/12
    a2=-1/18
    a3=-1/24
    _v = a1 * (1 - x[0]**2 - x[1]**2)**2 + a2 * (1 - x[0]**2 - x[1]**2)**3 + a3 * (1 - x[0]**2 - x[1]**2)**4
    _v = _v * v_scale
    return _v

def _w_exact(x):
    # return q/(64*12*(1-nu**2))*(1 - (x[0]**2 + x[1]**2))**2
    return w_scale * (1 - x[0]**2 - x[1]**2)**2

def transverse_load(x):
    _p = (40/3) * (1 - x[0]**2 - x[1]**2)**4 + (16/3) * (11 + x[0]**2 + x[1]**2)
    return f_scale * _p

f.interpolate(transverse_load)

v_exact.interpolate(_v_exact)
w_exact.interpolate(_w_exact)

state_exact = {"v": v_exact, "w": w_exact}
exact_energy = model.energy(state_exact)[0]

W_ext = f * w * dx
model.W_ext = W_ext


# W_ext = w * dx

penalisation = model.penalisation(state)

# Define the functional
L = energy - W_ext + penalisation

# test_v, test_w = ufl.split(ufl.TestFunctions(Q))
test_v, test_w = ufl.TestFunctions(Q)[AIRY], ufl.TestFunctions(Q)[TRANSVERSE]
F = ufl.derivative(L, q, ufl.TestFunction(Q)) + model.coupling_term(state, test_v, test_w)

# F_v = ufl.derivative(L, v, ufl.TestFunction(Q.sub(AIRY)))
# F_w = ufl.derivative(L, w, ufl.TestFunction(Q.sub(TRANSVERSE)))

# Ec_w = ufl.derivative(model.energy(state)[3], w, ufl.TestFunction(Q.sub(TRANSVERSE)))
# Em_v = ufl.derivative(model.energy(state)[2], w, ufl.TestFunction(Q.sub(AIRY)))

# labels = ["F", "F_v", "F_w", "Ec_w", "Em_v"]

# print(parameters.get("solvers").get("elasticity").get("snes"))

solver_parameters = {
    "snes_type": "newtonls",        # Solver type: NGMRES (Nonlinear GMRES)
    "snes_max_it": 100,          # Maximum number of iterations
    "snes_rtol": 1e-6,            # Relative tolerance for convergence
    "snes_atol": 1e-6,           # Absolute tolerance for convergence
    "snes_stol": 1e-6,           # Tolerance for the change in solution norm
    "snes_monitor": None,         # Function for monitoring convergence (optional)
    "snes_linesearch_type": "basic",  # Type of line search
}

solver = SNESSolver(
    F_form=F,
    u=q,
    bcs=bcs,
    bounds=None,
    # petsc_options=parameters.get("solvers").get("elasticity").get("snes"),
    petsc_options=solver_parameters,
    prefix='plate_fvk',
)
solver.solve()


# Solver runs

# for label, form in zip(labels, [F, F_v, F_w, Ec_w, Em_v]):
#     _F = create_vector(dolfinx.fem.form(form))
#     assemble_vector(_F, dolfinx.fem.form(form))
#     print(f"Norm of {label}: {_F.norm()}")
    

v, w = q.split()
v.name = "Airy"
w.name = "deflection"

V_v, dofs_v = Q.sub(0).collapse()
V_w, dofs_w = Q.sub(1).collapse()

D = _D
E = _E

from dolfinx.fem import assemble_scalar, form

from ufl import div, grad

def exact_bending_energy(v, w):
    laplacian = lambda f : ufl.div(grad(f))
    hessian = lambda f : ufl.grad(ufl.grad(f))

    #assemble_scalar(form( ufl.dot(ufl.grad(v_exact), ufl.grad(v_exact)) * ufl.dx ))
    return assemble_scalar( form( (D*nu/2 * ufl.inner(laplacian(w), laplacian(w)) + D*(1-nu)/2 * (ufl.inner(hessian(w), hessian(w))) )* ufl.dx) )

def exact_membrane_energy(v, w):
    laplacian = lambda f : div(grad(f))
    hessian = lambda f : grad(grad(f))
    return assemble_scalar( form( ( ((1+nu)/(2*E*h)) * ufl.inner(hessian(v), hessian(v)) - nu/(2*E*h) * ufl.inner(laplacian(v), laplacian(v)) ) * ufl.dx ) )


def exact_coupling_energy(v, w):
    laplacian = lambda f : div(grad(f))
    hessian = lambda f : grad(grad(f))
    cof = lambda v : ufl.Identity(2)*laplacian(v) - hessian(v)
    return assemble_scalar( form( 0.5* ufl.inner(cof(v), ufl.outer(grad(w),grad(w)) ) * ufl.dx ) )


for energy_type, energy_function in zip(["bending", "membrane", "coupling"], [exact_bending_energy, exact_membrane_energy, exact_coupling_energy]):
    print(f"Exact {energy_type} energy: {energy_function(v_exact, w_exact)}")
    

# _Fv = create_vector(dolfinx.fem.form(F_v))
# assemble_vector(_Fv, dolfinx.fem.form(F_v))

from disclinations.models import compute_energy_terms

energy_components = {
    "bending": model.energy(state)[1],
    "membrane": model.energy(state)[2],
    "coupling": model.energy(state)[3],
    # "external_work": -model.W_ext,
}
energy_terms = compute_energy_terms(energy_components, mesh.comm)
print(energy_terms)
#pdb.set_trace()

# __import__('pdb').set_trace()

# computed_energy_terms = {label: comm.allreduce(
#     dolfinx.fem.assemble_scalar(
#         dolfinx.fem.form(energy_term)),
#     op=MPI.SUM,
# ) for label, energy_term in energy_components.items()}

# for label, energy_term in computed_energy_terms.items():
#     print(f"{label}: {energy_term}")

import matplotlib.pyplot as plt

plt.figure()
ax = plot_mesh(mesh)
fig = ax.get_figure()
fig.savefig(f"{prefix}/mesh.png")

# ------------------------------

import pyvista
# from pyvista.plotting.utilities import xvfb

# xvfb.start_xvfb(wait=0.05)
pyvista.OFF_SCREEN = True

plotter = pyvista.Plotter(
        title="Displacement",
        window_size=[1200, 600],
        shape=(2, 2),
    )


scalar_plot = plot_scalar(v, plotter, subplot=(0, 0), V_sub=V_v, dofs=dofs_v)
scalar_plot = plot_scalar(w, plotter, subplot=(0, 1), V_sub=V_w, dofs=dofs_w)

scalar_plot = plot_scalar(v_exact, plotter, subplot=(1, 0), V_sub=V_v, dofs=dofs_v)
scalar_plot = plot_scalar(w_exact, plotter, subplot=(1, 1), V_sub=V_w, dofs=dofs_w)

scalar_plot.screenshot(f"{prefix}/test_fvk.png")
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

ax = _plt.gca()
ax2 = ax.twinx()

_plt, data = plot_profile(
    w_exact,
    points,
    None,
    subplot=(1, 2),
    lineproperties={
        "c": "r",
        "label": f"$w_e(x)$",
        "ls": "--"
    },
    fig=fig,
    ax=ax2,
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


_plt, data = plot_profile(
    v_exact,
    points,
    None,
    subplot=(1, 2),
    lineproperties={
        "c": "r",
        "label": f"$v_e(x)$",
        "ls": "--"
    },
    fig=fig,
    subplotnumber=2
)

_plt.legend()

_plt.savefig(f"{prefix}/test_fvk-profiles.png")


# 0---------------------------0
