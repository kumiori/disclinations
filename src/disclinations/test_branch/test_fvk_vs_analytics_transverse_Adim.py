import json
import logging
import os
import pdb
import sys
from pathlib import Path
from mpi4py import MPI
import warnings
import basix
import numpy as np
import yaml
import pyvista

import petsc4py
from petsc4py import PETSc

import matplotlib.pyplot as plt
import matplotlib.tri as tri

from disclinations.models.adimensional import A_NonlinearPlateFVK
from disclinations.meshes import mesh_bounding_box
from disclinations.meshes.primitives import mesh_circle_gmshapi
from disclinations.utils.viz import plot_scalar, plot_profile, plot_mesh
from disclinations.solvers import SNESSolver, SNESProblem

import dolfinx
from dolfinx import log
import dolfinx.io
from dolfinx.io import XDMFFile
from dolfinx.io import XDMFFile, gmshio
from dolfinx.fem.petsc import (assemble_matrix, create_vector, create_matrix, assemble_vector)
from dolfinx.fem import Constant, dirichletbc
from dolfinx.fem import assemble_scalar, form
import dolfinx.mesh
import dolfinx.plot

import ufl
from ufl import div, grad, CellDiameter, FacetNormal, dx

logging.basicConfig(level=logging.INFO)


REQUIRED_VERSION = "0.8.0"

if dolfinx.__version__ != REQUIRED_VERSION:
    warnings.warn(f"We need dolfinx version {REQUIRED_VERSION}, but found version {dolfinx.__version__}. Exiting.")
    sys.exit(1)

petsc4py.init(sys.argv)
log.set_log_level(log.LogLevel.WARNING)

COMM = MPI.COMM_WORLD

AIRY = 0
TRANSVERSE = 1

OUTDIR = os.path.join("output", "analytic_transverse_Adim")

# Create output folder if does not exists
if COMM.rank == 0:
    Path(OUTDIR).mkdir(parents=True, exist_ok=True)

def monitor(snes, it, norm):
    logging.info(f"Iteration {it}, residual {norm}")
    print(f"Iteration {it}, residual {norm}")
    return PETSc.SNES.ConvergedReason.ITERATING

def exact_bending_energy(v, w):
    laplacian = lambda f : ufl.div(grad(f))
    hessian = lambda f : ufl.grad(ufl.grad(f))
    return assemble_scalar( form( (D*nu/2 * ufl.inner(laplacian(w), laplacian(w)) + D*(1-nu)/2 * (ufl.inner(hessian(w), hessian(w))) )* ufl.dx) )

def exact_membrane_energy(v, w):
    laplacian = lambda f : div(grad(f))
    hessian = lambda f : grad(grad(f))
    return assemble_scalar( form( ( ((1+nu)/(2*E*thickness)) * ufl.inner(hessian(v), hessian(v))  - nu/(2*E*thickness) *  ufl.inner(laplacian(v), laplacian(v)) ) * ufl.dx ) )


def exact_coupling_energy(v, w):
    laplacian = lambda f : div(grad(f))
    hessian = lambda f : grad(grad(f))
    cof = lambda v : ufl.Identity(2)*laplacian(v) - hessian(v)
    return assemble_scalar( form( 0.5* ufl.inner(cof(v), ufl.outer(grad(w),grad(w)) ) * ufl.dx ) )

# LOAD PARAMETERS FILE
with open("parameters.yml") as f: parameters = yaml.load(f, Loader=yaml.FullLoader)

# GEOMETRIC PARAMETERS
thickness = parameters["model"]["thickness"]
a = parameters["geometry"]["radius"]/thickness

# ELASTIC PARAMETERS
nu = parameters["model"]["nu"]
E = parameters["model"]["E"]
D = E * thickness**3 / (12 * (1 - nu**2))
_E = E

# LOAD MESH
mesh_size = parameters["geometry"]["mesh_size"]
model_rank = 0
tdim = 2
gmsh_model, tdim = mesh_circle_gmshapi( parameters["geometry"]["geom_type"], 1, mesh_size, tdim)
mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, COMM, model_rank, tdim)
h = CellDiameter(mesh)
n = FacetNormal(mesh)

# DEFINE FUNCTIONAL SPACE
X = basix.ufl.element("P", str(mesh.ufl_cell()), parameters["model"]["order"]) 
Q = dolfinx.fem.functionspace(mesh, basix.ufl.mixed_element([X, X]))

# DEFINE FUNCTIONS
q = dolfinx.fem.Function(Q)
v, w = ufl.split(q)
q_exact = dolfinx.fem.Function(Q)
v_exact, w_exact = q_exact.split()
v_dim = E*(thickness**3)*v
w_dim = thickness*w
f = dolfinx.fem.Function(Q.sub(TRANSVERSE).collapse()[0])
state = {"v": v, "w": w}
state_exact = {"v": v_exact, "w": w_exact}
state_dimensional = {"v": v_dim, "w": w_dim}

# SET BOUDNARY CONDITIONS
mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
bndry_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
dofs_v = dolfinx.fem.locate_dofs_topological(V=Q.sub(AIRY), entity_dim=1, entities=bndry_facets)
dofs_w = dolfinx.fem.locate_dofs_topological(V=Q.sub(TRANSVERSE), entity_dim=1, entities=bndry_facets)
bcs_w = dirichletbc( np.array(0, dtype=PETSc.ScalarType), dofs_w, Q.sub(TRANSVERSE) )
bcs_v = dirichletbc( np.array(0, dtype=PETSc.ScalarType), dofs_v, Q.sub(AIRY) )
_bcs = {AIRY: bcs_v, TRANSVERSE: bcs_w}
bcs = list(_bcs.values())

# DEFINE THE VARIATIONAL PROBLEM
model = A_NonlinearPlateFVK(mesh, parameters["model"])
energy = model.energy(state)[0]

# DEFINE EXACT SOLUTIONS
def v_exact_function(x):
    a1=-1/12
    a2=-1/18
    a3=-1/24
    return D * ( a1 * (1 - x[0]**2 - x[1]**2)**2 + a2 * (1 - x[0]**2 - x[1]**2)**3 + a3 * (1 - x[0]**2 - x[1]**2)**4 )

def w_exact_function(x): return np.sqrt(2*D/(E*thickness)) * (1 - x[0]**2 - x[1]**2)**2

v_exact.interpolate(v_exact_function)
w_exact.interpolate(w_exact_function)
exact_energy = model.energy(state_exact)[0]
ex_bending_energy = exact_bending_energy(v_exact, w_exact)
ex_membrane_energy = exact_membrane_energy(v_exact, w_exact)
ex_coupl_energy = exact_coupling_energy(v_exact, w_exact)

# DEFINE THE FEM PROBLEM AND INSTANTIATE THE SOLVER

# External load
def transverse_load(x):
    f_scale = np.sqrt(2 * D**3 / (E * thickness))
    p = (40/3) * (1 - x[0]**2 - x[1]**2)**4 + (16/3) * (11 + x[0]**2 + x[1]**2)
    return (a**4) * (f_scale/E) * p

f.interpolate(transverse_load)

# External work
W_ext = f * w * dx

# Stabilization and symmetrization terms
penalisation = model.penalisation(state)

# Discrete energy functional
L = energy - W_ext + penalisation

F = ufl.derivative(L, q, ufl.TestFunction(Q))

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
    petsc_options=solver_parameters,
    prefix='plate_fvk',
)

# Solver run
solver.solve()

# COMPUTE DIMENSIONAL ENERGY
energy_components = {"bending": (1/a**2)*model.energy(state_dimensional)[1],
                    "membrane": (1/a**2)*model.energy(state_dimensional)[2],
                    "coupling": (1/a**2)*model.energy(state_dimensional)[3]}

energy_terms = {label: COMM.allreduce( dolfinx.fem.assemble_scalar( dolfinx.fem.form(energy_term)), op=MPI.SUM)
                         for label, energy_term in energy_components.items()}

# Print FE dimensioanal energy
for label, energy_term in energy_terms.items(): print(f"{label}: {energy_term}")

# Pring exact energy
print("Exact bending energy: ", ex_bending_energy)
print("Exact membrane energy: ", ex_membrane_energy)
print("Exact coupling energy: ", ex_coupl_energy)
    


# DEFINE SOLUTIONS FOR POSTPROCESSING

"""
    v and w belong to the class "ufl.indexed.Indexed"
    vpp and wpp belong to the class "dolfinx.fem.function.Function"
    For the computation we need v and w
    For the post-processing we need vpp, wpp
"""
vpp, wpp = q.split()
vpp = E*(thickness**3)*vpp
wpp = thickness*wpp
vpp.name = "Airy"
wpp.name = "deflection"

# PLOTS

# Plot the mesh
plt.figure()
ax = plot_mesh(mesh)
fig = ax.get_figure()
fig.savefig(f"{OUTDIR}/mesh.png")

# Plot the profiles
V_v, dofs_v = Q.sub(0).collapse()
V_w, dofs_w = Q.sub(1).collapse()

pyvista.OFF_SCREEN = True

plotter = pyvista.Plotter(title="Displacement", window_size=[1200, 600], shape=(2, 2) )

scalar_plot = plot_scalar(vpp, plotter, subplot=(0, 0), V_sub=V_v, dofs=dofs_v)
scalar_plot = plot_scalar(wpp, plotter, subplot=(0, 1), V_sub=V_w, dofs=dofs_w)

scalar_plot = plot_scalar(v_exact, plotter, subplot=(1, 0), V_sub=V_v, dofs=dofs_v)
scalar_plot = plot_scalar(w_exact, plotter, subplot=(1, 1), V_sub=V_w, dofs=dofs_w)

scalar_plot.screenshot(f"{OUTDIR}/test_fvk.png")
print("plotted scalar")

tol = 1e-3
xs = np.linspace(0 + tol, parameters["geometry"]["radius"] - tol, 101)
points = np.zeros((3, 101))
points[0] = xs

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

_plt, data = plot_profile( wpp, points, None, subplot=(1, 2), lineproperties={ "c": "k", "label": f"$w(x)$" }, fig=fig, subplotnumber=1 )

ax = _plt.gca()
ax2 = ax.twinx()

_plt, data = plot_profile( w_exact, points, None, subplot=(1, 2), lineproperties={"c": "r", "label": f"$w_e(x)$", "ls": "--" }, fig=fig, ax=ax2, subplotnumber=1 )

_plt, data = plot_profile( vpp, points, None, subplot=(1, 2), lineproperties={ "c": "k", "label": f"$v(x)$" }, fig=fig, subplotnumber=2 )


_plt, data = plot_profile( v_exact, points, None, subplot=(1, 2), lineproperties={ "c": "r", "label": f"$v_e(x)$", "ls": "--"}, fig=fig, subplotnumber=2 )

_plt.legend()

_plt.savefig(f"{OUTDIR}/test_fvk-profiles.png")
