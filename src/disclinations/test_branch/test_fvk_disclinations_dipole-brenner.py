# Solving a Foppl-von-Karman plate problem with a penalised formulation
# H^0_2 formulation with disclinations
#

import json
import logging
import os
import pdb
from pathlib import Path
from mpi4py import MPI
from petsc4py import PETSc
import sys
import warnings
import basix
import yaml
import numpy as np
import petsc4py
import yaml
from mpi4py import MPI
import pyvista
from pyvista.plotting.utilities import xvfb

import matplotlib.tri as tri
import matplotlib.pyplot as plt

import ufl
from ufl import (
    CellDiameter,
    FacetNormal,
    dx,
)

from disclinations.models import NonlinearPlateFVK_brenner
from disclinations.meshes import mesh_bounding_box
from disclinations.meshes.primitives import mesh_circle_gmshapi
from disclinations.utils.la import compute_cell_contributions, compute_disclination_loads
from disclinations.utils.viz import plot_scalar, plot_profile, plot_mesh
from disclinations.solvers import SNESSolver, SNESProblem

import dolfinx
import dolfinx.io
from dolfinx.io import XDMFFile, gmshio
from dolfinx import log
import dolfinx.mesh
import dolfinx.plot
from dolfinx.fem.petsc import (assemble_matrix, create_vector, create_matrix, assemble_vector)
from dolfinx.fem import Constant, dirichletbc
from dolfinx import plot

logging.basicConfig(level=logging.INFO)


required_version = "0.8.0"

if dolfinx.__version__ != required_version:
    warnings.warn(f"We need dolfinx version {required_version}, but found version {dolfinx.__version__}. Exiting.")
    sys.exit(1)


petsc4py.init(sys.argv)
log.set_log_level(log.LogLevel.WARNING)

AIRY = 0
TRANSVERSE = 1
PREFIX = os.path.join("output", "disclinations_dipole_brenner")
COMM = MPI.COMM_WORLD

# DISCLINATION DISTRIBUTION
DISCLINATION_POINTS_LIST = [[-0.2, 0.0, 0], [0.2, 0.0, 0]]
DISCLINATION_POWER_LIST = [-1, 1]

# Make output directory if does not exist
if COMM.rank == 0:
    Path(PREFIX).mkdir(parents=True, exist_ok=True)

def monitor(snes, it, norm):
    logging.info(f"Iteration {it}, residual {norm}")
    print(f"Iteration {it}, residual {norm}")
    return PETSc.SNES.ConvergedReason.ITERATING


# READ PARAMETERS FILE
with open("parameters.yml") as f:
    parameters = yaml.load(f, Loader=yaml.FullLoader)

# MATERIAL / GEOMETRIC PARAMETERS
nu = parameters["model"]["nu"]
thickness = parameters["model"]["thickness"]
_E = parameters["model"]["E"]
_D = _E * thickness**3 / (12 * (1 - nu**2))

# LOAD MESH
mesh_size = parameters["geometry"]["mesh_size"]
parameters["geometry"]["radius"] = 1
parameters["geometry"]["geom_type"] = "circle"
model_rank = 0
tdim = 2
gmsh_model, tdim = mesh_circle_gmshapi(
    parameters["geometry"]["geom_type"], 1, mesh_size, tdim
)
mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, COMM, model_rank, tdim)
h = CellDiameter(mesh)
n = FacetNormal(mesh)

# DEFINE FUNCTION SPACES
X = basix.ufl.element("P", str(mesh.ufl_cell()), parameters["model"]["order"])
Q = dolfinx.fem.functionspace(mesh, basix.ufl.mixed_element([X, X]))


# DEFINE DIRICHLET HOMOGENEOUS BOUNDARY CONDITIONS
mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
bndry_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
bndry_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
dofs_v = dolfinx.fem.locate_dofs_topological(V=Q.sub(AIRY), entity_dim=1, entities=bndry_facets)
dofs_w = dolfinx.fem.locate_dofs_topological(V=Q.sub(TRANSVERSE), entity_dim=1, entities=bndry_facets)
bcs_w = dirichletbc(
    np.array(0, dtype=PETSc.ScalarType),
    dofs_w, Q.sub(TRANSVERSE)
)
bcs_v = dirichletbc(
    np.array(0, dtype=PETSc.ScalarType),
    dofs_v, Q.sub(AIRY)
)
_bcs = {AIRY: bcs_v, TRANSVERSE: bcs_w}
bcs = list(_bcs.values())

# DEFINE FUNCTIONS
q = dolfinx.fem.Function(Q)
q_exact = dolfinx.fem.Function(Q)
v, w = ufl.split(q)
v_exact, w_exact = q_exact.split()
state = {"v": v, "w": w}


# DISCLINATION DISTRIBUTION
if mesh.comm.rank == 0:
    disclinations = [np.array([DISCLINATION_POINTS_LIST[0]], dtype=mesh.geometry.x.dtype),
              np.array([DISCLINATION_POINTS_LIST[1]], dtype=mesh.geometry.x.dtype)]
else:
    disclinations = [np.zeros((0, 3), dtype=mesh.geometry.x.dtype),
              np.zeros((0, 3), dtype=mesh.geometry.x.dtype)]


# DEFINE EXACT SOLUTIONS
def function_v_exact(x):
    distance = np.linalg.norm(disclinations[0] - disclinations[1]) # compute distance between dipole
    rq = (x[0]**2 + x[1]**2)
    _v = (1/(16*np.pi))*( ( x[1]**2 + (x[0]-distance/2)**2 )*( np.log(4.0) + np.log( x[1]**2 + (x[0]-distance/2)**2 ) - np.log( 4 - 4*x[0]*distance + rq*distance**2 ) ) - (1/4)*( 4*(x[1]**2) + ( 2*x[0]+distance)**2 ) * ( np.log(4) + np.log( x[1]**2 + (x[0]+distance/2)**2 ) - np.log( 4 + 4*x[0]*distance + rq*distance**2 ) ) )
    return _v * (_E*thickness)

def function_w_exact(x):
    _w = (1 - x[0]**2 - x[1]**2)**2
    _w = 0.0*_w
    return _w

v_exact.interpolate(function_v_exact)
w_exact.interpolate(function_w_exact)

def compute_exact_energy_dipole(v_exact):
    energy = 0.0
    eps = 1E-6
    for index, disc_coord in enumerate(DISCLINATION_POINTS_LIST):
        disc_coord_apprx = [disc_coord[0]+eps, disc_coord[1]+eps]
        energy += _E*thickness*0.5*DISCLINATION_POWER_LIST[index]*v_exact(disc_coord_apprx)
    return energy

exact_energy_dipole = compute_exact_energy_dipole(function_v_exact)

ex_membrane_energy = exact_energy_dipole
ex_bending_energy = 0.0
ex_coupl_energy = 0.0

# DEFINE THE FEM PROBLEM
model = NonlinearPlateFVK_brenner(mesh, parameters["model"])
energy = model.energy(state)[0]

# External work
dx = ufl.Measure("dx")
W_ext = Constant(mesh, np.array(0., dtype=PETSc.ScalarType)) * w * dx
penalisation = model.penalisation(state)

# Discrete energy functional
L = energy - W_ext + penalisation

# Disclination contribution
Q_v, Q_v_to_Q_dofs = Q.sub(AIRY).collapse()
b = compute_disclination_loads(disclinations, DISCLINATION_POWER_LIST, Q, V_sub_to_V_dofs=Q_v_to_Q_dofs, V_sub=Q_v)

test_v, test_w = ufl.TestFunctions(Q)[AIRY], ufl.TestFunctions(Q)[TRANSVERSE]
F = ufl.derivative(L, q, ufl.TestFunction(Q)) + model.coupling_term(state, test_v, test_w)

# Solver instance
solver = SNESSolver(
    F_form=F,
    u=q,
    bcs=bcs,
    petsc_options=parameters.get("solvers").get("elasticity").get("snes"),
    prefix='plate_fvk_disclinations',
    b0=b.vector,
    monitor=monitor,
)

solver.solve()


# COMPUTE ENERGY ERROR
lagrangian_components = {
    "total": model.energy(state)[0],
    "bending": model.energy(state)[1],
    "membrane": model.energy(state)[2],
    "coupling": model.energy(state)[3],
    "penalisation": model.penalisation(state),
}

# Assemble the energy terms and create the dictionary
energy_terms = {label: COMM.allreduce(
dolfinx.fem.assemble_scalar(
    dolfinx.fem.form(energy_term)),
op=MPI.SUM,
) for label, energy_term in lagrangian_components.items()}


print(yaml.dump(parameters["model"], default_flow_style=False))

# Compute error
error = np.abs(exact_energy_dipole - energy_terms['membrane'])

# PRINT RESULTS
print(f"Exact membrane energy: {exact_energy_dipole}")
print(f"Computed membrane energy: {energy_terms['membrane']}")
print(f"Abs error: {error}")
print(f"Rel error: {error/exact_energy_dipole:.3%}")
print(f"Error: {error/exact_energy_dipole:.1%}")


# PLOT MESH
plt.figure()
ax = plot_mesh(mesh)
fig = ax.get_figure()
fig.savefig(f"{PREFIX}/mesh.png")

# PLOTS: FUNCTION PROFILES
xvfb.start_xvfb(wait=0.05)
pyvista.OFF_SCREEN = True

plotter = pyvista.Plotter(
        title="Displacement",
        window_size=[1600, 600],
        shape=(1, 3),
    )

vpp, wpp = q.split()
vpp.name = "Airy"
wpp.name = "deflection"

V_v, dofs_v = Q.sub(0).collapse()
V_w, dofs_w = Q.sub(1).collapse()

_pv_points = np.array([p[0] for p in disclinations])
_pv_colours = np.array(-np.array(DISCLINATION_POWER_LIST))

scalar_plot = plot_scalar(wpp, plotter, subplot=(0, 0), V_sub=V_w, dofs=dofs_w,
                            lineproperties={'clim': [min(wpp.vector[:]), max(wpp.vector[:])]})
plotter.add_points(
        _pv_points,
        scalars = _pv_colours,
        style = 'points', 
        render_points_as_spheres=True, 
        point_size=15.0
)

scalar_plot = plot_scalar(vpp, plotter, subplot=(0, 1), V_sub=V_v, dofs=dofs_v,
                          lineproperties={'clim': [min(vpp.vector[:]), max(vpp.vector[:])]})
plotter.add_points(
        _pv_points,
        scalars = _pv_colours,
        style = 'points', 
        render_points_as_spheres=True, 
        point_size=15.0
)

plotter.subplot(0, 2)
cells, types, x = plot.vtk_mesh(V_v)
grid = pyvista.UnstructuredGrid(cells, types, x)
grid.point_data["v"] = vpp.x.array.real[dofs_v]
grid.set_active_scalars("v")

warped = grid.warp_by_scalar("v", scale_factor=100)
plotter.add_mesh(warped, show_edges=False)
plotter.add_points(
        _pv_points,
        scalars = _pv_colours,
        style = 'points', 
        render_points_as_spheres=True, 
        point_size=15.0
)

scalar_plot.screenshot(f"{PREFIX}/test_fvk.png")
print("plotted scalar")

npoints = 1001
tol = 1e-3
xs = np.linspace(-parameters["geometry"]["radius"] + tol, parameters["geometry"]["radius"] - tol, npoints)
points = np.zeros((3, npoints))
points[0] = xs

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

_plt, data = plot_profile(
    wpp,
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
    subplotnumber=1
)
_plt, data = plot_profile(
    vpp,
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

_plt.savefig(f"{PREFIX}/test_fvk-profiles.png")


# PLOTS: MOMENTS

# RK: should be dg0
DG_e = basix.ufl.element("DG", str(mesh.ufl_cell()), parameters["model"]["order"]-2)
DG = dolfinx.fem.functionspace(mesh, DG_e)

mxx = model.M(wpp)[0, 0]
mxx_expr = dolfinx.fem.Expression(mxx, DG.element.interpolation_points())
Mxx = dolfinx.fem.Function(DG)
Mxx.interpolate(mxx_expr)

pxx = model.P(vpp)[0, 0]
pxx_expr = dolfinx.fem.Expression(pxx, DG.element.interpolation_points())
Pxx = dolfinx.fem.Function(DG)
Pxx.interpolate(pxx_expr)

wxx = model.W(wpp)[0, 0]
wxx_expr = dolfinx.fem.Expression(wxx, DG.element.interpolation_points())
Wxx = dolfinx.fem.Function(DG)
Wxx.interpolate(wxx_expr)

plotter = pyvista.Plotter(
        title="Moment",
        window_size=[1600, 600],
        shape=(1, 3),
    )
try:
    plotter = plot_scalar(Mxx, plotter, subplot=(0, 0))
    plotter = plot_scalar(Pxx, plotter, subplot=(0, 1))
    plotter = plot_scalar(Wxx, plotter, subplot=(0, 2))

    plotter.screenshot(f"{PREFIX}/test_tensors.png")
    print("plotted scalar")
except Exception as e:
    print(e)
    
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

_plt, data = plot_profile(
    Mxx,
    points,
    None,
    subplot=(1, 2),
    lineproperties={
        "c": "k",
        "label": f"$Mxx(x)$"
    },
    fig=fig,
    subplotnumber=1
)

_plt.legend()

_plt, data = plot_profile(
    Pxx,
    points,
    None,
    subplot=(1, 2),
    lineproperties={
        "c": "k",
        "label": f"$Pxx(x)$"
    },
    fig=fig,
    subplotnumber=2
)

_plt.legend()

_plt.savefig(f"{PREFIX}/test_fvk-Mxx-profiles.png")
