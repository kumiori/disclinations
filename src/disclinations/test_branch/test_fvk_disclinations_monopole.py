# Solving a Foppl-von-Karman plate problem with a penalised formulation
# H^0_2 formulation with disclinations
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
from disclinations.models import NonlinearPlateFVK
from disclinations.meshes import mesh_bounding_box
from disclinations.meshes.primitives import mesh_circle_gmshapi
from disclinations.utils.la import compute_cell_contributions, compute_disclination_loads
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

gmsh_model, tdim = mesh_circle_gmshapi(
    parameters["geometry"]["geom_type"], 1, mesh_size, tdim
)
mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, model_rank, tdim)

outdir = "output"
prefix = os.path.join(outdir, "plate_fvk_disclinations_monopole")

if comm.rank == 0:
    Path(prefix).mkdir(parents=True, exist_ok=True)

h = CellDiameter(mesh)
n = FacetNormal(mesh)

# Function spaces

X = basix.ufl.element("P", str(mesh.ufl_cell()), parameters["model"]["order"]) 
Q = dolfinx.fem.functionspace(mesh, basix.ufl.mixed_element([X, X]))


# Material parameters

nu = parameters["model"]["nu"]
# _h = parameters["model"]["thickness"]
thickness = parameters["model"]["thickness"]
_E = parameters["model"]["E"]
_D = _E * thickness**3 / (12 * (1 - nu**2))

w_scale = np.sqrt(2*_D/(_E*thickness))
v_scale = _D
# p_scale = 12.*np.sqrt(6) * (1 - _nu**2)**3./2. / (_E * _h**4)
f_scale = np.sqrt(2 * _D**3 / (_E * thickness))

AIRY = 0
TRANSVERSE = 1

mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
bndry_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
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

dx = ufl.Measure("dx")

q = dolfinx.fem.Function(Q)
q_exact = dolfinx.fem.Function(Q)
v, w = ufl.split(q)
v_exact, w_exact = q_exact.split()

state = {"v": v, "w": w}


# Point sources
if mesh.comm.rank == 0:
    # point = np.array([[0.68, 0.36, 0]], dtype=mesh.geometry.x.dtype)
    disclinations = [np.array([[-0.0, 0.0, 0]], dtype=mesh.geometry.x.dtype)]
    signs = [1]
else:
    # point = np.zeros((0, 3), dtype=mesh.geometry.x.dtype)
    disclinations = [np.zeros((0, 3), dtype=mesh.geometry.x.dtype),
              np.zeros((0, 3), dtype=mesh.geometry.x.dtype)]


def _v_exact(x):
    rq = (x[0]**2 + x[1]**2)
    radius = parameters["geometry"]["radius"]
    _v = _E*signs[0]/(16.0*np.pi)*(rq*np.log(rq/radius**2) - rq + radius**2)

    # return _v * (_E*thickness)
    return _v * v_scale

def _w_exact(x):

    _w = 0.0 * x[0]
    return _w

v_exact.interpolate(_v_exact)
w_exact.interpolate(_w_exact)

# Define the variational problem
model = NonlinearPlateFVK(mesh, parameters["model"])
energy = model.energy(state)[0]

# Dead load (transverse)
W_ext = Constant(mesh, np.array(0., dtype=PETSc.ScalarType)) * w * dx
penalisation = model.penalisation(state)

# Define the functional
L = energy - W_ext + penalisation

Q_v, Q_v_to_Q_dofs = Q.sub(AIRY).collapse()

b = compute_disclination_loads(disclinations, signs, Q, V_sub_to_V_dofs=Q_v_to_Q_dofs, V_sub=Q_v)    

F = ufl.derivative(L, q, ufl.TestFunction(Q))

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

import matplotlib.pyplot as plt

plt.figure()
ax = plot_mesh(mesh)
fig = ax.get_figure()
fig.savefig(f"{prefix}/mesh.png")


energy_components = {"bending": model.energy(state)[1], "membrane": -model.energy(state)[2], "coupling": model.energy(state)[3], "external_work": -W_ext}
# penalty_components = {"dg1_w": dg1_w(w), "dg2": dg2(w), "dg1_v": dg1_v(v), "dgc": dgc(v, w)}
# boundary_components = {"bc1_w": bc1_w(w), "bc2": bc2(w), "bc1_v": bc1_v(v)}

# Assemble the energy terms and create the dictionary
computed_energy_terms = {label: comm.allreduce(
    dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(energy_term)),
    op=MPI.SUM,
) for label, energy_term in energy_components.items()}

# computed_penalty_terms = {label: comm.allreduce(
#     dolfinx.fem.assemble_scalar(
#         dolfinx.fem.form(penalty_term)),
#     op=MPI.SUM,
# ) for label, penalty_term in penalty_components.items()}

# computed_boundary_terms = {label: comm.allreduce(
#     dolfinx.fem.assemble_scalar(
#         dolfinx.fem.form(boundary_term)),
#     op=MPI.SUM,
# ) for label, boundary_term in boundary_components.items()}


print(computed_energy_terms)

# ------------------------------
# check energy vs exact energy of monopole
lagrangian_components = {
    "total": model.energy(state)[0],
    "bending": model.energy(state)[1],
    "membrane": model.energy(state)[2],
    "coupling": model.energy(state)[3],
    "penalisation": model.penalisation(state),
}

energy_terms = {label: comm.allreduce(
dolfinx.fem.assemble_scalar(
    dolfinx.fem.form(energy_term)),
op=MPI.SUM,
) for label, energy_term in lagrangian_components.items()}


exact_energy_monopole = parameters["model"]["E"] * parameters["geometry"]["radius"]**2 / (32 * np.pi)
# check, there is no thickness in here, compare with dipole

print(yaml.dump(parameters["model"], default_flow_style=False))

error = np.abs(exact_energy_monopole - energy_terms['membrane'])

print(f"Exact energy: {exact_energy_monopole}")
print(f"Computed energy: {energy_terms['membrane']}")
print(f"Abs error: {error}")
print(f"Rel error: {error/exact_energy_monopole:.3%}")
print(f"Error: {error/exact_energy_monopole:.1%}")


# ------------------------------


import pyvista
from pyvista.plotting.utilities import xvfb
from dolfinx import plot

xvfb.start_xvfb(wait=0.05)
pyvista.OFF_SCREEN = True

plotter = pyvista.Plotter(
        title="Displacement",
        window_size=[1600, 600],
        shape=(1, 3),
    )

v, w = q.split()
v.name = "Airy"
w.name = "deflection"

V_v, dofs_v = Q.sub(0).collapse()
V_w, dofs_w = Q.sub(1).collapse()

_pv_points = np.array([p[0] for p in disclinations])
_pv_colours = np.array(-np.array(signs))

scalar_plot = plot_scalar(w, plotter, subplot=(0, 0), V_sub=V_w, dofs=dofs_w,
                            lineproperties={'clim': [min(w.vector[:]), max(w.vector[:])]})
plotter.add_points(
        _pv_points,
        scalars = _pv_colours,
        style = 'points', 
        render_points_as_spheres=True, 
        point_size=15.0
)

scalar_plot = plot_scalar(v, plotter, subplot=(0, 1), V_sub=V_v, dofs=dofs_v,
                          lineproperties={'clim': [min(v.vector[:]), max(v.vector[:])]})
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
grid.point_data["v"] = v.x.array.real[dofs_v]
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

scalar_plot.screenshot(f"{prefix}/test_fvk.png")
print("plotted scalar")

npoints = 1001
tol = 1e-3
xs = np.linspace(-parameters["geometry"]["radius"] + tol, parameters["geometry"]["radius"] - tol, npoints)
points = np.zeros((3, npoints))
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


# RK: should be dg0
DG_e = basix.ufl.element("DG", str(mesh.ufl_cell()), parameters["model"]["order"]-2)
DG = dolfinx.fem.functionspace(mesh, DG_e)

mxx = model.M(w)[0, 0]
mxx_expr = dolfinx.fem.Expression(mxx, DG.element.interpolation_points())
Mxx = dolfinx.fem.Function(DG)
Mxx.interpolate(mxx_expr)

pxx = model.P(v)[0, 0]
pxx_expr = dolfinx.fem.Expression(pxx, DG.element.interpolation_points())
Pxx = dolfinx.fem.Function(DG)
Pxx.interpolate(pxx_expr)

wxx = model.W(w)[0, 0]
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

    plotter.screenshot(f"{prefix}/test_tensors.png")
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

_plt.savefig(f"{prefix}/test_fvk-Mxx-profiles.png")
