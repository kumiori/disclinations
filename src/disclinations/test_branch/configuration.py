"""
PURPOSE OF THE SCRIPT
Analyze configuration of interest
Using the Veriational FE formulationin its NON-dimensional formulation (see "models/adimensional.py").
Dimensionless parameters:
a := R/h >> 1
b := p0/E << 1

ARTICLE RELATED SECTION
"""

import json
import logging
import os
import pdb
import sys
from pathlib import Path
import importlib.resources as pkg_resources

import dolfinx
import dolfinx.plot
from dolfinx import log
import dolfinx.io
from dolfinx.io import XDMFFile, gmshio
import dolfinx.mesh
from dolfinx.fem import Constant, dirichletbc
from dolfinx.fem.petsc import (assemble_matrix, create_vector, create_matrix, assemble_vector)

import ufl
from ufl import (CellDiameter, FacetNormal, dx)

from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import petsc4py

import yaml
import warnings
import basix

import matplotlib
matplotlib.use('WebAgg')
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pyvista

#from disclinations.models import NonlinearPlateFVK
from disclinations.models.adimensional import A_NonlinearPlateFVK
from disclinations.meshes import mesh_bounding_box
from disclinations.meshes.primitives import mesh_circle_gmshapi
from disclinations.utils import load_parameters
from disclinations.utils.la import compute_cell_contributions, compute_disclination_loads
from disclinations.utils.viz import plot_scalar, plot_profile, plot_mesh
from disclinations.utils.sample_function import sample_function, interpolate_sample
from disclinations.solvers import SNESSolver, SNESProblem

PATH_TO_PARAMETERS_FILE = 'disclinations.test'

logging.basicConfig(level=logging.INFO)

REQUIRED_VERSION = "0.8.0"

if dolfinx.__version__ != REQUIRED_VERSION:
    warnings.warn(f"We need dolfinx version {REQUIRED_VERSION}, but found version {dolfinx.__version__}. Exiting.")
    sys.exit(1)


petsc4py.init(sys.argv)
log.set_log_level(log.LogLevel.WARNING)

comm = MPI.COMM_WORLD

def monitor(snes, it, norm):
    logging.info(f"Iteration {it}, residual {norm}")
    print(f"Iteration {it}, residual {norm}")
    return PETSc.SNES.ConvergedReason.ITERATING

# Set output directory
OUTDIR = os.path.join("output", "configuration")

if comm.rank == 0:
    Path(OUTDIR).mkdir(parents=True, exist_ok=True)


AIRY = 0
TRANSVERSE = 1

# SET DISCRETE DISTRIBUTION OF DISCLINATIONS
disclination_points_list = [[-0.7, 0.0, 0], [0.0, 0.7, 0], [0.7, 0.0, 0], [0.0, -0.7, 0],
                            [0.5, 0.5, 0], [0.5, -0.5, 0], [-0.5, -0.5, 0], [-0.5, 0.5, 0],
                            [0, 0, 0]] #
#disclination_points_list = [[0.0, 0.0, 0]]
disclination_power_list = [1, 1, 1, 1, -0.5, -0.5, -0.5, -0.5, 0] #[-0.5, -0.5, -0.5, -0.5, 1, 1, 1, 1, 0]

# READ PARAMETERS FILE
#with pkg_resources.path(PATH_TO_PARAMETERS_FILE, 'parameters.yml') as f:
    #parameters, _ = load_parameters(f)
with open('../test/parameters.yml') as f:
    parameters = yaml.load(f, Loader=yaml.FullLoader)

# COMPUTE DIMENSIONLESS PARAMETERS
a = parameters["geometry"]["radius"] / parameters["model"]["thickness"]
p0 = 100*parameters["model"]["thickness"]
b = p0 / parameters["model"]["E"]

print(10*"*")
print("Dimensionless parameters: ")
print("a := R/h = ", a)
print("b := p0/E = ", b)
print(10*"*")

# LOAD MESH
mesh_size = parameters["geometry"]["mesh_size"]
#parameters["geometry"]["radius"] = 1
parameters["geometry"]["geom_type"] = "circle"
model_rank = 0
tdim = 2
gmsh_model, tdim = mesh_circle_gmshapi( parameters["geometry"]["geom_type"], 1, mesh_size, tdim )
mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, model_rank, tdim)

h = CellDiameter(mesh)
n = FacetNormal(mesh)

# FUNCTION SPACES
X = basix.ufl.element("P", str(mesh.ufl_cell()), parameters["model"]["order"])
Q = dolfinx.fem.functionspace(mesh, basix.ufl.mixed_element([X, X]))

q = dolfinx.fem.Function(Q)
v, w = ufl.split(q)
state = {"v": v, "w": w}

# SET DIRICHLET BC
mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
bndry_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
bndry_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
dofs_v = dolfinx.fem.locate_dofs_topological(V=Q.sub(AIRY), entity_dim=1, entities=bndry_facets)
dofs_w = dolfinx.fem.locate_dofs_topological(V=Q.sub(TRANSVERSE), entity_dim=1, entities=bndry_facets)
bcs_v = dirichletbc( np.array(0, dtype=PETSc.ScalarType), dofs_v, Q.sub(AIRY) )
bcs_w = dirichletbc( np.array(0, dtype=PETSc.ScalarType), dofs_w, Q.sub(TRANSVERSE) )
_bcs = {AIRY: bcs_v, TRANSVERSE: bcs_w}
bcs = list(_bcs.values())


# DEFINE THE VARIATIONAL PROBLEM
model = A_NonlinearPlateFVK(mesh, parameters["model"])
energy = model.energy(state)[0]

# Volume load
def volume_load(x): return -((a**4)*b) * (x[0]**2 + x[1]**2)
f = dolfinx.fem.Function(Q.sub(TRANSVERSE).collapse()[0])
f.interpolate(volume_load)
dx = ufl.Measure("dx")
W_ext = f * w * dx # CFe: external work
penalisation = model.penalisation(state)

# Disclinations
if mesh.comm.rank == 0:
    disclinations = []
    for dp in disclination_points_list: disclinations.append(np.array([dp], dtype=mesh.geometry.x.dtype))
else:
    for dp in disclination_points_list: disclinations.append(np.zeros((0, 3), dtype=mesh.geometry.x.dtype))

# Functional
L = energy - W_ext + penalisation

# DEFINE THE FEM (WEAK) PROBLEM
F = ufl.derivative(L, q, ufl.TestFunction(Q))

Q_v, Q_v_to_Q_dofs = Q.sub(AIRY).collapse()

dp_list = [element*(a**2) for element in disclination_power_list]

b = compute_disclination_loads(disclinations, dp_list, Q, V_sub_to_V_dofs=Q_v_to_Q_dofs, V_sub=Q_v)

solver_parameters = {
        "snes_type": "newtonls",  # Solver type: NGMRES (Nonlinear GMRES)
        "snes_max_it": 50,  # Maximum number of iterations
        "snes_rtol": 1e-11,  # Relative tolerance for convergence
        "snes_atol": 1e-11,  # Absolute tolerance for convergence
        "snes_stol": 1e-11,  # Tolerance for the change in solution norm
        "snes_monitor": None,  # Function for monitoring convergence (optional)
        "snes_linesearch_type": "basic",  # Type of line search
    }

solver = SNESSolver(
    F_form=F,
    u=q,
    bcs=bcs,
    petsc_options=solver_parameters, #parameters.get("solvers").get("elasticity").get("snes"),
    prefix='plate_configuration',
    b0=b.vector,
    monitor=monitor,
)

solver.solve() # CFe: solve FEM problem
#convergence = solver.getConvergedReason()
#print("convergence: ", convergence)

# DISPLAY COMPUTED ENERGY VALUES
energy_scale = parameters["model"]["E"] * (parameters["model"]["thickness"]**3) / (a**2)
energy_components = {
    "bending": energy_scale*model.energy(state)[1],
    "membrane": energy_scale*model.energy(state)[2],
    "coupling": energy_scale*model.energy(state)[3],
    "external_work": p0* parameters["model"]["thickness"] * (parameters["geometry"]["radius"]**2) *W_ext
    }
# penalty_components = {"dg1_w": dg1_w(w), "dg2": dg2(w), "dg1_v": dg1_v(v), "dgc": dgc(v, w)}
# boundary_components = {"bc1_w": bc1_w(w), "bc2": bc2(w), "bc1_v": bc1_v(v)}

computed_energy_terms = {label: comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(energy_term)), op=MPI.SUM) for
                         label, energy_term in energy_components.items()}

print("Dimensional energy values: ", computed_energy_terms)

# DEFINE AIRY AND TRANSVERSE DISPLACEMENT FOR POST-PROCESSING
print("v_scale := Eh^3 = ", model.v_scale)
print("w_scale := h = ", model.w_scale)
v_pp, w_pp = q.split()
v_pp.vector.array = model.v_scale * v_pp.vector.array
w_pp.vector.array = model.w_scale * w_pp.vector.array


# PLOTS
info_filename = f"V_mesh_{parameters["geometry"]["mesh_size"]}_IP_{parameters["model"]["alpha_penalty"]}_E_{parameters["model"]["E"]:.2e}_h_{parameters["model"]["thickness"]:.2e}"

x_samples, y_samples, v_samples = sample_function(v_pp, parameters["geometry"]["radius"])
grid_x, grid_y, v_interp = interpolate_sample(x_samples, y_samples, v_samples, parameters["geometry"]["radius"])

x_samples, y_samples, w_samples = sample_function(w_pp, parameters["geometry"]["radius"])
grid_x, grid_y, w_interp = interpolate_sample(x_samples, y_samples, w_samples, parameters["geometry"]["radius"])

x0_samples = []
v0_samples = []
w0_samples = []
for i in range(len(y_samples)):
    if abs(y_samples[i]) < 5e-11:
        x0_samples.append(x_samples[i])
        v0_samples.append(v_samples[i])
        w0_samples.append(w_samples[i])

# Surface plots
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(grid_x, grid_y, v_interp, cmap='viridis', edgecolor='none')
cbar = fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)
cbar.set_label('[Nm]')
ax.set_xlabel('x axis [m]', fontsize=25)
ax.set_ylabel('y axis [m]', fontsize=25)
ax.set_zlabel('v [Nm]', fontsize=25)
ax.tick_params(axis='both', which='major', labelsize=20)  # X and Y axis major ticks
ax.tick_params(axis='z', which='major', labelsize=20)
ax.set_title('Airy\'s stress function', fontsize=30)
plt.gca().yaxis.get_offset_text().set_fontsize(35)
plt.show()
plt.savefig(f"{OUTDIR}/surface_Airy_{info_filename}.png", dpi=300)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(grid_x, grid_y, w_interp, cmap='coolwarm', edgecolor='none')
cbar = fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)
cbar.set_label('[m]')
ax.set_xlabel('x axis [m]', fontsize=25)
ax.set_ylabel('y axis [m]', fontsize=25)
ax.set_zlabel('$w$ [m]', fontsize=25)
ax.tick_params(axis='both', which='major', labelsize=20)  # X and Y axis major ticks
ax.tick_params(axis='z', which='major', labelsize=20)
ax.set_title('Transverse displacement', fontsize=30)
plt.gca().yaxis.get_offset_text().set_fontsize(35)
plt.show()
plt.savefig(f"{OUTDIR}/surface_Transv_displacement_{info_filename}.png", dpi=300)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(grid_x, grid_y, w_interp, cmap='coolwarm', edgecolor='none')
cbar = fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)
cbar.set_label('[m]')
ax.set_xlabel('x axis [m]', fontsize=25)
ax.set_ylabel('y axis [m]', fontsize=25)
ax.set_zlabel('$w$ [m]', fontsize=25)
ax.tick_params(axis='both', which='major', labelsize=20)  # X and Y axis major ticks
ax.tick_params(axis='z', which='major', labelsize=20)
ax.set_title('Transverse displacement', fontsize=30)
ax.set_zlim([-1,1])
plt.gca().yaxis.get_offset_text().set_fontsize(35)
plt.show()
plt.savefig(f"{OUTDIR}/surface_Transv_displacement_2_{info_filename}.png", dpi=300)

# Profile plots
plt.figure(figsize=(10, 6))
plt.plot(x0_samples, v0_samples, color='red', linestyle='-', label='v(x,0)', linewidth=3)
plt.xlabel('x axes [m]', fontsize=25)
plt.ylabel('Airy\' stress function [Nm]', fontsize=25)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.title('Airy\'s stress function',  fontsize=30)
plt.legend(fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig(f"{OUTDIR}/profile_Airy_{info_filename}.png", dpi=300)

plt.figure(figsize=(10, 6))
plt.plot(x0_samples, w0_samples, color='red', linestyle='-', label='w(x,0)', linewidth=3)
plt.xlabel('x axes [m]', fontsize=25)
plt.ylabel('Transverse displacement [m]', fontsize=25)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.title('Transverse displacement',  fontsize=30)
plt.legend(fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.gca().yaxis.get_offset_text().set_fontsize(35)
plt.show()
plt.savefig(f"{OUTDIR}/profile_Transv_displacement_{info_filename}.png", dpi=300)

# import matplotlib.pyplot as plt
#
# plt.figure()
# ax = plot_mesh(mesh)
# fig = ax.get_figure()
# fig.savefig(f"{prefix}/mesh.png")
#
#
#
#
# # ------------------------------
#
#
# import pyvista
# from pyvista.plotting.utilities import xvfb
# from dolfinx import plot
#
# xvfb.start_xvfb(wait=0.05)
# pyvista.OFF_SCREEN = True
#
# plotter = pyvista.Plotter(title="Displacement", window_size=[1600, 600], shape=(1, 3) )
#
# v, w = q.split()
# v.name = "Airy"
# w.name = "deflection"
#
# V_v, dofs_v = Q.sub(0).collapse()
# V_w, dofs_w = Q.sub(1).collapse()
#
# # _pv_points = np.array([p[0] for p in disclinations])
# # _pv_colours = np.array(-np.array(signs))
#
# scalar_plot = plot_scalar(w, plotter, subplot=(0, 0), V_sub=V_w, dofs=dofs_w, lineproperties={'clim': [min(w.vector[:]), max(w.vector[:])]})
# #plotter.add_points(
# #        disclinations,
# #        scalars = disclination_power_list,
# #        style = 'points',
# #        render_points_as_spheres=True,
# #        point_size=15.0
# #)
#
# scalar_plot = plot_scalar(v, plotter, subplot=(0, 1), V_sub=V_v, dofs=dofs_v,
#                           lineproperties={'clim': [min(v.vector[:]), max(v.vector[:])]})
# #plotter.add_points(
# #        disclinations,
# #        scalars = disclination_power_list,
# #        style = 'points',
# #        render_points_as_spheres=True,
# #        point_size=15.0
# #)
#
# plotter.subplot(0, 2)
# cells, types, x = plot.vtk_mesh(V_v)
# grid = pyvista.UnstructuredGrid(cells, types, x)
# grid.point_data["v"] = v.x.array.real[dofs_v]
# grid.set_active_scalars("v")
#
# warped = grid.warp_by_scalar("v", scale_factor=100)
# plotter.add_mesh(warped, show_edges=False)
# #plotter.add_points(
# #        disclinations,
# #        scalars = disclination_power_list,
# #        style = 'points',
# #        render_points_as_spheres=True,
# #        point_size=15.0
# #)
#
# scalar_plot.screenshot(f"{prefix}/test_fvk.png")
# print("plotted scalar")
#
# npoints = 1001
# tol = 1e-3
# xs = np.linspace(-parameters["geometry"]["radius"] + tol, parameters["geometry"]["radius"] - tol, npoints)
# points = np.zeros((3, npoints))
# points[0] = xs
#
# fig, axes = plt.subplots(1, 2, figsize=(18, 6))
#
# _plt, data = plot_profile(w, points, None, subplot=(1, 2), lineproperties={"c": "k", "label": f"$w(x)$"}, fig=fig, subplotnumber=1)
# _plt, data = plot_profile(v, points, None, subplot=(1, 2), lineproperties={"c": "k", "label": f"$v(x)$"}, fig=fig, subplotnumber=2)
# _plt.legend()
#
# _plt.savefig(f"{prefix}/test_fvk-profiles.png")


# Plot moments
# DG_e = basix.ufl.element("DG", str(mesh.ufl_cell()), parameters["model"]["order"]-2)
# DG = dolfinx.fem.functionspace(mesh, DG_e)
#
# mxx = model.M(w)[0, 0]
# mxx_expr = dolfinx.fem.Expression(mxx, DG.element.interpolation_points())
# Mxx = dolfinx.fem.Function(DG)
# Mxx.interpolate(mxx_expr)
#
# pxx = model.P(v)[0, 0]
# pxx_expr = dolfinx.fem.Expression(pxx, DG.element.interpolation_points())
# Pxx = dolfinx.fem.Function(DG)
# Pxx.interpolate(pxx_expr)
#
# wxx = model.W(w)[0, 0]
# wxx_expr = dolfinx.fem.Expression(wxx, DG.element.interpolation_points())
# Wxx = dolfinx.fem.Function(DG)
# Wxx.interpolate(wxx_expr)
#
# plotter = pyvista.Plotter(title="Moment", window_size=[1600, 600], shape=(1, 3))
# try:
#     plotter = plot_scalar(Mxx, plotter, subplot=(0, 0))
#     plotter = plot_scalar(Pxx, plotter, subplot=(0, 1))
#     plotter = plot_scalar(Wxx, plotter, subplot=(0, 2))
#     plotter.screenshot(f"{OUTDIR}/test_tensors.png")
#     print("plotted scalar")
# except Exception as e:
#     print(e)
#
# fig, axes = plt.subplots(1, 2, figsize=(18, 6))
#
# _plt, data = plot_profile(Mxx, points, None, subplot=(1, 2), lineproperties={"c": "k",  "label": f"$Mxx(x)$"}, fig=fig, subplotnumber=1)
# _plt.legend()
# _plt, data = plot_profile(Pxx, points, None, subplot=(1, 2), lineproperties={"c": "k", "label": f"$Pxx(x)$"}, fig=fig, subplotnumber=2)
# _plt.legend()
# _plt.savefig(f"{OUTDIR}/test_fvk-Mxx-profiles.png")
