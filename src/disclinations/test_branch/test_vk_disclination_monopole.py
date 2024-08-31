#import json
import logging
import os
import pdb
import sys
from pathlib import Path
import numpy as np
import math

import petsc4py
from petsc4py import PETSc
import ufl
from ufl import (CellDiameter, FacetNormal, dx,)
import yaml
from mpi4py import MPI

import dolfinx
import dolfinx.plot
from dolfinx import log
import dolfinx.mesh
import dolfinx.io
from dolfinx.io import XDMFFile, gmshio
from dolfinx.fem import Constant, dirichletbc
from dolfinx.fem.petsc import (assemble_matrix, create_vector, create_matrix, assemble_vector)
import basix

from disclinations.models import NonlinearPlateFVK
from disclinations.models.Fvk_plate import Fvk_plate
from disclinations.meshes import mesh_bounding_box
from disclinations.meshes.primitives import mesh_circle_gmshapi
from disclinations.utils.viz import plot_scalar, plot_profile, plot_mesh
from disclinations.utils.sample_function import sample_function, interpolate_sample
from disclinations.solvers import SNESSolver, SNESProblem

logging.basicConfig(level=logging.INFO)

petsc4py.init(sys.argv)
log.set_log_level(log.LogLevel.WARNING)

comm         = MPI.COMM_WORLD
AIRY         = 0
TRANSVERSE   = 1
TEST_PASSED  = False
ENENRGY_PERC_ERR_THRESHOLD = 1.0

with open("parameters.yml") as f: parameters = yaml.load(f, Loader=yaml.FullLoader)

model_rank = 0
tdim = 2

# Generate Mesh object
gmsh_model, tdim = mesh_circle_gmshapi(
    parameters["geometry"]["geom_type"], parameters["geometry"]["R"], parameters["geometry"]["mesh_size"], tdim
)
mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, model_rank, tdim)

outdir = "output"
dirTest = os.path.join(outdir, "Test - Disclination monopole")

if comm.rank == 0: Path(dirTest).mkdir(parents=True, exist_ok=True)


# Load Elastic / Geometric Parameters
_nu = parameters["model"]["nu"]
_h = parameters["model"]["thickness"]
_E = parameters["model"]["E"]
_D = (_E*_h**3)/(12*(1-_nu**2))

# Data
disclination_points_list = [[0.0,0.0,0.0]]
disclination_power = [1]

print(" ")
print("***************** TEST INFO *****************")
print("3D Young's modulus: ", _E)
print("Poisson coefficient: ", _nu)
print("Plate's thickness: ", _h)
print("Domain geometry: ", parameters["geometry"]["geom_type"])
print("Domain radius: ", parameters["geometry"]["R"])
print("Mesh size: ", parameters["geometry"]["mesh_size"])
print("Stabilization term: ", parameters["model"]["alpha_penalty"])
print("Finite Element polinomial order: ", parameters["model"]["order"])
print("***************** TEST BEGINS *****************")
print(" ")

# Define Function Space
fe = basix.ufl.element("P", str(mesh.ufl_cell()), parameters["model"]["order"])
fes = dolfinx.fem.functionspace(mesh, basix.ufl.mixed_element([fe, fe]))

q_exact = dolfinx.fem.Function(fes)
v_exact, w_exact = q_exact.split()

exact_energy_monopole = parameters["model"]["E"] * parameters["geometry"]["R"]**2 / (32 * np.pi)

def _v_exact(x):
    rq = (x[0]**2 + x[1]**2)
    _v = _E*disclination_power[0]/(16.0*np.pi)*(rq*np.log(rq/parameters["geometry"]["R"]**2) - rq + parameters["geometry"]["R"]**2)
    return _v

def _w_exact(x):
    _w = (1 - x[0]**2 - x[1]**2)**2
    _w = 0.0*_w
    return _w

v_exact.interpolate(_v_exact)
w_exact.interpolate(_w_exact)

plate = Fvk_plate(mesh, parameters["model"])
plate.set_out_dir()
plate.set_dirichlet_bc_airy(0)
plate.set_dirichlet_bc_trans_displ(0)
plate.set_disclination(disclination_points_list, disclination_power)
plate.compute()
plate.plot_profiles(parameters["geometry"]["R"])
w = plate.get_tranverse_disp()
v = plate.get_airy()

energy_absolute_error = exact_energy_monopole - plate.get_membrane_energy_value()
energy_percent_error  = 100*energy_absolute_error/exact_energy_monopole

if np.abs(energy_percent_error) < ENENRGY_PERC_ERR_THRESHOLD: TEST_PASSED = True

print(" ")
print("***************** TEST RESULTS *****************")
print(" ")
print("Plate's membrane energy: ", plate.get_membrane_energy_value())
print("Plate's bending energy: ", plate.get_bending_energy_value())
print("Plate's coupling energy: ", plate.get_coupling_energy_value())
print("Plate's penalization energy: ", plate.get_penalization_energy_value())
print(" ")
print("Exact monopole energy: ", exact_energy_monopole)
print(" ")
print("Absolute Error: ", energy_absolute_error)
print("Percent Error: ", energy_percent_error, " %")
print(" ")
if TEST_PASSED: print("Test Passed")
else: print("Test Failed")
print(" ")
print("***********************************************")

# PLOTS
x_samples, y_samples, v_samples = sample_function(v, parameters["geometry"]["R"])
grid_x, grid_y, v_interp = interpolate_sample(x_samples, y_samples, v_samples, parameters["geometry"]["R"])

x_samples, y_samples, v_exact_samples = sample_function(v_exact, parameters["geometry"]["R"])
grid_x, grid_y, v_exact_interp = interpolate_sample(x_samples, y_samples, v_exact_samples, parameters["geometry"]["R"])

x_samples, y_samples, w_samples = sample_function(w, parameters["geometry"]["R"])
grid_x, grid_y, w_interp = interpolate_sample(x_samples, y_samples, w_samples, parameters["geometry"]["R"])

x_samples, y_samples, w_exact_samples = sample_function(w_exact, parameters["geometry"]["R"])
grid_x, grid_y, w_exact_interp = interpolate_sample(x_samples, y_samples, w_exact_samples, parameters["geometry"]["R"])

x0_samples = []
v0_samples = []
v0_exact_samples = []
w0_samples = []
w0_exact_samples = []
for i in range(len(y_samples)):
    if abs(y_samples[i]) < 1e-10:
        x0_samples.append(x_samples[i])
        v0_samples.append(v_samples[i])
        v0_exact_samples.append(v_exact_samples[i])
        w0_samples.append(w_samples[i])
        w0_exact_samples.append(w_exact_samples[i])

import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

# SURFACE PLOT
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(grid_x, grid_y, v_interp, cmap='viridis', edgecolor='none')
fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)
ax.set_xlabel('x axis', fontsize=20)
ax.set_ylabel('y axis', fontsize=20)
ax.set_zlabel('v', fontsize=20)
ax.set_title('Airy\'strees function', fontsize=20)
plt.show()
plt.savefig(f"{dirTest}/surface_Airy.png", dpi=300)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(grid_x, grid_y, v_exact_interp-v_interp, cmap='viridis', edgecolor='none')
fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)
ax.set_xlabel('x axis', fontsize=20)
ax.set_ylabel('y axis', fontsize=20)
ax.set_zlabel('v - v_fem', fontsize=20)
ax.set_title('Airy\'strees function: difference between solutions - $v_ext - v_fem$')
plt.show()
plt.savefig(f"{dirTest}/surface_Airy_fem_vs_exact.png", dpi=300)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(grid_x, grid_y, w_interp, cmap='viridis', edgecolor='none')
fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)
ax.set_xlabel('x axis', fontsize=20)
ax.set_ylabel('y axis', fontsize=20)
ax.set_zlabel('$w$', fontsize=20)
ax.set_title('Transverse displacement', fontsize=20)
plt.show()
plt.savefig(f"{dirTest}/surface_Transv_displacement.png", dpi=300)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(grid_x, grid_y, w_exact_interp-w_interp, cmap='viridis', edgecolor='none')
fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)
ax.set_xlabel('x axis', fontsize=20)
ax.set_ylabel('y axis', fontsize=20)
ax.set_zlabel('Transverse displacement', fontsize=20)
ax.set_title('Transverse displacement: difference between solutions - w_ext - w_fem', fontsize=20)
plt.show()
plt.savefig(f"{dirTest}/surface_Transv_displacement_fem_vs_exact.png", dpi=300)

# PROFILE PLOTS
plt.figure(figsize=(10, 6))
plt.plot(x0_samples, v0_samples, color='red', linestyle='-', label='FE solution', linewidth=3)
plt.plot(x0_samples, v0_exact_samples, color='blue', linestyle='dotted', label='Exact solution', linewidth=3)
plt.xlabel('x axes', fontsize=20)
plt.ylabel('Airy\' stress function', fontsize=20)
plt.title('Comparison of FE and exact solutions - Airy\'s stress function',  fontsize=20)
plt.legend(fontsize=20)
plt.grid(True)

plt.tight_layout()
plt.show()
plt.savefig(f"{dirTest}/profile_FE and exact solutions - Airy\'s stress function.png", dpi=300)

plt.figure(figsize=(10, 6))
plt.plot(x0_samples, w0_samples, color='red', linestyle='-', label='FE solution', linewidth=3)
plt.plot(x0_samples, w0_exact_samples, color='blue', linestyle='dotted', label='Exact solution', linewidth=3)
plt.xlabel('x axes', fontsize=20)
plt.ylabel('Transverse displacement', fontsize=20)
plt.title('Comparison of FE and exact solutions - Transverse displacement',  fontsize=20)
plt.legend(fontsize=20)
plt.grid(True)

plt.tight_layout()
plt.show()
plt.savefig(f"{dirTest}/profile_FE and exact solutions - Transverse displacement.png", dpi=300)

# CONTOUR PLOT
# fig = plt.figure()
# plt.figure(figsize=(8, 6))
# contour = plt.contourf(grid_x, grid_y, v_interp, levels=100, cmap='viridis')
# plt.colorbar(contour)
# plt.xlabel('X Axis')
# plt.ylabel('Y Axis')
# plt.title('Airy\'strees function Contour Plot')
# plt.show()
# plt.savefig(f"{dirTest}/contour_Airy.png", dpi=300)

# SCATTER PLOT
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x_samples, y_samples, v_samples, c='b', marker='o')
# ax.set_xlabel('X Axis')
# ax.set_ylabel('Y Axis')
# ax.set_zlabel('Z Axis')
# ax.set_title('Airy\'strees function 3D samples')
# plt.show()
# plt.savefig(f"{dirTest}/3D_scatter_Airy.png", dpi=300)

"""
V_v, dofs_v = fes.sub(AIRY).collapse()
V_w, dofs_w = fes.sub(TRANSVERSE).collapse()
import matplotlib.pyplot as plt

plt.figure()
ax = plot_mesh(mesh)
fig = ax.get_figure()
fig.savefig(f"{dirTest}/mesh.png")
import pyvista
from pyvista.plotting.utilities import xvfb

xvfb.start_xvfb(wait=0.05)
pyvista.OFF_SCREEN = True

plotter = pyvista.Plotter(title="Displacement", window_size=[1200, 600], shape=(2, 2))

scalar_plot = plot_scalar(v, plotter, subplot=(0, 0), V_sub=V_v, dofs=dofs_v)
scalar_plot = plot_scalar(w, plotter, subplot=(0, 1), V_sub=V_w, dofs=dofs_w)

scalar_plot = plot_scalar(v_exact, plotter, subplot=(1, 0), V_sub=V_v, dofs=dofs_v)
scalar_plot = plot_scalar(w_exact, plotter, subplot=(1, 1), V_sub=V_w, dofs=dofs_w)

scalar_plot.screenshot(f"{dirTest}/test_fvk.png")

tol = 1e-3
#xs = np.linspace(-parameters["geometry"]["R"] + tol, parameters["geometry"]["R"] - tol, 101)
xs = np.linspace(-0.2, 0.2, 101)
ys = np.linspace(0.5, 0.5, 101)
points = np.zeros((3, 101))
points[0] = xs
points[1] = ys

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

_plt, data = plot_profile(w, points, None, subplot=(1, 2),
                          lineproperties={"c": "k", "label": f"$w(x)$"}, fig=fig, subplotnumber=1)

_plt, data = plot_profile(w_exact, points, None, subplot=(1, 2),
                          lineproperties={"c": "r", "label": f"$w_e(x)$", "ls": "--"},
                          fig=fig, subplotnumber=1)

_plt, data = plot_profile(v, points, None, subplot=(1, 2),
                          lineproperties={"c": "k", "label": f"$v(x)$"},
                          fig=fig, subplotnumber=2)

_plt, data = plot_profile(v_exact, points, None, subplot=(1, 2),
                          lineproperties={"c": "r", "label": f"$v_e(x)$", "ls": "--"},
                          fig=fig, subplotnumber=2)

_plt.legend()

_plt.savefig(f"{dirTest}/test_fvk-profiles.png")
"""
