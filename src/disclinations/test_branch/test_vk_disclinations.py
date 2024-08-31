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
from ufl import (CellDiameter, FacetNormal, dx)
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
from disclinations.solvers import SNESSolver, SNESProblem
from disclinations.utils.sample_function import sample_function, interpolate_sample

import matplotlib.pyplot as plt
import pyvista
from pyvista.plotting.utilities import xvfb

logging.basicConfig(level=logging.INFO)

petsc4py.init(sys.argv)
log.set_log_level(log.LogLevel.WARNING)

comm         = MPI.COMM_WORLD
AIRY         = 0
TRANSVERSE   = 1
TEST_INVALID = False
TEST_PASSED  = False
ENENRGY_PERC_ERR_THRESHOLD = 1.0

with open("parameters.yml") as f: parameters = yaml.load(f, Loader=yaml.FullLoader)

model_rank = 0
tdim = 2

# Generate Mesh object
gmsh_model, tdim = mesh_circle_gmshapi(parameters["geometry"]["geom_type"], parameters["geometry"]["R"], parameters["geometry"]["mesh_size"], tdim)

mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, model_rank, tdim)

outdir = "output"
dirTest = os.path.join(outdir, "Test - General distribution of disclinations")

if comm.rank == 0: Path(dirTest).mkdir(parents=True, exist_ok=True)

# Load Elastic / Geometric Parameters
nu = parameters["model"]["nu"]
h = parameters["model"]["thickness"]
E = parameters["model"]["E"]
D = (E*h**3)/(12*(1-nu**2))

if parameters["geometry"]["geom_type"] != "circle":
    TEST_INVALID = True
    REASON       = "Domain is not a circle. Please check parameters.yml file / geometry / geom_type"

if TEST_INVALID:
    print("Test is invalid")
    print("Reason: ", REASON)
    exit(1)

print(" ")
print("***************** TEST INFO *****************")
print("3D Young's modulus: ", E)
print("Poisson coefficient: ", nu)
print("Plate's thickness: ", h)
print("Domain geometry: ", parameters["geometry"]["geom_type"])
print("Domain radius: ", parameters["geometry"]["R"])
print("Mesh size: ", parameters["geometry"]["mesh_size"])
print("Stabilization term: ", parameters["model"]["alpha_penalty"])
print("Finite Element polinomial order: ", parameters["model"]["order"])
print("***************** TEST BEGINS *****************")
print(" ")

# Data
disclination_points_list = [[0.0, 0.3, 0.0], [0.3, 0.0, 0.0], [0.0, -0.3, 0.0], [-0.3, 0.0, 0.0]]
disclination_power_list  = [-1, 1, -1, +1]

# Define Function Space
fe  = basix.ufl.element("P", str(mesh.ufl_cell()), parameters["model"]["order"])
fes = dolfinx.fem.functionspace(mesh, basix.ufl.mixed_element([fe, fe]))

q_exact          = dolfinx.fem.Function(fes)
v_exact, w_exact = q_exact.split()

def green_function_unit_disc(x, x0):
    """
    Green function of the following problem
        Omega is the unit 2-ball centered in the origin
        Bilaplacian(green) = delta(x-x0) in Omega.
        green = 0 on the boundary
        normal derivative(green) = 0 on the boundary

    where delta(x-x0) is a Dirac delta centered at x0
    """
    rq = (x[0]**2.0 + x[1]**2.0)
    rq0 = (x0[0]**2.0 + x0[1]**2.0)
    a = ( x[0] - x0[0] )**2.0
    b = ( x[1] - x0[1] )**2.0
    c = 2.0 * ( x[0]*x0[0] + x[1]*x0[1] )
    d = rq * rq0
    return (1/(16.0*np.pi))*( (rq - 1.0)*(rq0 - 1.0) + ( a + b )*( np.log( a + b ) - np.log( d + 1.0 - c  ) )  )

def green_function_disc(x, x0, R):
    """
    Green function of the following problem
        Omega is the 2-ball of radius R centered in the origi
        Bilaplacian(green) = delta(x-x0) in Omega.
        green = 0 on the boundary
        normal derivative(green) = 0 on the boundary

    where delta(x-x0) is a Dirac delta centered at x0
    """
    x_1 = [x[0]/R, x[1]/R]
    x0_1 = [x0[0]/R, x0[1]/R]
    return (R**2.0)*green_function_unit_disc(x_1, x0_1)

def _v_exact(x):
    _v = 0.0
    for index, disc_coord in enumerate(disclination_points_list):
        _v += disclination_power_list[index]*green_function_disc(x, disc_coord, parameters["geometry"]["R"])
    return _v * (E*h)

def _w_exact(x):
    _w = (1 - x[0]**2 - x[1]**2)**2
    _w = 0.0*_w
    return _w

def compute_exact_energy():
    """
    The exact energy is computed using the following formula:
    e = 0.5*E*h*s*<v, theta> = 0.5*E*h*s*( v(x0_{1}) + v(x0_{2}) + ... + v(x0_{N}) )
    where
        theta is a discrete distribution of disclinations
        E: 3D Young's modulus
        h: Plate's Thickness
        s: Disclination Frank's Angle
        v: Airy's stress function
        <.,.> Duality product
        x0_{i}: support of the i-th disclination

    Note: evaluating v(x0) returns "nan". v is not defined on the disclination, but it has a well defined limit.
    Then we introduce the eps parameter to approximate v(x0) as v(x0+eps)
    """
    energy = 0.0
    eps = 1E-6
    for index, disc_coord in enumerate(disclination_points_list):
        disc_coord_apprx = [disc_coord[0]+eps, disc_coord[1]+eps]
        energy += E*h*0.5*disclination_power_list[index]*_v_exact(disc_coord_apprx)
    return energy

v_exact.interpolate(_v_exact)
w_exact.interpolate(_w_exact)

exact_energy = compute_exact_energy()

# Compute FE approximated solution
plate = Fvk_plate(mesh, parameters["model"])
plate.set_out_dir()
plate.set_dirichlet_bc_airy(0)
plate.set_dirichlet_bc_trans_displ(0)
plate.set_disclination(disclination_points_list, disclination_power_list)
plate.compute()
plate.plot_profiles(parameters["geometry"]["R"])

#plate.plot_3D_solutions()

w = plate.get_tranverse_disp()
v = plate.get_airy()


# Compute Error
v_diff = dolfinx.fem.Function(fes)
v_diff.x.array[:] = np.abs( v.x.array - v_exact.x.array )
v_exact_max = v_exact.vector.max()[1] if v_exact.vector.max()[1] > -v_exact.vector.min()[1] else -v_exact.vector.min()[1]
if v_exact_max != 0: error = 100.0*v_diff.vector.max()[1]/v_exact_max

energy_absolute_error = exact_energy - plate.get_membrane_energy_value()
energy_percent_error  = 100*energy_absolute_error/exact_energy

if np.abs(energy_percent_error) < ENENRGY_PERC_ERR_THRESHOLD: TEST_PASSED = True

print(" ")
print("***************** TEST RESULTS *****************")
print(" ")
print("Plate's membrane energy: ", plate.get_membrane_energy_value())
print("Plate's bending energy: ", plate.get_bending_energy_value())
print("Plate's coupling energy: ", plate.get_coupling_energy_value())
print("Plate's penalization energy: ", plate.get_penalization_energy_value())
print(" ")
print("Exact dipole energy: ", exact_energy)
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

"""
V_v, dofs_v = fes.sub(AIRY).collapse()
V_w, dofs_w = fes.sub(TRANSVERSE).collapse()

plt.figure()
ax = plot_mesh(mesh)
fig = ax.get_figure()
fig.savefig(f"{dirTest}/mesh.png")

xvfb.start_xvfb(wait=0.05)
pyvista.OFF_SCREEN = True

plotter = pyvista.Plotter(title="Displacement", window_size=[1200, 600], shape=(2, 2))

scalar_plot = plot_scalar(v, plotter, subplot=(0, 0), V_sub=V_v, dofs=dofs_v)
scalar_plot = plot_scalar(w, plotter, subplot=(0, 1), V_sub=V_w, dofs=dofs_w)

scalar_plot = plot_scalar(v_exact, plotter, subplot=(1, 0), V_sub=V_v, dofs=dofs_v)
scalar_plot = plot_scalar(w_exact, plotter, subplot=(1, 1), V_sub=V_w, dofs=dofs_w)

scalar_plot.screenshot(f"{dirTest}/test_fvk.png")

tol = 1e-3
xs = np.linspace(-parameters["geometry"]["R"] + tol, parameters["geometry"]["R"] - tol, 202)
points = np.zeros((3, 202))
points[0] = xs

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
