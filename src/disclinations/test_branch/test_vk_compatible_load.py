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

from ufl import (avg, ds, dS, outer, div, grad, inner, dot, jump)
from dolfinx.fem import FunctionSpace, Function, Expression, assemble_scalar, form

from disclinations.models import NonlinearPlateFVK
from disclinations.models.Fvk_plate import Fvk_plate
from disclinations.meshes import mesh_bounding_box
from disclinations.meshes.primitives import mesh_circle_gmshapi
from disclinations.utils.viz import plot_scalar, plot_profile, plot_mesh
from disclinations.solvers import SNESSolver, SNESProblem
from disclinations.utils.sample_function import sample_function, interpolate_sample

logging.basicConfig(level=logging.INFO)

petsc4py.init(sys.argv)
log.set_log_level(log.LogLevel.WARNING)

comm       = MPI.COMM_WORLD
AIRY       = 0
TRANSVERSE = 1

with open("parameters.yml") as f: parameters = yaml.load(f, Loader=yaml.FullLoader)

#parameters["geometry"]["geom_type"] = "circle"

model_rank = 0
tdim = 2

# Generate Mesh object
gmsh_model, tdim = mesh_circle_gmshapi(
    parameters["geometry"]["geom_type"], parameters["geometry"]["R"], parameters["geometry"]["mesh_size"], tdim
)
mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, model_rank, tdim)

outdir = "output"
#prefix = os.path.join(outdir, "Compatible Plate with Transverse Pressure")
dirTest = os.path.join(outdir, "Test - Compatible plate with transverse pressure")

if comm.rank == 0:
    Path(dirTest).mkdir(parents=True, exist_ok=True)


# Load Elastic / Geometric Parameters
nu = parameters["model"]["nu"]
h = parameters["model"]["thickness"]
E = parameters["model"]["E"]
D = (E*h**3)/(12*(1-nu**2))


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

# Define Function Space
fe  = basix.ufl.element("P", str(mesh.ufl_cell()), parameters["model"]["order"])
fes = dolfinx.fem.functionspace(mesh, basix.ufl.mixed_element([fe, fe]))

load = dolfinx.fem.Function(fes.sub(TRANSVERSE).collapse()[0])

q_exact = dolfinx.fem.Function(fes)
v_exact, w_exact = q_exact.split()

def transverse_load(x):
    f_scale = math.sqrt(2*D**3/(E*h))
    _f = (40/3) * (1 - x[0]**2 - x[1]**2)**4 + (16/3) * (11 + x[0]**2 + x[1]**2)
    return _f * f_scale


def _v_exact(x):
    v_scale = D
    a1=-1/12
    a2=-1/18
    a3=-1/24
    _v = a1 * (1 - x[0]**2 - x[1]**2)**2 + a2 * (1 - x[0]**2 - x[1]**2)**3 + a3 * (1 - x[0]**2 - x[1]**2)**4
    _v = _v * v_scale
    return _v

def _w_exact(x):
    w_scale = math.sqrt(2*D/(E*h))
    _w = (1 - x[0]**2 - x[1]**2)**2
    return _w*w_scale

def exact_bending_energy(w):
    laplacian = lambda f : ufl.div(grad(f))
    hessian = lambda f : ufl.grad(ufl.grad(f))

    #assemble_scalar(form( ufl.dot(ufl.grad(v_exact), ufl.grad(v_exact)) * ufl.dx ))
    return assemble_scalar( form( (D*nu/2 * ufl.inner(laplacian(w), laplacian(w)) + D*(1-nu)/2 * (ufl.inner(hessian(w), hessian(w))) )* ufl.dx) )

def exact_membrane_energy(v):
    laplacian = lambda f : div(grad(f))
    hessian = lambda f : grad(grad(f))
    return assemble_scalar( form( ( ((1+nu)/(2*E*h)) * ufl.inner(hessian(v), hessian(v)) - nu/(2*E*h) * ufl.inner(laplacian(v), laplacian(v)) ) * ufl.dx ) )


def exact_coupling_energy(v, w):
    laplacian = lambda f : div(grad(f))
    hessian = lambda f : grad(grad(f))
    cof = lambda v : ufl.Identity(2)*laplacian(v) - hessian(v)
    return assemble_scalar( form( 0.5* ufl.inner(cof(v), ufl.outer(grad(w),grad(w)) ) * ufl.dx ) )

v_exact.interpolate(_v_exact)
w_exact.interpolate(_w_exact)
load.interpolate(transverse_load)

plate = Fvk_plate(mesh, parameters["model"])
plate.set_out_dir()
plate.set_dirichlet_bc_airy(0)
plate.set_dirichlet_bc_trans_displ(0)
plate.set_load(load)
plate.compute()
plate.plot_profiles(parameters["geometry"]["R"])
w = plate.get_tranverse_disp()
v = plate.get_airy()

fem_membrane_e = plate.get_membrane_energy_value()
fem_bending_e  = plate.get_bending_energy_value()
fem_coupling_e = plate.get_coupling_energy_value()

exact_membrane_e = exact_membrane_energy(v_exact)
exact_bending_e  = exact_bending_energy(w_exact)
exact_coupling_e = exact_coupling_energy(v_exact, w_exact)

abs_membrane_e_error = exact_membrane_e - fem_membrane_e
abs_bending_e_error  = exact_bending_e  - fem_bending_e
abs_coupling_e_error = exact_coupling_e - fem_coupling_e

percent_membrane_e_error = 100*abs_membrane_e_error/exact_membrane_e
percent_bending_e_error  = 100*abs_bending_e_error/exact_bending_e
percent_coupling_e_error = 100*abs_coupling_e_error/exact_coupling_e

print(" ")
print("***************** TEST RESULTS *****************")
print(" ")
print("Plate's membrane energy: "    , fem_membrane_e)
print("Plate's bending energy: "     , fem_bending_e)
print("Plate's coupling energy: "    , fem_coupling_e)
print("Plate's penalization energy: ", plate.get_penalization_energy_value())
print(" ")
print("Exact membrane energy: ", exact_membrane_e )
print("Exact bending energy: " , exact_bending_e )
print("Exact coupling energy: ", exact_coupling_e )
print(" ")
print("Absolute membrane energy Error: ", abs_membrane_e_error)
print("Absolute bending energy Error: ", percent_bending_e_error)
print("Absolute coupling energy Error: ", percent_coupling_e_error)
print(" ")
print("Percent membrane energy Error: ", percent_membrane_e_error, " %")
print("Percent bending energy Error: " , percent_bending_e_error, " %")
print("Percent coupling energy Error: ", percent_coupling_e_error, " %")
print(" ")
#if TEST_PASSED: print("Test Passed")
#else: print("Test Failed")
print(" ")
print(" ")


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

#pdb.set_trace()

# SURFACE PLOTS
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
print("plotted scalar")

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
