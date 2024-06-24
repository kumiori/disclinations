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

#mesh = dolfinx.mesh.refine(mesh)
outdir = "output"
dirTest = os.path.join(outdir, "Test - Disclination dipole")

if comm.rank == 0: Path(dirTest).mkdir(parents=True, exist_ok=True)

# Load Elastic / Geometric Parameters
nu = parameters["model"]["nu"]
h = parameters["model"]["thickness"]
E = parameters["model"]["E"]
D = (E*h**3)/(12*(1-nu**2))

if parameters["geometry"]["geom_type"] != "circle":
    TEST_INVALID = True
    REASON       = "Domain is not a circle. Please check parameters.yml file / geometry / geom_type"

if parameters["geometry"]["geom_type"] == "circle" and parameters["geometry"]["R"] != 1.:
    TEST_INVALID = True
    REASON       = "Radius is not 1. Please check parameters.yml file / geometry / R"

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
disclination_points_list = [[0.2,0.0,0.0], [-0.2,0.0,0.0]]
distance = math.sqrt( (disclination_points_list[0][0] - disclination_points_list[1][0])**2 + (disclination_points_list[0][1] - disclination_points_list[1][1])**2 )
disclination_power_list = [1, -1]

# Define Function Space
fe  = basix.ufl.element("P", str(mesh.ufl_cell()), parameters["model"]["order"])
fes = dolfinx.fem.functionspace(mesh, basix.ufl.mixed_element([fe, fe]))

q_exact          = dolfinx.fem.Function(fes)
v_exact, w_exact = q_exact.split()

exact_energy_dipole = (disclination_power_list[0]**2) * (1.0 / (8 * np.pi)) *  distance**2 * (np.log(4+distance**2) - np.log(4 * distance))

def _v_exact(x):
    rq = (x[0]**2 + x[1]**2)
    _v = (1/(16*np.pi))*( ( x[1]**2 + (x[0]-distance/2)**2 )*( np.log(4.0) + np.log( x[1]**2 + (x[0]-distance/2)**2 ) - np.log( 4 - 4*x[0]*distance + rq*distance**2 ) ) - (1/4)*( 4*(x[1]**2) + ( 2*x[0]+distance)**2 ) * ( np.log(4) + np.log( x[1]**2 + (x[0]+distance/2)**2 ) - np.log( 4 + 4*x[0]*distance + rq*distance**2 ) ) )
    return _v * (E*h*disclination_power_list[0])

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
plate.set_disclination(disclination_points_list, disclination_power_list)
plate.compute()
plate.plot_profiles(parameters["geometry"]["R"])
w = plate.get_tranverse_disp()
v = plate.get_airy()

v_diff = dolfinx.fem.Function(fes)
v_diff.x.array[:] = np.abs( v.x.array - v_exact.x.array )
v_exact_max = v_exact.vector.max()[1] if v_exact.vector.max()[1] > -v_exact.vector.min()[1] else -v_exact.vector.min()[1]
if v_exact_max != 0: error = 100.0*v_diff.vector.max()[1]/v_exact_max

energy_absolute_error = exact_energy_dipole - plate.get_membrane_energy_value()
energy_percent_error  = 100*energy_absolute_error/exact_energy_dipole

if np.abs(energy_percent_error) < ENENRGY_PERC_ERR_THRESHOLD: TEST_PASSED = True

print(" ")
print("***************** TEST RESULTS *****************")
print(" ")
print("Plate's membrane energy: ", plate.get_membrane_energy_value())
print("Plate's bending energy: ", plate.get_bending_energy_value())
print("Plate's coupling energy: ", plate.get_coupling_energy_value())
print("Plate's penalization energy: ", plate.get_penalization_energy_value())
print(" ")
print("Exact dipole energy: ", exact_energy_dipole)
print(" ")
print("Absolute Error: ", )
print("Percent Error: ", energy_percent_error, " %")
print(" ")
if TEST_PASSED: print("Test Passed")
else: print("Test Failed")
print(" ")
print("***********************************************")

# PLOTS
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
