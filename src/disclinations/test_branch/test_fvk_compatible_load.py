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
from disclinations.meshes import mesh_bounding_box
from disclinations.meshes.primitives import mesh_circle_gmshapi
from disclinations.utils.viz import plot_scalar, plot_profile, plot_mesh
from disclinations.solvers import SNESSolver, SNESProblem

from Fvk_plate import Fvk_plate

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

print(" ")
print("***************** TEST RESULTS *****************")
print(" ")
print("Plate's membrane energy: "    , plate.get_membrane_energy_value())
print("Plate's bending energy: "     , plate.get_bending_energy_value())
print("Plate's coupling energy: "    , plate.get_coupling_energy_value())
print("Plate's penalization energy: ", plate.get_penalization_energy_value())
print(" ")


# PLOTS ------------------------------
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
