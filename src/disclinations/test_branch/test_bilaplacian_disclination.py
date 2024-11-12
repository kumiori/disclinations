 # Solving a simple biharmonic problem with disclinations
# H^1_0 formulation
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
from disclinations.meshes import mesh_bounding_box
from disclinations.meshes.primitives import mesh_circle_gmshapi

# from damage.utils import ColorPrint
from disclinations.solvers import SNESSolver
from disclinations.utils.la import compute_cell_contributions, compute_disclination_loads
from disclinations.utils.viz import plot_mesh
from disclinations.models import Biharmonic


from dolfinx import log
from dolfinx.io import XDMFFile
from mpi4py import MPI
from petsc4py import PETSc

logging.basicConfig(level=logging.INFO)


from dolfinx.fem.petsc import (assemble_matrix, create_vector, create_matrix, assemble_vector)

import sys

import dolfinx
import dolfinx.io
import dolfinx.mesh
import dolfinx.plot
import ufl
import yaml
from dolfinx.fem import Constant, dirichletbc
from dolfinx.io import XDMFFile, gmshio
from mpi4py import MPI
from ufl import dx
import matplotlib.pyplot as plt

petsc4py.init(sys.argv)
log.set_log_level(log.LogLevel.WARNING)

comm = MPI.COMM_WORLD

def monitor(snes, it, norm):
    logging.info(f"Iteration {it}, residual {norm}")
    return PETSc.SNES.ConvergedReason.ITERATING


with open("parameters.yml") as f:
    parameters = yaml.load(f, Loader=yaml.FullLoader)

mesh_size = parameters["geometry"]["mesh_size"]
parameters["geometry"]["radius"] = 1
parameters["geometry"]["geom_type"] = "circle"
order = parameters["model"]["order"]

model_rank = 0
tdim = 2

outdir = "output"
prefix = os.path.join(outdir, "biharmonic_monopole")

if comm.rank == 0:
    Path(prefix).mkdir(parents=True, exist_ok=True)


gmsh_model, tdim = mesh_circle_gmshapi(
    parameters["geometry"]["geom_type"], 1, mesh_size, tdim
)
mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, model_rank, tdim)

# X = ufl.FiniteElement("CG", mesh.ufl_cell(), 1)
V = dolfinx.fem.functionspace(mesh, ("Lagrange", order))

mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
bndry_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)

dofs = dolfinx.fem.locate_dofs_topological(V=V, entity_dim=1, entities=bndry_facets)

bcs = [dolfinx.fem.dirichletbc(value=np.array(0, dtype=PETSc.ScalarType), dofs=dofs, V=V)]

u = dolfinx.fem.Function(V)
u_exact = dolfinx.fem.Function(V)
state = {"u": u}

D = dolfinx.fem.Constant(mesh, 1.)
α = dolfinx.fem.Constant(mesh, 10.)
load = dolfinx.fem.Constant(mesh, .0)
h = ufl.CellDiameter(mesh)
h_avg = (h('+') + h('-')) / 2.0

tdim = mesh.topology.dim
num_cells = mesh.topology.index_map(tdim).size_local
_h = dolfinx.cpp.mesh.h(mesh._cpp_object, tdim, np.arange(num_cells, dtype=np.int32))

_u_scale = parameters["model"]["E"]

def _u_exact(x, _u_scale = 1, _s = 1, _R = 1):
    r = np.sqrt(x[0]**2 + x[1]**2)
    _u = _s * _u_scale / (8.*np.pi) * (r**2 * np.log(r/_R) - r**2/2. + _R**2/2.)
    
    return _u

u_exact.interpolate(_u_exact)

n = ufl.FacetNormal(mesh)

model = Biharmonic(mesh, parameters["model"])
W_ext = load * u * dx

L = model.energy(state) + model.penalisation(state) - W_ext

dx = ufl.Measure("dx")
dS = ufl.Measure("dS")


# Point sources
if mesh.comm.rank == 0:
    points = np.array([[0., 0., 0]], dtype=mesh.geometry.x.dtype)

    signs = [1]
else:
    point = np.zeros((0, 3), dtype=mesh.geometry.x.dtype)

b = compute_disclination_loads(points, signs, V)

F = ufl.derivative(L, u, ufl.TestFunction(V))

# Q_0, Q_0_to_Q_dofs = Q.sub(0).collapse()
# https://fenicsproject.discourse.group/t/meaning-of-collapse/10641/2

plt.figure()
ax = plot_mesh(mesh)
fig = ax.get_figure()

for point in points:
    ax.plot(point[0], point[1], 'ro')

fig.savefig(f"{prefix}/coarse-mesh.png")


# now b has to be added into the residual
solver = SNESSolver(
    F_form=F,
    u=u,
    bcs=bcs,
    petsc_options=parameters.get("solvers").get("elasticity").get("snes"),
    prefix='bilaplacian_disclination',
    b0=-b.vector
)
solver.solve()


from disclinations.utils.viz import plot_scalar, plot_profile
from dolfinx import plot

import pyvista
from pyvista.plotting.utilities import xvfb
xvfb.start_xvfb(wait=0.05)
pyvista.OFF_SCREEN = True

plotter = pyvista.Plotter(
        title="Displacement",
        window_size=[1600, 600],
        shape=(1, 2),
    )

scalar_plot = plot_scalar(u, plotter, subplot=(0, 0), lineproperties={"show_edges": True})
# plotter.subplot(0, 1)
_pv_points = points
_pv_colours = np.array(signs)
plotter.add_points(
        _pv_points,
        scalars = _pv_colours,
        style = 'points', 
        render_points_as_spheres=True, 
        point_size=15.0
)

plotter.subplot(0, 1)
cells, types, x = plot.vtk_mesh(V)
grid = pyvista.UnstructuredGrid(cells, types, x)
grid.point_data["u"] = u.x.array.real
grid.set_active_scalars("u")
# plotter = pyvista.Plotter()
# plotter.add_mesh(grid, show_edges=True)
warped = grid.warp_by_scalar("u", scale_factor=100)
plotter.add_mesh(warped, show_edges=False)

exact_energy_monopole = signs[0]**2 * parameters["model"]["E"] * parameters["model"]["thickness"] \
    * parameters["geometry"]["R"]**2 / (32 * np.pi)

print(exact_energy_monopole)
# Check the solution
check_components = {
    "energy": model.energy(state),
    "penalisation": model.penalisation(state),
}

# compute distance between points 
check_terms = {label: comm.allreduce(
dolfinx.fem.assemble_scalar(
    dolfinx.fem.form(energy_term)),
op=MPI.SUM,
) for label, energy_term in check_components.items()}

print(check_terms)
print(f"Exact energy: {exact_energy_monopole}")

print(f"Computed energy: {check_terms['energy'] + check_terms['penalisation']}")
# Print error norm
error = np.abs(exact_energy_monopole - check_terms['energy'])
print(f"Abs error: {error}")
print(f"Rel error: {error/exact_energy_monopole:.3%}")
print(f"Error: {error/exact_energy_monopole:.1%}")

# Print alpha value
print(f'Computed alpha: {parameters["model"]["alpha_penalty"]}')
# plotter.show_edges=True
scalar_plot.screenshot(f"{prefix}/test_bilaplacian_disclination.png")
print("plotted scalar")


# ---------------
# Plot profiles


tol = 1e-3
xs = np.linspace(0 + tol, parameters["geometry"]["radius"] - tol, 101)
points = np.zeros((3, 101))
points[0] = xs

fig, axes = plt.subplots(1, 1, figsize=(18, 6))

_plt, data = plot_profile(
    u,
    points,
    None,
    subplot=(1, 1),
    lineproperties={
        "c": "k",
        "label": f"$u(x)$"
    },
    fig=fig,
    subplotnumber=1
)

# ax = _plt.gca()
# axw = ax.twinx()

_plt, data = plot_profile(
    u_exact,
    points,
    None,
    subplot=(1, 1),
    lineproperties={
        "c": "r",
        "label": f"$u_e(x)$",
        "ls": "--"
    },
    fig=fig,
    subplotnumber=1
)
_plt.legend()

_plt.savefig(f"{prefix}/test_fvk-profiles.png")
