# Solving a simple Poisson problem
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
from disclinations.solvers import SNESSolver, SNESProblem
from disclinations.utils.la import compute_cell_contributions
from disclinations.utils.viz import plot_mesh

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
from ufl import (
    CellDiameter,
    FacetNormal,
    TestFunction,
    TrialFunction,
    avg,
    ds,
    dS,
    dx,
    div,
    grad,
    inner,
    dot,
    jump,
    outer,
    as_matrix,
    sym,
)

import math

import gc

petsc4py.init(sys.argv)
log.set_log_level(log.LogLevel.WARNING)

comm = MPI.COMM_WORLD

def monitor(snes, it, norm):
    logging.info(f"Iteration {it}, residual {norm}")
    return PETSc.SNES.ConvergedReason.ITERATING


with open("parameters.yml") as f:
    parameters = yaml.load(f, Loader=yaml.FullLoader)

mesh_size = parameters["geometry"]["mesh_size"]
#mesh_size = .5
parameters["geometry"]["radius"] = 1
parameters["geometry"]["geom_type"] = "circle"
order = parameters["model"]["order"]

model_rank = 0
tdim = 2

gmsh_model, tdim = mesh_circle_gmshapi(
    parameters["geometry"]["geom_type"], 1, mesh_size, tdim
)
mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, model_rank, tdim)

X = ufl.FiniteElement("CG", mesh.ufl_cell(), 1)
V = dolfinx.fem.functionspace(mesh, ("Lagrange", order))

mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
bndry_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)

dofs = dolfinx.fem.locate_dofs_topological(V=V, entity_dim=1, entities=bndry_facets)

bcs = [dolfinx.fem.dirichletbc(value=np.array(0, dtype=PETSc.ScalarType), dofs=dofs, V=V)]

D = dolfinx.fem.Constant(mesh, 1.)
#α = dolfinx.fem.Constant(mesh, 10.)
load = dolfinx.fem.Constant(mesh, 0.)
h = ufl.CellDiameter(mesh)
h_avg = (h('+') + h('-')) / 2.0

tdim = mesh.topology.dim
num_cells = mesh.topology.index_map(tdim).size_local
_h = dolfinx.cpp.mesh.h(mesh._cpp_object, tdim, np.arange(num_cells, dtype=np.int32))
# print(_h)

n = ufl.FacetNormal(mesh)

dx = ufl.Measure("dx")
dS = ufl.Measure("dS")


x_disclination = 0.2
y_disclination = 0.0
z_disclination = 0.0
dipole_arm = 0.4
dipole_elastic_energy = (1/D.value.take(0))*(dipole_arm**2)/(16*math.pi)*( math.log((4+dipole_arm**2)**2) - math.log(16*dipole_arm**2) )
#pdb.set_trace()

def computeEnergy(alpha, u):
    bending = (D/2 * (inner(div(grad(u)), div(grad(u))))) * dx
    W_ext = load * u * dx

    # Point sources
    b = dolfinx.fem.Function(V)
    b.x.array[:] = 0

    if mesh.comm.rank == 0:
        # point = np.array([[0.68, 0.36, 0]], dtype=mesh.geometry.x.dtype)

        points = [np.array([[-x_disclination, y_disclination, z_disclination]], dtype=mesh.geometry.x.dtype),
                np.array([[x_disclination, y_disclination, z_disclination]], dtype=mesh.geometry.x.dtype)]
        signs = [-1, 1]
    else:
        # point = np.zeros((0, 3), dtype=mesh.geometry.x.dtype)
        points = [np.zeros((0, 3), dtype=mesh.geometry.x.dtype),
                np.zeros((0, 3), dtype=mesh.geometry.x.dtype)]


    dg1 = lambda u: 1/2 * dot(jump(grad(u)), avg(grad(grad(u)) * n)) * dS

    dg2 = lambda u: 1/2 * alpha/avg(h) * inner(jump(grad(u)), jump(grad(u))) * dS
    dg3 = lambda u: 1/2 * alpha/h * inner(grad(u), grad(u)) * ds


    L = bending - dg1(u) + dg2(u) + dg3(u) - W_ext
    F = ufl.derivative(L, u, ufl.TestFunction(V))

    solver = SNESProblem(F, u, bcs, monitor = monitor)

    solver.snes.solve(None, u.vector)
    print(solver.snes.getConvergedReason())

    # cells, basis_values = compute_cell_contribution_point(V, point)
    _cells, _basis_values = compute_cell_contributions(V, points)

    # Q_0, Q_0_to_Q_dofs = Q.sub(0).collapse()
    # https://fenicsproject.discourse.group/t/meaning-of-collapse/10641/2

    for cell, basis_value, sign in zip(_cells, _basis_values, signs):
        # subspace_dofs = Q_0.dofmap.cell_dofs(cell)
        # dofs = np.array(Q_0_to_Q_dofs)[subspace_dofs]
        dofs = V.dofmap.cell_dofs(cell)
        b.x.array[dofs] += sign * basis_value

    dolfinx.fem.petsc.apply_lifting(b.vector, [solver.J_form], [bcs])
    b.vector.ghostUpdate(addv=PETSc.InsertMode.ADD,
                        mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.petsc.set_bc(b.vector, bcs)
    # dolfinx.fem.petsc.set_bc(b.vector, bcs, -1.)
    # b.x.scatter_forward()
    b.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                        mode=PETSc.ScatterMode.FORWARD)



    #plt.figure() # CFe: Added
    #ax = plot_mesh(mesh) # CFe: Added
    #fig = ax.get_figure() # CFe: Added

    #for point in points: # CFe: Added
    #    ax.plot(point[0][0], point[0][1], 'ro') # CFe: Added

    #fig.savefig(f"output/coarse-mesh.png") # CFe: Added


    # now b has to be added into the residual
    solver = SNESSolver(
        F_form=F,
        u=u,
        bcs=bcs,
        petsc_options=parameters.get("solvers").get("elasticity").get("snes"),
        prefix='bilaplacian_disclination',
        b0=b.vector
    )
    solver.solve()

    energy_value = dolfinx.fem.assemble_scalar(dolfinx.fem.form(bending)) # CFe: Added

    del bending, W_ext, b, points, L, F, solver
    gc.collect()

    return u, energy_value


import psutil
import os

# Function to check memory usage
def check_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Memory usage: {mem_info.rss / (1024 ** 2)} MB")

import numpy as np
alphaList = list(np.linspace(1.0, 1000.0, num=20))
uhList = []
energy_valueList = []
energy_errorList = []
energy_percentErrorList = []
for alpha in alphaList:
    u = dolfinx.fem.Function(V)
    check_memory()
    try:
        uh, energy = computeEnergy(alpha, u)

    except Exception as e:
        logging.exception("An error occurred: ", e)
    check_memory()
    print("Alpha: ", alpha)
    print("Elastic Energy: ", energy)
    print("Energy Error: ", energy-dipole_elastic_energy)
    uhList.append(uh)
    energy_valueList.append(energy)
    energy_errorList.append(energy-dipole_elastic_energy)
    energy_percentErrorList.append(100*(energy-dipole_elastic_energy)/dipole_elastic_energy)

print("alphaList: ", alphaList) # CFe: Added
print("****************************************")
print("energy_valueList: ", energy_valueList) # CFe: Added
print("****************************************")
print("energy_errorList: ", energy_errorList)
print("****************************************")
print("energy_percentErrorList: ", energy_percentErrorList)
print("****************************************")
print("dipole_elastic_energy: ", dipole_elastic_energy)
import matplotlib.pyplot as plt
plt.figure(figsize=(20, 10))
plt.plot(alphaList, energy_valueList, linewidth="4")
plt.axhline(y=dipole_elastic_energy, color='red', linewidth="4", linestyle='--', label=f'Exact Energy value = {dipole_elastic_energy}')
#plt.xscale("log")
plt.xlabel("Stabilization Term: Alpha", fontsize=25)
plt.ylabel("Energy", fontsize=25)
plt.xticks(ticks=np.arange(0, max(alphaList)+1, 1), fontsize=20)
plt.yticks(fontsize=20) #weight='bold'
plt.grid(True)
title = "Elastic Energy vs Stabilization Parameter (Alpha) - Mesh size: "+str(mesh_size)+" load: "+str(load.value.take(0))
plt.title(title, fontsize=20, weight='bold')
figureName = "Energy_biharmonic_disclination_alpha - Mesh size: "+str(mesh_size)+" alpha: "+str(max(alphaList))+"lin scale"+".png"
plt.savefig(figureName)

plt.figure(figsize=(20, 10))
plt.plot(alphaList, energy_percentErrorList, linewidth="4")
plt.axhline(y=0, color='red', linewidth="4", linestyle='--', label=f'Zero error')
#plt.xscale("log")
plt.xticks(ticks=np.arange(0, max(alphaList)+1, 1), fontsize=20, weight='bold')
plt.yticks(fontsize=20)
plt.xlabel("Stabilization Term: Alpha", fontsize=25)
plt.ylabel("Percent Energy Error [%]", fontsize=25)
plt.grid(True)
title = "Elastic Energy Error vs Stabilization Parameter (Alpha) - Mesh size: "+str(mesh_size)+" load: "+str(load.value.take(0))
plt.title(title, fontsize=20, weight='bold')
figureName = "Energy_Error_biharmonic_disclination_alpha- Mesh size: "+str(mesh_size)+" - alpha: "+str(max(alphaList))+" lin scale"+".png"
plt.savefig(figureName)


plt.figure(figsize=(20, 10))
plt.plot(alphaList, energy_valueList, linewidth="4")
plt.axhline(y=dipole_elastic_energy, color='red', linewidth="4", linestyle='--', label=f'Exact Energy value = {dipole_elastic_energy}')
plt.xscale("log")
plt.xlabel("Stabilization Term: Alpha", fontsize=25)
plt.ylabel("Energy", fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20) #weight='bold'
plt.grid(True)
title = "Elastic Energy vs Stabilization Parameter (Alpha) - Mesh size: "+str(mesh_size)+" load: "+str(load.value.take(0))
plt.title(title, fontsize=20, weight='bold')
figureName = "Energy_biharmonic_disclination_alpha - Mesh size: "+str(mesh_size)+" alpha: "+str(max(alphaList))+"logscale"+".png"
plt.savefig(figureName)

plt.figure(figsize=(20, 10))
plt.plot(alphaList, energy_percentErrorList, linewidth="4")
plt.axhline(y=0, color='red', linewidth="4", linestyle='--', label=f'Zero error')
plt.xscale("log")
plt.xticks(fontsize=20, weight='bold')
plt.yticks(fontsize=20)
plt.xlabel("Stabilization Term: Alpha", fontsize=25)
plt.ylabel("Percent Energy Error [%]", fontsize=25)
plt.grid(True)
title = "Elastic Energy Error vs Stabilization Parameter (Alpha) - Mesh size: "+str(mesh_size)+" load: "+str(load.value.take(0))
plt.title(title, fontsize=20, weight='bold')
figureName = "Energy_Error_biharmonic_disclination_alpha- Mesh size: "+str(mesh_size)+" alpha: "+str(max(alphaList))+" log scale"+".png"
plt.savefig(figureName)

#pdb.set_trace() # CFe: Added

# solver = SNESProblem(F, u, bcs, monitor = monitor)
# solver.snes.solve(None, q.vector)
# print(solver.snes.getConvergedReason())


# from disclinations.utils.viz import plot_scalar, plot_profile
# from dolfinx import plot
#
# import pyvista
# from pyvista.plotting.utilities import xvfb
# xvfb.start_xvfb(wait=0.05)
# pyvista.OFF_SCREEN = True
#
# plotter = pyvista.Plotter(
#         title="Displacement",
#         window_size=[1600, 600],
#         shape=(1, 2),
#     )
#
# scalar_plot = plot_scalar(u, plotter, subplot=(0, 0), lineproperties={"show_edges": True})
# # plotter.subplot(0, 1)
# _pv_points = np.array([p[0] for p in points])
# _pv_colours = np.array(signs)
# plotter.add_points(
#         _pv_points,
#         scalars = _pv_colours,
#         style = 'points',
#         render_points_as_spheres=True,
#         point_size=15.0
# )
#
# plotter.subplot(0, 1)
# cells, types, x = plot.vtk_mesh(V)
# grid = pyvista.UnstructuredGrid(cells, types, x)
# grid.point_data["u"] = u.x.array.real
# grid.set_active_scalars("u")
# # plotter = pyvista.Plotter()
# # plotter.add_mesh(grid, show_edges=True)
# warped = grid.warp_by_scalar("u", scale_factor=100)
# plotter.add_mesh(warped, show_edges=False)
#
#
#
# # plotter.show_edges=True
# scalar_plot.screenshot("output/test_bilaplacian_disclination.png")
# print("plotted scalar")

