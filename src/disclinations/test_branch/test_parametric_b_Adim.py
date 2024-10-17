"""
PURPOSE OF THE SCRIPT
Run experiments by varying a^2 while keeping constant a^4 * b.
The three FE formulation(Variational, Brenner, Carstensen) in their NON-dimensional form (see "models/adimensional.py") are used.
a := R/h
b := p0/E

ARTICLE RELATED SECTION
Section 6.2: "Parametric study by varying a and b"
"""

import gc
import hashlib
import json
import logging
import os
import pdb
import sys
from pathlib import Path
import basix
import numpy as np
import petsc4py
import pytest
import yaml
from mpi4py import MPI
from petsc4py import PETSc
import pandas as pd
import matplotlib.pyplot as plt
import importlib.resources as pkg_resources  # Python 3.7+ for accessing package files
import copy
from collections import namedtuple

import dolfinx
from dolfinx import fem
from dolfinx.common import list_timings
from dolfinx.fem import Constant, dirichletbc, locate_dofs_topological
from dolfinx.io import XDMFFile, gmshio

import ufl
from ufl import CellDiameter, FacetNormal, dx, sqrt

import disclinations
from disclinations.meshes.primitives import mesh_circle_gmshapi
from disclinations.models import (calculate_rescaling_factors, compute_energy_terms)
from disclinations.models.adimensional import A_NonlinearPlateFVK, A_NonlinearPlateFVK_brenner, A_NonlinearPlateFVK_carstensen
from disclinations.solvers import SNESSolver
from disclinations.utils import (Visualisation, memory_usage, monitor, table_timing_data, write_to_output)
from disclinations.utils.la import compute_disclination_loads
from disclinations.utils import _logger
from disclinations.models import create_disclinations
from disclinations.utils import (homogeneous_dirichlet_bc_H20, load_parameters, save_params_to_yaml)
from disclinations.utils import update_parameters, save_parameters
from disclinations.utils import create_or_load_circle_mesh
from disclinations.utils import table_timing_data, Visualisation

OUTDIR = os.path.join("output", "test_parametric_b_Adim") # CFe: output directory
PATH_TO_PARAMETERS_YML_FILE = 'disclinations.test'

# OUTPUT DIRECTORY
COMM = MPI.COMM_WORLD
if COMM.rank == 0: Path(OUTDIR).mkdir(parents=True, exist_ok=True)

NUM_RUNS = 10
#P0_RANGE = np.logspace(3, 5, NUM_RUNS) # sample NUM_RUNS between 10^3 to 10^6
P0_RANGE = np.linspace(10, 1000, NUM_RUNS) # load p0
LOG_SCALE = False #True

DISCLINATION_COORD_LIST = [[0.0, 0.0, 0]]
DISCLINATION_POWER_LIST = [1]

AIRY               = 0
TRANSVERSE = 1

VARIATIONAL  = 0
BRENNER         = 1
CARSTENSEN   = 2

HESSIAN_V  = 0
HESSIAN_W = 1
AIRY_V         = 2
TRANSV_W  = 3
CONV_ID      = 4

CostantData = namedtuple("CostantData", ["funcSpace", "bcs"])

petsc4py.init(["-petsc_type", "double"]) # CFe: ensuring double precision numbers

def L2norm(u): return np.sqrt( fem.assemble_scalar( fem.form( ufl.inner(u, u) * ufl.dx ) ) )

def hessianL2norm(u):
    hessian = lambda f : ufl.grad(ufl.grad(u))
    return np.sqrt( fem.assemble_scalar( fem.form( ufl.inner(hessian(u), hessian(u)) * ufl.dx ) ) )


def solveModel(FEMmodel, costantData, parameters, f):

    # Update parameter a
    a = parameters["geometry"]["radius"]/parameters["model"]["thickness"]

    # FUNCTION SPACE
    q = dolfinx.fem.Function(costantData.funcSpace)
    v, w = ufl.split(q)
    state = {"v": v, "w": w}

    # SELECT MODEL
    if FEMmodel == CARSTENSEN:
        model = A_NonlinearPlateFVK_carstensen(mesh, parameters["model"])
    elif FEMmodel == BRENNER:
        model = A_NonlinearPlateFVK_brenner(mesh, parameters["model"])
    elif FEMmodel == VARIATIONAL:
        model = A_NonlinearPlateFVK(mesh, parameters["model"])
    else:
        print(f"FEMmodel = {FEMmodel} is not acceptable. Script exiting")
        exit(0)

    # INSERT DISCLINATIONS
    disclinations = []
    if mesh.comm.rank == 0:
        for dc in DISCLINATION_COORD_LIST: disclinations.append( np.array([dc], dtype=mesh.geometry.x.dtype))
        dp_list = [dp*(a)**2 for dp in DISCLINATION_POWER_LIST]
    else:
        for dc in DISCLINATION_COORD_LIST: disclinations.append( np.zeros((0, 3), dtype=mesh.geometry.x.dtype) )
    Q_v, Q_v_to_Q_dofs = Q.sub(AIRY).collapse()
    print("dp_list: ", dp_list)
    b = compute_disclination_loads(disclinations, dp_list, Q, V_sub_to_V_dofs=Q_v_to_Q_dofs, V_sub=Q_v)

    # WEAK FEM FORMULATION
    W_ext = f * w * dx
    model.W_ext = W_ext
    energy = model.energy(state)[0]
    penalisation = model.penalisation(state)
    L = energy - W_ext + penalisation
    F = ufl.derivative(L, q, ufl.TestFunction(Q))
    if FEMmodel is not VARIATIONAL:
        test_v, test_w = ufl.TestFunctions(Q)[AIRY], ufl.TestFunctions(Q)[TRANSVERSE]
        F = F + model.coupling_term(state, test_v, test_w)

    # SOLVER INSTANCE
    solver_parameters = {
        "snes_type": "newtonls",  # Solver type: NGMRES (Nonlinear GMRES)
        "snes_max_it": 200,  # Maximum number of iterations
        "snes_rtol": 1e-12,  # Relative tolerance for convergence
        "snes_atol": 1e-12,  # Absolute tolerance for convergence
        "snes_stol": 1e-12,  # Tolerance for the change in solution norm
        "snes_monitor": None,  # Function for monitoring convergence (optional)
        "snes_linesearch_type": "basic",  # Type of line search
    }

    solver = SNESSolver(
        F_form=F,
        u=q,
        bcs=costantData.bcs,
        bounds=None,
        petsc_options=solver_parameters,
        prefix="plate_fvk",
        b0=b.vector,
    )

    # run solver
    n_iterations, convergence_id = solver.solve()

    return q, convergence_id


def run_experiment(FEMmodel, costantData, parameters, b):
    """
    Purpose: parametric study of the solution by varying b and keeping a constant
    """
    thickness = parameters["model"]["thickness"]
    E = parameters["model"]["E"]
    v_scale = E*(thickness**3)
    w_scale = thickness
    a = parameters["geometry"]["radius"] / thickness
    print("Running experiment with thickness:", parameters["model"]["thickness"])

    parameters = calculate_rescaling_factors(parameters) # CFe: update exact solution scale factors

    # CFe: update external load
    print("a**4 * b", (a**4)* (b))
    print("a: ", a)
    print("b: ", b)
    def nondim_transverse_load(x): return (a**4)* (b) * (1.0 + 0.0*x[0] + 0.0*x[1])

    Q = costantData.funcSpace
    f = dolfinx.fem.Function(Q.sub(TRANSVERSE).collapse()[0])
    f.interpolate(nondim_transverse_load)

    q, convergence_id = solveModel(FEMmodel, costantData, parameters, f)
    v, w = q.split()
    v_Nrm = L2norm(v)
    w_Nrm = L2norm(w)
    v_hessianNrm = hessianL2norm(v)
    w_hessianNrm = hessianL2norm(w)

    return v_scale*v_hessianNrm, w_scale*w_hessianNrm, v_scale*v_Nrm, w_scale*w_Nrm, convergence_id

if __name__ == "__main__":

    PARAMETER_NAME = "p0"

    EXPERIMENT_DIR = os.path.join(OUTDIR, PARAMETER_NAME)
    if COMM.rank == 0: Path(EXPERIMENT_DIR).mkdir(parents=True, exist_ok=True)

    # CFe: boolean, set it to true to plot with a logscale on the xaxis, False to plot with a linear scale on the xaxis
    #LOG_SCALE = False

    _experimental_data = []
    
    # LOAD PARAMETERS FILE
    with pkg_resources.path(PATH_TO_PARAMETERS_YML_FILE, 'parameters.yml') as f:
        parameters, _ = load_parameters(f)
        base_parameters = copy.deepcopy(parameters)
        # Remove thickness from the parameters, to compute the parametric series signature
        #if PARAMETER_CATEGORY in base_parameters and PARAMETER_NAME in base_parameters[PARAMETER_CATEGORY]:
           # del base_parameters[PARAMETER_CATEGORY][PARAMETER_NAME]
        
        base_signature = hashlib.md5(str(base_parameters).encode('utf-8')).hexdigest()

    b_range = [element/parameters["model"]["E"] for element in P0_RANGE]
    b_list = []

    logging.info(f"===================- {EXPERIMENT_DIR} -=================")

    # LOAD THE MESH
    mesh, mts, fts = create_or_load_circle_mesh(parameters, prefix=OUTDIR)
    h = CellDiameter(mesh)
    n = FacetNormal(mesh)

    # DEFINE THE FUNCTION SPACE
    X = basix.ufl.element("P", str(mesh.ufl_cell()), parameters["model"]["order"])
    Q = dolfinx.fem.functionspace(mesh, basix.ufl.mixed_element([X, X]))

    # DEFINE THE DIRICHLET BOUNDARY CONDITIONS
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    bndry_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    dofs_v = dolfinx.fem.locate_dofs_topological(V=Q.sub(AIRY), entity_dim=1, entities=bndry_facets)
    dofs_w = dolfinx.fem.locate_dofs_topological(V=Q.sub(TRANSVERSE), entity_dim=1, entities=bndry_facets)
    bcs_w = dirichletbc(np.array(0, dtype=PETSc.ScalarType), dofs_w, Q.sub(TRANSVERSE))
    bcs_v = dirichletbc(np.array(0, dtype=PETSc.ScalarType), dofs_v, Q.sub(AIRY))
    _bcs = {AIRY: bcs_v, TRANSVERSE: bcs_w}
    bcs = list(_bcs.values())

    # Values constant across the experiments
    costantData = CostantData(Q, bcs)
    non_converged_list = []

    # PERFORM EXPERIMENTS
    for b in b_range:
        #if changed := update_parameters(parameters, PARAMETER_NAME, param):
        data_var = run_experiment(VARIATIONAL, costantData, parameters, b)
        data_brn = run_experiment(BRENNER, costantData, parameters, b)
        data_car = run_experiment(CARSTENSEN, costantData, parameters, b)

        if (data_var[CONV_ID] > 0) and (data_brn[CONV_ID]  > 0) and (data_car[CONV_ID] > 0):
            b_list.append(b)
            _experimental_data.append([b, data_var[HESSIAN_V], data_brn[HESSIAN_V], data_car[HESSIAN_V],
                                    data_var[HESSIAN_W], data_brn[HESSIAN_W], data_car[HESSIAN_W],
                                    data_var[CONV_ID], data_brn[CONV_ID], data_car[CONV_ID],
                                    data_var[AIRY_V], data_brn[AIRY_V], data_car[AIRY_V],
                                    data_var[TRANSV_W], data_brn[TRANSV_W], data_car[TRANSV_W],
                                    parameters["geometry"]["mesh_size"], parameters["model"]["alpha_penalty"],
                                    parameters["model"]["thickness"], parameters["model"]["E"], parameters["model"]["nu"]])
        else:
            print(f"At least one of the three FEM models did not converged for {PARAMETER_NAME} = {b}")
            print("Convergence Variational model: ", data_var[CONV_ID])
            print("Convergence Brenner model: ", data_brn[CONV_ID])
            print("Convergence Carstensen model: ", data_car[CONV_ID])
            non_converged_list.append({"b": b, "Var_Con_ID": data_var[CONV_ID], "Brn_Con_ID": data_brn[CONV_ID], "Car_Con_ID": data_car[CONV_ID]})
    
    columns = [PARAMETER_NAME, "Hessian L2norm v (Variational)", "Hessian L2norm v (Brenner)", "Hessian L2norm v (Carstensen)",
               "Hessian L2norm w (Variational)", "Hessian L2norm w (Brenner)", "Hessian L2norm w (Carstensen)",
               "Convergence ID (Variational)", "Convergence ID (Brenner)", "Convergence ID (Carstensen)",
               "L2norm v (Variational)", "L2norm v (Brenner)", "L2norm v (Carstensen)",
               "L2norm w (Variational)", "L2norm w (Brenner)", "L2norm w (Carstensen)",
               "Mesh size", "Interior Penalty (IP)", "Thickness", "Young modulus", "Poisson ratio"]


    _logger.info(f"Saving experimental data to {EXPERIMENT_DIR}")
    #pdb.set_trace()

    # EXPORT RESULTS TO EXCEL FILE
    experimental_data = pd.DataFrame(_experimental_data, columns=columns)
    experimental_data.to_excel(f'{EXPERIMENT_DIR}/varying_a_mesh_{parameters["geometry"]["mesh_size"]}_IP_{parameters["model"]["alpha_penalty"]}_E_{parameters["model"]["E"]}.xlsx', index=False)

    # PRINT OUT RESULTS
    print(10*"*")
    print("Results")
    print(experimental_data)
    print(10*"*")
    print("Details on non-converged experiments")
    for el in non_converged_list: print(el)
    print(10*"*")

    # PLOTS L2 NORM
    plt.figure(figsize=(30, 20))
    plt.plot(b_list, experimental_data["L2norm v (Variational)"], marker='o', linestyle='solid', color='b', label='L2norm v (Variational)', linewidth=5, markersize=20)
    plt.plot(b_list, experimental_data["L2norm v (Brenner)"], marker='v', linestyle='dotted', color='r', label='L2norm v (Brenner)', linewidth=5, markersize=20)
    plt.plot(b_list, experimental_data["L2norm v (Carstensen)"], marker='^', linestyle='dashed', color='g', label='L2norm v (Carstensen)', linewidth=5, markersize=20)

    max_v = max([max(experimental_data["L2norm v (Variational)"]), max(experimental_data["L2norm v (Brenner)"]), max(experimental_data["L2norm v (Carstensen)"])])
    min_v = min([min(experimental_data["L2norm v (Variational)"]), min(experimental_data["L2norm v (Brenner)"]), min(experimental_data["L2norm v (Carstensen)"])])
    steps_v = (max_v - min_v)/NUM_RUNS

    if LOG_SCALE: plt.xscale('log') # CFe: use log xscale when needed
    plt.xticks(b_list)
    plt.yticks(np.arange(min_v, max_v, steps_v))
    plt.xlabel("b := p0/E", fontsize=35)
    plt.ylabel(r'$| v |_{L^2(\Omega)} [Nm^2]$', fontsize=35)
    plt.title(f'Airy function. Mesh size = {parameters["geometry"]["mesh_size"]}. IP = {parameters["model"]["alpha_penalty"]}. E = {parameters["model"]["E"]:.2e} Pa. h = {parameters["model"]["thickness"]:.2e} m', fontsize=40)
    plt.tick_params(axis='both', which='major', labelsize=35)
    plt.legend(fontsize=35)
    plt.grid(True)
    plt.gca().yaxis.get_offset_text().set_fontsize(35)
    plt.gca().xaxis.get_offset_text().set_fontsize(35)
    plt.savefig(EXPERIMENT_DIR+f'/varying_b_V_mesh_{parameters["geometry"]["mesh_size"]}_IP_{parameters["model"]["alpha_penalty"]}_E_{parameters["model"]["E"]:.2e}_h_{parameters["model"]["thickness"]:.2e}.png', dpi=300)
    plt.show()

    plt.figure(figsize=(30, 20))
    plt.plot(b_list, experimental_data["L2norm w (Variational)"], marker='o', linestyle='solid', color='b', label='L2norm w (Variational)', linewidth=5, markersize=20)
    plt.plot(b_list, experimental_data["L2norm w (Brenner)"], marker='v', linestyle='dotted', color='r', label='L2norm w (Brenner)', linewidth=5, markersize=20)
    plt.plot(b_list, experimental_data["L2norm w (Carstensen)"], marker='^', linestyle='dashed', color='g', label='L2norm w (Carstensen)', linewidth=5, markersize=20)

    max_w = max([max(experimental_data["L2norm w (Variational)"]), max(experimental_data["L2norm w (Brenner)"]), max(experimental_data["L2norm w (Carstensen)"])])
    min_w = min([min(experimental_data["L2norm w (Variational)"]), min(experimental_data["L2norm w (Brenner)"]), min(experimental_data["L2norm w (Carstensen)"])])
    steps_w = (max_w - min_w)/NUM_RUNS
    if steps_w == 0: steps_w = NUM_RUNS # CFe: if the deflection is not activated

    if LOG_SCALE: plt.xscale('log') # CFe: use log xscale when needed
    plt.xticks(b_list)

    plt.yticks(np.arange(min_w, max_w, steps_w))
    plt.xlabel("b := p0/E", fontsize=35)
    plt.ylabel(r'$| w |_{L^2(\Omega)} [m^2]$', fontsize=35)
    plt.title(f'Transverse displacement. Mesh size = {parameters["geometry"]["mesh_size"]}. IP = {parameters["model"]["alpha_penalty"]}. E = {parameters["model"]["E"]:.2e} Pa. h = {parameters["model"]["thickness"]:.2e} m', fontsize=40)
    plt.tick_params(axis='both', which='major', labelsize=35)
    plt.legend(fontsize=35)
    plt.gca().yaxis.get_offset_text().set_fontsize(35)
    plt.gca().xaxis.get_offset_text().set_fontsize(35)
    plt.grid(True)
    plt.savefig(EXPERIMENT_DIR+f'/varying_b_W_mesh_{parameters["geometry"]["mesh_size"]}_IP_{parameters["model"]["alpha_penalty"]}_E_{parameters["model"]["E"]:.2e}_h_{parameters["model"]["thickness"]:.2e}.png', dpi=300)
    plt.show()

    # PLOTS HESSIAN L2 NORM
    plt.figure(figsize=(30, 20))
    plt.plot(b_list, experimental_data["Hessian L2norm v (Variational)"], marker='o', linestyle='solid', color='b', label='Hessian L2norm v (Variational)', linewidth=5, markersize=20)
    plt.plot(b_list, experimental_data["Hessian L2norm v (Brenner)"], marker='v', linestyle='dotted', color='r', label='Hessian L2norm v (Brenner)', linewidth=5, markersize=20)
    plt.plot(b_list, experimental_data["Hessian L2norm v (Carstensen)"], marker='^', linestyle='dashed', color='g', label='Hessian L2norm v (Carstensen)', linewidth=5, markersize=20)

    max_v = max([max(experimental_data["Hessian L2norm v (Variational)"]), max(experimental_data["Hessian L2norm v (Brenner)"]), max(experimental_data["Hessian L2norm v (Carstensen)"])])
    min_v = min([min(experimental_data["Hessian L2norm v (Variational)"]), min(experimental_data["Hessian L2norm v (Brenner)"]), min(experimental_data["Hessian L2norm v (Carstensen)"])])
    steps_v = (max_v - min_v)/NUM_RUNS

    if LOG_SCALE: plt.xscale('log') # CFe: use log xscale when needed
    plt.xticks(b_list)
    plt.yticks(np.arange(min_v, max_v, steps_v))
    plt.xlabel("b := p0/E", fontsize=35)
    plt.ylabel(r'$| \nabla^2 v |_{L^2(\Omega)} [N]$', fontsize=35)
    plt.title(f'Airy function. Mesh size = {parameters["geometry"]["mesh_size"]}. IP = {parameters["model"]["alpha_penalty"]}. E = {parameters["model"]["E"]:.2e} Pa. h = {parameters["model"]["thickness"]:.2e} m', fontsize=40)
    plt.tick_params(axis='both', which='major', labelsize=35)
    plt.legend(fontsize=35)
    plt.gca().yaxis.get_offset_text().set_fontsize(35)
    plt.gca().xaxis.get_offset_text().set_fontsize(35)
    plt.grid(True)
    plt.savefig(EXPERIMENT_DIR+f'/varying_b_HessV_mesh_{parameters["geometry"]["mesh_size"]}_IP_{parameters["model"]["alpha_penalty"]}_E_{parameters["model"]["E"]:.2e}_h_{parameters["model"]["thickness"]:.2e}.png', dpi=300)
    plt.show()

    plt.figure(figsize=(30, 20))
    plt.plot(b_list, experimental_data["Hessian L2norm w (Variational)"], marker='o', linestyle='solid', color='b', label='Hessian L2norm w (Variational)', linewidth=5, markersize=20)
    plt.plot(b_list, experimental_data["Hessian L2norm w (Brenner)"], marker='v', linestyle='dotted', color='r', label='Hessian L2norm w (Brenner)', linewidth=5, markersize=20)
    plt.plot(b_list, experimental_data["Hessian L2norm w (Carstensen)"], marker='^', linestyle='dashed', color='g', label='Hessian L2norm w (Carstensen)', linewidth=5, markersize=20)

    max_w = max([max(experimental_data["Hessian L2norm w (Variational)"]), max(experimental_data["Hessian L2norm w (Brenner)"]), max(experimental_data["Hessian L2norm w (Carstensen)"])])
    min_w = min([min(experimental_data["Hessian L2norm w (Variational)"]), min(experimental_data["Hessian L2norm w (Brenner)"]), min(experimental_data["Hessian L2norm w (Carstensen)"])])
    steps_w = (max_w - min_w)/NUM_RUNS
    if steps_w == 0: steps_w = NUM_RUNS # CFe: if the deflection is not activated

    if LOG_SCALE: plt.xscale('log') # CFe: use log xscale when needed
    plt.xticks(b_list)

    plt.yticks(np.arange(min_w, max_w, steps_w))
    plt.xlabel("b := p0/E", fontsize=35)
    plt.ylabel(r'$| \nabla^2 w |_{L^2(\Omega)} $', fontsize=35)
    plt.title(f'Transverse displacement. Mesh size = {parameters["geometry"]["mesh_size"]}. IP = {parameters["model"]["alpha_penalty"]}. E = {parameters["model"]["E"]:.2e} Pa. h = {parameters["model"]["thickness"]:.2e} m', fontsize=40)
    plt.tick_params(axis='both', which='major', labelsize=35)
    plt.legend(fontsize=35)
    plt.gca().yaxis.get_offset_text().set_fontsize(35)
    plt.gca().xaxis.get_offset_text().set_fontsize(35)
    plt.grid(True)
    plt.savefig(EXPERIMENT_DIR+f'/varying_b_HessW_mesh_{parameters["geometry"]["mesh_size"]}_IP_{parameters["model"]["alpha_penalty"]}_E_{parameters["model"]["E"]:.2e}_h_{parameters["model"]["thickness"]:.2e}.png', dpi=300)
    plt.show()
