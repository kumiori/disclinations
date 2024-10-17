"""
PURPOSE OF THE SCRIPT
Compare the three FE formulations (Variational, Brenner, Carstensen) with a known solution while varying the plate's thickness.
The script uses the NON-dimensional FE formulations implemented in "models/adimensional.py"

ARTICLE RELATED SECTION
"Test 3: parametric study with different values of h"
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
import importlib.resources as pkg_resources  # Python 3.7+ for accessing package files
import copy
from collections import namedtuple

import ufl
from ufl import CellDiameter, FacetNormal, dx, sqrt

import disclinations
from disclinations.meshes.primitives import mesh_circle_gmshapi
from disclinations.models import (calculate_rescaling_factors, compute_energy_terms)
from disclinations.models.adimensional import A_NonlinearPlateFVK, A_NonlinearPlateFVK_brenner, A_NonlinearPlateFVK_carstensen
from disclinations.models import create_disclinations
from disclinations.solvers import SNESSolver
from disclinations.utils import (Visualisation, memory_usage, monitor,
                                 table_timing_data, write_to_output)
from disclinations.utils import _logger
from disclinations.utils.la import compute_disclination_loads
from disclinations.utils import (homogeneous_dirichlet_bc_H20, load_parameters,
                                 save_params_to_yaml)
from disclinations.utils import update_parameters, save_parameters
from disclinations.utils import create_or_load_circle_mesh
from disclinations.utils import table_timing_data, Visualisation

import dolfinx
from dolfinx import fem
from dolfinx.common import list_timings
from dolfinx.fem import Constant, dirichletbc, locate_dofs_topological
from dolfinx.io import XDMFFile, gmshio

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


prefix = os.path.join("output", "test_parametric_adim")
#series = base_signature[0::6]
PARAMETER_NAME = "thickness"
PARAMETER_CATEGORY = "model"
NUM_RUNS = 19
series = PARAMETER_NAME
experiment_dir = os.path.join(prefix, series)

PARAMETERS_FILE_PATH = 'disclinations.test'

AIRY                = 0
TRANSVERSE = 1

VARIATIONAL  = 0
BRENNER         = 1
CARSTENSEN   = 2

HESSIAN_V  = 0
HESSIAN_W = 1
AIRY_V         = 2
TRANSV_W  = 3
CONV_ID      = 4

comm = MPI.COMM_WORLD
CostantData = namedtuple("CostantData", ["funcSpace", "bcs"])

petsc4py.init(["-petsc_type", "double"]) # CFe: ensuring double precision numbers

def hessianL2norm(u):
    hessian = lambda f : ufl.grad(ufl.grad(u))
    return np.sqrt( fem.assemble_scalar( fem.form( ufl.inner(hessian(u), hessian(u)) * ufl.dx ) ) )

def L2norm(u):
    return np.sqrt( fem.assemble_scalar( fem.form( ufl.inner(u, u) * ufl.dx ) ) )

def solveModel(FEMmodel, costantData, parameters, f):

    # CFe: geometric and elastic data
    _E = parameters["model"]["E"]
    nu = parameters["model"]["nu"]
    thickness = parameters["model"]["thickness"]
    _D = _E * thickness**3 / (12 * (1 - nu**2))

    # CFe: function space data
    q = dolfinx.fem.Function(costantData.funcSpace)
    v, w = ufl.split(q)
    state = {"v": v, "w": w}

    if FEMmodel == CARSTENSEN:
        model = A_NonlinearPlateFVK_carstensen(mesh, parameters["model"])
    elif FEMmodel == BRENNER:
        model = A_NonlinearPlateFVK_brenner(mesh, parameters["model"])
    elif FEMmodel == VARIATIONAL:
        model = A_NonlinearPlateFVK(mesh, parameters["model"])
    else:
        print(f"FEMmodel = {FEMmodel} is not acceptable. Script exiting")
        exit(0)


    W_ext = f * w * dx
    model.W_ext = W_ext
    energy = model.energy(state)[0]

    penalisation = model.penalisation(state)

    L = energy - W_ext + penalisation

    test_v, test_w = ufl.TestFunctions(Q)[AIRY], ufl.TestFunctions(Q)[TRANSVERSE]
    F = ufl.derivative(L, q, ufl.TestFunction(Q))
    if FEMmodel is not VARIATIONAL:
        F = F + model.coupling_term(state, test_v, test_w)

    solver_parameters = {
        "snes_type": "newtonls",  # Solver type: NGMRES (Nonlinear GMRES)
        "snes_max_it": 100,  # Maximum number of iterations
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
    )
    n_iterations, convergence_id = solver.solve()
    return q, convergence_id


def run_experiment(FEMmodel, costantData, parameters, series):
    print("Running experiment of the series", series, " with thickness:", parameters["model"]["thickness"])

    parameters = calculate_rescaling_factors(parameters) # CFe: update exact solution scale factors

    v_scale = parameters["model"]["E"] * (parameters["model"]["thickness"]**3)
    w_scale = parameters["model"]["thickness"]

    # CFe: update external load
    a = parameters["geometry"]["radius"] / parameters["model"]["thickness"]
    b = parameters["model"]["f_scale"]/parameters["model"]["E"] # CFe: f_scale := sqrt(2 * D**3 / (E * thickness)) = E * (thickness**4) * sqrt( 2 ) / 12( 1 - nu**2 )=
    def nondim_transverse_load(x): return (a**4)*(b)*( (40 / 3) * (1 - x[0] ** 2 - x[1] ** 2) ** 4 + (16 / 3) * (11 + x[0] ** 2 + x[1] ** 2) )

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


def exactL2Nrm(costantData, parameters):
    """
    Purpose: compute the L2 norm of the exact solutions and their hessian matrices
    """

    # CFe: update exact solutions
    def _v_exact(x):
        return parameters["model"]["v_scale"]* ( - (1/12) * (1 - x[0]**2 - x[1]**2)**2  - (1/18) * (1 - x[0]**2 - x[1]**2)**3 - (1/24) * (1 - x[0]**2 - x[1]**2)**4 )

    def _w_exact(x): return parameters["model"]["w_scale"] * (1 - x[0]**2 - x[1]**2)**2

    # CFe: compute L2 norms for the exact solutions and their hessian matrices
    q_exact = dolfinx.fem.Function(costantData.funcSpace)
    v_exact, w_exact = q_exact.split()
    v_exact.interpolate(_v_exact)
    w_exact.interpolate(_w_exact)
    vExact_Nrm = L2norm(v_exact)
    wExact_Nrm = L2norm(w_exact)
    vExact_hessianNrm = hessianL2norm(v_exact)
    wExact_hessianNrm = hessianL2norm(w_exact)
    return vExact_hessianNrm, wExact_hessianNrm, vExact_Nrm, wExact_Nrm

if __name__ == "__main__":

    # CFe: choose the parameter
    # set PARAMETER_NAME equal to "thickness", "E" or "alpha_penalty"
    # set then PARAMETER_CATEGORY accordingly: "model" in all the three cases above.

    p_range = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    #p_range = np.linspace(0.01, 1, NUM_RUNS) # thickness

    # CFe: boolean, set it to true to plot with a logscale on the xaxis, False to plot with a linear scale on the xaxis
    LOG_SCALE = False

    _experimental_data = []
    
    # LOAD PARAMETERS FILE
    with pkg_resources.path(PARAMETERS_FILE_PATH, 'parameters.yml') as f:
        parameters, _ = load_parameters(f)
        base_parameters = copy.deepcopy(parameters)
        # Remove thickness from the parameters, to compute the parametric series signature
        if PARAMETER_CATEGORY in base_parameters and PARAMETER_NAME in base_parameters[PARAMETER_CATEGORY]:
            del base_parameters[PARAMETER_CATEGORY][PARAMETER_NAME]
        
        base_signature = hashlib.md5(str(base_parameters).encode('utf-8')).hexdigest()
    
    if comm.rank == 0: Path(experiment_dir).mkdir(parents=True, exist_ok=True)
            
    logging.info(f"===================- {experiment_dir} -=================")
    
    # LOAD MESH
    mesh, mts, fts = create_or_load_circle_mesh(parameters, prefix=prefix)
    h = CellDiameter(mesh)
    n = FacetNormal(mesh)

    # MAKE OUTPUT FOLDER IF DOES NOT EXIST
    if comm.rank == 0: Path(prefix).mkdir(parents=True, exist_ok=True)

    # DEFINE FUNCTION SPACE
    X = basix.ufl.element("P", str(mesh.ufl_cell()), parameters["model"]["order"])
    Q = dolfinx.fem.functionspace(mesh, basix.ufl.mixed_element([X, X]))

    # DEFINE HOMOGENEOUS DIRICHLET BOUNDARY CONDITIONS
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    bndry_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    dofs_v = dolfinx.fem.locate_dofs_topological(V=Q.sub(AIRY), entity_dim=1, entities=bndry_facets)
    dofs_w = dolfinx.fem.locate_dofs_topological(V=Q.sub(TRANSVERSE), entity_dim=1, entities=bndry_facets)
    bcs_w = dirichletbc(np.array(0, dtype=PETSc.ScalarType), dofs_w, Q.sub(TRANSVERSE))
    bcs_v = dirichletbc(np.array(0, dtype=PETSc.ScalarType), dofs_v, Q.sub(AIRY))
    _bcs = {AIRY: bcs_v, TRANSVERSE: bcs_w}
    bcs = list(_bcs.values())

    # DATA CONSTANT THOUGHTOUT THE EXPERIMENTS
    costantData = CostantData(Q, bcs)
    convergence_log = []

    with dolfinx.common.Timer(f"~Computation Experiment") as timer:

        for i, parameter in enumerate(p_range):
            if changed := update_parameters(parameters, PARAMETER_NAME, parameter):
                data_var = run_experiment(VARIATIONAL, costantData, parameters, series)
                data_brn = run_experiment(BRENNER, costantData, parameters, series)
                data_car = run_experiment(CARSTENSEN, costantData, parameters, series)
                data_exct = exactL2Nrm(costantData, parameters)
            else: 
                raise ValueError("Failed to update parameters")
            if (data_var[CONV_ID] > 0) & (data_brn[CONV_ID] > 0) & (data_car[CONV_ID] > 0 ):
                _experimental_data.append([parameter,
                                        data_var[HESSIAN_V], data_brn[HESSIAN_V], data_car[HESSIAN_V], data_exct[HESSIAN_V],
                                        data_var[HESSIAN_W], data_brn[HESSIAN_W], data_car[HESSIAN_W], data_exct[HESSIAN_W],
                                        data_var[CONV_ID], data_brn[CONV_ID], data_car[CONV_ID],
                                        data_var[AIRY_V], data_brn[AIRY_V], data_car[AIRY_V], data_exct[AIRY_V],
                                        data_var[TRANSV_W], data_brn[TRANSV_W], data_car[TRANSV_W], data_exct[TRANSV_W],
                                        parameters["geometry"]["mesh_size"], parameters["model"]["alpha_penalty"],
                                        parameters["model"]["thickness"], parameters["model"]["E"], parameters["model"]["nu"]])
            else:
                print("Some or more FE models failed to converge")
                print("data_var[CONV_ID]: ", data_var[CONV_ID])
                print("data_brn[CONV_ID]: ", data_brn[CONV_ID])
                print("data_car[CONV_ID]: ", data_car[CONV_ID])
                convergence_log.append({"thickness": parameter, "var": data_var[CONV_ID], "bnr": data_brn[CONV_ID], "car": data_car[CONV_ID]})
    
    columns = [PARAMETER_NAME,
               "Hessian L2norm v (Variational)", "Hessian L2norm v (Brenner)", "Hessian L2norm v (Carstensen)", "Hessian L2norm v (Analytical)",
               "Hessian L2norm w (Variational)", "Hessian L2norm w (Brenner)", "Hessian L2norm w (Carstensen)", "Hessian L2norm w (Analytical)",
               "Convergence ID (Variational)", "Convergence ID (Brenner)", "Convergence ID (Carstensen)",
               "L2norm v (Variational)", "L2norm v (Brenner)", "L2norm v (Carstensen)", "L2norm v (Analytical)",
               "L2norm w (Variational)", "L2norm w (Brenner)", "L2norm w (Carstensen)", "L2norm w (Analytical)",
               "Mesh size", "Interior Penalty (IP)", "Thickness", "Young modulus", "Poisson ratio"]


    experimental_data = pd.DataFrame(_experimental_data, columns=columns)
    
    _logger.info(f"Saving experimental data to {experiment_dir}")

    # COMPUTE THE PERCENTAGE ERROR
    experimental_data["Percent error - L2nrm w, Variational"] = [100*(experimental_data["L2norm w (Variational)"][i] - experimental_data["L2norm w (Analytical)"][i])/experimental_data["L2norm w (Analytical)"][i] for i in range(len(experimental_data["L2norm w (Variational)"]))]
    experimental_data["Percent error - L2nrm w, Brenner"] = [100*(experimental_data["L2norm w (Brenner)"][i] - experimental_data["L2norm w (Analytical)"][i])/experimental_data["L2norm w (Analytical)"][i] for i in range(len(experimental_data["L2norm w (Brenner)"]))]
    experimental_data["Percent error - L2nrm w, Carstensen"] = [100*(experimental_data["L2norm w (Carstensen)"][i] - experimental_data["L2norm w (Analytical)"][i])/experimental_data["L2norm w (Analytical)"][i] for i in range(len(experimental_data["L2norm w (Carstensen)"]))]
    experimental_data["Percent error - L2nrm v, Variational"] = [100*(experimental_data["L2norm v (Variational)"][i] - experimental_data["L2norm v (Analytical)"][i])/experimental_data["L2norm v (Analytical)"][i] for i in range(len(experimental_data["L2norm v (Variational)"]))]
    experimental_data["Percent error - L2nrm v, Brenner"] = [100*(experimental_data["L2norm v (Brenner)"][i] - experimental_data["L2norm v (Analytical)"][i])/experimental_data["L2norm v (Analytical)"][i] for i in range(len(experimental_data["L2norm v (Brenner)"]))]
    experimental_data["Percent error - L2nrm v, Carstensen"] = [100*(experimental_data["L2norm v (Carstensen)"][i] - experimental_data["L2norm v (Analytical)"][i])/experimental_data["L2norm v (Analytical)"][i] for i in range(len(experimental_data["L2norm v (Carstensen)"]))]

    # EXPORT RESULTS IN EXCEL FILE
    experimental_data.to_excel(f'{experiment_dir}/Models_comparison_mesh_{parameters["geometry"]["mesh_size"]}_IP_{parameters["model"]["alpha_penalty"]}_E_{parameters["model"]["E"]}.xlsx', index=False)

    # PRINT OUT RESULTS
    print(10*"*")
    print("Results")
    print(experimental_data)
    print(10*"*")

    # PLOTS
    x_ticks = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    #plt.plot(p_range, range(len(p_range)))  # Replace with your actual data

    plt.figure(figsize=(30, 20))
    plt.plot(experimental_data[PARAMETER_NAME], experimental_data["L2norm v (Variational)"], marker='o', linestyle='solid', color='b', label='v (Variational)', linewidth=5, markersize=20)
    plt.plot(experimental_data[PARAMETER_NAME], experimental_data["L2norm v (Brenner)"], marker='v', linestyle='dotted', color='r', label='v (Brenner)', linewidth=5, markersize=20)
    plt.plot(experimental_data[PARAMETER_NAME], experimental_data["L2norm v (Carstensen)"], marker='^', linestyle='dashed', color='g', label='v (Carstensen)', linewidth=5, markersize=20)
    plt.plot(experimental_data[PARAMETER_NAME], experimental_data["L2norm v (Analytical)"], marker='P', linestyle='dashdot', color='k', label='v (Analytical)', linewidth=5, markersize=20)

    max_v = max([max(experimental_data["L2norm v (Variational)"]), max(experimental_data["L2norm v (Brenner)"]), max(experimental_data["L2norm v (Carstensen)"]), max(experimental_data["L2norm v (Analytical)"])])
    min_v = min([min(experimental_data["L2norm v (Variational)"]), min(experimental_data["L2norm v (Brenner)"]), min(experimental_data["L2norm v (Carstensen)"]), min(experimental_data["L2norm v (Analytical)"])])
    steps_v = (max_v - min_v)/10

    if LOG_SCALE: plt.xscale('log') # CFe: use log xscale when needed
    plt.yticks(np.arange(min_v, max_v, steps_v))
    plt.xlabel(PARAMETER_NAME+" [m]", fontsize=35)
    plt.ylabel(r'$| v |_{L^2(\Omega)}$', fontsize=35)
    plt.title(f'Comparison between models. Airy function. Mesh size = {parameters["geometry"]["mesh_size"]}. IP = {parameters["model"]["alpha_penalty"]}. E = {parameters["model"]["E"]}', fontsize=40)
    plt.xticks(x_ticks)
    plt.gca().set_xticks(p_range, minor=True)
    plt.tick_params(axis='both', which='major', labelsize=35)
    plt.legend(fontsize=35)
    plt.grid(which='both')
    plt.savefig(experiment_dir+f'/models_comparison_V_mesh_{parameters["geometry"]["mesh_size"]}_IP_{parameters["model"]["alpha_penalty"]}_E_{parameters["model"]["E"]}.png', dpi=300)
    plt.show()

    plt.figure(figsize=(30, 20))
    plt.plot(experimental_data[PARAMETER_NAME], experimental_data["L2norm w (Variational)"], marker='o', linestyle='solid', color='b', label='L2norm w (Variational)', linewidth=5, markersize=20)
    plt.plot(experimental_data[PARAMETER_NAME], experimental_data["L2norm w (Brenner)"], marker='v', linestyle='dotted', color='r', label='L2norm w (Brenner)', linewidth=5, markersize=20)
    plt.plot(experimental_data[PARAMETER_NAME], experimental_data["L2norm w (Carstensen)"], marker='^', linestyle='dashed', color='g', label='L2norm w (Carstensen)', linewidth=5, markersize=20)
    plt.plot(experimental_data[PARAMETER_NAME], experimental_data["L2norm w (Analytical)"], marker='P', linestyle='dashdot', color='k', label='L2norm w (Analytical)', linewidth=5, markersize=20)
    #pdb.set_trace()

    max_w = max([max(experimental_data["L2norm w (Variational)"]), max(experimental_data["L2norm w (Brenner)"]), max(experimental_data["L2norm w (Carstensen)"]), max(experimental_data["L2norm w (Analytical)"])])
    min_w = min([min(experimental_data["L2norm w (Variational)"]), min(experimental_data["L2norm w (Brenner)"]), min(experimental_data["L2norm w (Carstensen)"]), min(experimental_data["L2norm w (Analytical)"])])
    steps_w = (max_w - min_w)/10

    if LOG_SCALE: plt.xscale('log') # CFe: use log xscale when needed
    plt.xticks(x_ticks )
    plt.yticks(np.arange(min_w, max_w, steps_w))
    plt.xlabel(PARAMETER_NAME+" [m]", fontsize=35)
    plt.ylabel(r'$| w |_{L^2(\Omega)}$', fontsize=35)
    plt.title(f'Comparison between models. Transverse displacement. Mesh size = {parameters["geometry"]["mesh_size"]}. IP = {parameters["model"]["alpha_penalty"]}. E = {parameters["model"]["E"]}', fontsize=40)
    plt.xticks(x_ticks)
    plt.gca().set_xticks(p_range, minor=True)
    plt.tick_params(axis='both', which='major', labelsize=35)
    plt.legend(fontsize=35)
    plt.grid(which='both')
    plt.savefig(experiment_dir+f'/models_comparison_W_mesh_{parameters["geometry"]["mesh_size"]}_IP_{parameters["model"]["alpha_penalty"]}_E_{parameters["model"]["E"]}.png', dpi=300)
    plt.show()

    # PERCENTAGE ERROR PLOT

    plt.figure(figsize=(30, 20))
    plt.plot(experimental_data[PARAMETER_NAME], experimental_data["Percent error - L2nrm v, Variational"], marker='o', linestyle='solid', color='b', label='L2norm v (Variational)', linewidth=5, markersize=20)
    plt.plot(experimental_data[PARAMETER_NAME], experimental_data["Percent error - L2nrm v, Brenner"], marker='v', linestyle='dotted', color='r', label='L2norm v (Brenner)', linewidth=5, markersize=20)
    plt.plot(experimental_data[PARAMETER_NAME], experimental_data["Percent error - L2nrm v, Carstensen"], marker='^', linestyle='dashed', color='g', label='L2norm v (Carstensen)', linewidth=5, markersize=20)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=5, label='0% error line')

    max_v = max([max(experimental_data["Percent error - L2nrm v, Variational"]), max(experimental_data["Percent error - L2nrm v, Brenner"]), max(experimental_data["Percent error - L2nrm v, Carstensen"]), 0])
    min_v = min([min(experimental_data["Percent error - L2nrm v, Variational"]), min(experimental_data["Percent error - L2nrm v, Brenner"]), min(experimental_data["Percent error - L2nrm v, Carstensen"]), 0])
    steps_v = (max_v - min_v)/10

    if LOG_SCALE: plt.xscale('log') # CFe: use log xscale when needed
    plt.xticks(x_ticks )
    plt.yticks(np.arange(min_v, max_v, steps_v))
    plt.xlabel(PARAMETER_NAME+" [m]", fontsize=35)
    plt.ylabel(r'$e_v$', fontsize=35)
    plt.title(f'Percentage errors. Airy function. Mesh size = {parameters["geometry"]["mesh_size"]}. IP = {parameters["model"]["alpha_penalty"]}. E = {parameters["model"]["E"]}', fontsize=40)
    plt.xticks(x_ticks)
    plt.gca().set_xticks(p_range, minor=True)
    plt.tick_params(axis='both', which='major', labelsize=35)
    plt.legend(fontsize=35)
    plt.grid(which='both')
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.savefig(experiment_dir+f'/perc_err_V_mesh_{parameters["geometry"]["mesh_size"]}_IP_{parameters["model"]["alpha_penalty"]}_E_{parameters["model"]["E"]}.png', dpi=300)
    plt.show()

    plt.figure(figsize=(30, 20))
    plt.plot(experimental_data[PARAMETER_NAME], experimental_data["Percent error - L2nrm w, Variational"], marker='o', linestyle='solid', color='b', label='L2norm w (Variational)', linewidth=5, markersize=20)
    plt.plot(experimental_data[PARAMETER_NAME], experimental_data["Percent error - L2nrm w, Brenner"], marker='v', linestyle='dotted', color='r', label='L2norm w (Brenner)', linewidth=5, markersize=20)
    plt.plot(experimental_data[PARAMETER_NAME], experimental_data["Percent error - L2nrm w, Carstensen"], marker='^', linestyle='dashed', color='g', label='L2norm w (Carstensen)', linewidth=5, markersize=20)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=5, label='0% error line')

    max_w = max([max(experimental_data["Percent error - L2nrm w, Variational"]), max(experimental_data["Percent error - L2nrm w, Brenner"]), max(experimental_data["Percent error - L2nrm w, Carstensen"]), 0])
    min_w = min([min(experimental_data["Percent error - L2nrm w, Variational"]), min(experimental_data["Percent error - L2nrm w, Brenner"]), min(experimental_data["Percent error - L2nrm w, Carstensen"]), 0])
    steps_w = (max_w - min_w)/10

    if LOG_SCALE: plt.xscale('log') # CFe: use log xscale when needed
    plt.xticks(x_ticks)
    plt.yticks(np.arange(min_w, max_w, steps_w))
    plt.xlabel(PARAMETER_NAME+" [m]", fontsize=35)
    plt.ylabel(r'$e_w$', fontsize=35)
    plt.title(f'Percentage errors. Transverse displacement. Mesh size = {parameters["geometry"]["mesh_size"]}. IP = {parameters["model"]["alpha_penalty"]}. E = {parameters["model"]["E"]}', fontsize=40)
    plt.xticks(x_ticks)
    plt.gca().set_xticks(p_range, minor=True)
    plt.tick_params(axis='both', which='major', labelsize=35)
    plt.legend(fontsize=35)
    plt.grid(which='both')
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.savefig(experiment_dir+f'/perc_err_W_mesh_{parameters["geometry"]["mesh_size"]}_IP_{parameters["model"]["alpha_penalty"]}_E_{parameters["model"]["E"]}.png', dpi=300)
    plt.show()
