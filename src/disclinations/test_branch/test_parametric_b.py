import gc
import hashlib
import json
import logging
import os
import pdb
import sys
from pathlib import Path

import basix
import disclinations
import dolfinx
import numpy as np
import petsc4py
import pytest
import ufl
import yaml
from disclinations.meshes.primitives import mesh_circle_gmshapi
from disclinations.models import (NonlinearPlateFVK, NonlinearPlateFVK_brenner,
                                  NonlinearPlateFVK_carstensen,
                                  calculate_rescaling_factors,
                                  compute_energy_terms)
from disclinations.solvers import SNESSolver
from disclinations.utils import (Visualisation, memory_usage, monitor,
                                 table_timing_data, write_to_output)
from disclinations.utils.la import compute_disclination_loads
from dolfinx import fem
from dolfinx.common import list_timings
from dolfinx.fem import Constant, dirichletbc, locate_dofs_topological
from dolfinx.io import XDMFFile, gmshio
from mpi4py import MPI
from petsc4py import PETSc
import pandas as pd
from disclinations.utils import _logger

import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
from ufl import CellDiameter, FacetNormal, dx, sqrt

# models = ["variational", "brenner", "carstensen"]
outdir = "output"

AIRY                = 0
TRANSVERSE = 1
CONV_ID        = 2

VARIATIONAL  = 0
BRENNER         = 1
CARSTENSEN   = 2

from disclinations.models import create_disclinations
from disclinations.utils import (homogeneous_dirichlet_bc_H20, load_parameters,
                                 save_params_to_yaml)
from disclinations.utils import update_parameters, save_parameters

from disclinations.utils import create_or_load_circle_mesh
import importlib.resources as pkg_resources  # Python 3.7+ for accessing package files
import copy

# CFe added
from collections import namedtuple
CostantData = namedtuple("CostantData", ["funcSpace", "bcs"])
# CFe end

petsc4py.init(["-petsc_type", "double"]) # CFe: ensuring double precision numbers

def hessianL2norm(u):
    hessian = lambda f : ufl.grad(ufl.grad(u))
    return np.sqrt( fem.assemble_scalar( fem.form( ufl.inner(hessian(u), hessian(u)) * ufl.dx ) ) )


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
        model = NonlinearPlateFVK_carstensen(mesh, parameters["model"])
    elif FEMmodel == BRENNER:
        model = NonlinearPlateFVK_brenner(mesh, parameters["model"])
    elif FEMmodel == VARIATIONAL:
        model = NonlinearPlateFVK(mesh, parameters["model"])
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
        # petsc_options=parameters.get("solvers").get("elasticity").get("snes"),
        petsc_options=solver_parameters,
        prefix="plate_fvk",
    )
    n_iterations, convergence_id = solver.solve()
    return q, convergence_id

def run_experiment(FEMmodel, costantData, parameters, param):
    """
    Purpose: parametric study of the solution by varying b := p0/E
    """
    print("Running experiment with load:", param)

    # CFe: update external load
    def nondim_transverse_load(x): return param*1.0 + 0.0*x[0] + 0.0*x[1]

    Q = costantData.funcSpace
    f = dolfinx.fem.Function(Q.sub(TRANSVERSE).collapse()[0])
    f.interpolate(nondim_transverse_load)

    q, convergence_id = solveModel(FEMmodel, costantData, parameters, f)
    v, w = q.split()
    v_hessianNrm = hessianL2norm(v)
    w_hessianNrm = hessianL2norm(w)

    return v_hessianNrm, w_hessianNrm, convergence_id

if __name__ == "__main__":

    # CFe: choose the parameter
    # set PARAMETER_NAME equal to "thickness", "E" or "alpha_penalty"
    # set then PARAMETER_CATEGORY accordingly: "model" in all the three cases above.
    PARAMETER_NAME = "Load p0"
    PARAMETER_CATEGORY = "model"
    NUM_RUNS = 10

    p_range = np.linspace(1, 10, NUM_RUNS) # load p0

    # CFe: boolean, set it to true to plot with a logscale on the xaxis, False to plot with a linear scale on the xaxis
    LOG_SCALE = False

    from disclinations.utils import table_timing_data, Visualisation
    _experimental_data = []
    outdir = "output"
    prefix = os.path.join(outdir, "test_parametric_load")
    
    with pkg_resources.path('disclinations.test', 'parameters.yml') as f:
        parameters, _ = load_parameters(f)
        base_parameters = copy.deepcopy(parameters)
        # Remove thickness from the parameters, to compute the parametric series signature
        if PARAMETER_CATEGORY in base_parameters and PARAMETER_NAME in base_parameters[PARAMETER_CATEGORY]:
            del base_parameters[PARAMETER_CATEGORY][PARAMETER_NAME]
        
        base_signature = hashlib.md5(str(base_parameters).encode('utf-8')).hexdigest()

    #series = base_signature[0::6]
    series = PARAMETER_NAME
    experiment_dir = os.path.join(prefix, series)
    
    if comm.rank == 0:
        Path(experiment_dir).mkdir(parents=True, exist_ok=True)
            
    logging.info(
        f"===================- {experiment_dir} -=================")
    
    mesh, mts, fts = create_or_load_circle_mesh(parameters, prefix=prefix)

    if comm.rank == 0:
        Path(prefix).mkdir(parents=True, exist_ok=True)
    h = CellDiameter(mesh)
    n = FacetNormal(mesh)

    X = basix.ufl.element("P", str(mesh.ufl_cell()), parameters["model"]["order"])
    Q = dolfinx.fem.functionspace(mesh, basix.ufl.mixed_element([X, X]))

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    bndry_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    dofs_v = dolfinx.fem.locate_dofs_topological(V=Q.sub(AIRY), entity_dim=1, entities=bndry_facets)
    dofs_w = dolfinx.fem.locate_dofs_topological(V=Q.sub(TRANSVERSE), entity_dim=1, entities=bndry_facets)

    bcs_w = dirichletbc(np.array(0, dtype=PETSc.ScalarType), dofs_w, Q.sub(TRANSVERSE))
    bcs_v = dirichletbc(np.array(0, dtype=PETSc.ScalarType), dofs_v, Q.sub(AIRY))

    _bcs = {AIRY: bcs_v, TRANSVERSE: bcs_w}
    bcs = list(_bcs.values())

    costantData = CostantData(Q, bcs)

    # CFe running experiments by varying a
    for i, param in enumerate(p_range):
        data_var = run_experiment(VARIATIONAL, costantData, parameters, param)
        data_brn = run_experiment(BRENNER, costantData, parameters, param)
        data_car = run_experiment(CARSTENSEN, costantData, parameters, param)


        if (data_var[CONV_ID] > 0) and (data_brn[CONV_ID]  > 0) and (data_car[CONV_ID] > 0):
            _experimental_data.append([param, data_var[AIRY], data_brn[AIRY], data_car[AIRY],
                                    data_var[TRANSVERSE], data_brn[TRANSVERSE], data_car[TRANSVERSE],
                                    data_var[CONV_ID], data_brn[CONV_ID], data_car[CONV_ID],
                                    parameters["geometry"]["mesh_size"], parameters["model"]["alpha_penalty"],
                                    parameters["model"]["thickness"], parameters["model"]["E"], parameters["model"]["nu"]])
        else:
            print(f"At least one of the three FEM models did not converged for {PARAMETER_NAME} = {param}")
            print("Convergence Variational model: ", data_var[CONV_ID])
            print("Convergence Brenner model: ", data_brn[CONV_ID])
            print("Convergence Carstensen model: ", data_car[CONV_ID])
    
    columns = [PARAMETER_NAME, "v (Variational)", "v (Brenner)", "v (Carstensen)",
               "w (Variational)", "w (Brenner)", "w (Carstensen)",
               "Convergence ID (Variational)", "Convergence ID (Brenner)", "Convergence ID (Carstensen)",
               "Mesh size", "Interior Penalty (IP)", "Thickness", "Young modulus", "Poisson ratio"]

    experimental_data = pd.DataFrame(_experimental_data, columns=columns)
    
    _logger.info(f"Saving experimental data to {experiment_dir}")
    #pdb.set_trace()

    experimental_data.to_excel(f'{experiment_dir}/varying_b_mesh_{parameters["geometry"]["mesh_size"]}_IP_{parameters["model"]["alpha_penalty"]}_E_{parameters["model"]["E"]}.xlsx', index=False)

    print(10*"*")
    print("Results")
    print(experimental_data)
    print(10*"*")

    # Plots
    plt.figure(figsize=(30, 20))
    plt.plot(experimental_data[PARAMETER_NAME], experimental_data["v (Variational)"], marker='o', linestyle='solid', color='b', label='v (Variational)', linewidth=5, markersize=20)
    plt.plot(experimental_data[PARAMETER_NAME], experimental_data["v (Brenner)"], marker='v', linestyle='dotted', color='r', label='v (Brenner)', linewidth=5, markersize=20)
    plt.plot(experimental_data[PARAMETER_NAME], experimental_data["v (Carstensen)"], marker='^', linestyle='dashed', color='g', label='v (Carstensen)', linewidth=5, markersize=20)

    max_v = max([max(experimental_data["v (Variational)"]), max(experimental_data["v (Brenner)"]), max(experimental_data["v (Carstensen)"])])
    min_v = min([min(experimental_data["v (Variational)"]), min(experimental_data["v (Brenner)"]), min(experimental_data["v (Carstensen)"])])
    steps_v = (max_v - min_v)/10

    if LOG_SCALE: plt.xscale('log') # CFe: use log xscale when needed
    plt.xticks(p_range)
    plt.yticks(np.arange(min_v, max_v, steps_v))
    plt.xlabel("b", fontsize=35)
    plt.ylabel(r'$| \nabla^2 v |_{L^2(\Omega)}$', fontsize=35)
    plt.title(f'Airy function. Mesh size = {parameters["geometry"]["mesh_size"]}. IP = {parameters["model"]["alpha_penalty"]}. E = {parameters["model"]["E"]}', fontsize=40)
    plt.tick_params(axis='both', which='major', labelsize=35)
    plt.legend(fontsize=35)
    plt.grid(True)
    plt.savefig(experiment_dir+f'/varying_b_V_mesh_{parameters["geometry"]["mesh_size"]}_IP_{parameters["model"]["alpha_penalty"]}_E_{parameters["model"]["E"]}.png', dpi=300)
    plt.show()

    plt.figure(figsize=(30, 20))
    plt.plot(experimental_data[PARAMETER_NAME], experimental_data["w (Variational)"], marker='o', linestyle='solid', color='b', label='w (Variational)', linewidth=5, markersize=20)
    plt.plot(experimental_data[PARAMETER_NAME], experimental_data["w (Brenner)"], marker='v', linestyle='dotted', color='r', label='w (Brenner)', linewidth=5, markersize=20)
    plt.plot(experimental_data[PARAMETER_NAME], experimental_data["w (Carstensen)"], marker='^', linestyle='dashed', color='g', label='w (Carstensen)', linewidth=5, markersize=20)

    max_w = max([max(experimental_data["w (Variational)"]), max(experimental_data["w (Brenner)"]), max(experimental_data["w (Carstensen)"])])
    min_w = min([min(experimental_data["w (Variational)"]), min(experimental_data["w (Brenner)"]), min(experimental_data["w (Carstensen)"])])
    steps_w = (max_w - min_w)/10

    if LOG_SCALE: plt.xscale('log') # CFe: use log xscale when needed
    plt.xticks(p_range)
    plt.yticks(np.arange(min_w, max_w, steps_w))
    plt.xlabel("b", fontsize=35)
    plt.ylabel(r'$| \nabla^2 w |_{L^2(\Omega)}$', fontsize=35)
    plt.title(f'Transverse displacement. Mesh size = {parameters["geometry"]["mesh_size"]}. IP = {parameters["model"]["alpha_penalty"]}. E = {parameters["model"]["E"]}', fontsize=40)
    plt.tick_params(axis='both', which='major', labelsize=35)
    plt.legend(fontsize=35)
    plt.grid(True)
    plt.savefig(experiment_dir+f'/varying_b_W_mesh_{parameters["geometry"]["mesh_size"]}_IP_{parameters["model"]["alpha_penalty"]}_E_{parameters["model"]["E"]}.png', dpi=300)
    plt.show()
