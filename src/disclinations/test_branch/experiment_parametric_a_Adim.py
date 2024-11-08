"""
PURPOSE OF THE SCRIPT
Run experiments by varying "a^4 * b" while keeping constant a^2.
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
import importlib.resources as pkg_resources  # Python 3.7+ for accessing package files
import copy
from collections import namedtuple
import gc
import matplotlib.pyplot as plt

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
from disclinations.utils.viz import plot_scalar, plot_profile, plot_mesh

# OUTPUT DIRECTORY
OUTDIR = os.path.join("output", "experiment_parametric_a_adim") # CFe: output directory
PATH_TO_PARAMETERS_YML_FILE = 'disclinations.test'

# NON LINEAR SEARCH TOLLERANCES
ABS_TOLLERANCE = 1e-13 # Absolute tollerance
REL_TOLLERANCE = 1e-13  # Relative tollerance
SOL_TOLLERANCE = 1e-13  # Solution tollerance

COMM = MPI.COMM_WORLD
if COMM.rank == 0: Path(OUTDIR).mkdir(parents=True, exist_ok=True)

DISCLINATION_COORD_LIST = [[0.0, 0.0, 0.0]]
DISCLINATION_POWER_LIST = [-1]
LOAD_SIGN = -1

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

BENDING_ERG      = 8
MEMBRANE_ERG = 9
COUPL_ERG          = 10
PENALIZATION    = 11
PENALIZATION_W                  = 9
PENALIZATION_V                   = 10
PENALIZATION_COUPLING   = 11

SMOOTHING = False

NUM_STEPS = 10 # Number ticks in the yaxis

CostantData = namedtuple("CostantData", ["funcSpace", "bcs"])

petsc4py.init(["-petsc_type", "double"]) # CFe: ensuring double precision numbers

def L2norm(u):
    return np.sqrt( fem.assemble_scalar( fem.form( ufl.inner(u, u) * ufl.dx ) ) )

def hessianL2norm(u):
    hessian = lambda f : ufl.grad(ufl.grad(u))
    return np.sqrt( fem.assemble_scalar( fem.form( ufl.inner(hessian(u), hessian(u)) * ufl.dx ) ) )


def run_experiment(FEMmodel, costantData, parameters, q_ig):
    """
    Purpose: parametric study of the solution by varying a := R/h
    """

    # UPDATE PARAMETER "a"
    a = 1/parameters["model"]["thickness"]

    # UPDATE PARAMETER "c"
    c = 1 #((a)**4)*1e-9  # Parameter c is kept constant
    def nondim_transverse_load(x): return c * LOAD_SIGN *(1.0 + 0.0*x[0] + 0.0*x[1])

    Q = costantData.funcSpace
    f = dolfinx.fem.Function(Q.sub(TRANSVERSE).collapse()[0])
    f.interpolate(nondim_transverse_load)


    # FUNCTION SPACE
    q = dolfinx.fem.Function(Q)
    q.vector.array = copy.deepcopy(q_ig.vector.array)
    q.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    v, w = ufl.split(q)
    state = {"v": v, "w": w}

    # SELECT MODEL
    if FEMmodel == CARSTENSEN:
        model = A_NonlinearPlateFVK_carstensen(mesh, parameters["model"])
    elif FEMmodel == BRENNER:
        model = A_NonlinearPlateFVK_brenner(mesh, parameters["model"])
    elif FEMmodel == VARIATIONAL:
        model = A_NonlinearPlateFVK(mesh, parameters["model"], SMOOTHING)
    else:
        print(f"FEMmodel = {FEMmodel} is not acceptable. Script exiting")
        exit(0)

    # INSERT DISCLINATIONS
    disclinations = []
    if mesh.comm.rank == 0:
        for dc in DISCLINATION_COORD_LIST: disclinations.append( np.array([dc], dtype=mesh.geometry.x.dtype))
        dp_list = [dp*(a)**2.0 for dp in DISCLINATION_POWER_LIST]
    else:
        for dc in DISCLINATION_COORD_LIST: disclinations.append( np.zeros((0, 3), dtype=mesh.geometry.x.dtype) )
    Q_v, Q_v_to_Q_dofs = Q.sub(AIRY).collapse()
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
        'snes_linesearch_type': 'basic', #'l2',  # L2-norm line search - CFe added
        'snes_linesearch_maxstep': 1.0, # CFe added
        #'snes_linesearch_damping': 0.5, #CFe added
        'ksp_type': 'gmres',  # Use GMRES for linear solver CFe added
        'pc_type': 'lu',  # LU preconditioner CFe added
        "snes_max_it": 200,  # Maximum number of iterations
        "snes_rtol": REL_TOLLERANCE,  # Relative tolerance for convergence
        "snes_atol": ABS_TOLLERANCE,  # Absolute tolerance for convergence
        "snes_stol": SOL_TOLLERANCE,  # Tolerance for the change in solution norm
        "snes_monitor": None,  # Function for monitoring convergence (optional)
        #"snes_linesearch_type": "none",  # Type of line search
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

    # FREE MEMORY
    solver.solver.destroy()
    gc.collect()

    # COMPUTE DIMENSIONAL ENERGY
    energy_terms = {
        "bending": model.compute_bending_energy(state, COMM),
        "membrane": model.compute_membrane_energy(state, COMM),
        "coupling": model.compute_coupling_energy(state, COMM),
        "penalty":  model.compute_penalisation(state, COMM),
        "penalty_w_dg1":  model.compute_penalisation_terms_w(state, COMM)[0],
        "penalty_w_dg2":  model.compute_penalisation_terms_w(state, COMM)[1],
        "penalty_w_bc1":  model.compute_penalisation_terms_w(state, COMM)[2],
        "penalty_w_bc3":  model.compute_penalisation_terms_w(state, COMM)[3],
        "penalty_w_dg3":  model.compute_penalisation_terms_w(state, COMM)[4],
        "penalty_w_tot":  model.compute_total_penalisation_w(state, COMM),
        "penalty_v_tot":  model.compute_total_penalisation_v(state, COMM),
        "penalty_coupling":  model.compute_penalisation_coupling(state, COMM),
        }

    # Print FE dimensioanal energy
    for label, energy_term in energy_terms.items(): print(f"{label}: {energy_term}")

    v, w = q.split()
    v_Nrm = L2norm(v)
    w_Nrm = L2norm(w)
    v_hessianNrm = hessianL2norm(v)
    w_hessianNrm = hessianL2norm(w)

    return_value_dic = {
            "w": w, "v": v, "q": q,
            "v_Nrm": v_Nrm, "w_Nrm": w_Nrm,
            "v_hessianNrm": v_hessianNrm, "w_hessianNrm": w_hessianNrm,
            "bending_energy": energy_terms["bending"],
            "membrane_energy": energy_terms["membrane"],
            "coupling_energy": energy_terms["coupling"],
            "penalty_energy": energy_terms["penalty"],
            "penalty_w_tot": energy_terms["penalty_w_tot"],
            "penalty_v_tot": energy_terms["penalty_v_tot"],
            "penalty_coupling": energy_terms["penalty_coupling"],
            "penalty_w_dg1": energy_terms["penalty_w_dg1"],
            "penalty_w_dg2": energy_terms["penalty_w_dg2"],
            "penalty_w_bc1": energy_terms["penalty_w_bc1"],
            "penalty_w_bc3": energy_terms["penalty_w_bc3"],
            "penalty_w_hessJump": energy_terms["penalty_w_dg3"],
            "convergence_id": convergence_id
        }

    # CHECK BOUNDARY CONDITIONS FOR V AND W
    from ufl import dot, grad
    n = ufl.FacetNormal(mesh)
    ds = ufl.Measure("ds", domain=mesh)
    normal_derivative_v = ufl.dot(ufl.grad(v), n)
    normal_derivative_w = ufl.dot(ufl.grad(w), n)
    L2norm_w = np.sqrt( fem.assemble_scalar(fem.form((w**2) * ds)) )
    L2norm_v = np.sqrt( fem.assemble_scalar(fem.form((v**2) * ds)) )
    L2norm_dwdn = np.sqrt( fem.assemble_scalar(fem.form((normal_derivative_w**2) * ds)) )
    L2norm_dvdn = np.sqrt( fem.assemble_scalar(fem.form((normal_derivative_v**2) * ds)) )
    print("boundary_integral_v: ", L2norm_v)
    print("boundary_integral_w: ", L2norm_w)
    print("L2norm_dwdn: ", L2norm_dwdn)
    print("L2norm_dvdn: ", L2norm_dvdn)

    return return_value_dic


if __name__ == "__main__":

    print("Smoothing: ", SMOOTHING)
    PARAMETER_NAME = "thickness"
    PARAMETER_CATEGORY = "model"
    NUM_RUNS = 15

    #a_list = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    #a_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 130, 160, 200, 230, 260, 300, 330]
    a_list = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    #a_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    #a_list = [15, 30, 40, 50, 60, 70, 80, 90, 100, 160] #, 300, 333
    #a_list  = np.linspace(100, 300, 3) #[95, 100, 105]
    #a_list = [300]
    a_range = []

    #x_range_plot = [10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    #x_range_plot = [10, 100, 200, 330]
    x_range_plot = a_list
    p_range = [1/el for el in a_list] # CFe: thickness

    # CFe: boolean, set it to true to plot with a logscale on the xaxis, False to plot with a linear scale on the xaxis
    LOG_SCALE = False

    _experimental_data = []

    # LOAD PARAMETERS FILE
    with pkg_resources.path(PATH_TO_PARAMETERS_YML_FILE, 'parameters.yml') as f:
        parameters, _ = load_parameters(f)
        base_parameters = copy.deepcopy(parameters)
        # Remove thickness from the parameters, to compute the parametric series signature
        if PARAMETER_CATEGORY in base_parameters and PARAMETER_NAME in base_parameters[PARAMETER_CATEGORY]:
            del base_parameters[PARAMETER_CATEGORY][PARAMETER_NAME]

        base_signature = hashlib.md5(str(base_parameters).encode('utf-8')).hexdigest()

    #series = base_signature[0::6]

    info_experiment = f"mesh_{parameters["geometry"]["mesh_size"]}_IP_{parameters["model"]["alpha_penalty"]:.2e}_smth_{SMOOTHING}_tol = {SOL_TOLLERANCE}_load={LOAD_SIGN}_s={DISCLINATION_POWER_LIST}"
    EXPERIMENT_DIR = os.path.join(OUTDIR, f"{PARAMETER_NAME}_{info_experiment}")
    if COMM.rank == 0: Path(EXPERIMENT_DIR).mkdir(parents=True, exist_ok=True)

    logging.info(f"===================- {EXPERIMENT_DIR} -=================")

    # LOAD THE MESH
    mesh, mts, fts = create_or_load_circle_mesh(parameters, prefix=OUTDIR)
    h = CellDiameter(mesh)
    n = FacetNormal(mesh)

    # PLOT THE MESH
    plt.figure()
    ax = plot_mesh(mesh)
    fig = ax.get_figure()
    fig.savefig(f"{EXPERIMENT_DIR}/mesh.png")

    # DEFINE THE FUNCTION SPACE
    X = basix.ufl.element("P", str(mesh.ufl_cell()), parameters["model"]["order"])
    Q = dolfinx.fem.functionspace(mesh, basix.ufl.mixed_element([X, X]))

    # INITIAL GUESS
    q_ig = dolfinx.fem.Function(Q)

    # DEFINE THE DIRICHLET BOUNDARY CONDITIONS
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    #from dolfinx import mesh as msh
    #msh.refine(mesh)

    #pdb.set_trace()
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
    sol_v_list = []
    sol_w_list = []
    for i, param in enumerate(p_range):
        print("Running experiment with a = ", np.round(1/param, 3))
        if changed := update_parameters(parameters, PARAMETER_NAME, param):
            # if param == 1/200:
            #     parameters["model"]["alpha_penalty"] = 100
            # elif (param <= 1/500) and (param != 1/700) and (param != 1/1000):
            #     parameters["model"]["alpha_penalty"] = 300
            # else:
            #     parameters["model"]["alpha_penalty"] = 200
            data_var = run_experiment(VARIATIONAL, costantData, parameters, q_ig)
            data_brn = data_var #run_experiment(BRENNER, costantData, parameters, q_ig)
            data_car = data_var #run_experiment(CARSTENSEN, costantData, parameters, q_ig)

        if (data_var["convergence_id"] > 0) and (data_brn["convergence_id"] > 0) and (data_car["convergence_id"] > 0):
            a_range.append(1/param)
            _experimental_data.append(
                [
                    param,
                    data_var["v_hessianNrm"], data_brn["v_hessianNrm"], data_car["v_hessianNrm"],
                    data_var["w_hessianNrm"], data_brn["w_hessianNrm"], data_car["w_hessianNrm"],
                    data_var["convergence_id"], data_brn["convergence_id"], data_car["convergence_id"],
                    data_var["v_Nrm"], data_brn["v_Nrm"], data_car["v_Nrm"],
                    data_var["w_Nrm"], data_brn["w_Nrm"], data_car["w_Nrm"],
                    parameters["geometry"]["mesh_size"], parameters["model"]["alpha_penalty"],
                    parameters["model"]["thickness"], parameters["model"]["E"], parameters["model"]["nu"],
                    data_var["bending_energy"],
                    data_var["membrane_energy"],
                    data_var["coupling_energy"],
                    data_var["penalty_energy"],
                    data_var["penalty_w_tot"], data_var["penalty_v_tot"], data_var["penalty_coupling"],
                    data_var["penalty_w_hessJump"],
                    data_var["penalty_w_dg1"],
                    data_var["penalty_w_dg2"],
                    data_var["penalty_w_bc1"],
                    data_var["penalty_w_bc3"],
                    ]
                )

            sol_w_list.append(data_var["w"])
            sol_v_list.append(data_var["v"])

            # Update the initial guess
            #if i == 0: q_ig.vector.array = data_var[7].vector.array
            #else: q_ig.vector.array = data_var[7].vector.array * p_range[i]/p_range[i-1]
            #print(" q_ig.vector.norm(PETSc.NormType.NORM_2): ",  q_ig.vector.norm(PETSc.NormType.NORM_2))
            #print(" data_var[7].vector.norm(PETSc.NormType.NORM_2): ",  data_var[7].vector.norm(PETSc.NormType.NORM_2) )
            #pdb.set_trace()
            q_ig.vector.array = copy.deepcopy(data_var["q"].vector.array)
            q_ig.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        else:
            print(f"At least one of the three FEM models did not converged for {PARAMETER_NAME} = {param}")
            print("Convergence Variational model: ", data_var["convergence_id"])
            print("Convergence Brenner model: ", data_brn["convergence_id"])
            print("Convergence Carstensen model: ", data_car["convergence_id"])
            non_converged_list.append({f"{PARAMETER_NAME}": param, "Var_Con_ID": data_var["convergence_id"], "Brn_Con_ID": data_brn["convergence_id"], "Car_Con_ID": data_car["convergence_id"]})
    
    columns = [PARAMETER_NAME, "Hessian L2norm v (Variational)", "Hessian L2norm v (Brenner)", "Hessian L2norm v (Carstensen)",
               "Hessian L2norm w (Variational)", "Hessian L2norm w (Brenner)", "Hessian L2norm w (Carstensen)",
               "Convergence ID (Variational)", "Convergence ID (Brenner)", "Convergence ID (Carstensen)",
               "L2norm v (Variational)", "L2norm v (Brenner)", "L2norm v (Carstensen)",
               "L2norm w (Variational)", "L2norm w (Brenner)", "L2norm w (Carstensen)",
               "Mesh size", "Interior Penalty (IP)", "Thickness", "Young modulus", "Poisson ratio",
               "Bending Erg (Var)", "Membrane Erg (Var)", "Coupling Erg (Var)", "Penalization Eng (Var)",
               "Penalization W Eng (Var)", "Penalization V Eng (Var)", "Penalization coupling Eng (Var)",
               "Penalization W Hess Jump (Var)",
               "Penalty w dg1 (Var)", "Penalty w dg2 (Var)", "Penalty w bc1 (Var)", "Penalty w bc3 (Var)"]


    _logger.info(f"Saving experimental data to {EXPERIMENT_DIR}")

    # EXPORT RESULTS TO EXCEL FILE
    experimental_data = pd.DataFrame(_experimental_data, columns=columns)
    experimental_data.to_excel(f'{EXPERIMENT_DIR}/varying_a.xlsx', index=False)

    # PRINT OUT RESULTS
    print(10*"*")
    print("Results")
    print(experimental_data)
    print(10*"*")
    print("Details on non-converged experiments")
    for el in non_converged_list: print(el)
    print(10*"*")
    #pdb.set_trace()

    # PLOTS L2 NORM
    FIGWIDTH = 15
    FIGHIGHT = 10
    FONTSIZE = 20
    MARKERSIZE = 17
    LINEWIDTH = 3
    x_values = []
    for element in x_range_plot:
        if element in a_range: x_values.append(element)

    plt.figure(figsize=(FIGWIDTH, FIGHIGHT))
    plt.xticks(a_range, [str(tick) if tick in x_values else '' for tick in a_range])
    plt.plot(a_range, experimental_data["L2norm v (Variational)"], marker='o', linestyle='solid', color='b', label='L2norm v (Variational)', linewidth=LINEWIDTH, markersize=MARKERSIZE)
    plt.plot(a_range, experimental_data["L2norm v (Brenner)"], marker='v', linestyle='dotted', color='r', label='L2norm v (Brenner)', linewidth=LINEWIDTH, markersize=MARKERSIZE)
    plt.plot(a_range, experimental_data["L2norm v (Carstensen)"], marker='^', linestyle='dashed', color='g', label='L2norm v (Carstensen)', linewidth=LINEWIDTH, markersize=MARKERSIZE)

    max_v = max([max(experimental_data["L2norm v (Variational)"]), max(experimental_data["L2norm v (Brenner)"]), max(experimental_data["L2norm v (Carstensen)"])])
    min_v = min([min(experimental_data["L2norm v (Variational)"]), min(experimental_data["L2norm v (Brenner)"]), min(experimental_data["L2norm v (Carstensen)"])])
    steps_v = (max_v - min_v)/NUM_STEPS
    if steps_v == 0: steps_v = NUM_STEPS # CFe: v is constant
    xrange_list = []

    if LOG_SCALE: plt.xscale('log') # CFe: use log xscale when needed
    plt.xticks(a_range, [str(tick) if tick in x_values else '' for tick in a_range])
    plt.yticks(np.arange(min_v, max_v, steps_v))
    plt.xlabel("a := R/h", fontsize=FONTSIZE)
    plt.ylabel(r'$| v |_{L^2(\Omega)} [Nm^2]$', fontsize=FONTSIZE)
    plt.title(f'Airy function. Mesh size = {parameters["geometry"]["mesh_size"]}. IP = {parameters["model"]["alpha_penalty"]}. tol = {SOL_TOLLERANCE}', fontsize=FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.grid(True)
    plt.gca().yaxis.get_offset_text().set_fontsize(FONTSIZE)
    plt.savefig(EXPERIMENT_DIR+f'/varying_a_V_{info_experiment}.png', dpi=300)
    #plt.show()

    plt.figure(figsize=(FIGWIDTH, FIGHIGHT))
    plt.plot(a_range, experimental_data["L2norm w (Variational)"], marker='o', linestyle='solid', color='b', label='L2norm w (Variational)', linewidth=LINEWIDTH, markersize=MARKERSIZE)
    plt.plot(a_range, experimental_data["L2norm w (Brenner)"], marker='v', linestyle='dotted', color='r', label='L2norm w (Brenner)', linewidth=LINEWIDTH, markersize=MARKERSIZE)
    plt.plot(a_range, experimental_data["L2norm w (Carstensen)"], marker='^', linestyle='dashed', color='g', label='L2norm w (Carstensen)', linewidth=LINEWIDTH, markersize=MARKERSIZE)

    max_w = max([max(experimental_data["L2norm w (Variational)"]), max(experimental_data["L2norm w (Brenner)"]), max(experimental_data["L2norm w (Carstensen)"])])
    min_w = min([min(experimental_data["L2norm w (Variational)"]), min(experimental_data["L2norm w (Brenner)"]), min(experimental_data["L2norm w (Carstensen)"])])
    steps_w = (max_w - min_w)/NUM_STEPS
    if steps_w == 0: steps_w = NUM_STEPS # CFe: if the deflection is not activated

    if LOG_SCALE: plt.xscale('log') # CFe: use log xscale when needed
    plt.xticks(a_range, [str(tick) if tick in x_values else '' for tick in a_range])
    plt.yticks(np.arange(min_w, max_w, steps_w))
    plt.xlabel("a := R/h", fontsize=FONTSIZE)
    plt.ylabel(r'$| w |_{L^2(\Omega)} [m^2]$', fontsize=FONTSIZE)
    plt.title(f'Transverse displacement. Mesh size = {parameters["geometry"]["mesh_size"]}. IP = {parameters["model"]["alpha_penalty"]}. tol = {SOL_TOLLERANCE}', fontsize=FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.gca().yaxis.get_offset_text().set_fontsize(FONTSIZE)
    plt.grid(True)
    plt.savefig(EXPERIMENT_DIR+f'/varying_a_W_{info_experiment}.png', dpi=300)
    #plt.show()

    # PLOTS HESSIAN L2 NORM
    plt.figure(figsize=(FIGWIDTH, FIGHIGHT))
    plt.plot(a_range, experimental_data["Hessian L2norm v (Variational)"], marker='o', linestyle='solid', color='b', label='Hessian L2norm v (Variational)', linewidth=LINEWIDTH, markersize=MARKERSIZE)
    plt.plot(a_range, experimental_data["Hessian L2norm v (Brenner)"], marker='v', linestyle='dotted', color='r', label='Hessian L2norm v (Brenner)', linewidth=LINEWIDTH, markersize=MARKERSIZE)
    plt.plot(a_range, experimental_data["Hessian L2norm v (Carstensen)"], marker='^', linestyle='dashed', color='g', label='Hessian L2norm v (Carstensen)', linewidth=LINEWIDTH, markersize=MARKERSIZE)

    max_v = max([max(experimental_data["Hessian L2norm v (Variational)"]), max(experimental_data["Hessian L2norm v (Brenner)"]), max(experimental_data["Hessian L2norm v (Carstensen)"])])
    min_v = min([min(experimental_data["Hessian L2norm v (Variational)"]), min(experimental_data["Hessian L2norm v (Brenner)"]), min(experimental_data["Hessian L2norm v (Carstensen)"])])

    steps_v = (max_v - min_v)/NUM_STEPS
    if steps_v == 0: steps_v = NUM_STEPS # CFe: v is constant

    if LOG_SCALE: plt.xscale('log') # CFe: use log xscale when needed
    plt.xticks(a_range, [str(tick) if tick in x_values else '' for tick in a_range])
    plt.yticks(np.arange(min_v, max_v, steps_v))
    plt.xlabel("a := R/h", fontsize=FONTSIZE)
    plt.ylabel(r'$| \nabla^2 v |_{L^2(\Omega)} [N]$', fontsize=FONTSIZE)
    plt.title(f'Airy function. Mesh size = {parameters["geometry"]["mesh_size"]}. IP = {parameters["model"]["alpha_penalty"]}. tol = {SOL_TOLLERANCE}', fontsize=FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.gca().yaxis.get_offset_text().set_fontsize(FONTSIZE)
    plt.grid(True)
    plt.savefig(EXPERIMENT_DIR+f'/varying_a_HessV_{info_experiment}.png', dpi=300)
    #plt.show()

    plt.figure(figsize=(FIGWIDTH, FIGHIGHT))
    plt.plot(a_range, experimental_data["Hessian L2norm w (Variational)"], marker='o', linestyle='solid', color='b', label='Hessian L2norm w (Variational)', linewidth=LINEWIDTH, markersize=MARKERSIZE)
    plt.plot(a_range, experimental_data["Hessian L2norm w (Brenner)"], marker='v', linestyle='dotted', color='r', label='Hessian L2norm w (Brenner)', linewidth=LINEWIDTH, markersize=MARKERSIZE)
    plt.plot(a_range, experimental_data["Hessian L2norm w (Carstensen)"], marker='^', linestyle='dashed', color='g', label='Hessian L2norm w (Carstensen)', linewidth=LINEWIDTH, markersize=MARKERSIZE)

    max_w = max([max(experimental_data["Hessian L2norm w (Variational)"]), max(experimental_data["Hessian L2norm w (Brenner)"]), max(experimental_data["Hessian L2norm w (Carstensen)"])])
    min_w = min([min(experimental_data["Hessian L2norm w (Variational)"]), min(experimental_data["Hessian L2norm w (Brenner)"]), min(experimental_data["Hessian L2norm w (Carstensen)"])])

    steps_w = (max_w - min_w)/NUM_STEPS
    if steps_w == 0: steps_w = NUM_STEPS # CFe: w is constant

    if LOG_SCALE: plt.xscale('log') # CFe: use log xscale when needed
    plt.xticks(a_range, [str(tick) if tick in x_values else '' for tick in a_range])
    plt.yticks(np.arange(min_w, max_w, steps_w))
    plt.xlabel("a := R/h", fontsize=FONTSIZE)
    plt.ylabel(r'$| \nabla^2 w |_{L^2(\Omega)} $', fontsize=FONTSIZE)
    plt.title(f'Transverse displacement. Mesh size = {parameters["geometry"]["mesh_size"]}. IP = {parameters["model"]["alpha_penalty"]}. tol = {SOL_TOLLERANCE}', fontsize=FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.gca().yaxis.get_offset_text().set_fontsize(FONTSIZE)
    plt.grid(True)
    plt.savefig(EXPERIMENT_DIR+f'/varying_a_HessW_{info_experiment}.png', dpi=300)
    #plt.show()

    # PLOT MEMBRANE ENERGY
    plt.figure(figsize=(FIGWIDTH, FIGHIGHT))
    plt.plot(a_range, experimental_data["Membrane Erg (Var)"], marker='o', linestyle='solid', color='b', label='Membrane Erg (Var)', linewidth=LINEWIDTH, markersize=MARKERSIZE)

    max_memEng = max(experimental_data["Membrane Erg (Var)"])
    min_memEng = min(experimental_data["Membrane Erg (Var)"])

    steps_memEng = abs( (max_memEng - min_memEng)/NUM_STEPS )
    if steps_memEng == 0: steps_memEng = NUM_STEPS
    if LOG_SCALE: plt.xscale('log') # CFe: use log xscale when needed
    plt.xticks(a_range, [str(tick) if tick in x_values else '' for tick in a_range])
    plt.yticks(np.arange(min_memEng, max_memEng, steps_memEng))
    plt.xlabel("a := R/h", fontsize=FONTSIZE)
    plt.ylabel('Membrane Energy', fontsize=FONTSIZE)
    plt.title(f'Membrane Energy. Mesh size = {parameters["geometry"]["mesh_size"]}. IP = {parameters["model"]["alpha_penalty"]}. tol = {SOL_TOLLERANCE}', fontsize=FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.gca().yaxis.get_offset_text().set_fontsize(FONTSIZE)
    plt.grid(True)
    plt.savefig(EXPERIMENT_DIR+f'/varying_a_MembEng_{info_experiment}.png', dpi=300)

    # PLOT BENDING ENERGY
    plt.figure(figsize=(FIGWIDTH, FIGHIGHT))
    plt.plot(a_range, experimental_data["Bending Erg (Var)"], marker='o', linestyle='solid', color='b', label='Bending Erg (Var)', linewidth=LINEWIDTH, markersize=MARKERSIZE)
    plt.axhline(y=0, color='black', linewidth=2)
    max_bendEng = max(experimental_data["Bending Erg (Var)"])
    min_bendEng = min(experimental_data["Bending Erg (Var)"])
    steps_bendEng = (max_bendEng - min_bendEng)/NUM_STEPS
    if steps_bendEng == 0: steps_bendEng = NUM_STEPS
    if LOG_SCALE: plt.xscale('log') # CFe: use log xscale when needed
    plt.xticks(a_range, [str(tick) if tick in x_values else '' for tick in a_range])
    plt.yticks(np.arange(min_bendEng, max_bendEng, steps_bendEng))
    plt.xlabel("a := R/h", fontsize=FONTSIZE)
    plt.ylabel('Bending Energy', fontsize=FONTSIZE)
    plt.title(f'Bending Energy. Mesh size = {parameters["geometry"]["mesh_size"]}. IP = {parameters["model"]["alpha_penalty"]}. tol = {SOL_TOLLERANCE}', fontsize=FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.gca().yaxis.get_offset_text().set_fontsize(FONTSIZE)
    plt.grid(True)
    plt.savefig(EXPERIMENT_DIR+f'/varying_a_BendEng_{info_experiment}.png', dpi=300)

    # PLOT COUPLING ENERGY
    plt.figure(figsize=(FIGWIDTH, FIGHIGHT))
    plt.plot(a_range, experimental_data["Coupling Erg (Var)"], marker='o', linestyle='solid', color='b', label='Coupling Erg (Var)', linewidth=LINEWIDTH, markersize=MARKERSIZE)
    plt.axhline(y=0, color='black', linewidth=2)

    max_couplingEng = max(experimental_data["Coupling Erg (Var)"])
    min_couplingEng = min(experimental_data["Coupling Erg (Var)"])

    steps_couplingEng = (max_couplingEng - min_couplingEng)/NUM_STEPS
    if steps_couplingEng == 0: steps_couplingEng = NUM_STEPS

    if LOG_SCALE: plt.xscale('log') # CFe: use log xscale when needed
    plt.xticks(a_range, [str(tick) if tick in x_values else '' for tick in a_range])
    plt.yticks(np.arange(min_couplingEng, max_couplingEng, steps_couplingEng))
    plt.xlabel("a := R/h", fontsize=FONTSIZE)
    plt.ylabel('Coupling Energy', fontsize=FONTSIZE)
    plt.title(f'Coupling Energy. Mesh size = {parameters["geometry"]["mesh_size"]}. IP = {parameters["model"]["alpha_penalty"]}. tol = {SOL_TOLLERANCE}', fontsize=FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.gca().yaxis.get_offset_text().set_fontsize(FONTSIZE)
    plt.grid(True)
    plt.savefig(EXPERIMENT_DIR+f'/varying_a_CouplEng_{info_experiment}.png', dpi=300)
    #plt.show()

    # PLOT PENALIZATION ENERGY

    # Penalization energy of Transverse Displacement w
    plt.figure(figsize=(FIGWIDTH, FIGHIGHT))
    plt.plot(a_range, experimental_data["Penalization W Eng (Var)"], marker='o', linestyle='solid', color='b', label='Penalization W Eng (Var)', linewidth=LINEWIDTH, markersize=MARKERSIZE)

    max_penalizWEng = max(experimental_data["Penalization W Eng (Var)"])
    min_penalizWEng = min(experimental_data["Penalization W Eng (Var)"])

    steps_penalizWEng = (max_penalizWEng - min_penalizWEng)/NUM_STEPS
    if steps_penalizWEng == 0: steps_penalizWEng = NUM_STEPS # CFe: w is constant

    if LOG_SCALE: plt.xscale('log') # CFe: use log xscale when needed
    plt.xticks(a_range, [str(tick) if tick in x_values else '' for tick in a_range])
    plt.yticks(np.arange(min_penalizWEng, max_penalizWEng, steps_penalizWEng))
    plt.xlabel("a := R/h", fontsize=FONTSIZE)
    plt.ylabel('Penalization Energy w', fontsize=FONTSIZE)
    plt.title(f'Penalization Energy w. Mesh size = {parameters["geometry"]["mesh_size"]}. IP = {parameters["model"]["alpha_penalty"]}. tol = {SOL_TOLLERANCE}', fontsize=FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.gca().yaxis.get_offset_text().set_fontsize(FONTSIZE)
    plt.grid(True)
    plt.savefig(EXPERIMENT_DIR+f'/varying_a_PenEngW_{info_experiment}.png', dpi=300)

    # Penalization energy of the Jump of the Hessian of the Transverse Displacement w
    plt.figure(figsize=(FIGWIDTH, FIGHIGHT))
    plt.plot(a_range, experimental_data["Penalization W Hess Jump (Var)"], marker='o', linestyle='solid', color='b', label='Penalization W Hess Jump (Var)', linewidth=LINEWIDTH, markersize=MARKERSIZE)

    max_penalizWEng = max(experimental_data["Penalization W Hess Jump (Var)"])
    min_penalizWEng = min(experimental_data["Penalization W Hess Jump (Var)"])

    steps_penalizWEng = (max_penalizWEng - min_penalizWEng)/NUM_STEPS
    if steps_penalizWEng == 0: steps_penalizWEng = NUM_STEPS # CFe: w is constant

    if LOG_SCALE: plt.xscale('log') # CFe: use log xscale when needed
    plt.xticks(a_range, [str(tick) if tick in x_values else '' for tick in a_range])
    plt.yticks(np.arange(min_penalizWEng, max_penalizWEng, steps_penalizWEng))
    plt.xlabel("a := R/h", fontsize=FONTSIZE)
    plt.ylabel('Penalization Hessian w', fontsize=FONTSIZE)
    plt.title(f'Penalization Energy w. Mesh size = {parameters["geometry"]["mesh_size"]}. IP = {parameters["model"]["alpha_penalty"]}. tol = {SOL_TOLLERANCE}', fontsize=FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.gca().yaxis.get_offset_text().set_fontsize(FONTSIZE)
    plt.grid(True)
    plt.savefig(EXPERIMENT_DIR+f'/varying_a_PenHessJumpW_{info_experiment}.png', dpi=300)

    # Penalization energy of Airy's function v
    plt.figure(figsize=(FIGWIDTH, FIGHIGHT))
    plt.plot(a_range, experimental_data["Penalization V Eng (Var)"], marker='o', linestyle='solid', color='b', label='Penalization V Eng (Var)', linewidth=LINEWIDTH, markersize=MARKERSIZE)

    max_penalizVEng = max(experimental_data["Penalization V Eng (Var)"])
    min_penalizVEng = min(experimental_data["Penalization V Eng (Var)"])
    steps_penalizVEng = (max_penalizVEng - min_penalizVEng)/NUM_STEPS
    if steps_penalizVEng == 0: steps_penalizVEng = NUM_STEPS # CFe: w is constant

    if LOG_SCALE: plt.xscale('log') # CFe: use log xscale when needed
    plt.xticks(a_range, [str(tick) if tick in x_values else '' for tick in a_range])
    plt.yticks(np.arange(min_penalizVEng, max_penalizVEng, steps_penalizVEng))
    plt.xlabel("a := R/h", fontsize=FONTSIZE)
    plt.ylabel('Penalization Energy v', fontsize=FONTSIZE)
    plt.title(f'Penalization Energy v. Mesh size = {parameters["geometry"]["mesh_size"]}. IP = {parameters["model"]["alpha_penalty"]}. tol = {SOL_TOLLERANCE}', fontsize=FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.gca().yaxis.get_offset_text().set_fontsize(FONTSIZE)
    plt.grid(True)
    plt.savefig(EXPERIMENT_DIR+f'/varying_a_PenEngV_{info_experiment}.png', dpi=300)

    # PLOT DIFFERENCES BETWEEN ENERGIES AND PENALTY ENERGIES

    w_dg1_var_list = experimental_data["Penalty w dg1 (Var)"]
    w_dg2_var_list = experimental_data["Penalty w dg2 (Var)"]
    w_bc1_var_list = experimental_data["Penalty w bc1 (Var)"]
    w_bc3_var_list = experimental_data["Penalty w bc3 (Var)"]
    w_dg3_var_list = experimental_data["Penalization W Hess Jump (Var)"]

    fig, ax = plt.subplots(2, 2, figsize=(FIGWIDTH, FIGHIGHT))
    ax[0,0].plot(a_range, w_dg1_var_list, marker='o', linestyle='solid', color='b', label="w_dg1", linewidth=LINEWIDTH, markersize=MARKERSIZE)
    ax[0,0].legend(fontsize=FONTSIZE)
    ax[0,0].grid(True)
    ax[0,0].set_xlabel("a := R/h", fontsize=FONTSIZE)
    plt.gca().yaxis.get_offset_text().set_fontsize(FONTSIZE)
    ax[0,1].plot(a_range, w_dg2_var_list, marker='v', linestyle='solid', color='r', label="w_dg2", linewidth=LINEWIDTH, markersize=MARKERSIZE)
    ax[0,1].legend(fontsize=FONTSIZE)
    ax[0,1].grid(True)
    ax[0,1].set_xlabel("a := R/h", fontsize=FONTSIZE)
    plt.gca().yaxis.get_offset_text().set_fontsize(FONTSIZE)
    ax[1,0].plot(a_range, w_bc1_var_list, marker='P', linestyle='solid', color='k', label="w_bc1", linewidth=LINEWIDTH, markersize=MARKERSIZE)
    ax[1,0].legend(fontsize=FONTSIZE)
    ax[1,0].grid(True)
    plt.gca().yaxis.get_offset_text().set_fontsize(FONTSIZE)
    ax[1,0].set_xlabel("a := R/h", fontsize=FONTSIZE)
    ax[1,1].plot(a_range, w_bc3_var_list, marker='*', linestyle='solid', color='g', label="w_bc3", linewidth=LINEWIDTH, markersize=MARKERSIZE)
    ax[1,1].legend(fontsize=FONTSIZE)
    ax[1,1].grid(True)
    ax[1,1].set_xlabel("a := R/h", fontsize=FONTSIZE)
    plt.xticks(a_range, [str(tick) if tick in x_values else '' for tick in a_range])
    fig.suptitle(f'Penalization Energy v. Mesh size = {parameters["geometry"]["mesh_size"]}. IP = {parameters["model"]["alpha_penalty"]}. tol = {SOL_TOLLERANCE}', fontsize=FONTSIZE)
    plt.gca().yaxis.get_offset_text().set_fontsize(FONTSIZE)
    plt.savefig(EXPERIMENT_DIR+f'/varying_a_PenEngsW_{info_experiment}.png', dpi=300)

    # PLOT PROFILES

    tol = 1e-5
    xs = np.linspace(-parameters["geometry"]["radius"] + tol, parameters["geometry"]["radius"] - tol, 3001)
    points = np.zeros((3, 3001))
    points[0] = xs

    for i, sol_w in enumerate(sol_w_list):

        fig, axes = plt.subplots(1, 1, figsize=(24, 18))
        axes.yaxis.get_offset_text().set_size(FONTSIZE) # Size of the order of magnitude
        plt, data = plot_profile(sol_w, points, None, subplot=(1, 1), lineproperties={"c": "r", "lw":5, "label": f"w_{i}", "ls": ":"}, fig=fig, subplotnumber=1)
        plt.xlabel("x", fontsize=FONTSIZE)
        plt.ylabel("Transverse displacement", fontsize=FONTSIZE)
        plt.xticks(fontsize=FONTSIZE)
        plt.yticks(fontsize=FONTSIZE)
        plt.title(f"Transverse displacement. Thickness = {1/a_range[i]} m; a = {a_range[i]}; Mesh = {parameters["geometry"]["mesh_size"]}. IP = {parameters["model"]["alpha_penalty"]}. E = {parameters["model"]["E"]:.2e} tol = {SOL_TOLLERANCE}", size = FONTSIZE)
        plt.grid(True)
        plt.legend(fontsize=FONTSIZE)
        plt.savefig(f"{EXPERIMENT_DIR}/_{a_range[i]}a-w-profiles_{info_experiment}.png")

        fig, axes = plt.subplots(1, 1, figsize=(24, 18))
        sol_w.vector.array = sol_w.vector.array / a_range[i]
        plt, data = plot_profile(sol_w, points, None, subplot=(1, 1), lineproperties={"c": "r", "lw":5, "label": f"w_{i}", "ls": ":"}, fig=fig, subplotnumber=1)
        plt.xlabel("x [m]", fontsize=FONTSIZE)
        plt.ylabel("Transverse displacement [m]", fontsize=FONTSIZE)
        plt.xticks(fontsize=FONTSIZE)
        plt.yticks(fontsize=FONTSIZE)
        plt.ylim(-1, 1)
        plt.title(f"Transverse displacement [1:1] aspect ratio. Thickness = {1/a_range[i]} m; a = {a_range[i]}; Mesh = {parameters["geometry"]["mesh_size"]}. IP = {parameters["model"]["alpha_penalty"]}", size = FONTSIZE)
        plt.grid(True)
        plt.legend(fontsize=FONTSIZE)
        plt.savefig(f"{EXPERIMENT_DIR}/_{a_range[i]}b-w-profiles_{info_experiment}.png")

        fig, axes = plt.subplots(1, 1, figsize=(24, 18))
        plt, data = plot_profile(sol_v_list[i], points, None, subplot=(1, 1), lineproperties={"c": "b", "lw":5, "label": f"v_{i}", "ls": ":"}, fig=fig, subplotnumber=1)
        plt.xlabel("x", fontsize=FONTSIZE)
        plt.ylabel("Airy", fontsize=FONTSIZE)
        plt.xticks(fontsize=FONTSIZE)
        plt.yticks(fontsize=FONTSIZE)
        plt.title(f"Airy. Thickness = {1/a_range[i]} m; a = {a_range[i]}; Mesh = {parameters["geometry"]["mesh_size"]}. IP = {parameters["model"]["alpha_penalty"]}. E = {parameters["model"]["E"]:.2e} tol = {SOL_TOLLERANCE}", size = FONTSIZE)
        plt.grid(True)
        plt.legend(fontsize=FONTSIZE)
        plt.savefig(f"{EXPERIMENT_DIR}/_{a_range[i]}a-v-profiles_{info_experiment}.png")
