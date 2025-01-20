"""
PURPOSE OF THE SCRIPT
Run experiments by varying "a^4 * b" while keeping constant a^2.
The three FE formulation(Variational, Brenner, Carstensen) in their NON-dimensional form (see "models/adimensional.py") are used.
a := R/h
b := p0/E
c := a^4 b

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
from matplotlib.ticker import PercentFormatter, ScalarFormatter, MaxNLocator
import pyvista

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

from visuals import visuals
visuals.matplotlibdefaults(useTex=False)

# OUTPUT DIRECTORY
OUTDIR = os.path.join("output", "experiment_parametric_gamma") # CFe: output directory
PATH_TO_PARAMETERS_YML_FILE = 'disclinations.test'

# NON LINEAR SEARCH TOLLERANCES
ABS_TOLLERANCE = 1e-12 # Absolute tolerance
REL_TOLLERANCE = 1e-12  # Relative tolerance
SOL_TOLLERANCE = 1e-12  # Solution tolerance

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

X_COORD = 0
Y_COORD = 1

NUM_STEPS = 10 # Number ticks in the yaxis

SMOOTHING = False

COMPARISON = False # Comparison between FE models

CostantData = namedtuple("CostantData", ["funcSpace", "bcs"])

petsc4py.init(["-petsc_type", "double"]) # CFe: ensuring double precision numbers

def L2norm(u):
    return np.sqrt( fem.assemble_scalar( fem.form( ufl.inner(u, u) * ufl.dx ) ) )

def hessianL2norm(u):
    hessian = lambda f : ufl.grad(ufl.grad(u))
    return np.sqrt( fem.assemble_scalar( fem.form( ufl.inner(hessian(u), hessian(u)) * ufl.dx ) ) )

def ouward_unit_normal_x(x): return x[X_COORD] / np.sqrt(x[X_COORD]**2 + x[Y_COORD]**2)
def ouward_unit_normal_y(x): return x[Y_COORD] / np.sqrt(x[X_COORD]**2 + x[Y_COORD]**2)

def run_experiment(FEMmodel, costantData, parameters, q_ig, gamma, beta):
    """
    Purpose: parametric study of the solution by varying f := (R/h)^4 p0/E
    """

    # UPDATE EXTERNAL LOAD
    f0 = gamma * ( beta ** 4 )
    def nondim_transverse_load(x): return f0 * LOAD_SIGN *(1.0 + 0.0*x[0] + 0.0*x[1])
    print("f0 = ", f0)
    Q = costantData.funcSpace
    f = dolfinx.fem.Function(Q.sub(TRANSVERSE).collapse()[0])
    f.interpolate(nondim_transverse_load)

    # FUNCTION SPACE
    q = dolfinx.fem.Function(Q)
    #q.vector.array = copy.deepcopy(q_ig.vector.array)
    #q.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    v, w = ufl.split(q)
    state = {"v": v, "w": w}

    # SELECT MODEL
    if FEMmodel == CARSTENSEN:
        model = A_NonlinearPlateFVK_carstensen(mesh, parameters["model"])
    elif FEMmodel == BRENNER:
        model = A_NonlinearPlateFVK_brenner(mesh, parameters["model"])
    elif FEMmodel == VARIATIONAL:
        model = A_NonlinearPlateFVK(mesh, parameters["model"], smooth = SMOOTHING)
    else:
        print(f"FEMmodel = {FEMmodel} is not acceptable. Script exiting")
        exit(0)

    # INSERT DISCLINATIONS
    disclinations = []
    #beta = 1/parameters["model"]["thickness"]
    a = beta
    if mesh.comm.rank == 0:
        for dc in DISCLINATION_COORD_LIST: disclinations.append( np.array([dc], dtype=mesh.geometry.x.dtype))
        dp_list = [dp*(beta)**2.0 for dp in DISCLINATION_POWER_LIST]
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

    print("bending_energy = ", energy_terms["bending"])
    print("membrane_energy = ", energy_terms["membrane"])
    print("coupling_energy = ", energy_terms["coupling"])

    v_var, w_var = ufl.split(q_ig)
    state_var = {"v": v_var, "w": w_var}
    bending_energy_var = model.compute_bending_energy(state_var, COMM),
    membrane_energy_var = model.compute_membrane_energy(state_var, COMM),
    coupling_energy_var = model.compute_coupling_energy(state_var, COMM),
    print("bending_energy_var = ", bending_energy_var)
    print("membrane_energy_var = ", membrane_energy_var)
    print("coupling_energy_var = ", coupling_energy_var)

    # Compute the normal derivative for v and w
    from ufl import dot, grad
    n = ufl.FacetNormal(mesh)
    normal_derivative_v = ufl.dot(ufl.grad(v), n)
    normal_derivative_w = ufl.dot(ufl.grad(w), n)
    ds = ufl.Measure("ds", domain=mesh)
    boundary_integral_w = fem.assemble_scalar(fem.form((w**2) * ds))
    boundary_integral_v = fem.assemble_scalar(fem.form((v**2) * ds))
    boundary_integral_normal_der_w = fem.assemble_scalar(fem.form((normal_derivative_w**2) * ds))
    boundary_integral_normal_der_v = fem.assemble_scalar(fem.form((normal_derivative_v**2) * ds))
    print("boundary_integral_v: ", boundary_integral_v)
    print("boundary_integral_w: ", boundary_integral_w)
    print("boundary_integral_normal_der_v: ", boundary_integral_normal_der_v)
    print("boundary_integral_w: ", boundary_integral_normal_der_w)

     # FREE MEMORY
    solver.solver.destroy()
    gc.collect()

    return return_value_dic


if __name__ == "__main__":

    PARAMETER_NAME = "p0"
    PARAMETER_CATEGORY = "model"
    NUM_RUNS = 15

    print("Smoothing: ", SMOOTHING)

    f_list = [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000]
    gamma_list = [6.25E-7, 1.875E-6, 6.25E-6, 1.875E-5, 6.25E-5, 1.875E-4, 6.25E-4, 1.875E-3, 6.25E-3, 1.875E-2]
    gamma_range = []

    #x_range_plot = [0.1, 1, 10, 100, 1000]
    #x_range_plot_log = [0.1, 1, 10, 100, 1000, 3000]
    x_range_plot = [6.25E-7, 6.25E-6, 6.25E-5, 6.25E-4, 6.25E-3, 1.875E-2]
    x_range_plot_log = [6.25E-7, 6.25E-6, 6.25E-5, 6.25E-4, 6.25E-3, 1.875E-2]

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

    #parameters["model"]["thickness"] = 0.05
    #parameters["geometry"]["mesh_size"] = 0.03

    thickness = parameters["model"]["thickness"]
    mesh_size = parameters["geometry"]["mesh_size"]
    IP = parameters["model"]["alpha_penalty"]
    nu = parameters["model"]["nu"] # Poisson ratio
    c_nu =  1/(12*(1-nu**2))
    E = parameters["model"]["E"] # Young's modulus
    R = parameters["geometry"]["radius"] # Plate's radius

    beta = R/thickness
    a = beta

    print("mesh size: ", mesh_size)
    print("beta: ", beta)
    print("Thickness: ", thickness)
    #series = base_signature[0::6]

    info_experiment = f"mesh_{mesh_size}_IP_{IP:.2e}_smth_{SMOOTHING}_tol={SOL_TOLLERANCE}_load={LOAD_SIGN}_s={DISCLINATION_POWER_LIST}"

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
    V_v, dofs2_v = Q.sub(AIRY).collapse()
    V_w, dofs2_w = Q.sub(1).collapse()

    # INITIAL GUESS
    q_ig = dolfinx.fem.Function(Q)

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
    w_varModel_list = []
    v_varModel_list = []
    w_brnModel_list = []
    v_brnModel_list = []
    w_carModel_list = []
    v_carModel_list = []

    # LOOP THROUGH ALL C VALUES
    #f_value replaced for gamma_value
    for i, gamma_value in enumerate(gamma_list):
        print("Running experiment with gamma = ", gamma_value)
        #q_ig.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        data_var = run_experiment(VARIATIONAL, costantData, parameters, q_ig, gamma_value, beta)
        if COMPARISON:
            data_brn = run_experiment(BRENNER, costantData, parameters, q_ig, gamma_value, beta)
            data_car = run_experiment(CARSTENSEN, costantData, parameters, q_ig, gamma_value, beta)
        else:
            data_brn = data_var
            data_car = data_var
        #q_ig.vector.array = copy.deepcopy(data_brn["q"].vector.array)
        #q_ig.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        #q_ig.vector.array = copy.deepcopy(data_brn["q"].vector.array)
        #q_ig.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        if (data_var["convergence_id"] > 0) and (data_brn["convergence_id"] > 0) and (data_car["convergence_id"] > 0):
            #f_range replaced for gamma_range
            gamma_range.append(gamma_value)
            _experimental_data.append(
                [
                    gamma_value,
                    data_var["convergence_id"], data_brn["convergence_id"], data_car["convergence_id"],
                    data_var["v_Nrm"], data_brn["v_Nrm"], data_car["v_Nrm"],
                    data_var["w_Nrm"], data_brn["w_Nrm"], data_car["w_Nrm"],
                    data_var["v_hessianNrm"], data_brn["v_hessianNrm"], data_car["v_hessianNrm"],
                    data_var["w_hessianNrm"], data_brn["w_hessianNrm"], data_car["w_hessianNrm"],
                    data_var["bending_energy"], data_var["membrane_energy"], data_var["coupling_energy"], data_var["penalty_energy"],
                    data_brn["bending_energy"], data_brn["membrane_energy"], data_brn["coupling_energy"], data_brn["penalty_energy"],
                    data_car["bending_energy"], data_car["membrane_energy"], data_car["coupling_energy"], data_car["penalty_energy"],
                    data_var["penalty_w_tot"], data_var["penalty_v_tot"], data_var["penalty_coupling"],
                    data_brn["penalty_w_tot"], data_brn["penalty_v_tot"], data_brn["penalty_coupling"],
                    data_car["penalty_w_tot"], data_car["penalty_v_tot"], data_car["penalty_coupling"],
                    data_var["penalty_w_dg1"], data_var["penalty_w_dg2"], data_var["penalty_w_bc1"], data_var["penalty_w_bc3"],
                    data_brn["penalty_w_dg1"], data_brn["penalty_w_dg2"], data_brn["penalty_w_bc1"], data_brn["penalty_w_bc3"],
                    data_car["penalty_w_dg1"], data_car["penalty_w_dg2"], data_car["penalty_w_bc1"], data_car["penalty_w_bc3"],
                    data_var["penalty_w_hessJump"], data_brn["penalty_w_hessJump"], data_car["penalty_w_hessJump"],
                    parameters["model"]["thickness"], parameters["model"]["E"], parameters["model"]["nu"],
                    parameters["geometry"]["mesh_size"], parameters["model"]["alpha_penalty"],
                    ]
                )

            w_varModel_list.append(data_var["w"])
            w_brnModel_list.append(data_brn["w"])
            w_carModel_list.append(data_car["w"])
            v_varModel_list.append(data_var["v"])
            v_brnModel_list.append(data_brn["v"])
            v_carModel_list.append(data_car["v"])

            # Update the initial guess
            #q_ig.vector.array = copy.deepcopy(data_var["q"].vector.array)
            #q_ig.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        else:
            print(f"At least one of the three FEM models did not converged for {PARAMETER_NAME} = {gamma_value}")
            print("Convergence Variational model: ", data_var["convergence_id"])
            print("Convergence Brenner model: ", data_brn["convergence_id"])
            print("Convergence Carstensen model: ", data_car["convergence_id"])
            non_converged_list.append({f"{PARAMETER_NAME}": gamma_value, "Var_Con_ID": data_var["convergence_id"], "Brn_Con_ID": data_brn["convergence_id"], "Car_Con_ID": data_car["convergence_id"]})

    columns = ["c", "Convergence ID (Variational)", "Convergence ID (Brenner)", "Convergence ID (Carstensen)",
                "L2norm v (Variational)", "L2norm v (Brenner)", "L2norm v (Carstensen)",
                "L2norm w (Variational)", "L2norm w (Brenner)", "L2norm w (Carstensen)",
                "Hessian L2norm v (Variational)", "Hessian L2norm v (Brenner)", "Hessian L2norm v (Carstensen)",
                "Hessian L2norm w (Variational)", "Hessian L2norm w (Brenner)", "Hessian L2norm w (Carstensen)",
                "Bending Erg (Var)", "Membrane Erg (Var)", "Coupling Erg (Var)", "Penalization Eng (Var)",
                "Bending Erg (Brn)", "Membrane Erg (Brn)", "Coupling Erg (Brn)", "Penalization Eng (Brn)",
                "Bending Erg (Car)", "Membrane Erg (Car)", "Coupling Erg (Car)", "Penalization Eng (Car)",
                "Penalization W Eng (Var)", "Penalization V Eng (Var)", "Penalization coupling Eng (Var)",
                "Penalization W Eng (Brn)", "Penalization V Eng (Brn)", "Penalization coupling Eng (Brn)",
                "Penalization W Eng (Car)", "Penalization V Eng (Car)", "Penalization coupling Eng (Car)",
                "Penalty w dg1 (Var)", "Penalty w dg2 (Var)", "Penalty w bc1 (Var)", "Penalty w bc3 (Var)",
                "Penalty w dg1 (Brn)", "Penalty w dg2 (Brn)", "Penalty w bc1 (Brn)", "Penalty w bc3 (Brn)",
                "Penalty w dg1 (Car)", "Penalty w dg2 (Car)", "Penalty w bc1 (Car)", "Penalty w bc3 (Car)",
                "Penalization W Hess Jump (Var)", "Penalization W Hess Jump (Brn)", "Penalization W Hess Jump (Car)",
                "Thickness", "Young modulus", "Poisson ratio", "Mesh size", "Interior Penalty (IP)"
                ]

    _logger.info(f"Saving experimental data to {EXPERIMENT_DIR}")

    # Bending and membrane energies predicted by Kircchoff-Love model
    kMembraneEng = (DISCLINATION_POWER_LIST[0]**2)*(beta**4)/(32*np.pi)
    kMembraneEngList = [kMembraneEng for gamma in gamma_range]
    kBendingEngList = [( ( ( gamma* (beta)**4 )**2) * np.pi * (R**6) ) / ( 384 * c_nu ) for gamma in gamma_range]

    # Compute sigma_nn stress field
    V_v, dofs2_v = Q.sub(AIRY).collapse()
    V_w, dofs2_w = Q.sub(TRANSVERSE).collapse()
    sigma_xx = dolfinx.fem.Function(V_v)
    sigma_xy = dolfinx.fem.Function(V_v)
    sigma_yy = dolfinx.fem.Function(V_v)

    # Contribute to the total membranale stress due to the transverse displacement
    #sigmaxx_w = ( Eyoung / (1-nu**2) ) * (1/2) * ( ufl.grad( sol_w_list[index_solution] )[0]**2 + nu * ufl.grad( sol_w_list[index_solution] )[1]**2 )
    #sigmayy_w = ( Eyoung / (1-nu**2) ) * (1/2) * ( ufl.grad( sol_w_list[index_solution] )[1]**2 + nu * ufl.grad( sol_w_list[index_solution] )[0]**2 )
    #sigmaxy_w = ( Eyoung / (1-nu**2) ) * (1/2) * ( ufl.grad( sol_w_list[index_solution] )[0]  * ufl.grad( sol_w_list[index_solution] )[1] )
    import pyvista
    scalar_bar_args = {
        "vertical": True,
        "title_font_size": 30,  # Increase the title font size
        "label_font_size": 30,  # Increase the label font size
        "width": 0.18,          # Adjust the width of the scalar bar
        "height": 0.8,          # Adjust the height of the scalar bar
        "position_x": 0.9,      # X position (between 0 and 1, relative to the viewport)
        "position_y": 0.1       # Y position (between 0 and 1, relative to the viewport)
    }

    #Q_v, Q_v_to_Q_dofs = Q.sub(AIRY).collapse()
    index_solution = -1 # Last solution corresponding to the maximum value of f
    topology, cells, geometry = dolfinx.plot.vtk_mesh(V_v)
    grid = pyvista.UnstructuredGrid(topology, cells, geometry)
    def hessian(u): return ufl.grad(ufl.grad(u))
    sigma_xx_expr = dolfinx.fem.Expression( hessian(v_varModel_list[index_solution])[Y_COORD, Y_COORD], V_v.element.interpolation_points() )
    sigma_xy_expr = dolfinx.fem.Expression( - hessian(v_varModel_list[index_solution])[X_COORD, Y_COORD], V_v.element.interpolation_points() )
    sigma_yy_expr = dolfinx.fem.Expression( hessian(v_varModel_list[index_solution])[X_COORD, X_COORD], V_v.element.interpolation_points() )

    sigma_xx.interpolate(sigma_xx_expr)
    sigma_xy.interpolate(sigma_xy_expr)
    sigma_yy.interpolate(sigma_yy_expr)

    n_x = dolfinx.fem.Function(V_v)
    n_y = dolfinx.fem.Function(V_v)
    n_x.interpolate(ouward_unit_normal_x)
    n_y.interpolate(ouward_unit_normal_y)

    sigma_n_x = dolfinx.fem.Function(V_v)
    sigma_n_y = dolfinx.fem.Function(V_v)
    sigma_nx_expr = dolfinx.fem.Expression( sigma_xx*n_x + sigma_xy*n_y, V_v.element.interpolation_points() )
    sigma_ny_expr = dolfinx.fem.Expression( sigma_xy*n_x + sigma_yy*n_y, V_v.element.interpolation_points() )
    sigma_n_x.interpolate(sigma_nx_expr)
    sigma_n_y.interpolate(sigma_ny_expr)
    sigma_n = np.column_stack((sigma_n_x.x.array.real, sigma_n_y.x.array.real, np.zeros_like(sigma_n_x.x.array.real)))
    sigma_nn = dolfinx.fem.Function(V_v)
    sigma_nn_expr = dolfinx.fem.Expression( sigma_n_x*n_x + sigma_n_y*n_y , V_v.element.interpolation_points() )
    sigma_nn.interpolate(sigma_nn_expr)

    # EXPORT RESULTS TO EXCEL FILE
    experimental_data = pd.DataFrame(_experimental_data, columns=columns)
    experimental_data.to_excel(f'{EXPERIMENT_DIR}/vary_gamma.xlsx', index=False)



    # PRINT OUT RESULTS
    print(10*"*")
    print("Results")
    print(experimental_data)
    print(10*"*")
    print("Details on non-converged experiments")
    for el in non_converged_list: print(el)
    print(10*"*")

    # PLOTS L2 NORM
    FIGWIDTH = 15
    FIGHIGHT = 11
    FONTSIZE = 30
    MARKERSIZE = 20
    LINEWIDTH = 5
    x_values = []
    for element in x_range_plot:
        if element in gamma_range: x_values.append(element)

    if False:
        plt.figure(figsize=(FIGWIDTH, FIGHIGHT))
        plt.xticks(gamma_range, [str(tick) if tick in x_values else '' for tick in gamma_range])
        plt.plot(gamma_range, experimental_data["L2norm v (Variational)"], marker='o', linestyle='solid', color='b', label='VAR', linewidth=LINEWIDTH, markersize=MARKERSIZE)
        if COMPARISON: plt.plot(gamma_range, experimental_data["L2norm v (Brenner)"], marker='v', linestyle='dotted', color='r', label='BNRS17', linewidth=LINEWIDTH, markersize=MARKERSIZE)
        if COMPARISON: plt.plot(gamma_range, experimental_data["L2norm v (Carstensen)"], marker='^', linestyle='dashed', color='g', label='CMN18', linewidth=LINEWIDTH, markersize=MARKERSIZE)

        max_v = max([max(experimental_data["L2norm v (Variational)"]), max(experimental_data["L2norm v (Brenner)"]), max(experimental_data["L2norm v (Carstensen)"])])
        min_v = min([min(experimental_data["L2norm v (Variational)"]), min(experimental_data["L2norm v (Brenner)"]), min(experimental_data["L2norm v (Carstensen)"])])
        steps_v = (max_v - min_v)/NUM_STEPS
        if steps_v == 0: steps_v = NUM_STEPS # CFe: v is constant
        xrange_list = []

        if LOG_SCALE: plt.xscale('log') # CFe: use log xscale when needed
        plt.xticks(gamma_range, [str(tick) if tick in x_values else '' for tick in gamma_range])
        plt.yticks(np.arange(min_v, max_v, steps_v))
        plt.xlabel(r"$\gamma$ = $\frac{p_0}{E}$", fontsize=FONTSIZE)
        plt.ylabel(r'$| v |_{L^2(\Omega)} [Nm^2]$', fontsize=FONTSIZE)
        ax = plt.gca() # use scientific notation for y axis
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_fontsize(FONTSIZE)
        plt.title(r'$| v |_{L^2(\Omega)}'+f'. Mesh size = {mesh_size}. IP = {IP}. tol = {SOL_TOLLERANCE}', fontsize=FONTSIZE)
        plt.tick_params(axis='both', which='major', labelsize=FONTSIZE)
        plt.legend(fontsize=FONTSIZE)
        plt.grid(True)
        plt.gca().yaxis.get_offset_text().set_fontsize(FONTSIZE)
        plt.savefig(EXPERIMENT_DIR+f'/vary_gamma_V_{info_experiment}.png', dpi=300)
        #plt.show()


        plt.figure(figsize=(FIGWIDTH, FIGHIGHT))
        plt.plot(gamma_range, experimental_data["L2norm w (Variational)"], marker='o', linestyle='solid', color='b', label='VAR', linewidth=LINEWIDTH, markersize=MARKERSIZE)
        if COMPARISON: plt.plot(gamma_range, experimental_data["L2norm w (Brenner)"], marker='v', linestyle='dotted', color='r', label='BNRS17', linewidth=LINEWIDTH, markersize=MARKERSIZE)
        if COMPARISON: plt.plot(gamma_range, experimental_data["L2norm w (Carstensen)"], marker='^', linestyle='dashed', color='g', label='CMN18', linewidth=LINEWIDTH, markersize=MARKERSIZE)

        max_w = max([max(experimental_data["L2norm w (Variational)"]), max(experimental_data["L2norm w (Brenner)"]), max(experimental_data["L2norm w (Carstensen)"])])
        min_w = min([min(experimental_data["L2norm w (Variational)"]), min(experimental_data["L2norm w (Brenner)"]), min(experimental_data["L2norm w (Carstensen)"])])
        steps_w = (max_w - min_w)/NUM_STEPS
        if steps_w == 0: steps_w = NUM_STEPS # CFe: if the deflection is not activated

        if LOG_SCALE: plt.xscale('log') # CFe: use log xscale when needed
        plt.xticks(gamma_range, [str(tick) if tick in x_values else '' for tick in gamma_range])
        plt.yticks(np.arange(min_w, max_w, steps_w))
        plt.xlabel(r"$\gamma$ = $\frac{p_0}{E}$", fontsize=FONTSIZE)
        plt.ylabel(r'$| w |_{L^2(\Omega)} [m^2]$', fontsize=FONTSIZE)
        ax = plt.gca() # use scientific notation for y axis
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_fontsize(FONTSIZE)
        plt.title(r'$| w |_{L^2(\Omega)}'+f'. Mesh size = {mesh_size}. IP = {IP}. tol = {SOL_TOLLERANCE}', fontsize=FONTSIZE)
        plt.tick_params(axis='both', which='major', labelsize=FONTSIZE)
        plt.legend(fontsize=FONTSIZE)
        plt.gca().yaxis.get_offset_text().set_fontsize(FONTSIZE)
        plt.grid(True)
        plt.savefig(EXPERIMENT_DIR+f'/vary_gamma_W_{info_experiment}.png', dpi=300)
        #plt.show()

        # PLOTS HESSIAN L2 NORM
        plt.figure(figsize=(FIGWIDTH, FIGHIGHT))
        plt.plot(gamma_range, experimental_data["Hessian L2norm v (Variational)"], marker='o', linestyle='solid', color='b', label='VAR', linewidth=LINEWIDTH, markersize=MARKERSIZE)
        if COMPARISON: plt.plot(gamma_range, experimental_data["Hessian L2norm v (Brenner)"], marker='v', linestyle='dotted', color='r', label='BNRS17', linewidth=LINEWIDTH, markersize=MARKERSIZE)
        if COMPARISON: plt.plot(gamma_range, experimental_data["Hessian L2norm v (Carstensen)"], marker='^', linestyle='dashed', color='g', label='CMN18', linewidth=LINEWIDTH, markersize=MARKERSIZE)

        max_v = max([max(experimental_data["Hessian L2norm v (Variational)"]), max(experimental_data["Hessian L2norm v (Brenner)"]), max(experimental_data["Hessian L2norm v (Carstensen)"])])
        min_v = min([min(experimental_data["Hessian L2norm v (Variational)"]), min(experimental_data["Hessian L2norm v (Brenner)"]), min(experimental_data["Hessian L2norm v (Carstensen)"])])

        steps_v = (max_v - min_v)/NUM_STEPS
        if steps_v == 0: steps_v = NUM_STEPS # CFe: v is constant

        if LOG_SCALE: plt.xscale('log') # CFe: use log xscale when needed
        plt.xticks(gamma_range, [str(tick) if tick in x_values else '' for tick in gamma_range])
        plt.yticks(np.arange(min_v, max_v, steps_v))
        plt.xlabel(r"$\gamma$ = $\frac{p_0}{E}$", fontsize=FONTSIZE)
        plt.ylabel(r'$| \nabla^2 v |_{L^2(\Omega)} [N]$', fontsize=FONTSIZE)
        ax = plt.gca() # use scientific notation for y axis
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_fontsize(FONTSIZE)
        plt.title(r'$| \nabla^2 v |_{L^2(\Omega)}'+f'. Mesh size = {mesh_size}. IP = {IP}. tol = {SOL_TOLLERANCE}', fontsize=FONTSIZE)
        plt.tick_params(axis='both', which='major', labelsize=FONTSIZE)
        plt.legend(fontsize=FONTSIZE)
        plt.gca().yaxis.get_offset_text().set_fontsize(FONTSIZE)
        plt.grid(True)
        plt.savefig(EXPERIMENT_DIR+f'/vary_gamma_HessV_{info_experiment}.png', dpi=300)
        #plt.show()

        plt.figure(figsize=(FIGWIDTH, FIGHIGHT))
        plt.plot(gamma_range, experimental_data["Hessian L2norm w (Variational)"], marker='o', linestyle='solid', color='b', label='VAR', linewidth=LINEWIDTH, markersize=MARKERSIZE)
        if COMPARISON: plt.plot(gamma_range, experimental_data["Hessian L2norm w (Brenner)"], marker='v', linestyle='dotted', color='r', label='BNRS17', linewidth=LINEWIDTH, markersize=MARKERSIZE)
        if COMPARISON: plt.plot(gamma_range, experimental_data["Hessian L2norm w (Carstensen)"], marker='^', linestyle='dashed', color='g', label='CMN18', linewidth=LINEWIDTH, markersize=MARKERSIZE)

        max_w = max([max(experimental_data["Hessian L2norm w (Variational)"]), max(experimental_data["Hessian L2norm w (Brenner)"]), max(experimental_data["Hessian L2norm w (Carstensen)"])])
        min_w = min([min(experimental_data["Hessian L2norm w (Variational)"]), min(experimental_data["Hessian L2norm w (Brenner)"]), min(experimental_data["Hessian L2norm w (Carstensen)"])])

        steps_w = (max_w - min_w)/NUM_STEPS
        if steps_w == 0: steps_w = NUM_STEPS # CFe: w is constant

        if LOG_SCALE: plt.xscale('log') # CFe: use log xscale when needed
        plt.xticks(gamma_range, [str(tick) if tick in x_values else '' for tick in gamma_range])
        plt.yticks(np.arange(min_w, max_w, steps_w))
        plt.xlabel(r"$\gamma$ = $\frac{p_0}{E}$", fontsize=FONTSIZE)
        plt.ylabel(r'$| \nabla^2 w |_{L^2(\Omega)} $', fontsize=FONTSIZE)
        ax = plt.gca() # use scientific notation for y axis
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_fontsize(FONTSIZE)
        plt.title(r'$| \nabla^2 w |_{L^2(\Omega)}'+f' Mesh size = {mesh_size}. IP = {IP}. tol = {SOL_TOLLERANCE}', fontsize=FONTSIZE)
        plt.tick_params(axis='both', which='major', labelsize=FONTSIZE)
        plt.legend(fontsize=FONTSIZE)
        plt.gca().yaxis.get_offset_text().set_fontsize(FONTSIZE)
        plt.grid(True)
        plt.savefig(EXPERIMENT_DIR+f'/vary_gamma_HessW_{info_experiment}.png', dpi=300)
        #plt.show()

        # PLOT MEMBRANE ENERGY
        plt.figure(figsize=(FIGWIDTH, FIGHIGHT))
        plt.plot(gamma_range, experimental_data["Membrane Erg (Var)"], marker='o', linestyle='solid', color='b', label='VAR', linewidth=LINEWIDTH, markersize=MARKERSIZE)
        plt.axhline(0, color='black', linewidth=2, label="zero value")
        plt.axhline(kMembraneEng, color='red', linewidth=2, label="Kirchhoff-Love prediction")
        if COMPARISON: plt.plot(gamma_range, experimental_data["Membrane Erg (Brn)"], marker='v', linestyle='dotted', color='r', label='BNRS17', linewidth=LINEWIDTH, markersize=MARKERSIZE)
        if COMPARISON: plt.plot(gamma_range, experimental_data["Membrane Erg (Car)"], marker='^', linestyle='dashed', color='g', label='CMN18', linewidth=LINEWIDTH, markersize=MARKERSIZE)

        max_memEng = max(max(experimental_data["Membrane Erg (Var)"]), max(experimental_data["Membrane Erg (Brn)"]), max(experimental_data["Membrane Erg (Car)"]), 0, kMembraneEng )
        min_memEng = min( min(experimental_data["Membrane Erg (Var)"]), min(experimental_data["Membrane Erg (Brn)"]), min(experimental_data["Membrane Erg (Car)"]), 0, kMembraneEng )

        steps_memEng = abs( (max_memEng - min_memEng)/NUM_STEPS )
        if steps_memEng == 0: steps_memEng = NUM_STEPS
        if LOG_SCALE: plt.xscale('log') # CFe: use log xscale when needed
        plt.xticks(gamma_range, [str(tick) if tick in x_values else '' for tick in gamma_range])
        plt.yticks(np.arange(min_memEng, max_memEng, steps_memEng))
        plt.xlabel(r"$\gamma$ = $\frac{p_0}{E}$", fontsize=FONTSIZE)
        plt.ylabel('Membrane Energy', fontsize=FONTSIZE)
        ax = plt.gca() # use scientific notation for y axis
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_fontsize(FONTSIZE)
        plt.title(f'Membrane Energy. Mesh size = {mesh_size}. IP = {IP}. tol = {SOL_TOLLERANCE}', fontsize=FONTSIZE)
        plt.tick_params(axis='both', which='major', labelsize=FONTSIZE)
        plt.legend(fontsize=FONTSIZE)
        plt.gca().yaxis.get_offset_text().set_fontsize(FONTSIZE)
        plt.grid(True)
        plt.savefig(EXPERIMENT_DIR+f'/vary_gamma_MembEng_{info_experiment}.png', dpi=300)

        # PLOT BENDING ENERGY
        plt.figure(figsize=(FIGWIDTH, FIGHIGHT))
        plt.plot(gamma_range, experimental_data["Bending Erg (Var)"], marker='o', linestyle='solid', color='b', label='VAR', linewidth=LINEWIDTH, markersize=MARKERSIZE)
        if COMPARISON: plt.plot(gamma_range, experimental_data["Bending Erg (Brn)"], marker='v', linestyle='dotted', color='r', label='BNRS17', linewidth=LINEWIDTH, markersize=MARKERSIZE)
        if COMPARISON: plt.plot(gamma_range, experimental_data["Bending Erg (Car)"], marker='^', linestyle='dashed', color='g', label='CMN18', linewidth=LINEWIDTH, markersize=MARKERSIZE)
        plt.axhline(y=0, color='black', linewidth=2)
        max_bendEng = max( max(experimental_data["Bending Erg (Var)"]), max(experimental_data["Bending Erg (Brn)"]), max(experimental_data["Bending Erg (Car)"]) )
        min_bendEng = min( min(experimental_data["Bending Erg (Var)"]), min(experimental_data["Bending Erg (Var)"]), min(experimental_data["Bending Erg (Var)"]) )
        steps_bendEng = (max_bendEng - min_bendEng)/NUM_STEPS
        if steps_bendEng == 0: steps_bendEng = NUM_STEPS
        if LOG_SCALE: plt.xscale('log') # CFe: use log xscale when needed
        plt.xticks(gamma_range, [str(tick) if tick in x_values else '' for tick in gamma_range])
        plt.yticks(np.arange(min_bendEng, max_bendEng, steps_bendEng))
        plt.xlabel(r"$\gamma$ = $\frac{p_0}{E}$", fontsize=FONTSIZE)
        plt.ylabel('Bending Energy', fontsize=FONTSIZE)
        ax = plt.gca() # use scientific notation for y axis
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_fontsize(FONTSIZE)
        plt.title(f'Bending Energy. Mesh size = {mesh_size}. IP = {IP}. tol = {SOL_TOLLERANCE}', fontsize=FONTSIZE)
        plt.tick_params(axis='both', which='major', labelsize=FONTSIZE)
        plt.legend(fontsize=FONTSIZE)
        plt.gca().yaxis.get_offset_text().set_fontsize(FONTSIZE)
        plt.grid(True)
        plt.savefig(EXPERIMENT_DIR+f'/vary_gamma_BendEng_{info_experiment}.png', dpi=300)

        # PLOT COUPLING ENERGY
        plt.figure(figsize=(FIGWIDTH, FIGHIGHT))
        plt.plot(gamma_range, experimental_data["Coupling Erg (Var)"], marker='o', linestyle='solid', color='b', label='VAR', linewidth=LINEWIDTH, markersize=MARKERSIZE)
        if COMPARISON: plt.plot(gamma_range, experimental_data["Coupling Erg (Brn)"], marker='v', linestyle='dotted', color='r', label='BNRS17', linewidth=LINEWIDTH, markersize=MARKERSIZE)
        if COMPARISON: plt.plot(gamma_range, experimental_data["Coupling Erg (Car)"], marker='^', linestyle='dashed', color='g', label='CMN18', linewidth=LINEWIDTH, markersize=MARKERSIZE)
        plt.axhline(y=0, color='black', linewidth=2)

        max_couplingEng = max( max(experimental_data["Coupling Erg (Var)"]), max(experimental_data["Coupling Erg (Brn)"]), max(experimental_data["Coupling Erg (Car)"]) )
        min_couplingEng = min( min(experimental_data["Coupling Erg (Var)"]), min(experimental_data["Coupling Erg (Brn)"]), min(experimental_data["Coupling Erg (Car)"]) )

        steps_couplingEng = (max_couplingEng - min_couplingEng)/NUM_STEPS
        if steps_couplingEng == 0: steps_couplingEng = NUM_STEPS

        if LOG_SCALE: plt.xscale('log') # CFe: use log xscale when needed
        plt.xticks(gamma_range, [str(tick) if tick in x_values else '' for tick in gamma_range])
        plt.yticks(np.arange(min_couplingEng, max_couplingEng, steps_couplingEng))
        plt.xlabel(r"$\gamma$ = $\frac{p_0}{E}$", fontsize=FONTSIZE)
        plt.ylabel('Coupling Energy', fontsize=FONTSIZE)
        ax = plt.gca() # use scientific notation for y axis
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_fontsize(FONTSIZE)
        plt.title(f'Coupling Energy. Mesh size = {mesh_size}. IP = {IP}. tol = {SOL_TOLLERANCE}', fontsize=FONTSIZE)
        plt.tick_params(axis='both', which='major', labelsize=FONTSIZE)
        plt.legend(fontsize=FONTSIZE)
        plt.gca().yaxis.get_offset_text().set_fontsize(FONTSIZE)
        plt.grid(True)
        plt.savefig(EXPERIMENT_DIR+f'/varying_c_CouplEng_{info_experiment}.png', dpi=300)
        #plt.show()

# PLOT LOG-LOG ENERGIES
    plt.figure(figsize=(FIGWIDTH, FIGHIGHT))
    print("gamma range: ", gamma_range)
    #plt.xticks(gamma_range, [str(tick) if tick in x_range_plot_log else '' for tick in gamma_range])
    plt.plot(gamma_range, experimental_data["Membrane Erg (Var)"], marker='o', linestyle='none', label='Membrane', linewidth=LINEWIDTH, markersize=MARKERSIZE) #, color='b'
    plt.plot(gamma_range, experimental_data["Bending Erg (Var)"], marker='^', linestyle='none', label='Bending', linewidth=LINEWIDTH, markersize=MARKERSIZE) #, color='r'
    plt.plot(gamma_range, experimental_data["Coupling Erg (Var)"], marker='v', linestyle='none', label='Coupling', linewidth=LINEWIDTH, markersize=MARKERSIZE) #, color='k'
    plt.plot(gamma_range, kMembraneEngList, marker='.', linestyle='--', label='KL model: membrane', linewidth=LINEWIDTH, markersize=0*MARKERSIZE) #, color='g'
    plt.plot(gamma_range, kBendingEngList, marker='s', linestyle='-.', label='KL model: bending', linewidth=LINEWIDTH, markersize=0*MARKERSIZE) #, color='y'
    plt.xscale('log')
    plt.yscale('log')
    plt.axhline(y=0, color='black', linewidth=2)
    plt.xlabel(r"$\gamma$ = $\frac{p_0}{E}$", fontsize=FONTSIZE)
    plt.ylabel('Dimensionless energy', fontsize=FONTSIZE)
    #plt.title(f'Log-log plot energies. Mesh size = {mesh_size}. IP = {IP}. tol = {SOL_TOLLERANCE}', fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.gca().yaxis.get_offset_text().set_fontsize(FONTSIZE)
    #ax = plt.gca()  # Get the current axis
    #ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    #ax.xaxis.get_offset_text().set_fontsize(FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    plt.grid(True)
    visuals.setspines()
    plt.savefig(EXPERIMENT_DIR+f'/vary_gamma_loglog_energies_{info_experiment}.png', dpi=300)

    if False:
        # PLOT  HESSIAN VS LAPLACIAN L2-NORM W: the absolute error
        nu = parameters["model"]["nu"]
        c_nu = 1/(12*(1-nu**2))
        wHessLap_var_list = []
        wHessLap_brn_list = []
        wHessLap_car_list = []
        for i in range(len(experimental_data["Bending Erg (Var)"])):
            wHessLap_var_list.append( (2/c_nu)*experimental_data["Bending Erg (Var)"][i] - (experimental_data["Hessian L2norm w (Variational)"][i])**2 )
            wHessLap_brn_list.append( (2/c_nu)*experimental_data["Bending Erg (Brn)"][i] - (experimental_data["Hessian L2norm w (Brenner)"][i])**2 )
            #wHessLap_car_list.append( (2/c_nu)*experimental_data["Bending Erg (Car)"][i] - (experimental_data["Hessian L2norm w (Carstensen)"][i])**2 )
            wHessLap_car_list.append(0)

        plt.figure(figsize=(FIGWIDTH, FIGHIGHT))
        plt.plot(gamma_range, wHessLap_var_list, marker='o', linestyle='solid', color='b', label='VAR', linewidth=LINEWIDTH, markersize=MARKERSIZE)
        if COMPARISON: plt.plot(gamma_range, wHessLap_brn_list, marker='v', linestyle='dotted', color='r', label='BNRS17', linewidth=LINEWIDTH, markersize=MARKERSIZE)
        if COMPARISON: plt.plot(gamma_range, wHessLap_car_list, marker='^', linestyle='dashed', color='g', label='CMN18', linewidth=LINEWIDTH, markersize=MARKERSIZE)
        plt.axhline(y=0, color='black', linewidth=2)
        max_wHessLap = max( max(wHessLap_var_list), max(wHessLap_brn_list), max(wHessLap_car_list) )
        min_wHessLap = min( min(wHessLap_var_list), min(wHessLap_brn_list), min(wHessLap_car_list) )
        steps_wHessLap = (max_wHessLap - min_wHessLap)/NUM_STEPS
        if steps_wHessLap == 0: steps_wHessLap = NUM_STEPS
        if LOG_SCALE: plt.xscale('log') # CFe: use log xscale when needed
        plt.xticks(gamma_range, [str(tick) if tick in x_values else '' for tick in gamma_range])
        plt.yticks(np.arange(min_wHessLap, max_wHessLap, steps_wHessLap))
        plt.xlabel(r"$\gamma$ = $\frac{p_0}{E}$", fontsize=FONTSIZE)
        plt.ylabel(r'$| \Delta w |^2_{L^2} - | \nabla^2 w |^2_{L^2}$', fontsize=FONTSIZE)
        ax = plt.gca() # use scientific notation for y axis
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_fontsize(FONTSIZE)
        plt.title(r'$| \Delta w |^2_{L^2} - | \nabla^2 w |^2_{L^2}$', fontsize=FONTSIZE)
        plt.tick_params(axis='both', which='major', labelsize=FONTSIZE)
        plt.legend(fontsize=FONTSIZE)
        plt.gca().yaxis.get_offset_text().set_fontsize(FONTSIZE)
        plt.grid(True)
        plt.savefig(EXPERIMENT_DIR+f'/vary_gamma_HessVsLapW_{info_experiment}.png', dpi=300)

        # PLOT  HESSIAN VS LAPLACIAN L2-NORM W: the percent error
        wHessLapPercErr_var_list = []
        wHessLapPercErr_brn_list = []
        wHessLapPercErr_car_list = []
        for i in range(len(experimental_data["Bending Erg (Var)"])):
            wHessLapPercErr_var_list.append( 100 * wHessLap_var_list[i] / ( (2/c_nu)*experimental_data["Bending Erg (Var)"][i] ) )
            wHessLapPercErr_brn_list.append( 100 * wHessLap_brn_list[i]  / ( (2/c_nu)*experimental_data["Bending Erg (Brn)"][i]  ) )
            #wHessLap_car_list.append( (2/c_nu)*experimental_data["Bending Erg (Car)"][i] - (experimental_data["Hessian L2norm w (Carstensen)"][i])**2 )
            wHessLapPercErr_car_list.append(0)

        plt.figure(figsize=(FIGWIDTH, FIGHIGHT))
        plt.plot(gamma_range, wHessLapPercErr_var_list, marker='o', linestyle='solid', color='b', label='VAR', linewidth=LINEWIDTH, markersize=MARKERSIZE)
        if COMPARISON: plt.plot(gamma_range, wHessLapPercErr_brn_list, marker='v', linestyle='dotted', color='r', label='BNRS17', linewidth=LINEWIDTH, markersize=MARKERSIZE)
        if COMPARISON: plt.plot(gamma_range, wHessLapPercErr_car_list, marker='^', linestyle='dashed', color='g', label='CMN18', linewidth=LINEWIDTH, markersize=MARKERSIZE)
        plt.axhline(y=0, color='black', linewidth=2)
        max_wHessLap = max( max(wHessLapPercErr_var_list), max(wHessLapPercErr_brn_list), max(wHessLapPercErr_car_list) )
        min_wHessLap = min( min(wHessLapPercErr_var_list), min(wHessLapPercErr_brn_list), min(wHessLapPercErr_car_list) )
        steps_wHessLap = (max_wHessLap - min_wHessLap)/NUM_STEPS
        if steps_wHessLap == 0: steps_wHessLap = NUM_STEPS
        if LOG_SCALE: plt.xscale('log') # CFe: use log xscale when needed
        #plt.text(f'Max (Var): {max(wHessLapPercErr_var_list):.2f}%', ha='right', color='blue')
        plt.xticks(gamma_range, [str(tick) if tick in x_values else '' for tick in gamma_range])
        plt.yticks(np.arange(min_wHessLap, max_wHessLap, steps_wHessLap))
        plt.xlabel(r"$\gamma$ = $\frac{p_0}{E}$", fontsize=FONTSIZE)
        plt.ylabel(r'$ ( | \Delta w |^2_{L^2} - | \nabla^2 w |^2_{L^2} ) / | \Delta w |^2_{L^2} $', fontsize=FONTSIZE)
        ax = plt.gca() # use scientific notation for y axis
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_fontsize(FONTSIZE)
        plt.title(r'$ ( | \Delta w |^2_{L^2} - | \nabla^2 w |^2_{L^2} ) / | \Delta w |^2_{L^2} $', fontsize=FONTSIZE)
        plt.tick_params(axis='both', which='major', labelsize=FONTSIZE)
        plt.legend(fontsize=FONTSIZE)
        plt.gca().yaxis.get_offset_text().set_fontsize(FONTSIZE)
        plt.gca().yaxis.set_major_formatter(PercentFormatter())
        plt.grid(True)
        plt.savefig(EXPERIMENT_DIR+f'/vary_gamma_HessVsLapWPerc_{info_experiment}.png', dpi=300)

        # PLOT  HESSIAN VS LAPLACIAN L2-NORM V
        vHessLap_var_list = []
        vHessLap_brn_list = []
        vHessLap_car_list = []
        for i in range(len(experimental_data["Membrane Erg (Var)"])):
            vHessLap_var_list.append( 2*experimental_data["Membrane Erg (Var)"][i] - (experimental_data["Hessian L2norm v (Variational)"][i])**2 )
            vHessLap_brn_list.append( 2*experimental_data["Membrane Erg (Brn)"][i] - (experimental_data["Hessian L2norm v (Brenner)"][i])**2 )
            vHessLap_car_list.append( 2*experimental_data["Membrane Erg (Car)"][i] - (experimental_data["Hessian L2norm v (Carstensen)"][i])**2 )

        plt.figure(figsize=(FIGWIDTH, FIGHIGHT))
        plt.plot(gamma_range, vHessLap_var_list, marker='o', linestyle='solid', color='b', label='VAR', linewidth=LINEWIDTH, markersize=MARKERSIZE)
        if COMPARISON: plt.plot(gamma_range, vHessLap_var_list, marker='v', linestyle='dotted', color='r', label='BNRS17', linewidth=LINEWIDTH, markersize=MARKERSIZE)
        if COMPARISON: plt.plot(gamma_range, vHessLap_car_list, marker='^', linestyle='dashed', color='g', label='CMN18', linewidth=LINEWIDTH, markersize=MARKERSIZE)
        plt.axhline(y=0, color='black', linewidth=2)
        max_vHessLap = max( max(vHessLap_var_list), max(vHessLap_brn_list), max(vHessLap_car_list), 0 )
        min_vHessLap = min( min(vHessLap_var_list), min(vHessLap_brn_list), min(vHessLap_car_list), 0 )
        steps_vHessLap = (max_vHessLap - min_vHessLap)/NUM_STEPS
        if steps_wHessLap == 0: steps_wHessLap = NUM_STEPS
        if LOG_SCALE: plt.xscale('log') # CFe: use log xscale when needed
        plt.xticks(gamma_range, [str(tick) if tick in x_values else '' for tick in gamma_range])
        plt.yticks(np.arange(min_vHessLap, max_vHessLap, steps_vHessLap))
        plt.xlabel(r"$\gamma$ = $\frac{p_0}{E}$", fontsize=FONTSIZE)
        plt.ylabel(r'$| \Delta v |^2_{L^2} - | \nabla^2 v |^2_{L^2}$', fontsize=FONTSIZE)
        ax = plt.gca() # use scientific notation for y axis
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_fontsize(FONTSIZE)
        plt.title(r'$| \Delta v |^2_{L^2} - | \nabla^2 v |^2_{L^2}$', fontsize=FONTSIZE)
        plt.tick_params(axis='both', which='major', labelsize=FONTSIZE)
        plt.legend(fontsize=FONTSIZE)
        plt.gca().yaxis.get_offset_text().set_fontsize(FONTSIZE)
        plt.grid(True)
        plt.savefig(EXPERIMENT_DIR+f'/vary_gamma_HessVsLapV_{info_experiment}.png', dpi=300)

    # PLOT SIGMA NN
    subplotter = pyvista.Plotter(shape=(1, 1))
    subplotter.subplot(0, 0)
    grid["sigma_nn"] = sigma_nn.x.array.real
    grid.set_active_scalars("sigma_nn")
    subplotter.add_text("sigma_nn", position="upper_edge", font_size=14, color="black")
    scalar_bar_args["title"] = "sigma_rr "
    subplotter.add_mesh( grid.warp_by_scalar( scale_factor = 1 / max(np.abs(sigma_nn.x.array.real )) ), show_edges=False, edge_color="white", show_scalar_bar=True, scalar_bar_args=scalar_bar_args, cmap="coolwarm")
    IMG_WIDTH = 1600
    IMG_HEIGHT = 1200
    PNG_SCALE = 2
    subplotter.window_size = (IMG_WIDTH, IMG_HEIGHT)
    subplotter.screenshot(f"{EXPERIMENT_DIR}/visualization_sigma_rr_{index_solution}_{info_experiment}.png", scale = PNG_SCALE)
    subplotter.export_html(f"{EXPERIMENT_DIR}/visualization_sigma_rr_{index_solution}_{info_experiment}.html")

    if False:
        # PLOT PENALIZATION ENERGY

        # Total Penalization on the Transverse Displacement w
        plt.figure(figsize=(FIGWIDTH, FIGHIGHT))
        plt.plot(gamma_range, experimental_data["Penalization W Eng (Var)"], marker='o', linestyle='solid', color='b', label='VAR', linewidth=LINEWIDTH, markersize=MARKERSIZE)
        if COMPARISON: plt.plot(gamma_range, experimental_data["Penalization W Eng (Brn)"], marker='v', linestyle='dotted', color='r', label='BNRS17', linewidth=LINEWIDTH, markersize=MARKERSIZE)
        if COMPARISON: plt.plot(gamma_range, experimental_data["Penalization W Eng (Car)"], marker='^', linestyle='dashed', color='g', label='CMN18', linewidth=LINEWIDTH, markersize=MARKERSIZE)

        max_penalizWEng = max( max(experimental_data["Penalization W Eng (Var)"]), max(experimental_data["Penalization W Eng (Brn)"]), max(experimental_data["Penalization W Eng (Car)"]) )
        min_penalizWEng = min( min(experimental_data["Penalization W Eng (Var)"]), min(experimental_data["Penalization W Eng (Brn)"]), min(experimental_data["Penalization W Eng (Car)"]) )

        steps_penalizWEng = (max_penalizWEng - min_penalizWEng)/NUM_STEPS
        if steps_penalizWEng == 0: steps_penalizWEng = NUM_STEPS # CFe: w is constant

        if LOG_SCALE: plt.xscale('log') # CFe: use log xscale when needed
        plt.xticks(gamma_range, [str(tick) if tick in x_values else '' for tick in gamma_range])
        plt.yticks(np.arange(min_penalizWEng, max_penalizWEng, steps_penalizWEng))
        plt.xlabel(r"$\gamma$ = $\frac{p_0}{E}$", fontsize=FONTSIZE)
        plt.ylabel('Penalization Energy w', fontsize=FONTSIZE)
        ax = plt.gca() # use scientific notation for y axis
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_fontsize(FONTSIZE)
        plt.title(f'Penalization Energy w. Mesh size = {mesh_size}. IP = {IP}. tol = {SOL_TOLLERANCE}', fontsize=FONTSIZE)
        plt.tick_params(axis='both', which='major', labelsize=FONTSIZE)
        plt.legend(fontsize=FONTSIZE)
        plt.gca().yaxis.get_offset_text().set_fontsize(FONTSIZE)
        plt.grid(True)
        plt.savefig(EXPERIMENT_DIR+f'/vary_gamma_PenEngW_{info_experiment}.png', dpi=300)

        # Penalization energy of the Jump of the Hessian of the Transverse Displacement w
        plt.figure(figsize=(FIGWIDTH, FIGHIGHT))
        plt.plot(gamma_range, experimental_data["Penalization W Hess Jump (Var)"], marker='o', linestyle='solid', color='b', label='VAR', linewidth=LINEWIDTH, markersize=MARKERSIZE)
        if COMPARISON: plt.plot(gamma_range, experimental_data["Penalization W Hess Jump (Brn)"], marker='v', linestyle='dotted', color='r', label='BNRS17', linewidth=LINEWIDTH, markersize=MARKERSIZE)

        max_penalizWEng = max( max(experimental_data["Penalization W Hess Jump (Var)"]), max(experimental_data["Penalization W Hess Jump (Brn)"]) )
        min_penalizWEng = min( min(experimental_data["Penalization W Hess Jump (Var)"]), min(experimental_data["Penalization W Hess Jump (Brn)"]) )

        steps_penalizWEng = (max_penalizWEng - min_penalizWEng)/NUM_STEPS
        if steps_penalizWEng == 0: steps_penalizWEng = NUM_STEPS # CFe: w is constant

        if LOG_SCALE: plt.xscale('log') # CFe: use log xscale when needed
        plt.xticks(gamma_range, [str(tick) if tick in x_values else '' for tick in gamma_range])
        plt.yticks(np.arange(min_penalizWEng, max_penalizWEng, steps_penalizWEng))
        plt.xlabel(r"$\gamma$ = $\frac{p_0}{E}$", fontsize=FONTSIZE)
        plt.ylabel('Penalization Hessian w', fontsize=FONTSIZE)
        ax = plt.gca() # use scientific notation for y axis
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_fontsize(FONTSIZE)
        plt.title(f'Penalization Hessian w. Mesh size = {mesh_size}. IP = {IP}. tol = {SOL_TOLLERANCE}', fontsize=FONTSIZE)
        plt.tick_params(axis='both', which='major', labelsize=FONTSIZE)
        plt.legend(fontsize=FONTSIZE)
        plt.gca().yaxis.get_offset_text().set_fontsize(FONTSIZE)
        plt.grid(True)
        plt.savefig(EXPERIMENT_DIR+f'/vary_gamma_PenHessJumpW_{info_experiment}.png', dpi=300)

        # Total Penalization on the Airy's function v
        plt.figure(figsize=(FIGWIDTH, FIGHIGHT))
        plt.plot(gamma_range, experimental_data["Penalization V Eng (Var)"], marker='o', linestyle='solid', color='b', label='VAR', linewidth=LINEWIDTH, markersize=MARKERSIZE)
        if COMPARISON: plt.plot(gamma_range, experimental_data["Penalization V Eng (Brn)"], marker='v', linestyle='dotted', color='r', label='BNRS17', linewidth=LINEWIDTH, markersize=MARKERSIZE)
        if COMPARISON: plt.plot(gamma_range, experimental_data["Penalization V Eng (Car)"], marker='^', linestyle='dashed', color='g', label='CMN18', linewidth=LINEWIDTH, markersize=MARKERSIZE)

        max_penalizVEng = max( max(experimental_data["Penalization V Eng (Var)"]), max(experimental_data["Penalization V Eng (Brn)"]), max(experimental_data["Penalization V Eng (Car)"]) )
        min_penalizVEng = min( min(experimental_data["Penalization V Eng (Var)"]), min(experimental_data["Penalization V Eng (Brn)"]), min(experimental_data["Penalization V Eng (Car)"]) )
        steps_penalizVEng = (max_penalizVEng - min_penalizVEng)/NUM_STEPS
        if steps_penalizVEng == 0: steps_penalizVEng = NUM_STEPS # CFe: w is constant

        if LOG_SCALE: plt.xscale('log') # CFe: use log xscale when needed
        plt.xticks(gamma_range, [str(tick) if tick in x_values else '' for tick in gamma_range])
        plt.yticks(np.arange(min_penalizVEng, max_penalizVEng, steps_penalizVEng))
        plt.xlabel(r"$\gamma$ = $\frac{p_0}{E}$", fontsize=FONTSIZE)
        plt.ylabel('Penalization Energy v', fontsize=FONTSIZE)
        ax = plt.gca() # use scientific notation for y axis
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_fontsize(FONTSIZE)
        plt.title(f'Penalization Energy v. Mesh size = {mesh_size}. IP = {IP}. tol = {SOL_TOLLERANCE}', fontsize=FONTSIZE)
        plt.tick_params(axis='both', which='major', labelsize=FONTSIZE)
        plt.legend(fontsize=FONTSIZE)
        plt.gca().yaxis.get_offset_text().set_fontsize(FONTSIZE)
        plt.grid(True)
        plt.savefig(EXPERIMENT_DIR+f'/vary_gamma_PenEngV_{info_experiment}.png', dpi=300)


        # PLOT DIFFERENCES BETWEEN ENERGIES AND PENALTY ENERGIES
        w_dg1_var_list = experimental_data["Penalty w dg1 (Var)"]
        w_dg2_var_list = experimental_data["Penalty w dg2 (Var)"]
        w_bc1_var_list = experimental_data["Penalty w bc1 (Var)"]
        w_bc3_var_list = experimental_data["Penalty w bc3 (Var)"]
        w_dg3_var_list = experimental_data["Penalization W Hess Jump (Var)"]

        w_dg1_brn_list = experimental_data["Penalty w dg1 (Brn)"]
        w_dg2_brn_list = experimental_data["Penalty w dg2 (Brn)"]
        w_bc1_brn_list = experimental_data["Penalty w bc1 (Brn)"]
        w_bc3_brn_list = experimental_data["Penalty w bc3 (Brn)"]
        w_dg3_brn_list = experimental_data["Penalization W Hess Jump (Brn)"]

        fig, ax = plt.subplots(2, 2, figsize=(FIGWIDTH, FIGHIGHT))
        ax[0,0].plot(gamma_range, w_dg1_var_list, marker='o', linestyle='solid', color='b', label="w_dg1 (VAR)", linewidth=LINEWIDTH, markersize=MARKERSIZE)
        ax[0,0].plot(gamma_range, w_dg1_brn_list, marker='v', linestyle='dotted', color='r', label="w_dg1 (BNRS17)", linewidth=LINEWIDTH, markersize=MARKERSIZE)
        ax[0,0].legend(fontsize=FONTSIZE)
        ax[0,0].grid(True)
        ax[0,0].set_xlabel(r"$\gamma$ = $\frac{p_0}{E}$", fontsize=FONTSIZE)
        plt.gca().yaxis.get_offset_text().set_fontsize(FONTSIZE)
        ax[0,1].plot(gamma_range, w_dg2_var_list, marker='o', linestyle='solid', color='b', label="w_dg2 (VAR)", linewidth=LINEWIDTH, markersize=MARKERSIZE)
        ax[0,1].plot(gamma_range, w_dg2_brn_list, marker='v', linestyle='dotted', color='r', label="w_dg2 (BNRS17)", linewidth=LINEWIDTH, markersize=MARKERSIZE)
        ax[0,1].legend(fontsize=FONTSIZE)
        ax[0,1].grid(True)
        ax[0,1].set_xlabel(r"$\gamma$ = $\frac{p_0}{E}$", fontsize=FONTSIZE)
        plt.gca().yaxis.get_offset_text().set_fontsize(FONTSIZE)
        ax[1,0].plot(gamma_range, w_bc1_var_list, marker='o', linestyle='solid', color='b', label="w_bc1 (Var)", linewidth=LINEWIDTH, markersize=MARKERSIZE)
        ax[1,0].plot(gamma_range, w_bc1_brn_list, marker='v', linestyle='dotted', color='r', label="w_bc1 (BNRS17)", linewidth=LINEWIDTH, markersize=MARKERSIZE)
        ax[1,0].legend(fontsize=FONTSIZE)
        ax[1,0].grid(True)
        plt.gca().yaxis.get_offset_text().set_fontsize(FONTSIZE)
        ax[1,0].set_xlabel(r"$\gamma$ = $\frac{p_0}{E}$", fontsize=FONTSIZE)
        ax[1,1].plot(gamma_range, w_bc3_var_list, marker='o', linestyle='solid', color='b', label="w_bc3 (VAR)", linewidth=LINEWIDTH, markersize=MARKERSIZE)
        ax[1,1].plot(gamma_range, w_bc3_brn_list, marker='v', linestyle='dotted', color='r', label="w_bc3 (BNRS17)", linewidth=LINEWIDTH, markersize=MARKERSIZE)
        ax[1,1].legend(fontsize=FONTSIZE)
        ax[1,1].grid(True)
        ax[1,1].set_xlabel(r"$\gamma$ = $\frac{p_0}{E}$", fontsize=FONTSIZE)
        plt.xticks(gamma_range, [str(tick) if tick in x_values else '' for tick in gamma_range])
        fig.suptitle(f'Penalization Energies w. Mesh size = {mesh_size}. IP = {IP}. tol = {SOL_TOLLERANCE}', fontsize=FONTSIZE)
        plt.gca().yaxis.get_offset_text().set_fontsize(FONTSIZE)
        plt.savefig(EXPERIMENT_DIR+f'/varying_c_PenEngsW_{info_experiment}.png', dpi=300)

        # PLOT PROFILES

        tol = 1e-4
        xs = np.linspace(-parameters["geometry"]["radius"] + tol, parameters["geometry"]["radius"] - tol, 1001)
        points = np.zeros((3, 1001))
        points[0] = xs


        for i in range(len(w_varModel_list)):

            fig, axes = plt.subplots(1, 1, figsize=(FIGWIDTH, FIGHIGHT))
            axes.yaxis.get_offset_text().set_size(FONTSIZE) # Size of the order of magnitude
            plt, data = plot_profile(w_varModel_list[i], points, None, subplot=(1, 1), lineproperties={"c": "r", "lw":5, "label": "VAR", "ls": "solid"}, fig=fig, subplotnumber=1)
            if COMPARISON: plt, data = plot_profile(w_brnModel_list[i], points, None, subplot=(1, 1), lineproperties={"c": "r", "lw":5, "label": "BNRS17", "ls": "-"}, fig=fig, subplotnumber=1)
            plt.xlabel(r"$\xi_1$", fontsize=FONTSIZE)
            plt.ylabel("w", fontsize=FONTSIZE)
            plt.xticks(fontsize=FONTSIZE)
            plt.yticks(fontsize=FONTSIZE)
            ax = plt.gca() # use scientific notation for y axis
            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax.yaxis.get_offset_text().set_fontsize(FONTSIZE)
            plt.title(fr"Transverse displacement. $\gamma$ = {gamma_range[i]}; Mesh = {mesh_size}. IP = {IP}. E = {E:.2e} tol = {SOL_TOLLERANCE}", size = FONTSIZE)
            plt.grid(True)
            plt.legend(fontsize=FONTSIZE)
            plt.savefig(f"{EXPERIMENT_DIR}/_{gamma_range[i]}a-w-profiles_{info_experiment}.png")

            fig, axes = plt.subplots(1, 1, figsize=(FIGWIDTH, FIGHIGHT))
            plt, data = plot_profile(v_varModel_list[i], points, None, subplot=(1, 1), lineproperties={"c": "b", "lw":5, "label": "VAR", "ls": "solid"}, fig=fig, subplotnumber=1)
            if COMPARISON: plt, data = plot_profile(v_brnModel_list[i], points, None, subplot=(1, 1), lineproperties={"c": "r", "lw":5, "label": "BNRS17", "ls": "-"}, fig=fig, subplotnumber=1)
            plt.xlabel(r"$\xi_1$", fontsize=FONTSIZE)
            plt.ylabel("v", fontsize=FONTSIZE)
            plt.xticks(fontsize=FONTSIZE)
            plt.yticks(fontsize=FONTSIZE)
            plt.title(fr"Airy. $\gamma$ = {gamma_range[i]}. Mesh = {mesh_size}. IP = {IP}. E = {E:.2e}. tol = {SOL_TOLLERANCE}", size = FONTSIZE)
            plt.grid(True)
            plt.legend(fontsize=FONTSIZE)
            plt.savefig(f"{EXPERIMENT_DIR}/_{gamma_range[i]}a-v-profiles_{info_experiment}.png")

            # fig, axes = plt.subplots(1, 1, figsize=(FIGWIDTH, FIGHIGHT))
            # w_varModel_list[i].vector.array = w_varModel_list[i].vector.array * parameters["model"]["thickness"]
            # w_brnModel_list[i].vector.array = w_brnModel_list[i].vector.array * parameters["model"]["thickness"]
            # plt, data = plot_profile(w_varModel_list[i], points, None, subplot=(1, 1), lineproperties={"c": "b", "lw":5, "label": "VAR", "ls": "solid"}, fig=fig, subplotnumber=1)
            # if COMPARISON: plt, data = plot_profile(w_brnModel_list[i], points, None, subplot=(1, 1), lineproperties={"c": "r", "lw":5, "label": "BNRS17", "ls": "-"}, fig=fig, subplotnumber=1)
            # plt.xlabel("x [m]", fontsize=FONTSIZE)
            # plt.ylabel("w [m]", fontsize=FONTSIZE)
            # plt.xticks(fontsize=FONTSIZE)
            # plt.yticks(fontsize=FONTSIZE)
            # ax = plt.gca() # use scientific notation for y axis
            # ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            # ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            # ax.yaxis.get_offset_text().set_fontsize(FONTSIZE)
            # plt.ylim(-1, 1)
            # plt.title(f"Transverse displacement [1:1] aspect ratio. gamma = {gamma_range[i]}; Mesh = {parameters["geometry"]["mesh_size"]}. IP = {parameters["model"]["alpha_penalty"]}", size = FONTSIZE)
            # plt.grid(True)
            # plt.legend(fontsize=FONTSIZE)
            # plt.savefig(f"{EXPERIMENT_DIR}/_{gamma_range[i]}b-w-profiles_{info_experiment}.png")

    if pyvista.OFF_SCREEN: pyvista.start_xvfb(wait=0.1)
    transparent = False
    topology, cells, geometry = dolfinx.plot.vtk_mesh(V_v)
    grid = pyvista.UnstructuredGrid(topology, cells, geometry)

    # PLOT PROFILES
    grid.point_data["w_0.1"] = w_varModel_list[0].vector.array[dofs2_w]
    grid.point_data["w_100"] = w_varModel_list[4].vector.array[dofs2_w]
    grid.point_data["w_3000"] = w_varModel_list[-1].vector.array[dofs2_w]
    #grid.set_active_scalars("w")
    y0 = 0
    tolerance = 1e-2
    points = grid.points
    x_values = points[np.abs(points[:, 1] - y0) < tolerance, 0]  # Select x-coordinates at xi2 = 0
    w_0 = grid['w_0.1'][np.abs(points[:, 1] - y0) < tolerance]
    w_6 = grid['w_100'][np.abs(points[:, 1] - y0) < tolerance]
    w_9 = grid['w_3000'][np.abs(points[:, 1] - y0) < tolerance]
    sorted_indices = np.argsort(x_values) # Sort data for plotting
    x_sorted = x_values[sorted_indices]
    w_0sorted = w_0[sorted_indices]
    w_6sorted = w_6[sorted_indices]
    w_9sorted = w_9[sorted_indices]
    w_0normalized = w_0sorted / np.max(np.abs(w_0sorted))
    w_6normalized = w_6sorted / np.max(np.abs(w_6sorted))
    w_9normalized = w_9sorted / np.max(np.abs(w_9sorted))
    scale_w0 = f"{np.max(np.abs(w_0sorted)):.2e}"
    scale_w6 = f"{np.max(np.abs(w_6sorted)):.2e}"
    scale_w9 = f"{np.max(np.abs(w_9sorted)):.2e}"
    plt.figure(figsize=(FIGWIDTH, FIGHIGHT))
    plt.plot(x_sorted, w_0normalized, label=fr'$\gamma$: {gamma_list[0]:.2e}, max(|w|): {scale_w0}', linestyle='solid', linewidth=LINEWIDTH) #, color='blue'
    plt.plot(x_sorted, w_6normalized, label=fr'$\gamma$: {gamma_list[6]:.2e}, max(|w|): {scale_w6}', linestyle='dotted', linewidth=LINEWIDTH) #, color='red'
    plt.plot(x_sorted, w_9normalized, label=fr'$\gamma$: {gamma_list[9]:.2e}, max(|w|): {scale_w9}', color='black', linestyle='dashed', linewidth=LINEWIDTH)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.xlabel(r"$\xi_1$", fontsize=FONTSIZE)
    plt.ylabel(r"$ \frac{w}{\max(|w|)}$", fontsize=FONTSIZE)
    plt.title(fr"Profile of w at $\xi_2$ = {y0}", fontsize=FONTSIZE)
    ax = plt.gca() # use scientific notation for y axis
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))  # Adjust the number of bins to choose the number of ticks
    ax.yaxis.get_offset_text().set_fontsize(FONTSIZE)
    plt.legend(fontsize=0.8*FONTSIZE)
    #plt.grid(True)
    plt.savefig(f"{EXPERIMENT_DIR}/profiles_w_comparison_y_{y0}_{info_experiment}.png", dpi=300)
    plt.savefig(f"{EXPERIMENT_DIR}/profiles_w_comparison_y_{y0}_{info_experiment}.pdf", dpi=300)

    grid.point_data["v_0.1"] = v_varModel_list[0].vector.array[dofs2_v]
    grid.point_data["v_100"] = v_varModel_list[4].vector.array[dofs2_v]
    grid.point_data["v_3000"] = v_varModel_list[-1].vector.array[dofs2_v]
    points = grid.points
    x_values = points[np.abs(points[:, 1] - y0) < tolerance, 0]  # Select x-coordinates at xi2 = 0
    v_0 = grid['v_0.1'][np.abs(points[:, 1] - y0) < tolerance]
    v_6 = grid['v_100'][np.abs(points[:, 1] - y0) < tolerance]
    v_9 = grid['v_3000'][np.abs(points[:, 1] - y0) < tolerance]
    sorted_indices = np.argsort(x_values) # Sort data for plotting
    x_sorted = x_values[sorted_indices]
    v_0sorted = v_0[sorted_indices]
    v_6sorted = v_6[sorted_indices]
    v_9sorted = v_9[sorted_indices]
    v_0normalized = v_0sorted / np.max(np.abs(v_0sorted))
    v_6normalized = v_6sorted / np.max(np.abs(v_6sorted))
    v_9normalized = v_9sorted / np.max(np.abs(v_9sorted))
    scale_v0 = f"{np.max(np.abs(v_0sorted)):.1e}"
    scale_v6 = f"{np.max(np.abs(v_6sorted)):.1e}"
    scale_v9 = f"{np.max(np.abs(v_9sorted)):.1e}"
    plt.figure(figsize=(FIGWIDTH, FIGHIGHT))
    plt.plot(x_sorted, v_0normalized, label=fr'$\gamma$: {gamma_list[0]:.2e}, max(|v|): {scale_v0}', linestyle='solid', linewidth=LINEWIDTH) #, color='blue'
    plt.plot(x_sorted, v_6normalized, label=fr'$\gamma$: {gamma_list[6]:.2e}, max(|v|): {scale_v6}', linestyle='dotted', linewidth=LINEWIDTH) #, color='red'
    plt.plot(x_sorted, v_9normalized, label=fr'$\gamma$: {gamma_list[9]:.2e}, max(|v|): {scale_v9}', color='black', linestyle='dashed', linewidth=LINEWIDTH)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.xlabel(r"$\xi_1$", fontsize=FONTSIZE)
    plt.ylabel(r"$ \frac{v}{\max(|v|)}$", fontsize=FONTSIZE)
    plt.title(fr"Profile of v at $\xi_2$ = {y0}", fontsize=FONTSIZE)
    ax = plt.gca() # use scientific notation for y axis
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))  # Adjust the number of bins to choose the number of ticks
    ax.yaxis.get_offset_text().set_fontsize(FONTSIZE)
    plt.legend(fontsize=0.8*FONTSIZE)
    #plt.grid(True)
    plt.savefig(f"{EXPERIMENT_DIR}/profiles_v_comparison_y_{y0}_{info_experiment}.png", dpi=300)
    plt.savefig(f"{EXPERIMENT_DIR}/profiles_v_comparison_y_{y0}_{info_experiment}.pdf", dpi=300)
    print(f"===================- {EXPERIMENT_DIR} -=================")
