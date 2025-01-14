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
from matplotlib.ticker import PercentFormatter, ScalarFormatter, MaxNLocator

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
OUTDIR = os.path.join("output", "experiment_parametric_beta_adim") # CFe: output directory
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
COMPARISON = False # Comparison between FE models

NUM_STEPS = 10 # Number ticks in the yaxis

CostantData = namedtuple("CostantData", ["funcSpace", "bcs"])

petsc4py.init(["-petsc_type", "double"]) # CFe: ensuring double precision numbers

index_solution = 0
X_COORD = 0
Y_COORD = 1

def L2norm(u): return np.sqrt( fem.assemble_scalar( fem.form( ufl.inner(u, u) * ufl.dx ) ) )

def hessianL2norm(u):
    hessian = lambda f : ufl.grad(ufl.grad(u))
    return np.sqrt( fem.assemble_scalar( fem.form( ufl.inner(hessian(u), hessian(u)) * ufl.dx ) ) )

def bracket(f, g):
    J = ufl.as_matrix([[0, -1], [1, 0]])
    return ufl.inner(ufl.grad(ufl.grad(f)), J.T * ufl.grad(ufl.grad(g)) * J)

def ouward_unit_normal_x(x): return x[X_COORD] / np.sqrt(x[X_COORD]**2 + x[Y_COORD]**2)
def ouward_unit_normal_y(x): return x[Y_COORD] / np.sqrt(x[X_COORD]**2 + x[Y_COORD]**2)

def mongeAmpereForm(f1, f2, mesh, order = 1):
        DG_e = basix.ufl.element("DG", str(mesh.ufl_cell()), order)
        DG = dolfinx.fem.functionspace(mesh, DG_e)
        kappa = bracket(f1, f2)
        kappa_expr = dolfinx.fem.Expression(kappa, DG.element.interpolation_points())
        Kappa = dolfinx.fem.Function(DG)
        Kappa.interpolate(kappa_expr)

        return Kappa

def run_experiment(FEMmodel, costantData, parameters, q_ig):
    """
    Purpose: parametric study of the solution by varying beta := R/h
    """

    # UPDATE PARAMETER "beta"
    beta = 1/parameters["model"]["thickness"]

    # UPDATE PARAMETER "f"
    f0 = 1 # Parameter f is kept constant
    def nondim_transverse_load(x): return f0 * LOAD_SIGN *(1.0 + 0.0*x[0] + 0.0*x[1])

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
        model = A_NonlinearPlateFVK(mesh, parameters["model"], SMOOTHING)
    else:
        print(f"FEMmodel = {FEMmodel} is not acceptable. Script exiting")
        exit(0)

    # INSERT DISCLINATIONS
    disclinations = []
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

    v2, w2 = q.split()
    state2 = {"v": v2, "w": w2}

    # COMPUTE DIMENSIONAL ENERGY
    energy_terms = {
        "bending": model.compute_bending_energy(state2, COMM),
        "membrane": model.compute_membrane_energy(state2, COMM),
        "coupling": model.compute_coupling_energy(state2, COMM),
        "penalty":  model.compute_penalisation(state2, COMM),
        "penalty_w_dg1":  model.compute_penalisation_terms_w(state2, COMM)[0],
        "penalty_w_dg2":  model.compute_penalisation_terms_w(state2, COMM)[1],
        "penalty_w_bc1":  model.compute_penalisation_terms_w(state2, COMM)[2],
        "penalty_w_bc3":  model.compute_penalisation_terms_w(state2, COMM)[3],
        "penalty_w_dg3":  model.compute_penalisation_terms_w(state2, COMM)[4],
        "penalty_w_tot":  model.compute_total_penalisation_w(state2, COMM),
        "penalty_v_tot":  model.compute_total_penalisation_v(state2, COMM),
        "penalty_coupling":  model.compute_penalisation_coupling(state2, COMM),
        }

    # Print FE dimensioanal energy
    for label, energy_term in energy_terms.items(): print(f"{label}: {energy_term}")

    v_Nrm = L2norm(v2)
    w_Nrm = L2norm(w2)
    v_hessianNrm = hessianL2norm(v2)
    w_hessianNrm = hessianL2norm(w2)

    return_value_dic = {
            "w": w2, "v": v2, "q": q,
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
    normal_derivative_v = ufl.dot(ufl.grad(v2), n)
    normal_derivative_w = ufl.dot(ufl.grad(w2), n)
    L2norm_w = np.sqrt( fem.assemble_scalar(fem.form((w2**2) * ds)) )
    L2norm_v = np.sqrt( fem.assemble_scalar(fem.form((v2**2) * ds)) )
    L2norm_dwdn = np.sqrt( fem.assemble_scalar(fem.form((normal_derivative_w**2) * ds)) )
    L2norm_dvdn = np.sqrt( fem.assemble_scalar(fem.form((normal_derivative_v**2) * ds)) )
    L1norm_dwdn = np.sqrt( fem.assemble_scalar(fem.form(( np.abs(normal_derivative_w) ) * ds)) )
    L1norm_dvdn = np.sqrt( fem.assemble_scalar(fem.form(( np.abs(normal_derivative_v) ) * ds)) )
    print("boundary_integral_v: ", L2norm_v)
    print("boundary_integral_w: ", L2norm_w)
    print("L1norm_dwdn: ", L1norm_dwdn)
    print("L2norm_dwdn: ", L2norm_dwdn)
    print("L1norm_dvdn: ", L1norm_dvdn)
    print("L2norm_dvdn: ", L2norm_dvdn)

    # FREE MEMORY
    solver.solver.destroy()
    gc.collect()

    return return_value_dic


if __name__ == "__main__":

    print("Smoothing: ", SMOOTHING)
    PARAMETER_NAME = "thickness"
    PARAMETER_CATEGORY = "model"
    NUM_RUNS = 15

    #a_list changed for beta_list
    #a_range changed for beta_range
    beta_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    #beta_list = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    beta_range = []

    x_range_plot = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    x_range_plot_log = [10, 20, 30, 70, 100]
    p_range = [1/el for el in beta_list] # CFe: thickness

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
    mesh_size = parameters["geometry"]["mesh_size"]
    IP = parameters["model"]["alpha_penalty"]
    nu = parameters["model"]["nu"] # Poisson ratio
    E = parameters["model"]["E"] # Young's modulus
    R = parameters["geometry"]["radius"] # Plate's radius
    c_nu =  1/(12*(1-nu**2))

    info_experiment = f"mesh_{mesh_size}_IP_{IP:.2e}_smth_{SMOOTHING}_tol = {SOL_TOLLERANCE}_load={LOAD_SIGN}_s={DISCLINATION_POWER_LIST}"
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
    visuals.setspines()
    fig = ax.get_figure()
    fig.savefig(f"{EXPERIMENT_DIR}/mesh.png")

    # DEFINE THE FUNCTION SPACE
    X = basix.ufl.element("P", str(mesh.ufl_cell()), parameters["model"]["order"])
    Q = dolfinx.fem.functionspace(mesh, basix.ufl.mixed_element([X, X]))

    # INITIAL GUESS
    q_ig = dolfinx.fem.Function(Q)

    # DEFINE THE DIRICHLET BOUNDARY CONDITIONS
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)

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
    sol_q_list = []

    for i, param in enumerate(p_range):
        print("Running experiment with beta = ", np.round(1/param, 3))
        if changed := update_parameters(parameters, PARAMETER_NAME, param):
            data_var = run_experiment(VARIATIONAL, costantData, parameters, q_ig)
            if COMPARISON:
                data_brn = run_experiment(BRENNER, costantData, parameters, q_ig)
                data_car = run_experiment(CARSTENSEN, costantData, parameters, q_ig)
            else:
                data_brn = data_var
                data_car = data_var

        if (data_var["convergence_id"] > 0) and (data_brn["convergence_id"] > 0) and (data_car["convergence_id"] > 0):
            beta_range.append(1/param)
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
            sol_q_list.append(data_var["q"])

            # Update the initial guess
            #if i == 0: q_ig.vector.array = data_var[7].vector.array
            #else: q_ig.vector.array = data_var[7].vector.array * p_range[i]/p_range[i-1]
            #print(" q_ig.vector.norm(PETSc.NormType.NORM_2): ",  q_ig.vector.norm(PETSc.NormType.NORM_2))
            #print(" data_var[7].vector.norm(PETSc.NormType.NORM_2): ",  data_var[7].vector.norm(PETSc.NormType.NORM_2) )
            #pdb.set_trace()
            #q_ig.vector.array = copy.deepcopy(data_var["q"].vector.array)
            #q_ig.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

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

    # KIRCHHOFF-LOVE ENERGIES
    f0 = 1
    kMembraneEngList = [(a**4)/(32*np.pi) for a in beta_range]
    kBendingEngList = [( (f0**2) * np.pi * (R**6) ) / ( 384 * c_nu ) for a in beta_range]

    # COMPUTE MONGE-AMPERE FORM
    bracket_vw_list = []
    bracket_ww_list = []
    bracket_vv_list = []
    for i in range(len(sol_q_list)):
        v_airy, w_td = ufl.split(sol_q_list[i])
        bracket_ww_list.append(mongeAmpereForm(sol_w_list[i], sol_w_list[i], mesh))
        bracket_vw_list.append(mongeAmpereForm(sol_v_list[i], sol_w_list[i], mesh))
        bracket_vv_list.append(mongeAmpereForm(sol_v_list[i], sol_v_list[i], mesh))

    # Instance of Pandas DF
    experimental_data = pd.DataFrame(_experimental_data, columns=columns)

    max_memEng = max(experimental_data["Membrane Erg (Var)"])
    min_memEng = min(experimental_data["Membrane Erg (Var)"])
    max_bendEng = max(experimental_data["Bending Erg (Var)"])
    min_bendEng = min(experimental_data["Bending Erg (Var)"])
    max_couplingEng = max(experimental_data["Coupling Erg (Var)"])
    min_couplingEng = min(experimental_data["Coupling Erg (Var)"])

    powerLaw_memEng = ( np.log(max_memEng) - np.log(min_memEng) ) / ( np.log(beta_range[-1]) - np.log(beta_range[0])  )
    powerLaw_bendEng = ( np.log(max_bendEng) - np.log(min_bendEng) ) / ( np.log(beta_range[0]) - np.log(beta_range[-1])  )
    powerLaw_couplingEng = ( np.log(np.abs(max_couplingEng)) - np.log(np.abs(min_couplingEng)) ) / ( np.log(beta_range[0]) - np.log(beta_range[-1])  )

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
    topology, cells, geometry = dolfinx.plot.vtk_mesh(V_v)
    grid = pyvista.UnstructuredGrid(topology, cells, geometry)
    def hessian(u): return ufl.grad(ufl.grad(u))
    sigma_xx_expr = dolfinx.fem.Expression( hessian(sol_v_list[index_solution])[Y_COORD, Y_COORD], V_v.element.interpolation_points() )
    sigma_xy_expr = dolfinx.fem.Expression( - hessian(sol_v_list[index_solution])[X_COORD, Y_COORD], V_v.element.interpolation_points() )
    sigma_yy_expr = dolfinx.fem.Expression( hessian(sol_v_list[index_solution])[X_COORD, X_COORD], V_v.element.interpolation_points() )

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
    experimental_data["Power law membrane eng"] = powerLaw_memEng
    experimental_data["Power law bending eng"] = powerLaw_bendEng
    experimental_data["Power law coupling eng"] = powerLaw_couplingEng
    # experimental_data.to_excel(f'{EXPERIMENT_DIR}/varying_a.xlsx', index=False)

    # PRINT OUT RESULTS
    print(10*"*")
    print("Results")
    print(experimental_data)
    print(10*"*")
    print("Details on non-converged experiments")
    for el in non_converged_list: print(el)
    print("Power law membrane energy: ", powerLaw_memEng)
    print("Power law bending energy: ", powerLaw_bendEng)
    print("Power law coupling energy: ", powerLaw_couplingEng)
    print(10*"*")

    # PLOTS L2 NORM
    FIGWIDTH = 17
    FIGHEIGHT = 11
    FONTSIZE = 30
    MARKERSIZE = 17
    LINEWIDTH = 7
    x_values = []
    for element in x_range_plot:
        if element in beta_range: x_values.append(element)

    plt.figure(figsize=(FIGWIDTH, FIGHEIGHT))
    plt.xticks(beta_range, [str(tick) if tick in x_values else '' for tick in beta_range])
    plt.plot(beta_range, experimental_data["L2norm v (Variational)"], marker='o', linestyle='solid', label='VAR', linewidth=LINEWIDTH, markersize=MARKERSIZE)
    if COMPARISON: plt.plot(beta_range, experimental_data["L2norm v (Brenner)"], marker='v', linestyle='dotted', label='BNRS17', linewidth=LINEWIDTH, markersize=MARKERSIZE)
    if COMPARISON: plt.plot(beta_range, experimental_data["L2norm v (Carstensen)"], marker='^', linestyle='dashed', label='CMN18', linewidth=LINEWIDTH, markersize=MARKERSIZE)

    max_v = max([max(experimental_data["L2norm v (Variational)"]), max(experimental_data["L2norm v (Brenner)"]), max(experimental_data["L2norm v (Carstensen)"])])
    min_v = min([min(experimental_data["L2norm v (Variational)"]), min(experimental_data["L2norm v (Brenner)"]), min(experimental_data["L2norm v (Carstensen)"])])
    steps_v = (max_v - min_v)/NUM_STEPS
    if steps_v == 0: steps_v = NUM_STEPS # CFe: v is constant
    xrange_list = []

    if LOG_SCALE: plt.xscale('log') # CFe: use log xscale when needed
    plt.xticks(beta_range, [str(tick) if tick in x_values else '' for tick in beta_range])
    if min_v != max_v:  plt.yticks(np.arange(min_v, max_v, steps_v))
    #plt.xlabel("a := R/h", fontsize=FONTSIZE)
    plt.xlabel(r"$\beta$ = $\frac{R}{h}$", fontsize=FONTSIZE)
    plt.ylabel(r'$| v |_{L^2(\Omega)} [Nm^2]$', fontsize=FONTSIZE)
    plt.title(f'Airy function. Mesh size = {mesh_size}. IP = {IP}. tol = {SOL_TOLLERANCE}', fontsize=FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.grid(True)
    plt.gca().yaxis.get_offset_text().set_fontsize(FONTSIZE)
    visuals.setspines()
    plt.savefig(EXPERIMENT_DIR+f'/varying_a_V_{info_experiment}.png', dpi=300)
    plt.savefig(EXPERIMENT_DIR+f'/varying_a_V_{info_experiment}.pdf', dpi=300)
    #plt.show()

    plt.figure(figsize=(FIGWIDTH, FIGHEIGHT))
    plt.plot(beta_range, experimental_data["L2norm w (Variational)"], marker='o', linestyle='solid', label='VAR', linewidth=LINEWIDTH, markersize=MARKERSIZE)
    if COMPARISON: plt.plot(beta_range, experimental_data["L2norm w (Brenner)"], marker='v', linestyle='dotted', label='BNRS17', linewidth=LINEWIDTH, markersize=MARKERSIZE)
    if COMPARISON: plt.plot(beta_range, experimental_data["L2norm w (Carstensen)"], marker='^', linestyle='dashed', label='CMN18', linewidth=LINEWIDTH, markersize=MARKERSIZE)

    max_w = max([max(experimental_data["L2norm w (Variational)"]), max(experimental_data["L2norm w (Brenner)"]), max(experimental_data["L2norm w (Carstensen)"])])
    min_w = min([min(experimental_data["L2norm w (Variational)"]), min(experimental_data["L2norm w (Brenner)"]), min(experimental_data["L2norm w (Carstensen)"])])
    steps_w = (max_w - min_w)/NUM_STEPS
    if steps_w == 0: steps_w = NUM_STEPS # CFe: if the deflection is not activated

    if LOG_SCALE: plt.xscale('log') # CFe: use log xscale when needed
    plt.xticks(beta_range, [str(tick) if tick in x_values else '' for tick in beta_range])
    if min_w != max_w:  plt.yticks(np.arange(min_w, max_w, steps_w))
    plt.xlabel(r"$\beta$ = $\frac{R}{h}$", fontsize=FONTSIZE)
    plt.ylabel(r'$| w |_{L^2(\Omega)} [m^2]$', fontsize=FONTSIZE)
    plt.title(f'Transverse displacement. Mesh size = {parameters["geometry"]["mesh_size"]}. IP = {parameters["model"]["alpha_penalty"]}. tol = {SOL_TOLLERANCE}', fontsize=FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.gca().yaxis.get_offset_text().set_fontsize(FONTSIZE)
    plt.grid(True)
    visuals.setspines()
    plt.savefig(EXPERIMENT_DIR+f'/varying_a_W_{info_experiment}.png', dpi=300)
    plt.savefig(EXPERIMENT_DIR+f'/varying_a_W_{info_experiment}.pdf', dpi=300)
    #plt.show()

    # PLOTS HESSIAN L2 NORM
    plt.figure(figsize=(FIGWIDTH, FIGHEIGHT))
    plt.plot(beta_range, experimental_data["Hessian L2norm v (Variational)"], marker='o', linestyle='solid', label='VAR', linewidth=LINEWIDTH, markersize=MARKERSIZE)
    if COMPARISON: plt.plot(beta_range, experimental_data["Hessian L2norm v (Brenner)"], marker='v', linestyle='dotted', label='BNRS17', linewidth=LINEWIDTH, markersize=MARKERSIZE)
    if COMPARISON: plt.plot(beta_range, experimental_data["Hessian L2norm v (Carstensen)"], marker='^', linestyle='dashed', label='CMN18', linewidth=LINEWIDTH, markersize=MARKERSIZE)

    max_v = max([max(experimental_data["Hessian L2norm v (Variational)"]), max(experimental_data["Hessian L2norm v (Brenner)"]), max(experimental_data["Hessian L2norm v (Carstensen)"])])
    min_v = min([min(experimental_data["Hessian L2norm v (Variational)"]), min(experimental_data["Hessian L2norm v (Brenner)"]), min(experimental_data["Hessian L2norm v (Carstensen)"])])

    steps_v = (max_v - min_v)/NUM_STEPS
    if steps_v == 0: steps_v = NUM_STEPS # CFe: v is constant

    if LOG_SCALE: plt.xscale('log') # CFe: use log xscale when needed
    plt.xticks(beta_range, [str(tick) if tick in x_values else '' for tick in beta_range])
    if min_v != max_v:  plt.yticks(np.arange(min_v, max_v, steps_v))
    plt.xlabel(r"$\beta$ = $\frac{R}{h}$", fontsize=FONTSIZE)
    plt.ylabel(r'$| \nabla^2 v |_{L^2(\Omega)} [N]$', fontsize=FONTSIZE)
    plt.title(f'Airy function. Mesh size = {mesh_size}. IP = {IP}. tol = {SOL_TOLLERANCE}', fontsize=FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.gca().yaxis.get_offset_text().set_fontsize(FONTSIZE)
    plt.grid(True)
    visuals.setspines()
    plt.savefig(EXPERIMENT_DIR+f'/varying_a_HessV_{info_experiment}.png', dpi=300)
    #plt.show()

    plt.figure(figsize=(FIGWIDTH, FIGHEIGHT))
    plt.plot(beta_range, experimental_data["Hessian L2norm w (Variational)"], marker='o', linestyle='solid', label='VAR', linewidth=LINEWIDTH, markersize=MARKERSIZE)
    if COMPARISON: plt.plot(beta_range, experimental_data["Hessian L2norm w (Brenner)"], marker='v', linestyle='dotted', label='BNRS17', linewidth=LINEWIDTH, markersize=MARKERSIZE)
    if COMPARISON: plt.plot(beta_range, experimental_data["Hessian L2norm w (Carstensen)"], marker='^', linestyle='dashed', label='CMN18', linewidth=LINEWIDTH, markersize=MARKERSIZE)

    max_w = max([max(experimental_data["Hessian L2norm w (Variational)"]), max(experimental_data["Hessian L2norm w (Brenner)"]), max(experimental_data["Hessian L2norm w (Carstensen)"])])
    min_w = min([min(experimental_data["Hessian L2norm w (Variational)"]), min(experimental_data["Hessian L2norm w (Brenner)"]), min(experimental_data["Hessian L2norm w (Carstensen)"])])

    steps_w = (max_w - min_w)/NUM_STEPS
    if steps_w == 0: steps_w = NUM_STEPS # CFe: w is constant

    if LOG_SCALE: plt.xscale('log') # CFe: use log xscale when needed
    plt.xticks(beta_range, [str(tick) if tick in x_values else '' for tick in beta_range])
    if min_w != max_w:  plt.yticks(np.arange(min_w, max_w, steps_w))
    plt.xlabel(r"$\beta$ = $\frac{R}{h}$", fontsize=FONTSIZE)
    plt.ylabel(r'$| \nabla^2 w |_{L^2(\Omega)} $', fontsize=FONTSIZE)
    plt.title(f'Transverse displacement. Mesh size = {mesh_size}. IP = {IP}. tol = {SOL_TOLLERANCE}', fontsize=FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.gca().yaxis.get_offset_text().set_fontsize(FONTSIZE)
    plt.grid(True)
    visuals.setspines()
    plt.savefig(EXPERIMENT_DIR+f'/varying_a_HessW_{info_experiment}.png', dpi=300)
    #plt.show()

    # PLOT MEMBRANE ENERGY
    plt.figure(figsize=(FIGWIDTH, FIGHEIGHT))
    plt.plot(beta_range, experimental_data["Membrane Erg (Var)"], marker='o', linestyle='solid', label='VAR', linewidth=LINEWIDTH, markersize=MARKERSIZE)

    steps_memEng = abs( (max_memEng - min_memEng)/NUM_STEPS )
    if steps_memEng == 0: steps_memEng = NUM_STEPS
    if LOG_SCALE: plt.xscale('log') # CFe: use log xscale when needed
    plt.xticks(beta_range, [str(tick) if tick in x_values else '' for tick in beta_range])
    if max_memEng != min_memEng: plt.yticks(np.arange(min_memEng, max_memEng, steps_memEng))
    plt.xlabel(r"$\beta$ = $\frac{R}{h}$", fontsize=FONTSIZE)
    plt.ylabel('Membrane Energy', fontsize=FONTSIZE)
    plt.title(f'Membrane Energy. Mesh size = {mesh_size}. IP = {IP}. tol = {SOL_TOLLERANCE}', fontsize=FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.gca().yaxis.get_offset_text().set_fontsize(FONTSIZE)
    visuals.setspines()
    plt.grid(True)
    plt.savefig(EXPERIMENT_DIR+f'/varying_a_MembEng_{info_experiment}.png', dpi=300)

    # LOG-PLOT MEMBRANE ENERGY
    plt.figure(figsize=(FIGWIDTH, FIGHEIGHT))
    plt.plot(beta_range, experimental_data["Membrane Erg (Var)"], marker='o', linestyle='solid', label='VAR', linewidth=LINEWIDTH, markersize=MARKERSIZE)
    plt.plot(beta_range, kMembraneEngList, marker='.', linestyle='--', label='KL model', linewidth=LINEWIDTH, markersize=MARKERSIZE)

    steps_memEng = abs( (max_memEng - min_memEng)/NUM_STEPS )
    if steps_memEng == 0: steps_memEng = NUM_STEPS
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(beta_range, [str(tick) if tick in x_range_plot_log else '' for tick in beta_range])
    if max_memEng != min_memEng: plt.yticks(np.arange(min_memEng, max_memEng, steps_memEng))
    plt.xlabel(r"$\beta$ = $\frac{R}{h}$", fontsize=FONTSIZE)
    plt.ylabel('Membrane Energy', fontsize=FONTSIZE)
    plt.title(f'Membrane Energy. Mesh size = {mesh_size}. IP = {IP}. tol = {SOL_TOLLERANCE}', fontsize=FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.gca().yaxis.get_offset_text().set_fontsize(FONTSIZE)
    plt.grid(True)
    visuals.setspines()
    plt.savefig(EXPERIMENT_DIR+f'/varying_a_MembEng_log_{info_experiment}.png', dpi=300)

    # PLOT BENDING ENERGY
    plt.figure(figsize=(FIGWIDTH, FIGHEIGHT))
    plt.plot(beta_range, experimental_data["Bending Erg (Var)"], marker='o', linestyle='solid', label='VAR', linewidth=LINEWIDTH, markersize=MARKERSIZE)
    plt.axhline(y=0, linewidth=2)
    steps_bendEng = (max_bendEng - min_bendEng)/NUM_STEPS
    if steps_bendEng == 0: steps_bendEng = NUM_STEPS
    if LOG_SCALE: plt.xscale('log') # CFe: use log xscale when needed
    plt.xticks(beta_range, [str(tick) if tick in x_values else '' for tick in beta_range])
    if max_bendEng != min_bendEng: plt.yticks(np.arange(min_bendEng, max_bendEng, steps_bendEng))
    plt.xlabel(r"$\beta$ = $\frac{R}{h}$", fontsize=FONTSIZE)
    plt.ylabel('Bending Energy', fontsize=FONTSIZE)
    plt.title(f'Bending Energy. Mesh size = {mesh_size}. IP = {IP}. tol = {SOL_TOLLERANCE}', fontsize=FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.gca().yaxis.get_offset_text().set_fontsize(FONTSIZE)
    plt.grid(True)
    visuals.setspines()
    plt.savefig(EXPERIMENT_DIR+f'/varying_a_BendEng_{info_experiment}.png', dpi=300)

    # LOG-PLOT BENDING ENERGY
    #(np.log(experimental_data["Bending Erg (Var)"][-1]) - np.log(experimental_data["Bending Erg (Var)"][0]))/(np.log(100) - np.log(10))
    plt.figure(figsize=(FIGWIDTH, FIGHEIGHT))
    plt.plot(beta_range, experimental_data["Bending Erg (Var)"], marker='o', linestyle='solid', label='VAR', linewidth=LINEWIDTH, markersize=MARKERSIZE)
    if steps_memEng == 0: steps_memEng = NUM_STEPS
    plt.xscale('log')
    plt.yscale('log')
    plt.axhline(y=0,c='k', linewidth=2)
    max_bendEng = max(experimental_data["Bending Erg (Var)"])
    min_bendEng = min(experimental_data["Bending Erg (Var)"])
    steps_bendEng = (max_bendEng - min_bendEng)/NUM_STEPS
    if steps_bendEng == 0: steps_bendEng = NUM_STEPS
    if LOG_SCALE: plt.xscale('log') # CFe: use log xscale when needed
    plt.xticks(beta_range, [str(tick) if tick in x_range_plot_log else '' for tick in beta_range])
    plt.xlabel(r"$\beta$ = $\frac{R}{h}$", fontsize=FONTSIZE)
    plt.ylabel('Bending Energy', fontsize=FONTSIZE)
    plt.title(f'Bending Energy. Mesh size = {mesh_size}. IP = {IP}. tol = {SOL_TOLLERANCE}', fontsize=FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.gca().yaxis.get_offset_text().set_fontsize(FONTSIZE)
    plt.grid(True)
    visuals.setspines()
    plt.savefig(EXPERIMENT_DIR+f'/varying_a_BendEng_log_{info_experiment}.png', dpi=300)

    # PLOT COUPLING ENERGY
    plt.figure(figsize=(FIGWIDTH, FIGHEIGHT))
    plt.plot(beta_range, experimental_data["Coupling Erg (Var)"], marker='o', linestyle='solid', label='VAR', linewidth=LINEWIDTH, markersize=MARKERSIZE)
    plt.axhline(y=0,c='k', linewidth=2)

    steps_couplingEng = (max_couplingEng - min_couplingEng)/NUM_STEPS
    if steps_couplingEng == 0: steps_couplingEng = NUM_STEPS

    if LOG_SCALE: plt.xscale('log') # CFe: use log xscale when needed
    plt.xticks(beta_range, [str(tick) if tick in x_values else '' for tick in beta_range])
    if max_couplingEng != min_couplingEng: plt.yticks(np.arange(min_couplingEng, max_couplingEng, steps_couplingEng))
    plt.xlabel(r"$\beta$ = $\frac{R}{h}$", fontsize=FONTSIZE)
    plt.ylabel('Coupling Energy', fontsize=FONTSIZE)
    plt.title(f'Coupling Energy. Mesh size = {mesh_size}. IP = {IP}. tol = {SOL_TOLLERANCE}', fontsize=FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.gca().yaxis.get_offset_text().set_fontsize(FONTSIZE)
    plt.grid(True)
    visuals.setspines()
    plt.savefig(EXPERIMENT_DIR+f'/varying_a_CouplEng_{info_experiment}.png', dpi=300)
    #plt.show()

    # PLOT COUPLING ENERGY LOG-LOG
    plt.figure(figsize=(FIGWIDTH, FIGHEIGHT))
    plt.plot(beta_range, experimental_data["Coupling Erg (Var)"], marker='o', linestyle='solid', label='VAR', linewidth=LINEWIDTH, markersize=MARKERSIZE)
    if steps_memEng == 0: steps_memEng = NUM_STEPS
    plt.xscale('log')
    plt.yscale('log')
    plt.axhline(y=0,c='k', linewidth=2)
    plt.xticks(beta_range, [str(tick) if tick in x_range_plot_log else '' for tick in beta_range])
    plt.xlabel(r"$\beta$ = $\frac{R}{h}$", fontsize=FONTSIZE)
    plt.ylabel('Coupling Energy', fontsize=FONTSIZE)
    plt.title(f'Coupling Energy. Mesh size = {mesh_size}. IP = {IP}. tol = {SOL_TOLLERANCE}', fontsize=FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.gca().yaxis.get_offset_text().set_fontsize(FONTSIZE)
    plt.grid(True)
    visuals.setspines()
    plt.savefig(EXPERIMENT_DIR+f'/varying_a_CouplEng_log_{info_experiment}.png', dpi=300)



    # PLOT LOG-LOG ENERGIES
    plt.figure(figsize=(FIGWIDTH, FIGHEIGHT))
    plt.plot(beta_range, experimental_data["Membrane Erg (Var)"], marker='o', linestyle='none', label='Membrane', linewidth=LINEWIDTH, markersize=MARKERSIZE)
    plt.plot(beta_range, experimental_data["Bending Erg (Var)"], marker='^', linestyle='none', label='Bending', linewidth=LINEWIDTH, markersize=MARKERSIZE)
    plt.plot(beta_range, experimental_data["Coupling Erg (Var)"], marker='v', linestyle='none', label='Coupling', linewidth=LINEWIDTH, markersize=MARKERSIZE)
    plt.plot(beta_range, kMembraneEngList, marker='.', linestyle='--', label='KL model: membrane', linewidth=LINEWIDTH, markersize=0*MARKERSIZE)
    plt.plot(beta_range, kBendingEngList, marker='.', linestyle='--', label='KL model: bending', linewidth=LINEWIDTH, markersize=0*MARKERSIZE)
    plt.xscale('log')
    plt.yscale('log')
    plt.axhline(y=0,c='k', linewidth=2)
    plt.xticks(beta_range, [str(tick) if tick in x_range_plot_log else '' for tick in beta_range])
    plt.xlabel(r"$\beta$ = $\frac{R}{h}$", fontsize=FONTSIZE)
    plt.ylabel('Dimensionless energy', fontsize=FONTSIZE)
    #plt.title(f'Log-log plot energies. Mesh size = {mesh_size}. IP = {IP}. tol = {SOL_TOLLERANCE}', fontsize=FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.gca().yaxis.get_offset_text().set_fontsize(FONTSIZE)
    plt.grid(True)
    visuals.setspines()
    plt.savefig(EXPERIMENT_DIR+f'/varying_a_loglog_energies_{info_experiment}.png', dpi=300)

    # PLOT SIGMA NN
    subplotter = pyvista.Plotter(shape=(1, 1))
    subplotter.subplot(0, 0)
    grid["sigma_nn"] = sigma_nn.x.array.real
    grid.set_active_scalars("sigma_nn")
    subplotter.add_text("sigma_rr", position="upper_edge", font_size=14,color="k")
    scalar_bar_args["title"] = "sigma_rr"
    subplotter.add_mesh( grid.warp_by_scalar( scale_factor = 1 / max(np.abs(sigma_nn.x.array.real )) ), 
                        show_edges=False, show_scalar_bar=True, scalar_bar_args=scalar_bar_args, cmap="coolwarm")
    IMG_WIDTH = 1600
    IMG_HEIGHT = 1200
    PNG_SCALE = 2
    subplotter.window_size = (IMG_WIDTH, IMG_HEIGHT)
    subplotter.screenshot(f"{EXPERIMENT_DIR}/visualization_sigma_nn_{index_solution}_{info_experiment}.png", scale = PNG_SCALE)
    # subplotter.screenshot(f"{EXPERIMENT_DIR}/visualization_sigma_nn_{index_solution}_{info_experiment}.pdf]", scale = PNG_SCALE)
    # subplotter.export_html(f"{EXPERIMENT_DIR}/visualization_sigma_nn_{index_solution}_{info_experiment}.html")

    # PLOT PENALIZATION ENERGY

    # Penalization energy of Transverse Displacement w
    plt.figure(figsize=(FIGWIDTH, FIGHEIGHT))
    plt.plot(beta_range, experimental_data["Penalization W Eng (Var)"], marker='o', linestyle='solid', label='VAR', linewidth=LINEWIDTH, markersize=MARKERSIZE)

    max_penalizWEng = max(experimental_data["Penalization W Eng (Var)"])
    min_penalizWEng = min(experimental_data["Penalization W Eng (Var)"])

    steps_penalizWEng = (max_penalizWEng - min_penalizWEng)/NUM_STEPS
    if steps_penalizWEng == 0: steps_penalizWEng = NUM_STEPS # CFe: w is constant

    if LOG_SCALE: plt.xscale('log') # CFe: use log xscale when needed
    plt.xticks(beta_range, [str(tick) if tick in x_values else '' for tick in beta_range])
    plt.yticks(np.arange(min_penalizWEng, max_penalizWEng, steps_penalizWEng))
    plt.xlabel(r"$\beta$ = $\frac{R}{h}$", fontsize=FONTSIZE)
    plt.ylabel('Penalization Energy w', fontsize=FONTSIZE)
    plt.title(f'Penalization Energy w. Mesh size = {mesh_size}. IP = {IP}. tol = {SOL_TOLLERANCE}', fontsize=FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.gca().yaxis.get_offset_text().set_fontsize(FONTSIZE)
    plt.grid(True)
    visuals.setspines()
    plt.savefig(EXPERIMENT_DIR+f'/varying_a_PenEngW_{info_experiment}.png', dpi=300)

    # Penalization energy of the Jump of the Hessian of the Transverse Displacement w
    plt.figure(figsize=(FIGWIDTH, FIGHEIGHT))
    plt.plot(beta_range, experimental_data["Penalization W Hess Jump (Var)"], marker='o', linestyle='solid', label='VAR', linewidth=LINEWIDTH, markersize=MARKERSIZE)

    max_penalizWEng = max(experimental_data["Penalization W Hess Jump (Var)"])
    min_penalizWEng = min(experimental_data["Penalization W Hess Jump (Var)"])

    steps_penalizWEng = (max_penalizWEng - min_penalizWEng)/NUM_STEPS
    if steps_penalizWEng == 0: steps_penalizWEng = NUM_STEPS # CFe: w is constant

    if LOG_SCALE: plt.xscale('log') # CFe: use log xscale when needed
    plt.xticks(beta_range, [str(tick) if tick in x_values else '' for tick in beta_range])
    plt.yticks(np.arange(min_penalizWEng, max_penalizWEng, steps_penalizWEng))
    plt.xlabel(r"$\beta$ = $\frac{R}{h}$", fontsize=FONTSIZE)
    plt.ylabel('Penalization Hessian w', fontsize=FONTSIZE)
    plt.title(f'Penalization Energy w. Mesh size = {mesh_size}. IP = {IP}. tol = {SOL_TOLLERANCE}', fontsize=FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.gca().yaxis.get_offset_text().set_fontsize(FONTSIZE)
    plt.grid(True)
    visuals.setspines()
    plt.savefig(EXPERIMENT_DIR+f'/varying_a_PenHessJumpW_{info_experiment}.png', dpi=300)
    plt.savefig(EXPERIMENT_DIR+f'/varying_a_PenHessJumpW_{info_experiment}.pdf', dpi=300)

    # Penalization energy of Airy's function v
    plt.figure(figsize=(FIGWIDTH, FIGHEIGHT))
    plt.plot(beta_range, experimental_data["Penalization V Eng (Var)"], marker='o', linestyle='solid', label='VAR', linewidth=LINEWIDTH, markersize=MARKERSIZE)

    max_penalizVEng = max(experimental_data["Penalization V Eng (Var)"])
    min_penalizVEng = min(experimental_data["Penalization V Eng (Var)"])
    steps_penalizVEng = (max_penalizVEng - min_penalizVEng)/NUM_STEPS
    if steps_penalizVEng == 0: steps_penalizVEng = NUM_STEPS # CFe: w is constant

    if LOG_SCALE: plt.xscale('log') # CFe: use log xscale when needed
    plt.xticks(beta_range, [str(tick) if tick in x_values else '' for tick in beta_range])
    plt.yticks(np.arange(min_penalizVEng, max_penalizVEng, steps_penalizVEng))
    plt.xlabel(r"$\beta$ = $\frac{R}{h}$", fontsize=FONTSIZE)
    plt.ylabel('Penalization Energy v', fontsize=FONTSIZE)
    plt.title(f'Penalization Energy v. Mesh size = {mesh_size}. IP = {IP}. tol = {SOL_TOLLERANCE}', fontsize=FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.gca().yaxis.get_offset_text().set_fontsize(FONTSIZE)
    plt.grid(True)
    visuals.setspines()
    plt.savefig(EXPERIMENT_DIR+f'/varying_a_PenEngV_{info_experiment}.png', dpi=300)
    plt.savefig(EXPERIMENT_DIR+f'/varying_a_PenEngV_{info_experiment}.pdf', dpi=300)

    # PLOT DIFFERENCES BETWEEN ENERGIES AND PENALTY ENERGIES

    w_dg1_var_list = experimental_data["Penalty w dg1 (Var)"]
    w_dg2_var_list = experimental_data["Penalty w dg2 (Var)"]
    w_bc1_var_list = experimental_data["Penalty w bc1 (Var)"]
    w_bc3_var_list = experimental_data["Penalty w bc3 (Var)"]
    w_dg3_var_list = experimental_data["Penalization W Hess Jump (Var)"]

    fig, ax = plt.subplots(2, 2, figsize=(FIGWIDTH, FIGHEIGHT))
    ax[0,0].plot(beta_range, w_dg1_var_list, marker='o', linestyle='solid', label="w_dg1", linewidth=LINEWIDTH, markersize=MARKERSIZE)
    ax[0,0].legend(fontsize=FONTSIZE)
    ax[0,0].grid(True)
    ax[0,0].set_xlabel(r"$\beta$ = $\frac{R}{h}$", fontsize=FONTSIZE)
    plt.gca().yaxis.get_offset_text().set_fontsize(FONTSIZE)
    ax[0,1].plot(beta_range, w_dg2_var_list, marker='v', linestyle='solid', label="w_dg2", linewidth=LINEWIDTH, markersize=MARKERSIZE)
    ax[0,1].legend(fontsize=FONTSIZE)
    ax[0,1].grid(True)
    ax[0,1].set_xlabel(r"$\beta$ = $\frac{R}{h}$", fontsize=FONTSIZE)
    plt.gca().yaxis.get_offset_text().set_fontsize(FONTSIZE)
    ax[1,0].plot(beta_range, w_bc1_var_list, marker='P', linestyle='solid', label="w_bc1", linewidth=LINEWIDTH, markersize=MARKERSIZE)
    ax[1,0].legend(fontsize=FONTSIZE)
    ax[1,0].grid(True)
    plt.gca().yaxis.get_offset_text().set_fontsize(FONTSIZE)
    ax[1,0].set_xlabel(r"$\beta$ = $\frac{R}{h}$", fontsize=FONTSIZE)
    ax[1,1].plot(beta_range, w_bc3_var_list, marker='*', linestyle='solid', label="w_bc3", linewidth=LINEWIDTH, markersize=MARKERSIZE)
    ax[1,1].legend(fontsize=FONTSIZE)
    ax[1,1].grid(True)
    ax[1,1].set_xlabel(r"$\beta$ = $\frac{R}{h}$", fontsize=FONTSIZE)
    plt.xticks(beta_range, [str(tick) if tick in x_values else '' for tick in beta_range])
    fig.suptitle(f'Penalization Energy v. Mesh size = {mesh_size}. IP = {IP}. tol = {SOL_TOLLERANCE}', fontsize=FONTSIZE)
    plt.gca().yaxis.get_offset_text().set_fontsize(FONTSIZE)
    visuals.setspines()
    plt.savefig(EXPERIMENT_DIR+f'/varying_a_PenEngsW_{info_experiment}.png', dpi=300)
    plt.savefig(EXPERIMENT_DIR+f'/varying_a_PenEngsW_{info_experiment}.pdf', dpi=300)

    # PLOT PROFILES

    tol = 1e-5
    xs = np.linspace(-R + tol, R - tol, 3001)
    points = np.zeros((3, 3001))
    points[0] = xs

    for i, sol_w in enumerate(sol_w_list):

        fig, axes = plt.subplots(1, 1, figsize=(FIGWIDTH, FIGHEIGHT))
        axes.yaxis.get_offset_text().set_size(FONTSIZE) # Size of the order of magnitude
        plt, data = plot_profile(sol_w, points, None, subplot=(1, 1), lineproperties={"c": "r", "lw":5, "ls": "solid"}, fig=fig, subplotnumber=1)
        plt.xlabel(r"$\xi_1$", fontsize=FONTSIZE)
        plt.ylabel("w", fontsize=FONTSIZE)
        plt.xticks(fontsize=FONTSIZE)
        plt.yticks(fontsize=FONTSIZE)
        ax = plt.gca() # use scientific notation for y axis
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_fontsize(FONTSIZE)
        plt.title(fr"Transverse displacement. $\beta$ = {beta_range[i]}; Mesh = {mesh_size}. IP = {IP}. E = {E:.2e} tol = {SOL_TOLLERANCE}", size = FONTSIZE)
        plt.grid(True)
        plt.legend(fontsize=FONTSIZE)
        visuals.setspines()
        plt.savefig(f"{EXPERIMENT_DIR}/_{beta_range[i]}a-w-profiles_{info_experiment}.png")
        plt.savefig(f"{EXPERIMENT_DIR}/_{beta_range[i]}a-w-profiles_{info_experiment}.pdf")

        # fig, axes = plt.subplots(1, 1, figsize=(FIGWIDTH, FIGHEIGHT))
        # sol_w.vector.array = sol_w.vector.array / beta_range[i]
        # plt, data = plot_profile(sol_w, points, None, subplot=(1, 1), lineproperties={"c": "r", "lw":5, "ls": "solid"}, fig=fig, subplotnumber=1)
        # plt.xlabel("x [m]", fontsize=FONTSIZE)
        # plt.ylabel("Transverse displacement [m]", fontsize=FONTSIZE)
        # plt.xticks(fontsize=FONTSIZE)
        # plt.yticks(fontsize=FONTSIZE)
        # plt.ylim(-1, 1)
        # plt.title(f"Transverse displacement [1:1] aspect ratio. Thickness = {1/beta_range[i]} m; a = {beta_range[i]}; Mesh = {parameters["geometry"]["mesh_size"]}. IP = {parameters["model"]["alpha_penalty"]}", size = FONTSIZE)
        # plt.grid(True)
        # plt.legend(fontsize=FONTSIZE)
        # plt.savefig(f"{EXPERIMENT_DIR}/_{beta_range[i]}b-w-profiles_{info_experiment}.png")

        fig, axes = plt.subplots(1, 1, figsize=(FIGWIDTH, FIGHEIGHT))
        plt, data = plot_profile(sol_v_list[i], points, None, subplot=(1, 1), lineproperties={"lw":5, "ls": "solid"}, fig=fig, subplotnumber=1)
        plt.xlabel(r"$\xi_1$", fontsize=FONTSIZE)
        plt.ylabel("v", fontsize=FONTSIZE)
        plt.xticks(fontsize=FONTSIZE)
        plt.yticks(fontsize=FONTSIZE)
        ax = plt.gca() # use scientific notation for y axis
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_fontsize(FONTSIZE)
        plt.title(fr"Airy. $\beta$ = {beta_range[i]}; Mesh = {mesh_size}. IP = {IP}. E = {E:.2e} tol = {SOL_TOLLERANCE}", size = FONTSIZE)
        plt.grid(True)
        plt.legend(fontsize=FONTSIZE)
        plt.savefig(f"{EXPERIMENT_DIR}/_{beta_range[i]}a-v-profiles_{info_experiment}.png")
        plt.savefig(f"{EXPERIMENT_DIR}/_{beta_range[i]}a-v-profiles_{info_experiment}.pdf")

        # Profiles of Monge-Ampere Bracket v, w
        fig, axes = plt.subplots(1, 1, figsize=(FIGWIDTH, FIGHEIGHT))
        plt, data = plot_profile(bracket_vw_list[i], points, None, subplot=(1, 1), lineproperties={"lw":5, "ls": "solid"}, fig=fig, subplotnumber=1)
        plt.xlabel(r"$\xi_1$", fontsize=FONTSIZE)
        plt.ylabel(r"$[v, w]$", fontsize=FONTSIZE)
        plt.xticks(fontsize=FONTSIZE)
        plt.yticks(fontsize=FONTSIZE)
        ax = plt.gca() # use scientific notation for y axis
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_fontsize(FONTSIZE)
        plt.title(fr"[v, w]. $\beta$ = {beta_range[i]}; Mesh = {mesh_size}. IP = {IP}. E = {E:.2e}. tol = {SOL_TOLLERANCE}", size = FONTSIZE)
        plt.grid(True)
        plt.legend(fontsize=FONTSIZE)
        plt.savefig(f"{EXPERIMENT_DIR}/_{beta_range[i]}a-[v_w]-profiles_{info_experiment}.png")
        plt.savefig(f"{EXPERIMENT_DIR}/_{beta_range[i]}a-[v_w]-profiles_{info_experiment}.pdf")

        # Profiles of Monge-Ampere Bracket w, w
        fig, axes = plt.subplots(1, 1, figsize=(FIGWIDTH, FIGHEIGHT))
        plt, data = plot_profile(bracket_ww_list[i], points, None, subplot=(1, 1), lineproperties={"lw":5, "ls": "solid"}, fig=fig, subplotnumber=1)
        plt.xlabel(r"$\xi_1$", fontsize=FONTSIZE)
        plt.ylabel(r"$[w, w]$", fontsize=FONTSIZE)
        plt.xticks(fontsize=FONTSIZE)
        plt.yticks(fontsize=FONTSIZE)
        ax = plt.gca() # use scientific notation for y axis
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_fontsize(FONTSIZE)
        plt.title(fr"[w, w]. $\beta$ = {beta_range[i]}; Mesh = {mesh_size}. IP = {IP}. E = {E:.2e} tol = {SOL_TOLLERANCE}", size = FONTSIZE)
        plt.grid(True)
        plt.legend(fontsize=FONTSIZE)
        plt.savefig(f"{EXPERIMENT_DIR}/_{beta_range[i]}a-[w_w]-profiles_{info_experiment}.png")
        plt.savefig(f"{EXPERIMENT_DIR}/_{beta_range[i]}a-[w_w]-profiles_{info_experiment}.pdf")


        # Profiles of Monge-Ampere Bracket v, v
        fig, axes = plt.subplots(1, 1, figsize=(FIGWIDTH, FIGHEIGHT))
        plt, data = plot_profile(bracket_vv_list[i], points, None, subplot=(1, 1), lineproperties={"lw":5, "ls": "solid"}, fig=fig, subplotnumber=1)
        plt.xlabel(r"$\xi_1$", fontsize=FONTSIZE)
        plt.ylabel(r"$[v, v]$", fontsize=FONTSIZE)
        plt.xticks(fontsize=FONTSIZE)
        plt.yticks(fontsize=FONTSIZE)
        ax = plt.gca() # use scientific notation for y axis
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_fontsize(FONTSIZE)
        plt.title(fr"[v, v]. $\beta$ = {beta_range[i]}; Mesh = {mesh_size}. IP = {IP}. E = {E:.2e} tol = {SOL_TOLLERANCE}", size = FONTSIZE)
        plt.grid(True)
        plt.legend(fontsize=FONTSIZE)
        plt.savefig(f"{EXPERIMENT_DIR}/_{beta_range[i]}a-[v_v]-profiles_{info_experiment}.png")
        plt.savefig(f"{EXPERIMENT_DIR}/_{beta_range[i]}a-[v_v]-profiles_{info_experiment}.pdf")



    if pyvista.OFF_SCREEN: pyvista.start_xvfb(wait=0.1)
    transparent = False
    topology, cells, geometry = dolfinx.plot.vtk_mesh(V_v)
    grid = pyvista.UnstructuredGrid(topology, cells, geometry)

    # PLOT PROFILES
    grid.point_data["w_1"] = sol_w_list[0].vector.array[dofs2_w]
    grid.point_data["w_2"] = sol_w_list[-1].vector.array[dofs2_w]
    y0 = 0
    tolerance = 1e-2
    points = grid.points
    x_values = points[np.abs(points[:, 1] - y0) < tolerance, 0]  # Select x-coordinates at y = 0
    w_1 = grid['w_1'][np.abs(points[:, 1] - y0) < tolerance]
    w_2 = grid['w_2'][np.abs(points[:, 1] - y0) < tolerance]
    sorted_indices = np.argsort(x_values) # Sort data for plotting
    x_sorted = x_values[sorted_indices]
    w_1sorted = w_1[sorted_indices]
    w_2sorted = w_2[sorted_indices]
    w_1normalized = w_1sorted / np.max(np.abs(w_1sorted))
    w_2normalized = w_2sorted / np.max(np.abs(w_2sorted))
    scale_w1 = f"{np.max(np.abs(w_1sorted)):.2e}"
    scale_w2 = f"{np.max(np.abs(w_2sorted)):.2e}"
    plt.figure(figsize=(FIGWIDTH, FIGHEIGHT))
    plt.plot(x_sorted, w_1normalized, label=fr'$\beta$: 10, max(|w|): {scale_w1}', linestyle='solid', linewidth=LINEWIDTH)
    plt.plot(x_sorted, w_2normalized, label=fr'$\beta$: 100, max(|w|): {scale_w2}',c='k', linestyle='dashed', linewidth=LINEWIDTH)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.xlabel(r"$\xi_1$", fontsize=FONTSIZE)
    plt.ylabel(r"$ \frac{w}{\max(|w|)}$", fontsize=FONTSIZE)
    plt.title(fr"Profile of w at $\xi_2$ = {y0}", fontsize=FONTSIZE)
    ax = plt.gca() # use scientific notation for y axis
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(FONTSIZE)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))  # Adjust the number of bins to choose the number of ticks
    plt.legend(fontsize=FONTSIZE)
    #plt.grid(True)
    plt.savefig(f"{EXPERIMENT_DIR}/profiles_w_comparison_y_{y0}_{info_experiment}.png", dpi=300)
    plt.savefig(f"{EXPERIMENT_DIR}/profiles_w_comparison_y_{y0}_{info_experiment}.pdf", dpi=300)

    grid.point_data["v_1"] = sol_v_list[0].vector.array[dofs2_v]
    grid.point_data["v_2"] = sol_v_list[-1].vector.array[dofs2_v]
    points = grid.points
    x_values = points[np.abs(points[:, 1] - y0) < tolerance, 0]  # Select x-coordinates at y = 0
    v_1 = grid['v_1'][np.abs(points[:, 1] - y0) < tolerance]
    v_2 = grid['v_2'][np.abs(points[:, 1] - y0) < tolerance]
    sorted_indices = np.argsort(x_values) # Sort data for plotting
    x_sorted = x_values[sorted_indices]
    v_1sorted = v_1[sorted_indices]
    v_2sorted = v_2[sorted_indices]
    v_1normalized = v_1sorted / np.max(np.abs(v_1sorted))
    v_2normalized = v_2sorted / np.max(np.abs(v_2sorted))
    scale_v1 = f"{np.max(np.abs(v_1sorted)):.1e}"
    scale_v2 = f"{np.max(np.abs(v_2sorted)):.1e}"
    plt.figure(figsize=(FIGWIDTH, FIGHEIGHT))
    plt.plot(x_sorted, v_1normalized, label=fr'$\beta$: 10, max(|v|): {scale_v1}', linestyle='solid', linewidth=LINEWIDTH)
    plt.plot(x_sorted, v_2normalized, label=fr'$\beta$: 100, max(|v|): {scale_v2}',c='k', linestyle='dashed', linewidth=LINEWIDTH)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.xlabel(r"$\xi_1$", fontsize=FONTSIZE)
    plt.ylabel(r"$ \frac{v}{\max(|v|)}$", fontsize=FONTSIZE)
    plt.title(fr"Profile of v at $\xi_2$ = {y0}", fontsize=FONTSIZE)
    ax = plt.gca() # use scientific notation for y axis
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(FONTSIZE)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))  # Adjust the number of bins to choose the number of ticks
    plt.legend(fontsize=FONTSIZE)
    #plt.grid(True)
    plt.savefig(f"{EXPERIMENT_DIR}/profiles_v_comparison_y_{y0}_{info_experiment}.png", dpi=300)
    plt.savefig(f"{EXPERIMENT_DIR}/profiles_v_comparison_y_{y0}_{info_experiment}.pdf", dpi=300)
    print(f"===================- {EXPERIMENT_DIR} -=================")
