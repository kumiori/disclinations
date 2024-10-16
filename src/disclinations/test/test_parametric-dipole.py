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
                                 table_timing_data, write_to_output, 
                                 homogeneous_dirichlet_bc_H20,
                                 initialise_exact_solution_dipole)
from disclinations.utils.la import compute_disclination_loads
from dolfinx import fem
from dolfinx.common import list_timings
from dolfinx.fem import Constant, dirichletbc, locate_dofs_topological
from dolfinx.io import XDMFFile, gmshio
from mpi4py import MPI
from petsc4py import PETSc
import pandas as pd
from disclinations.utils import _logger

comm = MPI.COMM_WORLD
from ufl import CellDiameter, FacetNormal, dx

# models = ["variational", "brenner", "carstensen"]
outdir = "output"

AIRY = 0
TRANSVERSE = 1

from disclinations.models import create_disclinations
from disclinations.utils import (homogeneous_dirichlet_bc_H20, load_parameters,
                                 save_params_to_yaml)
from disclinations.utils import update_parameters, save_parameters

from disclinations.utils import create_or_load_circle_mesh
import importlib.resources as pkg_resources  # Python 3.7+ for accessing package files
import copy

def run_experiment(mesh, parameters, series, variant = "variational"):
    print("Running experiment of the series", series, " with thickness:", parameters["model"]["thickness"])
    
    parameters = calculate_rescaling_factors(parameters)

    prefix = os.path.join(outdir, f"plate_fvk_parametric_dipole-{series}")
    if comm.rank == 0:
        Path(prefix).mkdir(parents=True, exist_ok=True)

    # Function spaces
    params = parameters
    
    X = basix.ufl.element("P", str(mesh.ufl_cell()), params["model"]["order"])
    Q = dolfinx.fem.functionspace(mesh, basix.ufl.mixed_element([X, X]))

    V_P1 = dolfinx.fem.functionspace(
        mesh, ("CG", 1)
    )

    DG_e = basix.ufl.element("DG", str(mesh.ufl_cell()), params["model"]["order"] - 2)
    DG = dolfinx.fem.functionspace(mesh, DG_e)

    T_e = basix.ufl.element("P", str(mesh.ufl_cell()), params["model"]["order"] - 2)
    T = dolfinx.fem.functionspace(mesh, T_e)
    
    boundary_conditions = homogeneous_dirichlet_bc_H20(mesh, Q)

    # Point sources
    if mesh.comm.rank == 0:
        # point = np.array([[0.68, 0.36, 0]], dtype=mesh.geometry.x.dtype)
        points = [np.array([[-0.2, 0.0, 0]], dtype=mesh.geometry.x.dtype),
                np.array([[0.2, -0.0, 0]], dtype=mesh.geometry.x.dtype)]
        signs = [-1, 1]
    else:
        # point = np.zeros((0, 3), dtype=mesh.geometry.x.dtype)
        points = [np.zeros((0, 3), dtype=mesh.geometry.x.dtype),
                np.zeros((0, 3), dtype=mesh.geometry.x.dtype)]

    disclinations, params = create_disclinations(
        mesh, params, points=points, signs=signs
    )

    q = dolfinx.fem.Function(Q)
    v, w = ufl.split(q)
    state = {"v": v, "w": w}
    _W_ext = W_ext = Constant(mesh, np.array(0.0, dtype=PETSc.ScalarType)) * w * dx
    # Define the variational problem

    Q_v, Q_v_to_Q_dofs = Q.sub(AIRY).collapse()
    b = compute_disclination_loads(
        disclinations,
        params["loading"]["signs"],
        Q,
        V_sub_to_V_dofs=Q_v_to_Q_dofs,
        V_sub=Q_v,
    )
    test_v, test_w = ufl.TestFunctions(Q)[AIRY], ufl.TestFunctions(Q)[TRANSVERSE]

    # 6. Define variational form (depends on the model)
    if variant == "variational":
        model = NonlinearPlateFVK(mesh, params["model"])
        energy = model.energy(state)[0]
        # Dead load (transverse)
        model.W_ext = _W_ext
        penalisation = model.penalisation(state)

        L = energy - model.W_ext + penalisation
        F = ufl.derivative(L, q, ufl.TestFunction(Q))

    elif variant == "brenner":
        # F = define_brenner_form(fem, params)
        model = NonlinearPlateFVK_brenner(mesh, params["model"])
        energy = model.energy(state)[0]

        # Dead load (transverse)
        model.W_ext = _W_ext
        penalisation = model.penalisation(state)

        L = energy - model.W_ext + penalisation
        # F = ufl.derivative(L, q, ufl.TestFunction(Q))
        F = ufl.derivative(L, q, ufl.TestFunction(Q)) + model.coupling_term(state, test_v, test_w)

    elif variant == "carstensen":
        model = NonlinearPlateFVK_carstensen(mesh, params["model"])
        energy = model.energy(state)[0]

        # Dead load (transverse)
        model.W_ext = _W_ext
        penalisation = model.penalisation(state)

        L = energy - model.W_ext + penalisation
        F = ufl.derivative(L, q, ufl.TestFunction(Q)) + model.coupling_term(state, test_v, test_w)

    # 7. Set up the solver
    solver = SNESSolver(
        F_form=F,
        u=q,
        bcs=boundary_conditions,
        petsc_options=params["solvers"]["elasticity"]["snes"],
        prefix="plate_fvk_dipole",
        b0=b.vector,
        monitor=monitor,
    )

    solver.solve()
    exact_solution = initialise_exact_solution_dipole(Q, params)
    
    pdb.set_trace()
        
    return np.random.rand(1)

if __name__ == "__main__":
    from disclinations.utils import table_timing_data, Visualisation
    _experimental_data = []
    outdir = "output"
    prefix = os.path.join(outdir, "test_parametric")
    
    with pkg_resources.path('disclinations.test', 'parameters.yml') as f:
        # parameters = yaml.load(f, Loader=yaml.FullLoader)
        parameters, _ = load_parameters(f)
        base_parameters = copy.deepcopy(parameters)
        # Remove thickness from the parameters, to compute the parametric series signature
        if "model" in base_parameters and "thickness" in base_parameters["model"]:
            del base_parameters["model"]["thickness"]
        
        base_signature = hashlib.md5(str(base_parameters).encode('utf-8')).hexdigest()

    series = base_signature[0::6]
    experiment_dir = os.path.join(prefix, series)
    num_runs = 10
    
    if comm.rank == 0:
        Path(experiment_dir).mkdir(parents=True, exist_ok=True)
            
    logging.info(
        f"===================- {experiment_dir} -=================")
    
    mesh, mts, fts = create_or_load_circle_mesh(parameters, prefix=prefix)

    
    with dolfinx.common.Timer(f"~Computation Experiment") as timer:

        for i, thickness in enumerate(np.linspace(0.1, 1, num_runs)):
            if changed := update_parameters(parameters, "thickness", thickness):
               data = run_experiment(mesh, parameters, series)
            else: 
                raise ValueError("Failed to update parameters")
            _experimental_data.append(data)
    
    experimental_data = pd.DataFrame(_experimental_data)
    
    _logger.info(f"Saving experimental data to {experiment_dir}")
    print(experimental_data)