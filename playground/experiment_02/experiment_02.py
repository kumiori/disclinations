#!/usr/bin/env python3

import hashlib
import json
import logging
import os
import sys
import yaml
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import gc

import basix
import ufl
import dolfinx
import dolfinx.mesh
import dolfinx.plot

from dolfinx.common import list_timings
from dolfinx.fem import Constant, dirichletbc
from dolfinx.io import XDMFFile, gmshio
from mpi4py import MPI
from ufl import (
    CellDiameter,
    FacetNormal,
    dx,
)

from dolfinx import log
from dolfinx.io import XDMFFile
from mpi4py import MPI
from petsc4py import PETSc

from disclinations.meshes.primitives import mesh_circle_gmshapi
from disclinations.models import (
    NonlinearPlateFVK,
    NonlinearPlateFVK_brenner,
    NonlinearPlateFVK_carstensen,
    calculate_rescaling_factors,
    compute_energy_terms,
    _transverse_load_exact_solution,
)
from disclinations.solvers import SNESSolver
from disclinations.utils.la import compute_disclination_loads
from disclinations.utils.viz import plot_scalar, plot_profile, plot_mesh
from disclinations.utils import update_parameters, memory_usage, save_parameters,homogeneous_dirichlet_bc_H20
from disclinations.utils import _logger, snes_solver_stats,save_params_to_yaml,create_or_load_circle_mesh
from disclinations.models import create_disclinations
from disclinations.models import assemble_penalisation_terms
from disclinations.utils.la import compute_norms

logging.basicConfig(level=logging.INFO)
comm = MPI.COMM_WORLD
AIRY = 0
TRANSVERSE = 1

def postprocess(state, model, mesh, params, exact_solution, prefix):
    with dolfinx.common.Timer(f"~Postprocessing and Vis") as timer:

        fem_energies = {}
        exact_energies = {}

        _logger.info("\nEnergy Analysis:")

        if exact_solution is not None:
            _v_exact, _w_exact = exact_solution
        else:
            _v_exact, _w_exact = None, None

        for i, energy_name in enumerate(
            ["total", "bending", "membrane", "coupling", "external_work"]
        ):
            if exact_solution is not None:
                exact_energy = dolfinx.fem.assemble_scalar(
                    dolfinx.fem.form(model.energy(exact_solution)[i])
                )
                _exact_energy = mesh.comm.allreduce(exact_energy, op=MPI.SUM)
                exact_energies[energy_name] = _exact_energy

            if energy_name != "external_work":
                fem_energy = dolfinx.fem.assemble_scalar(
                    dolfinx.fem.form(model.energy(state)[i])
                )
            else:
                fem_energy = dolfinx.fem.assemble_scalar(dolfinx.fem.form(model.W_ext))

            _fem_energy = mesh.comm.allreduce(fem_energy, op=MPI.SUM)
            fem_energies[energy_name] = _fem_energy

        penalisation_terms = assemble_penalisation_terms(model)
        norms = compute_norms(state["v"], state["w"], mesh)

        extra_fields = [
            {"field": _v_exact, "name": "v_exact"},
            {"field": _w_exact, "name": "w_exact"},
            {
                "field": model.M(state["w"]),  # Tensor expression
                "name": "M",
                "components": "tensor",
            },
            {
                "field": model.P(state["v"]),  # Tensor expression
                "name": "P",
                "components": "tensor",
            },
            {
                "field": model.gaussian_curvature(state["w"]),  # Tensor expression
                "name": "Kappa",
                "components": "tensor",
            },
        ]
        # write_to_output(prefix, q, extra_fields)
        return fem_energies, norms, None, None, penalisation_terms


def run_experiment(mesh: None, parameters: dict, series: str):
    _logger.info(
        f"Running experiment of the series {series} with parameter: {parameters['model']['γ_adim']}"
    )

    parameters = calculate_rescaling_factors(parameters)
    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()

    prefix = os.path.join(experiment_dir, signature)
    if comm.rank == 0:
        Path(prefix).mkdir(parents=True, exist_ok=True)

    # Function spaces
    X = basix.ufl.element("P", str(mesh.ufl_cell()), parameters["model"]["order"])
    Q = dolfinx.fem.functionspace(mesh, basix.ufl.mixed_element([X, X]))

    V_P1 = dolfinx.fem.functionspace(mesh, ("CG", 1))

    DG_e = basix.ufl.element(
        "DG", str(mesh.ufl_cell()), parameters["model"]["order"] - 2
    )
    DG = dolfinx.fem.functionspace(mesh, DG_e)

    T_e = basix.ufl.element("P", str(mesh.ufl_cell()), parameters["model"]["order"] - 2)
    T = dolfinx.fem.functionspace(mesh, T_e)

    boundary_conditions = homogeneous_dirichlet_bc_H20(mesh, Q)

    # Loading
    # Disclinations
    # if mesh.comm.rank == 0:
    #     # point = np.array([[0.68, 0.36, 0]], dtype=mesh.geometry.x.dtype)
    #     points = [np.array([[-0.2, 0.0, 0]], dtype=mesh.geometry.x.dtype),
    #             np.array([[0.2, 0.0, 0]], dtype=mesh.geometry.x.dtype),
    #             np.array([[0.0, 0.2, 0]], dtype=mesh.geometry.x.dtype),
    #             np.array([[0.0, -.2, 0]], dtype=mesh.geometry.x.dtype),
    #             ]
    #     signs = [1, 1, 1, 1]
    # else:
    #     # point = np.zeros((0, 3), dtype=mesh.geometry.x.dtype)
    #     points = [np.zeros((0, 3), dtype=mesh.geometry.x.dtype),
    #             np.zeros((0, 3), dtype=mesh.geometry.x.dtype)]
    #     signs = np.zeros_like(points)
        
    q = dolfinx.fem.Function(Q)
    v, w = ufl.split(q)
    state = {"v": v, "w": w}

    Q_v, Q_v_to_Q_dofs = Q.sub(AIRY).collapse()

    disclinations, parameters = create_disclinations(
        mesh, parameters
    )
    
    b = compute_disclination_loads(
        disclinations,
        parameters["loading"]["signs"],
        Q,
        V_sub_to_V_dofs=Q_v_to_Q_dofs,
        V_sub=Q_v,
    )
    b.vector.scale(parameters["model"]["β_adim"] ** 2.0)
    
    W_ext = parameters["model"]["γ_adim"] * dolfinx.fem.Constant(mesh, -1.) * w * dx

    model = NonlinearPlateFVK(mesh, parameters["model"], adimensional=True)
    energy = model.energy(state)[0]
    # Dead load (transverse)
    model.W_ext = W_ext
    penalisation = model.penalisation(state)


    # Define the functional
    L = energy - W_ext + penalisation
    F = ufl.derivative(L, q, ufl.TestFunction(Q))
    save_params_to_yaml(parameters, os.path.join(prefix, "parameters.yml"))
    
    solver = SNESSolver(
        F_form=F,
        u=q,
        bcs=boundary_conditions,
        bounds=None,
        petsc_options=parameters["solvers"]["nonlinear"],
        prefix='plate_fvk',
        b0=b.vector,
    )
    solver.solve()
    
    solver_stats = snes_solver_stats(solver.solver)
    del solver
    
    
    # Postprocessing and viz
    with dolfinx.common.Timer(f"~Postprocessing and Vis") as timer:
        energies, norms, abs_error, rel_error, penalisation = postprocess(
            state, model, mesh, params=parameters, exact_solution=None, prefix=prefix
        )
        import pyvista
        from pyvista.plotting.utilities import xvfb

        xvfb.start_xvfb(wait=0.05)
        pyvista.OFF_SCREEN = True


        κ = model.gaussian_curvature(w, order = parameters["model"]["order"]-2)
        # heatmap of bracket(w, w) = det Hessian = k_1 * k_2 = kappa (gaussian curvature)
        
        plotter = pyvista.Plotter(
                title="Curvature",
                window_size=[600, 600],
                shape=(1, 1),
            )
        scalar_plot = plot_scalar(κ, plotter, subplot=(0, 0))
        plotter.add_title("Gaussian curvature", font_size=12)
        plotter.remove_scalar_bar()

        scalar_bar = plotter.add_scalar_bar("k", title_font_size=18, label_font_size=14)

        scalar_plot.screenshot(f"{prefix}/gaussian_curvature.png")
        print("plotted curvature")

    return energies, norms, abs_error, rel_error, penalisation, solver_stats

def load_parameters(file_path):
    """
    Load parameters from a YAML file.

    Args:
        file_path (str): Path to the YAML parameter file.

    Returns:
        dict: Loaded parameters.
    """
    import hashlib

    with open(file_path) as f:
        parameters = yaml.load(f, Loader=yaml.FullLoader)

    parameters["geometry"]["radius"] = 1
    parameters["geometry"]["geom_type"] = "circle"
    parameters["solvers"]["nonlinear"] = {
        "snes_type": "newtonls",      # Solver type: NGMRES (Nonlinear GMRES)
        "snes_max_it": 100,           # Maximum number of iterations
        "snes_rtol": 1e-6,            # Relative tolerance for convergence
        "snes_atol": 1e-15,           # Absolute tolerance for convergence
        "snes_stol": 1e-5,           # Tolerance for the change in solution norm
        "snes_monitor": None,         # Function for monitoring convergence (optional)
        "snes_linesearch_type": "basic",  # Type of line search
    }
    parameters["model"]["γ_adim"] = 1.0
    parameters["model"]["β_adim"] = 1.

    parameters["loading"] = {
        "points": [np.array([[-0.2, 0.0, 0]]),
                np.array([[0.2, 0.0, 0]]),
                np.array([[0.0, 0.2, 0]]),
                np.array([[0.0, -.2, 0]])],
        "signs": [1, 1, 1, 1]
                }
    
    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()

    return parameters, signature

import importlib.resources as pkg_resources  # Python 3.7+ for accessing package files
import copy
if __name__ == "__main__":
    from disclinations.utils import table_timing_data, Visualisation
    _experimental_data = []
    outdir = "output"
    prefix = outdir
    
    with pkg_resources.path("disclinations.test", "parameters.yml") as f:
        parameters, _ = load_parameters(f)
        base_parameters = copy.deepcopy(parameters)

        base_signature = hashlib.md5(str(base_parameters).encode("utf-8")).hexdigest()

    series = base_signature[0::6]
    experiment_dir = os.path.join(prefix, series)

    max_memory = 0
    num_runs = 10
    
    if comm.rank == 0:
        Path(experiment_dir).mkdir(parents=True, exist_ok=True)

    _logger.info(f"Running series {series} with {num_runs} runs")
    _logger.info(f"===================- {experiment_dir} -=================")
    # postprocess = Visualisation(experiment_dir)

    with dolfinx.common.Timer(f"~Computation Experiment") as timer:

        # Mesh is fixed for all runs
        mesh = None
        mesh_size = parameters["geometry"]["mesh_size"]
        tdim = 2
        model_rank = 0
        mesh, mts, fts = create_or_load_circle_mesh(parameters, prefix=prefix)
        import matplotlib.pyplot as plt

        plt.figure()
        ax = plot_mesh(mesh)
        fig = ax.get_figure()
        
        for point in parameters["loading"]["points"]:
            ax.plot(point[0][0], point[0][1], 'ro')

        fig.savefig(f"{experiment_dir}/mesh.png")
            
        for i, gamma in enumerate(np.linspace(0.1, 100, num_runs)):
            # Check memory usage before computation
            mem_before = memory_usage()
            _logger.critical(f"===================- γ_adim = {gamma} -=================")
            
            # parameters["model"]["thickness"] = thickness

            if changed := update_parameters(parameters, "γ_adim", float(gamma)):
                signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()
            else:
                raise ValueError("Failed to update parameters")
            
            energies, norms, _, _, penalisation, solver_stats = run_experiment(mesh, parameters, series)
            
            run_data = {
                "signature": signature,
                "param_value": gamma,
                **energies,
                **norms,
                # "abs_error_membrane": abs_errors[0],
                # "abs_error_bending": abs_errors[1],
                # "abs_error_coupling": abs_errors[2],
                # "rel_error_membrane": rel_errors[0],
                # "rel_error_bending": rel_errors[1],
                # "rel_error_coupling": rel_errors[2],
                **penalisation,  # Unpack penalization terms into individual columns
                **solver_stats,  # Unpack solver statistics into individual columns
            }
            _experimental_data.append(run_data)
            
            # Check memory usage after computation
            mem_after = memory_usage()
            max_memory = max(max_memory, mem_after)
            
            # Log memory usage
            logging.info(f"Run {i}/{num_runs}: Memory Usage (MB) - Before: {mem_before}, After: {mem_after}")
            # Perform garbage collection
            gc.collect()
    
    experimental_data = pd.DataFrame(_experimental_data)
    print(experimental_data)
    
    with dolfinx.common.Timer(f"~Postprocessing and Vis") as timer:

        if mesh.comm.rank == 0:
            experimental_data.to_pickle(f"{experiment_dir}/experimental_data.pkl")

            with open(f"{experiment_dir}/experimental_data.json", "w") as file:
                json.dump(_experimental_data, file)
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(
            experimental_data["param_value"],
            experimental_data["bending"],
            label="Bending Energy",
            marker="o",
        )
        plt.plot(
            experimental_data["param_value"],
            experimental_data["membrane"],
            label="Membrane Energy",
            marker="o",
        )
        plt.plot(
            experimental_data["param_value"],
            experimental_data["coupling"],
            label="Coupling Energy",
            marker="o",
        )
        plt.plot(
            experimental_data["param_value"],
            experimental_data["external_work"],
            label="External Work",
            marker="o",
        )

        # Customize the plot
        plt.title(
            f'Energy Terms vs Parameter sign {parameters["loading"]["signs"]} penalisation {parameters["model"]["alpha_penalty"]}, '
        )
        plt.xlabel("a")
        plt.ylabel("Energy")
        plt.yscale("log")  # Using log scale for better visibility
        plt.xscale("log")  # Using log scale for better visibility
        plt.legend()

        plt.savefig(f"{experiment_dir}/energy_terms.png")

        plt.figure(figsize=(10, 6))
        plt.plot(
            experimental_data["param_value"],
            experimental_data["snes_final_residual_norm"],
            label="Residual norm",
            marker="o",
        )
        plt.savefig(f"{experiment_dir}/residuals.png")

    # import matplotlib.pyplot as plt
    
    # # Plot energy terms versus thickness
    # plt.figure(figsize=(10, 6))
    # plt.plot(experimental_data["thickness"], experimental_data["membrane_energy"], label="Membrane Energy")
    # plt.plot(experimental_data["thickness"], experimental_data["bending_energy"], label="Bending Energy")
    # plt.plot(experimental_data["thickness"], experimental_data["coupling_energy"], label="Coupling Energy")
    # plt.xlabel("Thickness")
    # plt.ylabel("Energy")
    # plt.title("Energy Terms vs Thickness")
    # plt.legend()
    # plt.savefig(f"{experiment_dir}/energy_terms.png")

    # # Plot L2 norm terms versus thickness
    # plt.figure(figsize=(10, 6))
    # plt.plot(experimental_data["thickness"], experimental_data["w_L2"], label=r"$w_{L^2}$")
    # plt.plot(experimental_data["thickness"], experimental_data["v_L2"], label=r"$v_{L^2}$")
    # plt.xlabel("Thickness")
    # plt.ylabel("L2 Norms")
    # plt.title("L2 Norm Terms vs Thickness")
    # plt.legend()

    # plt.savefig(f"{experiment_dir}/norms_fields.png")


    timings = table_timing_data()
    print(timings)
    list_timings(MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])
    








    # list_timings(MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])
    # postprocess.visualise_results(experimental_data)
    # postprocess.save_table(timings, "timing_data")
    # postprocess.save_table(experimental_data, "postprocessing_data")
    
    