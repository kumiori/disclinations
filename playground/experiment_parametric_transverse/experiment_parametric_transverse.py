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
    _transverse_load_exact_solution,
    calculate_rescaling_factors,
)
from disclinations.solvers import SNESSolver
from disclinations.utils.la import compute_norms
from disclinations.utils.viz import plot_scalar, plot_profile, plot_mesh
from disclinations.utils import (
    update_parameters,
    save_params_to_yaml,
    snes_solver_stats,
    memory_usage,
    create_or_load_circle_mesh,
    homogeneous_dirichlet_bc_H20,
    _logger,
)
from disclinations.models import (
    assemble_penalisation_terms,
    initialise_exact_solution_compatible_transverse,
)

logging.basicConfig(level=logging.INFO)
comm = MPI.COMM_WORLD
AIRY = 0
TRANSVERSE = 1


def postprocess(state, q, model, mesh, params, exact_solution, prefix):
    with dolfinx.common.Timer(f"~Postprocessing and Vis") as timer:
        # Compute energies for the exact solution
        exact_energies = {}
        fem_energies = {}
        abs_errors = {}
        rel_errors = {}
        # the exact solution is adimensional, to perform energy comparison

        _logger.info("\nEnergy Analysis:")

        for i, energy_name in enumerate(["total", "bending", "membrane", "coupling"]):
            exact_energy = dolfinx.fem.assemble_scalar(
                dolfinx.fem.form(model.energy(exact_solution)[i])
            )
            _exact_energy = mesh.comm.allreduce(exact_energy, op=MPI.SUM)
            exact_energies[energy_name] = _exact_energy

            fem_energy = dolfinx.fem.assemble_scalar(
                dolfinx.fem.form(model.energy(state)[i])
            )
            _fem_energy = mesh.comm.allreduce(fem_energy, op=MPI.SUM)
            fem_energies[energy_name] = _fem_energy

            # Compute absolute and relative errors
            abs_error = abs(fem_energies[energy_name] - exact_energies[energy_name])
            rel_error = (
                abs_error / abs(exact_energies[energy_name])
                if exact_energies[energy_name] != 0
                else float("inf")
            )
            abs_errors[energy_name] = abs_error
            rel_errors[energy_name] = rel_error

            # Log detailed energy information
            _logger.info(f"{energy_name.capitalize()} Energy Analysis:")
            _logger.info(f"  Exact Energy: {_exact_energy:.2e}")
            _logger.info(f"  FEM Energy: {_fem_energy:.2e}")
            _logger.info(f"  Absolute Error: {abs_error:.0e}")
            _logger.info(f"  Relative Error: {rel_error:.2%}\n")

        penalisation_terms = assemble_penalisation_terms(model)
        norms = compute_norms(state["v"], state["w"], mesh)

        _v_exact, _w_exact = exact_solution["v"], exact_solution["w"]

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
        ]

        # write_to_output(prefix, q, extra_fields)

        # Convert errors to numpy arrays for easy handling
        abs_error_array = np.array(list(abs_errors.values()))
        rel_error_array = np.array(list(rel_errors.values()))

        # Optionally, return the computed values for further use
    return fem_energies, norms, abs_error_array, rel_error_array, penalisation_terms


def run_experiment(variant, mesh, parameters, experiment_folder):
    # Setup, output and file handling
    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()[0:6]

    outdir = "output"
    prefix = os.path.join(experiment_dir, signature)
    if comm.rank == 0:
        Path(prefix).mkdir(parents=True, exist_ok=True)

    parameters = calculate_rescaling_factors(parameters)

    # Parameters

    X = basix.ufl.element("P", str(mesh.ufl_cell()), parameters["model"]["order"])
    Q = dolfinx.fem.functionspace(mesh, basix.ufl.mixed_element([X, X]))

    q = dolfinx.fem.Function(Q)
    f = dolfinx.fem.Function(Q.sub(TRANSVERSE).collapse()[0])

    v, w = ufl.split(q)
    state = {"v": v, "w": w}

    # Loading

    # Transverse load (analytical solution)
    transverse_load = lambda x: _transverse_load_exact_solution(
        x, parameters, adimensional=True
    )
    exact_solution = initialise_exact_solution_compatible_transverse(
        Q, parameters, adimensional=True
    )

    f.interpolate(transverse_load)
    boundary_conditions = homogeneous_dirichlet_bc_H20(mesh, Q)

    c_nu = 1 / (12 * (1 - parameters["model"]["nu"] ** 2))

    f_scale = (
        np.sqrt(2 * c_nu**3)
        * parameters["model"]["E"]
        * parameters["model"]["thickness"] ** 4
        / parameters["geometry"]["radius"] ** 4
    )
    # W_ext = parameters["model"]["β_adim"]**4 * parameters["model"]["γ_adim"] * f * w * dx
    # W_ext = parameters["model"]["β_adim"]**4 * Constant(mesh, np.array(-1.0, dtype=PETSc.ScalarType)) * w * dx

    # W_ext = (
    #     parameters["model"]["β_adim"] ** 4
    #     * parameters["model"]["γ_adim"]
    #     * f_scale
    #     * f
    #     * w
    #     * dx
    # )

    W_ext = np.sqrt(2 * c_nu**3) * parameters["model"]["γ_adim"] * f * w * dx

    model = NonlinearPlateFVK(mesh, parameters["model"], adimensional=True)
    energy = model.energy(state)[0]
    penalisation = model.penalisation(state)

    # Define the functional
    L = energy - W_ext + penalisation
    F = ufl.derivative(L, q, ufl.TestFunction(Q))

    solver = SNESSolver(
        F_form=F,
        u=q,
        bcs=boundary_conditions,
        bounds=None,
        petsc_options=parameters["solvers"]["nonlinear"],
        prefix="plate_fvk",
    )
    solver.solve()
    solver_stats = snes_solver_stats(solver.solver)

    del solver

    energies, norms, abs_error, rel_error, penalisation = postprocess(
        state,
        q,
        model,
        mesh,
        params=parameters,
        exact_solution=exact_solution,
        prefix=prefix,
    )

    # Postprocessing and viz
    with dolfinx.common.Timer(f"~Postprocessing and Vis") as timer:
        import pyvista
        from pyvista.plotting.utilities import xvfb

        xvfb.start_xvfb(wait=0.05)
        pyvista.OFF_SCREEN = True

        # import matplotlib.pyplot as plt

        # plt.figure()
        # ax = plot_mesh(mesh)
        # fig = ax.get_figure()
        # fig.savefig(f"{prefix}/mesh.png")

        κ = model.gaussian_curvature(w, order=parameters["model"]["order"] - 2)
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
    # parameters["geometry"]["mesh_size"] = 0.03
    parameters["geometry"]["mesh_size"] = 0.02
    # parameters["solvers"]["elasticity"] = {
    #     "snes_type": "newtonls",      # Solver type: NGMRES (Nonlinear GMRES)
    #     "snes_max_it": 100,           # Maximum number of iterations
    #     "snes_rtol": 1e-6,            # Relative tolerance for convergence
    #     "snes_atol": 1e-10,           # Absolute tolerance for convergence
    #     "snes_stol": 1e-5,           # Tolerance for the change in solution norm
    #     "snes_monitor": None,         # Function for monitoring convergence (optional)
    #     "snes_linesearch_type": "basic",  # Type of line search
    # }
    parameters["model"]["β_adim"] = np.nan
    parameters["model"]["γ_adim"] = 1

    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()
    print(yaml.dump(parameters, default_flow_style=False))

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
    num_runs = 2
    if comm.rank == 0:
        Path(experiment_dir).mkdir(parents=True, exist_ok=True)

    save_params_to_yaml(
        base_parameters, os.path.join(experiment_dir, "parameters.yaml")
    )

    _logger.info(f"Running series {series} with {num_runs} runs")
    _logger.info(f"===================- {experiment_dir} -=================")

    with dolfinx.common.Timer(f"~Computation Experiment") as timer:
        # Mesh is fixed for all runs
        mesh = None
        mesh_size = parameters["geometry"]["mesh_size"]
        tdim = 2
        model_rank = 0

        mesh, mts, fts = create_or_load_circle_mesh(parameters, prefix=prefix)

        for i, a in enumerate(np.logspace(np.log10(10), np.log10(100), num=num_runs)):
            # Check memory usage before computation
            mem_before = memory_usage()
            _logger.critical(f"===================- β_adim = {a} -=================")

            if changed := update_parameters(parameters, "β_adim", float(a)):
                parameters["model"]["thickness"] = parameters["geometry"]["radius"] / a
                signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()
            else:
                raise ValueError("Failed to update parameters")

            energies, norms, abs_errors, rel_errors, penalisation, solver_stats = (
                run_experiment("variational", mesh, parameters, prefix)
            )
            run_data = {
                "signature": signature,
                # "variant":
                "param_value": a,
                **energies,
                **norms,
                "abs_error_membrane": abs_errors[0],
                "abs_error_bending": abs_errors[1],
                "abs_error_coupling": abs_errors[2],
                "rel_error_membrane": rel_errors[0],
                "rel_error_bending": rel_errors[1],
                "rel_error_coupling": rel_errors[2],
                **penalisation,  # Unpack penalization terms into individual columns
                **solver_stats,  # Unpack solver statistics into individual columns
            }
            _experimental_data.append(run_data)

            # Check memory usage after computation
            mem_after = memory_usage()
            max_memory = max(max_memory, mem_after)

            # Log memory usage
            logging.info(
                f"Run {i}/{num_runs}: Memory Usage (MB) - Before: {mem_before}, After: {mem_after}"
            )
            # Perform garbage collection
            gc.collect()

    experimental_data = pd.DataFrame(_experimental_data)

    with dolfinx.common.Timer(f"~Postprocessing and Vis") as timer:
        if mesh.comm.rank == 0:
            experimental_data.to_pickle(f"{experiment_dir}/experimental_data.pkl")

            with open(f"{experiment_dir}/experimental_data.json", "w") as file:
                json.dump(_experimental_data, file)

    print(experimental_data)
    import matplotlib.pyplot as plt

    # Plot energy terms versus thickness
    plt.figure(figsize=(10, 6))
    plt.plot(
        experimental_data["param_value"],
        experimental_data["membrane"],
        label="Membrane Energy",
    )
    plt.plot(
        experimental_data["param_value"],
        experimental_data["bending"],
        label="Bending Energy",
    )
    plt.plot(
        experimental_data["param_value"],
        experimental_data["coupling"],
        label="Coupling Energy",
    )
    plt.xlabel("param_value")
    plt.ylabel("Energy")
    plt.title("Energy Terms vs param_value")
    plt.legend()
    plt.savefig(f"{experiment_dir}/energy_terms.png")

    # Plot L2 norm terms versus thickness
    plt.figure(figsize=(10, 6))
    plt.plot(
        experimental_data["param_value"], experimental_data["w_L2"], label=r"$w_{L^2}$"
    )
    plt.plot(
        experimental_data["param_value"], experimental_data["v_L2"], label=r"$v_{L^2}$"
    )
    plt.xlabel("param_value")
    plt.ylabel("L2 Norms")
    plt.title("L2 Norm Terms vs param_value")
    plt.legend()

    plt.savefig(f"{experiment_dir}/norms_fields.png")

    timings = table_timing_data()

    list_timings(MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])
    # postprocess.visualise_results(experimental_data)
    # postprocess.save_table(timings, "timing_data")
    # postprocess.save_table(experimental_data, "postprocessing_data")
