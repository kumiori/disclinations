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
    create_disclinations,
    _transverse_load_exact_solution,
)

from disclinations.solvers import SNESSolver
from disclinations.utils.la import compute_disclination_loads
from disclinations.utils.viz import plot_scalar, plot_profile, plot_mesh
from disclinations.utils import (
    update_parameters,
    memory_usage,
    save_parameters,
    save_params_to_yaml,
)
from disclinations.utils import (
    create_or_load_circle_mesh,
    basic_postprocess,
    homogeneous_dirichlet_bc_H20,
)
import importlib.resources as pkg_resources  # Python 3.7+ for accessing package files
import copy
from disclinations.utils import _logger, snes_solver_stats
import pytest

from disclinations.models import (
    assemble_penalisation_terms,
    initialise_exact_solution_monopole,
)
from disclinations.utils.la import compute_norms

models = ["variational", "brenner", "carstensen"]

logging.basicConfig(level=logging.INFO)
comm = MPI.COMM_WORLD


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
            _logger.info(f"  Exact Energy: {_exact_energy:.5e}")
            _logger.info(f"  FEM Energy: {_fem_energy:.5e}")
            _logger.info(f"  Absolute Error: {abs_error:.0e}")
            _logger.info(f"  Relative Error: {rel_error:.2%}\n")

        penalization_terms = assemble_penalisation_terms(model)
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
        return fem_energies, norms, abs_error_array, rel_error_array, penalization_terms


def postprocess_and_visualize(model, state, exact_solution, data, prefix, parameters):
    with dolfinx.common.Timer(f"~Postprocessing and Vis") as timer:
        import pyvista
        from pyvista.plotting.utilities import xvfb

        xvfb.start_xvfb(wait=0.05)
        pyvista.OFF_SCREEN = True
        w = state["w"]

        κ = model.gaussian_curvature(w, order=parameters["model"]["order"] - 2)

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


# def run_experiment(mesh: None, parameters: dict, series: str):
@pytest.mark.parametrize("variant", models)
def run_experiment(
    mesh, parameters, experiment_dir, variant="variational", initial_guess=None
):
    # Setup, output and file handling
    _logger.info(
        f"Running experiment of the series {series} with parameter: {parameters['model']['β_adim']}"
    )
    parameters = calculate_rescaling_factors(parameters)

    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()[0:6]
    data = {
        "membrane_energy": [],
        "bending_energy": [],
        "coupling_energy": [],
        "w_L2": [],
        "v_L2": [],
        "thickness": [],
    }

    outdir = "output"
    prefix = os.path.join(outdir, series, signature)

    if comm.rank == 0:
        Path(prefix).mkdir(parents=True, exist_ok=True)

    # MPI communicator and global variables

    AIRY = 0
    TRANSVERSE = 1

    # Parameters

    mesh_size = parameters["geometry"]["mesh_size"]
    nu = parameters["model"]["nu"]
    thickness = parameters["model"]["thickness"]
    radius = parameters["geometry"]["radius"]
    E = parameters["model"]["E"]

    _D = E * thickness**3 / (12 * (1 - nu**2))
    w_scale = np.sqrt(2 * _D / (E * thickness))
    v_scale = _D
    f_scale = np.sqrt(2 * _D**3 / (E * thickness))
    f0 = (
        parameters["model"]["E"]
        * (parameters["model"]["thickness"] / parameters["geometry"]["radius"]) ** 4
    )
    b_scale = parameters["model"]["β_adim"] ** 2

    # Mesh
    model_rank = 0
    tdim = 2
    # order = 3
    if mesh is None:
        with dolfinx.common.Timer("~Mesh Generation") as timer:
            gmsh_model, tdim = mesh_circle_gmshapi(
                parameters["geometry"]["geom_type"], 1, mesh_size, tdim
            )
            mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, model_rank, tdim)

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)

    # Functional setting

    X = basix.ufl.element("P", str(mesh.ufl_cell()), parameters["model"]["order"])
    Q = dolfinx.fem.functionspace(mesh, basix.ufl.mixed_element([X, X]))

    q = dolfinx.fem.Function(Q)
    q_exact = dolfinx.fem.Function(Q)
    f = dolfinx.fem.Function(Q.sub(TRANSVERSE).collapse()[0])

    v, w = ufl.split(q)
    v_exact, w_exact = q_exact.split()
    state = {"v": v, "w": w}

    # Loading
    # Point sources
    # if mesh.comm.rank == 0:
    #     points = [
    #         np.array([[-0.3, 0.0, 0]], dtype=mesh.geometry.x.dtype),
    #         np.array([[0.3, 0.0, 0]], dtype=mesh.geometry.x.dtype),
    #     ]
    #     disclination_power_list = [-1, 1]
    # else:
    #     points = [
    #         np.zeros((0, 3), dtype=mesh.geometry.x.dtype),
    #         np.zeros((0, 3), dtype=mesh.geometry.x.dtype),
    #     ]
    #     disclination_power_list = [0, 0]

    Q_v, Q_v_to_Q_dofs = Q.sub(AIRY).collapse()
    # disclinations, parameters = create_disclinations(
    #     mesh, parameters, points=points, signs=disclination_power_list
    # )
    # b = compute_disclination_loads(
    #     disclinations,
    #     disclination_power_list,
    #     Q,
    #     V_sub_to_V_dofs=Q_v_to_Q_dofs,
    #     V_sub=Q_v,
    # )
    disclinations, parameters = create_disclinations(mesh, parameters)
    b = compute_disclination_loads(
        disclinations,
        parameters["loading"]["signs"],
        Q,
        V_sub_to_V_dofs=Q_v_to_Q_dofs,
        V_sub=Q_v,
    )

    b.vector.scale(parameters["model"]["β_adim"] ** 2.0)

    # Transverse load (analytical solution)
    _transverse_load = lambda x: _transverse_load_exact_solution(
        x, parameters, adimensional=False
    )

    exact_solution = initialise_exact_solution_monopole(Q, parameters, adimensional=True)

    f.interpolate(_transverse_load)

    boundary_conditions = homogeneous_dirichlet_bc_H20(mesh, Q)

    W_ext = (
        dolfinx.fem.Constant(mesh, np.array(0.0, dtype=PETSc.ScalarType))
        * parameters["model"]["γ_adim"]
        * parameters["model"]["β_adim"] ** 4
        * f
        * w
        * dx
    )

    model = NonlinearPlateFVK(mesh, parameters["model"], adimensional=True)
    energy = model.energy(state)[0]
    model.W_ext = W_ext
    penalisation = model.penalisation(state)

    # Define the functional
    L = energy - W_ext + penalisation
    F = ufl.derivative(L, q, ufl.TestFunction(Q))

    save_params_to_yaml(parameters, os.path.join(prefix, "parameters.yaml"))
    if MPI.COMM_WORLD.rank == 0:
        with open(f"{prefix}/signature.md5", "w") as f:
            f.write(signature)

    solver = SNESSolver(
        F_form=F,
        u=q,
        bcs=boundary_conditions,
        bounds=None,
        petsc_options=parameters["solvers"]["nonlinear"],
        prefix="plate_fvk",
        b0=b.vector,
    )
    solver.solve()

    solver_stats = snes_solver_stats(solver.solver)

    del solver

    # Postprocessing and viz
    energies, norms, abs_error, rel_error, penalisation = postprocess(
        state,
        q,
        model,
        mesh,
        params=parameters,
        exact_solution=exact_solution,
        prefix=prefix,
    )
    V_P1 = dolfinx.fem.functionspace(mesh, ("CG", 1))
    V_P1 = dolfinx.fem.functionspace(
        mesh, ("CG", 1)
    )  # "CG" stands for continuous Galerkin (Lagrange)

    with XDMFFile(
        comm,
        os.path.join(outdir, series, "fields-vs-thickness.xdmf"),
        "a",
        encoding=XDMFFile.Encoding.HDF5,
    ) as file:
        _v, _w = q.split()
        _v.name = "potential"
        _w.name = "displacement"

        interpolation = dolfinx.fem.Function(V_P1)
        _logger.info(f"saving interpolation, {parameters['model']['β_adim']}")

        interpolation.interpolate(_v)
        interpolation.name = "v"

        file.write_function(
            interpolation, parameters["model"]["β_adim"]
        )  # Specify unique mesh_xpath for velocity

        interpolation.interpolate(_w)
        interpolation.name = "w"
        file.write_function(
            interpolation, parameters["model"]["β_adim"]
        )  # Specify unique mesh_xpath for velocity

    postprocess_and_visualize(model, state, exact_solution, data, prefix, parameters)

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
    parameters["model"]["higher_regularity"] = False

    parameters["loading"] = {
        "points": [
            np.array([[0.0, 0.0, 0]]).tolist(),
        ],
        "signs": [1],
    }

    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()
    print(yaml.dump(parameters, default_flow_style=False))

    return parameters, signature


if __name__ == "__main__":
    from disclinations.utils import table_timing_data, Visualisation

    _experimental_data = []
    outdir = "output"
    prefix = outdir
    _simulation_data = []

    with pkg_resources.path("disclinations.test", "parameters.yml") as f:
        # parameters = yaml.load(f, Loader=yaml.FullLoader)
        parameters, _ = load_parameters(f)

        base_parameters = copy.deepcopy(parameters)
        # Remove thickness from the parameters, to compute the parametric series signature
        if "model" in base_parameters and "thickness" in base_parameters["model"]:
            del base_parameters["model"]["thickness"]

        base_signature = hashlib.md5(str(base_parameters).encode("utf-8")).hexdigest()

    series = base_signature[0::6]

    experiment_dir = os.path.join(outdir, series)
    num_runs = 11

    if comm.rank == 0:
        Path(experiment_dir).mkdir(parents=True, exist_ok=True)

    logging.info(f"===================- {experiment_dir} -=================")
    _logger.info(f"Running {num_runs} experiments for series {series}")
    mesh, mts, fts = create_or_load_circle_mesh(parameters, prefix=prefix)

    with dolfinx.common.Timer("~Computation Experiment") as timer:
        # Mesh is fixed for all runs
        model_rank = 0
        initial_guess = None

        with XDMFFile(
            comm,
            os.path.join(outdir, series, "fields-vs-thickness.xdmf"),
            "w",
            encoding=XDMFFile.Encoding.HDF5,
        ) as file:
            file.write_mesh(mesh)

        for i, a in enumerate(np.logspace(np.log10(10), np.log10(1000), num=15)):
            parameters["model"]["β_adim"] = None
            parameters["model"]["γ_adim"] = 1

            # Check memory usage before computation
            mem_before = memory_usage()

            if changed := update_parameters(parameters, "β_adim", float(a)):
                signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()
                energies, norms, abs_errors, rel_errors, penalisation, solver_stats = (
                    run_experiment(mesh, parameters, experiment_dir, initial_guess=None)
                )
            else:
                abs_error, rel_error = None, None
                raise ValueError("Failed to update parameters")

            run_data = {
                "signature": signature,
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

            # Log memory usage
            _logger.info(
                f"Run {i}/{num_runs}: Memory Usage (MB) - Before: {mem_before}, After: {mem_after}"
            )
            # Perform garbage collection
            gc.collect()

    experimental_data = pd.DataFrame(_experimental_data)
    timings = table_timing_data()
    _logger.info(f"Saving experimental data to {experiment_dir}")

    import matplotlib.pyplot as plt

    __import__("pdb").set_trace()
    # Plot energy terms versus thickness
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        experimental_data["param_value"],
        experimental_data["membrane"],
        label="Membrane Energy",
        marker="o",
    )
    ax2 = ax.twinx()
    ax2.plot(
        experimental_data["param_value"],
        experimental_data["bending"],
        label="Bending Energy",
        marker="o",
    )
    ax2.plot(
        experimental_data["param_value"],
        experimental_data["coupling"],
        label="Coupling Energy",
        marker="o",
    )
    ax.set_xlabel("a")
    ax.set_ylabel("Energy")
    ax.loglog()
    plt.title("Energy Terms vs a")

    plt.legend()

    plt.savefig(f"{experiment_dir}/energy_terms.png")

    # Plot L2 norm terms versus thickness
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        experimental_data["param_value"],
        experimental_data["w_L2"],
        label=r"$w_{L^2}$",
        marker="o",
    )
    ax2 = ax.twinx()
    ax2.plot(
        experimental_data["param_value"],
        experimental_data["v_L2"],
        label=r"$v_{L^2}$",
        marker="o",
    )
    ax.set_xlabel("Thickness")
    ax.set_ylabel("L2 Norms w")
    ax2.set_ylabel("L2 Norm v")
    plt.title("L2 Norm Terms vs a")
    plt.legend()
    plt.loglog()

    plt.savefig(f"{experiment_dir}/norms_fields.png")

    list_timings(MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])

    Visualisation(experiment_dir).visualise_results(experimental_data)
    Visualisation(experiment_dir).save_table(timings, "timing_data")
    Visualisation(experiment_dir).save_table(experimental_data, "postprocessing_data")
