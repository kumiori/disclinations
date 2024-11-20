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
from disclinations.models import (
    NonlinearPlateFVK,
    NonlinearPlateFVK_brenner,
    NonlinearPlateFVK_carstensen,
    calculate_rescaling_factors,
    compute_energy_terms,
    _transverse_load_exact_solution,
)
from disclinations.solvers import SNESSolver
from disclinations.utils import (
    Visualisation,
    memory_usage,
    monitor,
    table_timing_data,
    write_to_output,
    homogeneous_dirichlet_bc_H20,
    # initialise_exact_solution_dipole,
    # exact_energy_dipole,
)
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
from disclinations.utils import (
    homogeneous_dirichlet_bc_H20,
    #  load_parameters,
    save_params_to_yaml,
)
from disclinations.utils import update_parameters, save_parameters

from disclinations.utils import create_or_load_circle_mesh
import importlib.resources as pkg_resources  # Python 3.7+ for accessing package files
import copy


def compute_energy_errors(energy_terms, exact_energy_dipole):
    computed_membrane_energy = energy_terms["membrane"]
    error = np.abs(exact_energy_dipole - computed_membrane_energy)

    print(f"Exact energy: {exact_energy_dipole:.3e}")
    print(f"Computed energy: {computed_membrane_energy:.3e}")
    print(f"Abs error: {error:.3%}")
    print(f"Rel error: {error/exact_energy_dipole:.3%}")

    return error, error / exact_energy_dipole


from disclinations.models import assemble_penalisation_terms

from disclinations.utils.la import compute_norms
from disclinations.utils import _logger, snes_solver_stats


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


def run_experiment(
    # mesh, parameters, experiment_dir, variant="variational", initial_guess=None
    variant,
    mesh,
    parameters,
    experiment_folder,
):
    _logger.info(
        f"Running experiment of the series {series} with parameter: {parameters['model']['β_adim']}"
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

    # Point sources
    if mesh.comm.rank == 0:
        # point = np.array([[0.68, 0.36, 0]], dtype=mesh.geometry.x.dtype)
        points = [np.array([[0.0, 0.0, 0]], dtype=mesh.geometry.x.dtype)]
        signs = [-1]
    else:
        # point = np.zeros((0, 3), dtype=mesh.geometry.x.dtype)
        points = [np.zeros((0, 3), dtype=mesh.geometry.x.dtype)]

    disclinations, parameters = create_disclinations(
        mesh, parameters, points=points, signs=signs
    )

    q = dolfinx.fem.Function(Q)

    if initial_guess is not None:
        _logger.info("Using nontrivial initial guess")
        q = initial_guess

    v, w = ufl.split(q)
    f = dolfinx.fem.Function(Q.sub(TRANSVERSE).collapse()[0])

    def transverse_load(x):
        return _transverse_load_exact_solution(x, parameters, adimensional=True)

    f.interpolate(transverse_load)

    state = {"v": v, "w": w}
    assert parameters["model"]["γ_adim"] == 1

    W_ext = parameters["model"]["γ_adim"] * f * w * dx
    # W_ext = Constant(mesh, np.array(-1.0, dtype=PETSc.ScalarType)) * w * dx

    # Define the variational problem
    Q_v, Q_v_to_Q_dofs = Q.sub(AIRY).collapse()
    b = compute_disclination_loads(
        disclinations,
        parameters["loading"]["signs"],
        Q,
        V_sub_to_V_dofs=Q_v_to_Q_dofs,
        V_sub=Q_v,
    )

    # Scale the disclination loads by adimensional factor
    b.vector.scale(parameters["model"]["β_adim"] ** 2.0)

    # 6. Define variational form (depends on the model)
    if variant == "variational":
        model = NonlinearPlateFVK(mesh, parameters["model"], adimensional=True)
        energy = model.energy(state)[0]

        # Dead load (transverse)
        model.W_ext = W_ext
        penalisation = model.penalisation(state)

        L = energy - model.W_ext + penalisation
        F = ufl.derivative(L, q, ufl.TestFunction(Q))

    else:
        raise NotImplementedError(f"Model {model} not implemented.")

    save_params_to_yaml(parameters, os.path.join(prefix, "parameters.yml"))

    # 7. Set up the solver

    solver = SNESSolver(
        F_form=F,
        u=q,
        bcs=boundary_conditions,
        petsc_options=parameters["solvers"]["nonlinear"]["snes"],
        prefix="plate_fvk_relative_work",
        b0=b.vector,
        monitor=monitor,
    )

    solver.solve()
    solver_stats = snes_solver_stats(solver.solver)

    # energies, norms, abs_error, rel_error, penalisation = postprocess(
    #     state, model, mesh, params=params, exact_solution=exact_solution, prefix=prefix
    # )
    # # # 11. Display error results

    # # 12. Assert that the relative error is within an acceptable range
    # sanity_check(abs_error, rel_error, penalisation, params)

    # return energies, norms, abs_error, rel_error, penalisation, solver_stats

    # energy_terms = postprocess(
    #     state, model, mesh, params=parameters, exact_solution=None, prefix=prefix
    # )

    energies, norms, abs_error, rel_error, penalisation = postprocess(
        state, model, mesh, params=parameters, exact_solution=None, prefix=prefix
    )

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

    parameters["model"]["β_adim"] = np.nan
    parameters["model"]["γ_adim"] = 1.0
    parameters["model"]["higher_regularity"] = True

    # Remove thickness from the parameters, to compute the parametric series signature
    if "model" in parameters and "thickness" in parameters["model"]:
        del parameters["model"]["thickness"]

    # parameters["model"]["alpha_penalty"] = 200

    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()
    _logger.info(yaml.dump(parameters, default_flow_style=False))

    return parameters, signature


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
    num_runs = 100

    if comm.rank == 0:
        Path(experiment_dir).mkdir(parents=True, exist_ok=True)
    _logger.info(f"Running series {series} with {num_runs} runs")
    _logger.info(f"===================- {experiment_dir} -=================")

    mesh, mts, fts = create_or_load_circle_mesh(parameters, prefix=prefix)

    with dolfinx.common.Timer(f"~Computation Experiment") as timer:

        initial_guess = None

        # for i, a in enumerate(np.linspace(10, 1000, num_runs)):
        for i, a in enumerate(np.logspace(np.log10(10), np.log10(1000), num=num_runs)):
            parameters["model"]["β_adim"] = None
            parameters["model"]["γ_adim"] = 1

            _logger.critical(f"===================- β_adim = {a} -=================")

            if changed := update_parameters(parameters, "β_adim", float(a)):
                parameters["model"]["thickness"] = parameters["geometry"]["radius"] / a
                signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()
            else:
                raise ValueError("Failed to update parameters")

            energies, norms, _, _, penalisation, solver_stats = run_experiment(
                "variational", mesh, parameters, prefix
            )
            run_data = {
                "signature": signature,
                "param_value": a,
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

    experimental_data = pd.DataFrame(_experimental_data)

    _logger.info(f"Saving experimental data to {experiment_dir}")
    print(experimental_data)

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
