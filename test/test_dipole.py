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
    create_disclinations,
    initialise_exact_solution_dipole,
)
from disclinations.solvers import SNESSolver
from disclinations.utils import (
    Visualisation,
    memory_usage,
    monitor,
    table_timing_data,
    write_to_output,
)
from disclinations.utils.la import compute_disclination_loads
from disclinations.utils import _logger
from dolfinx import fem
from dolfinx.common import list_timings
from dolfinx.io import XDMFFile, gmshio
from mpi4py import MPI
from petsc4py import PETSc
from disclinations.utils import create_or_load_circle_mesh, print_energy_analysis

comm = MPI.COMM_WORLD
from ufl import CellDiameter, FacetNormal, dx

models = ["variational", "brenner", "carstensen"]
outdir = "output"

AIRY = 0
TRANSVERSE = 1

from disclinations.utils import (
    homogeneous_dirichlet_bc_H20,
    save_params_to_yaml,
)


def load_parameters(filename):
    with open(filename, "r") as f:
        params = yaml.safe_load(f)

    params["model"]["thickness"] = 0.01
    params["model"]["E"] = 1
    # params["model"]["alpha_penalty"] = 300

    signature = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()

    return params, signature


@pytest.mark.parametrize("variant", models)
def test_model_computation(variant):
    """
    Parametric unit test for testing three different models:
    variational, brenner, and carstensen.
    """

    # 1. Load parameters from YML file
    params, signature = load_parameters(f"parameters.yaml")

    params = calculate_rescaling_factors(params)
    c_nu = 1 / (12 * (1 - params["model"]["nu"] ** 2))

    _logger.info("Model parameters")
    _logger.info(params["model"])

    # params = load_parameters(f"{model}_params.yml")
    # 2. Construct or load mesh
    prefix = os.path.join(outdir, "validation_dipole")
    if comm.rank == 0:
        Path(prefix).mkdir(parents=True, exist_ok=True)

    mesh, mts, fts = create_or_load_circle_mesh(params, prefix=prefix)

    # 3. Construct FEM approximation
    # Function spaces

    X = basix.ufl.element("P", str(mesh.ufl_cell()), params["model"]["order"])
    Q = dolfinx.fem.functionspace(mesh, basix.ufl.mixed_element([X, X]))

    # 4. Construct boundary conditions
    boundary_conditions = homogeneous_dirichlet_bc_H20(mesh, Q)

    q = dolfinx.fem.Function(Q)
    v, w = ufl.split(q)

    state = {"v": v, "w": w}

    _W_ext = dolfinx.fem.Constant(mesh, np.array(0.0, dtype=PETSc.ScalarType)) * w * dx

    test_v, test_w = ufl.TestFunctions(Q)[AIRY], ufl.TestFunctions(Q)[TRANSVERSE]
    Q_v, Q_v_to_Q_dofs = Q.sub(AIRY).collapse()

    # disclinations, parameters = create_disclinations(
    #     mesh, parameters, points=points, signs=signs
    # )

    # Point sources
    if mesh.comm.rank == 0:
        points = [
            np.array([[-0.2, 0.0, 0]], dtype=mesh.geometry.x.dtype),
            np.array([[0.2, -0.0, 0]], dtype=mesh.geometry.x.dtype),
        ]
        disclination_power_list = [-1, 1]
    else:
        points = [
            np.zeros((0, 3), dtype=mesh.geometry.x.dtype),
            np.zeros((0, 3), dtype=mesh.geometry.x.dtype),
        ]
        disclination_power_list = [0, 0]

    disclinations, params = create_disclinations(
        mesh, params, points=points, signs=disclination_power_list
    )
    b = compute_disclination_loads(
        disclinations,
        disclination_power_list,
        Q,
        V_sub_to_V_dofs=Q_v_to_Q_dofs,
        V_sub=Q_v,
    )

    # 5. Initialize exact solutions (for comparison later)
    exact_solution = initialise_exact_solution_dipole(Q, params, adimensional=True)

    # 6. Define variational form (depends on the model)
    if variant == "variational":
        model = NonlinearPlateFVK(mesh, params["model"], adimensional=True)
        energy = model.energy(state)[0]

        # Dead load (transverse)
        model.W_ext = _W_ext
        penalisation = model.penalisation(state)

        L = energy - model.W_ext + penalisation
        F = ufl.derivative(L, q, ufl.TestFunction(Q))

    elif variant == "brenner":
        # F = define_brenner_form(fem, params)
        model = NonlinearPlateFVK_brenner(mesh, params["model"], adimensional=True)
        energy = model.energy(state)[0]

        # Dead load (transverse)
        model.W_ext = _W_ext
        penalisation = model.penalisation(state)

        L = energy - model.W_ext + penalisation
        F = ufl.derivative(L, q, ufl.TestFunction(Q)) + model.coupling_term(
            state, test_v, test_w
        )

    elif variant == "carstensen":
        model = NonlinearPlateFVK_carstensen(mesh, params["model"], adimensional=True)
        energy = model.energy(state)[0]

        # Dead load (transverse)
        model.W_ext = _W_ext
        penalisation = model.penalisation(state)

        L = energy - model.W_ext + penalisation
        F = ufl.derivative(L, q, ufl.TestFunction(Q)) + model.coupling_term(
            state, test_v, test_w
        )

    save_params_to_yaml(params, os.path.join(prefix, "parameters.yml"))

    if MPI.COMM_WORLD.rank == 0:
        with open(f"{prefix}/signature.md5", "w") as f:
            f.write(signature)

    # 7. Set up the solver
    solver = SNESSolver(
        F_form=F,
        u=q,
        bcs=boundary_conditions,
        petsc_options=params["solvers"]["nonlinear"]["snes"],
        prefix="plate_fvk_dipole",
        b0=b.vector,
        monitor=monitor,
    )

    solver.solve()

    # 9. Postprocess (if any)
    # 10. Compute absolute and relative error with respect to the exact solution

    abs_error, rel_error, penalisation = postprocess(
        state, model, mesh, params=params, exact_solution=exact_solution, prefix=prefix
    )
    # # 11. Display error results

    # 12. Assert that the relative error is within an acceptable range
    sanity_check(abs_error, rel_error, penalisation, params)


from disclinations.models import assemble_penalisation_terms


def sanity_check(abs_errors, rel_errors, penalization_terms, params):
    atol = float(params["solvers"]["nonlinear"]["snes"]["snes_atol"])
    rtol = float(params["solvers"]["nonlinear"]["snes"]["snes_rtol"])

    _logger.info("\nSanity Check Report:")

    # Check each error against tolerance
    for energy_type, abs_err, rel_err in zip(
        ["total", "bending", "membrane", "coupling"], abs_errors, rel_errors
    ):
        abs_check = abs_err < atol
        rel_check = rel_err < rtol

        _logger.info(f"{energy_type.capitalize()} Energy Error Check:")
        _logger.info(
            f"  Absolute Error: {abs_err:.2e} {'(PASS)' if abs_check else '(FAIL)'}"
        )
        _logger.info(
            f"  Relative Error: {rel_err:.2e} {'(PASS)' if rel_check else '(FAIL)'}"
        )

        if not abs_check or not rel_check:
            _logger.warning(
                f"{energy_type.capitalize()} energy error exceeds tolerance: "
                f"abs {abs_err:.2e} / {atol}, rel {rel_err:.2e} / {rtol}"
            )

    # Verify if penalization terms meet expectations
    max_penalization = max(penalization_terms.values())
    if max_penalization > rtol:
        _logger.warning(
            f"Max penalization term {max_penalization:.2e} exceeds relative tolerance {rtol:.2e}"
        )
    else:
        _logger.info("All penalization terms within expected tolerance range.")

    return all(
        abs_err < atol and rel_err < rtol
        for abs_err, rel_err in zip(abs_errors, rel_errors)
    )


def postprocess(state, model, mesh, params, exact_solution, prefix):
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

        penalization_terms = assemble_penalisation_terms(model)

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
        return abs_error_array, rel_error_array, penalization_terms


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    max_memory = 0
    mem_before = memory_usage()

    # pytest.main()

    with dolfinx.common.Timer(f"~Computation Experiment") as timer:
        test_model_computation("variational")
        # test_model_computation("brenner")
        # test_model_computation("carstensen")

    # mem_after = memory_usage()
    # max_memory = max(max_memory, mem_after)
    # logging.info(f"Run Memory Usage (MB) - Before: {mem_before}, After: {mem_after}")
    # gc.collect()
    timings = table_timing_data()
    list_timings(MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])
