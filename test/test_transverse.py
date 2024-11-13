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
    initialise_exact_solution_compatible_transverse,
)
from disclinations.solvers import SNESSolver
from disclinations.utils import (
    Visualisation,
    memory_usage,
    monitor,
    table_timing_data,
    write_to_output,
)
from disclinations.utils import _logger
from disclinations.utils.la import compute_disclination_loads
from dolfinx import fem
from dolfinx.common import list_timings
from dolfinx.fem import Constant, dirichletbc, locate_dofs_topological
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

from disclinations.models import create_disclinations
from disclinations.utils import (
    homogeneous_dirichlet_bc_H20,
    #  load_parameters,
    save_params_to_yaml,
)


def load_parameters(filename):
    with open(filename, "r") as f:
        params = yaml.safe_load(f)

    params["model"]["thickness"] = 0.05
    # params["model"]["E"] = 1e10
    params["model"]["E"] = 1
    params["model"]["alpha_penalty"] = 300

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

    _logger.info("Model parameters")
    _logger.info(params["model"])
    pdb.set_trace()
    # params = load_parameters(f"{model}_params.yml")
    # 2. Construct or load mesh
    prefix = os.path.join(outdir, "validation_transverse_load")
    if comm.rank == 0:
        Path(prefix).mkdir(parents=True, exist_ok=True)

    mesh, mts, fts = create_or_load_circle_mesh(params, prefix=prefix)

    # 3. Construct FEM approximation
    # Function spaces

    X = basix.ufl.element("P", str(mesh.ufl_cell()), params["model"]["order"])
    Q = dolfinx.fem.functionspace(mesh, basix.ufl.mixed_element([X, X]))

    # 4. Construct boundary conditions
    boundary_conditions = homogeneous_dirichlet_bc_H20(mesh, Q)

    # 5. Initialize exact solutions (for comparison later)
    exact_solution = initialise_exact_solution_compatible_transverse(
        Q, params, adimensional=True
    )

    q = dolfinx.fem.Function(Q)
    v, w = ufl.split(q)

    f = dolfinx.fem.Function(Q.sub(TRANSVERSE).collapse()[0])

    transverse_load = lambda x: _transverse_load_exact_solution(x, params)

    f.interpolate(transverse_load)

    state = {"v": v, "w": w}

    adim_coeff = params["geometry"]["radius"] / params["model"]["thickness"]
    _W_ext = adim_coeff**4 * f * w * dx

    test_v, test_w = ufl.TestFunctions(Q)[AIRY], ufl.TestFunctions(Q)[TRANSVERSE]

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
        prefix="plate_fvk_transverse",
        monitor=monitor,
    )

    solver.solve()
    # 9. Postprocess (if any)
    # pdb.set_trace()
    # 10. Compute absolute and relative error with respect to the exact solution

    abs_error, rel_error = postprocess(
        state, model, mesh, params=params, exact_solution=exact_solution, prefix=prefix
    )
    # # 11. Display error results
    _logger.critical(
        f"Model: {model}, Absolute Error: {abs_error:.2e}, Relative Error: {rel_error:.1%}"
    )

    # 12. Assert that the relative error is within an acceptable range
    rel_tol = float(params["solvers"]["nonlinear"]["snes"]["snes_rtol"])

    # assert (
    #     rel_error < rel_tol
    # ), f"Relative error too high ({rel_error:.2e}>{rel_tol:.2e}) for {model} model."


def postprocess(state, model, mesh, params, exact_solution, prefix):
    with dolfinx.common.Timer(f"~Postprocessing and Vis") as timer:
        energy_components = {
            "bending": model.energy(state)[1],
            "membrane": model.energy(state)[2],
            "coupling": model.energy(state)[3],
            "external_work": -model.W_ext,
        }

        energy_terms = compute_energy_terms(energy_components, mesh.comm)

        exact_energy_transverse = comm.allreduce(
            dolfinx.fem.assemble_scalar(
                dolfinx.fem.form(model.energy(exact_solution)[0])
            ),
            op=MPI.SUM,
        )

        # print(energy_terms)
        _logger.info(f"Exact energy (transverse): {exact_energy_transverse}")

        _logger.info(
            yaml.dump(params["model"], sort_keys=True, default_flow_style=False)
        )

        abs_error, rel_error = print_energy_analysis(
            energy_terms, exact_energy_transverse
        )

        _penalisation = [model._dgw, model._dgv, model._dgc, model._bcv, model._bcw]

        for label, penalisation_term in zip(
            ["dgw", "dgv", "dgc", "bcv", "bcw"], _penalisation
        ):
            _logger.critical(
                f"Penalisation term {label}: {dolfinx.fem.assemble_scalar(dolfinx.fem.form(penalisation_term))}"
            )
        pdb.set_trace()

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

        return abs_error, rel_error


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    max_memory = 0
    mem_before = memory_usage()

    # pytest.main()

    with dolfinx.common.Timer(f"~Computation Experiment") as timer:
        test_model_computation("variational")
        test_model_computation("brenner")
        test_model_computation("carstensen")

    # mem_after = memory_usage()
    # max_memory = max(max_memory, mem_after)
    # logging.info(f"Run Memory Usage (MB) - Before: {mem_before}, After: {mem_after}")
    # gc.collect()
    timings = table_timing_data()
    list_timings(MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])
