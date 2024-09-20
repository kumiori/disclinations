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
                                 table_timing_data, write_to_output)
from disclinations.utils.la import compute_disclination_loads
from dolfinx import fem
from dolfinx.common import list_timings
from dolfinx.fem import Constant, dirichletbc, locate_dofs_topological
from dolfinx.io import XDMFFile, gmshio
from mpi4py import MPI
from petsc4py import PETSc

comm = MPI.COMM_WORLD
from ufl import CellDiameter, FacetNormal, dx

models = ["variational", "brenner", "carstensen"]
outdir = "output"

AIRY = 0
TRANSVERSE = 1

from disclinations.models import create_disclinations
from disclinations.utils import (homogeneous_dirichlet_bc_H20, load_parameters,
                                 save_params_to_yaml)


@pytest.mark.parametrize("variant", models)
def test_model_computation(variant):
    """
    Parametric unit test for testing three different models:
    variational, brenner, and carstensen.
    """

    # 1. Load parameters from YML file
    params, signature = load_parameters(f"parameters.yml")
    params = calculate_rescaling_factors(params)

    # params = load_parameters(f"{model}_params.yml")
    # 2. Construct or load mesh
    prefix = os.path.join(outdir, "plate_fvk_transverse")
    if comm.rank == 0:
        Path(prefix).mkdir(parents=True, exist_ok=True)

    mesh, mts, fts = create_or_load_mesh(params, prefix=prefix)

    # 3. Construct FEM approximation
    h = CellDiameter(mesh)
    n = FacetNormal(mesh)

    # Function spaces

    X = basix.ufl.element("P", str(mesh.ufl_cell()), params["model"]["order"])
    Q = dolfinx.fem.functionspace(mesh, basix.ufl.mixed_element([X, X]))

    V_P1 = dolfinx.fem.functionspace(
        mesh, ("CG", 1)
    )  # "CG" stands for continuous Galerkin (Lagrange)

    DG_e = basix.ufl.element("DG", str(mesh.ufl_cell()), params["model"]["order"] - 2)
    DG = dolfinx.fem.functionspace(mesh, DG_e)

    T_e = basix.ufl.element("P", str(mesh.ufl_cell()), params["model"]["order"] - 2)
    T = dolfinx.fem.functionspace(mesh, T_e)

    # 4. Construct boundary conditions
    boundary_conditions = homogeneous_dirichlet_bc_H20(mesh, Q)

    disclinations, params = create_disclinations(
        mesh, params, points=[-0.0, 0.0, 0], signs=[1.0]
    )
    # 5. Initialize exact solutions (for comparison later)
    exact_solution = initialise_exact_solution(Q, params)

    q = dolfinx.fem.Function(Q)
    v, w = ufl.split(q)
    f = dolfinx.fem.Function(Q.sub(TRANSVERSE).collapse()[0])
    transverse_load = lambda x: _transverse_load(x, params)
    f.interpolate(transverse_load)

    state = {"v": v, "w": w}
    _W_ext = f * w * dx
    
    
    # Define the variational problem

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
        F = ufl.derivative(L, q, ufl.TestFunction(Q)) + model.coupling_term(state)

    elif variant == "carstensen":
        model = NonlinearPlateFVK_carstensen(mesh, params["model"])
        energy = model.energy(state)[0]

        # Dead load (transverse)
        model.W_ext = _W_ext
        penalisation = model.penalisation(state)

        L = energy - model.W_ext + penalisation
        F = ufl.derivative(L, q, ufl.TestFunction(Q)) + model.coupling_term(state)

    # 7. Set up the solver
    solver = SNESSolver(
        F_form=F,
        u=q,
        bcs=boundary_conditions,
        petsc_options=params["solvers"]["elasticity"]["snes"],
        prefix="plate_fvk_transverse",
        monitor=monitor,
    )

    solver.solve()
    save_params_to_yaml(params, "params_with_scaling.yml")

    # 9. Postprocess (if any)
    abs_error, rel_error = postprocess(
        state, model, mesh, params=params, exact_solution=exact_solution, prefix=prefix
    )

    # 10. Compute absolute and relative error with respect to the exact solution
    # abs_error, rel_error = compute_error(solution, exact_solution)

    # # 11. Display error results
    # print(f"Model: {model}, Absolute Error: {abs_error}, Relative Error: {rel_error}")

    # 12. Assert that the relative error is within an acceptable range
    rel_tol = float(params["solvers"]["elasticity"]["snes"]["snes_rtol"])
    # assert (
    #     rel_error < rel_tol
    # ), f"Relative error too high ({rel_error:.2e}>{rel_tol:.2e}) for {model} model."

def create_or_load_mesh(parameters, prefix):
    """
    Create a new mesh if it doesn't exist, otherwise load the existing one.

    Args:
    - parameters (dict): A dictionary containing the geometry and mesh parameters.
    - comm (MPI.Comm): MPI communicator.
    - outdir (str): Directory to store the mesh file.

    Returns:
    - mesh: The generated or loaded mesh.
    - mts: Mesh topology data structure.
    - fts: Facet topology data structure.
    """
    # Extract geometry and mesh size parameters
    mesh_size = parameters["geometry"]["mesh_size"]
    parameters["geometry"]["radius"] = 1  # Assuming the radius is 1
    parameters["geometry"]["geom_type"] = "circle"
    geometry_json = json.dumps(parameters["geometry"], sort_keys=True)
    sha_hash = hashlib.sha256(geometry_json.encode()).hexdigest()

    # Set up file prefix for mesh storage
    mesh_file_path = f"{prefix}/mesh-{sha_hash}.xdmf"
    with dolfinx.common.Timer("~Mesh Generation") as timer:
        # Check if the mesh file already exists
        if os.path.exists(mesh_file_path):
            print("Loading existing mesh...")
            with XDMFFile(comm, mesh_file_path, "r") as file:
                mesh = file.read_mesh()
                mts = None  # Assuming facet tags are needed
                # mts = file.read_meshtags(mesh, "facet")  # Assuming facet tags are needed
                fts = None  # Modify as needed if facet topology structure is available
            return mesh, mts, fts

        else:
            # If no mesh file exists, create a new mesh
            print("Creating new mesh...")
            model_rank = 0
            tdim = 2

            gmsh_model, tdim = mesh_circle_gmshapi(
                parameters["geometry"]["geom_type"], 1, mesh_size, tdim
            )

            mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, model_rank, tdim)

            # Save the mesh for future use
            os.makedirs(prefix, exist_ok=True)
            with XDMFFile(
                comm, mesh_file_path, "w", encoding=XDMFFile.Encoding.HDF5
            ) as file:
                file.write_mesh(mesh)

        return mesh, mts, fts

def initialise_exact_solution(Q, params):
    """
    Initialize the exact solutions for v and w using the provided parameters.

    Args:
    - Q: The function space.
    - params: A dictionary of parameters containing geometric properties (e.g., radius).

    Returns:
    - v_exact: Exact solution for v.
    - w_exact: Exact solution for w.
    """

    # Extract the necessary parameters
    v_scale = params["model"]["v_scale"]

    q_exact = dolfinx.fem.Function(Q)
    v_exact, w_exact = q_exact.split()

    # Define the exact solution for v
    def _v_exact(x):
        rq = x[0] ** 2 + x[1] ** 2
        
        a1=-1/12
        a2=-1/18
        a3=-1/24

        _v = a1 * rq ** 2 + a2 * rq ** 3 + a3 * rq ** 4
        
        return _v * v_scale  # Apply scaling

    # Define the exact solution for w
    def _w_exact(x):
        w_scale = params["model"]["w_scale"]
        
        return w_scale * (1 - x[0]**2 - x[1]**2)**2  # Zero function as per your code

    # Interpolate the exact solutions over the mesh
    v_exact.interpolate(_v_exact)
    w_exact.interpolate(_w_exact)

    return v_exact, w_exact

def _transverse_load(x, params):
    f_scale = params["model"]["f_scale"]
    _p = (40/3) * (1 - x[0]**2 - x[1]**2)**4 + (16/3) * (11 + x[0]**2 + x[1]**2)
    return f_scale * _p


def print_energy_analysis(energy_terms, exact_energy_transverse):
    """Print computed energy vs exact energy analysis."""
    computed_membrane_energy = energy_terms["membrane"]
    error = np.abs(exact_energy_transverse - computed_membrane_energy)

    print(f"Exact energy: {exact_energy_transverse}")
    print(f"Computed energy: {computed_membrane_energy}")
    print(f"Abs error: {error:.3%}")
    print(f"Rel error: {error/exact_energy_transverse:.3%}")

    return error, error / exact_energy_transverse

def postprocess(state, model, mesh, params, exact_solution, prefix):
    with dolfinx.common.Timer(f"~Postprocessing and Vis") as timer:
        energy_components = {
            "bending": model.energy(state)[1],
            "membrane": model.energy(state)[2],
            "coupling": model.energy(state)[3],
            "external_work": -model.W_ext,
        }

        energy_terms = compute_energy_terms(energy_components, mesh.comm)
        _exact_state = {"v": exact_solution[0], "w": exact_solution[1]}
        exact_energy_transverse = comm.allreduce(
            dolfinx.fem.assemble_scalar(
                dolfinx.fem.form(model.energy(_exact_state)[0])),
            op=MPI.SUM)
        
        print(yaml.dump(params["model"], sort_keys=True, default_flow_style=False))
        abs_error, rel_error = print_energy_analysis(
            energy_terms, exact_energy_transverse
        )

        _v_exact, _w_exact = exact_solution
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

    mem_after = memory_usage()
    max_memory = max(max_memory, mem_after)
    logging.info(f"Run Memory Usage (MB) - Before: {mem_before}, After: {mem_after}")
    gc.collect()
    timings = table_timing_data()
    list_timings(MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])
    