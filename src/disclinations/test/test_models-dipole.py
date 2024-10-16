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
from dolfinx import fem
from dolfinx.common import list_timings
from dolfinx.fem import Constant, dirichletbc, locate_dofs_topological
from dolfinx.io import XDMFFile, gmshio
from mpi4py import MPI
from petsc4py import PETSc
from disclinations.utils import create_or_load_circle_mesh

comm = MPI.COMM_WORLD
from ufl import CellDiameter, FacetNormal, dx

models = ["variational", "brenner", "carstensen"]
outdir = "output"

AIRY = 0
TRANSVERSE = 1


@pytest.mark.parametrize("variant", models)
def test_model_computation(variant):
    """
    Parametric unit test for testing three different models:
    variational, brenner, and carstensen.
    """

    # 1. Load parameters from YML file
    params, signature = load_parameters(f"parameters.yml")
    params = calculate_rescaling_factors(params)
    pdb.set_trace()
    # params = load_parameters(f"{model}_params.yml")
    # 2. Construct or load mesh
    prefix = os.path.join(outdir, "plate_fvk_disclinations_dipole")
    if comm.rank == 0:
        Path(prefix).mkdir(parents=True, exist_ok=True)

    mesh, mts, fts = create_or_load_circle_mesh(params, prefix=prefix)

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
    # 5. Initialize exact solutions (for comparison later)
    # exact_solution = initialise_exact_solution(Q, params)

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
        prefix="plate_fvk_disclinations",
        b0=b.vector,
        monitor=monitor,
    )

    solver.solve()
    save_params_to_yaml(params, "params_with_scaling.yml")

    # 9. Postprocess (if any)
    abs_error, rel_error = postprocess(
        state, model, mesh, params=params, exact_solution=None, prefix=prefix
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
    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()

    return parameters, signature


def homogeneous_dirichlet_bc_H20(mesh, Q):
    """
    Apply homogeneous Dirichlet boundary conditions (H^2_0 Sobolev space)
    to both AIRY and TRANSVERSE fields.

    Args:
    - mesh: The mesh of the domain.
    - Q: The function space.

    Returns:
    - A list of boundary conditions (bcs) for the problem.
    """
    # Create connectivity between topological dimensions
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)

    # Identify the boundary facets
    bndry_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)

    # Locate DOFs for AIRY field
    dofs_v = locate_dofs_topological(V=Q.sub(AIRY), entity_dim=1, entities=bndry_facets)

    # Locate DOFs for TRANSVERSE field
    dofs_w = locate_dofs_topological(
        V=Q.sub(TRANSVERSE), entity_dim=1, entities=bndry_facets
    )

    # Create homogeneous Dirichlet BC (value = 0) for both fields
    bcs_v = dirichletbc(np.array(0, dtype=PETSc.ScalarType), dofs_v, Q.sub(AIRY))
    bcs_w = dirichletbc(np.array(0, dtype=PETSc.ScalarType), dofs_w, Q.sub(TRANSVERSE))

    # Return the boundary conditions as a list
    return [bcs_v, bcs_w]


def initialise_exact_solution(Q, params):
    """
    Initialize the exact solutions for v and w using the provided parameters.
    TODO: Do we have an exact solution for the dipole case?
    
    Args:
    - Q: The function space.
    - params: A dictionary of parameters containing geometric properties (e.g., radius).

    Returns:
    - v_exact: Exact solution for v.
    - w_exact: Exact solution for w.
    """

    return None, None
    # return v_exact, w_exact


def calculate_rescaling_factors(params):
    """
    Calculate rescaling factors and store them in the params dictionary.

    Args:
    - params (dict): Dictionary containing geometry, material, and model parameters.

    Returns:
    - params (dict): Updated dictionary with rescaling factors.
    """
    # Extract necessary parameters
    _E = params["model"]["E"]
    nu = params["model"]["nu"]
    # _D = params["model"]["D"]
    thickness = params["model"]["thickness"]
    _D = _E * thickness**3 / (12 * (1 - nu**2))
    
    # Calculate rescaling factors
    w_scale = np.sqrt(2 * _D / (_E * thickness))
    v_scale = _D
    f_scale = np.sqrt(2 * _D**3 / (_E * thickness))

    # Store rescaling factors in the params dictionary
    params["model"]["w_scale"] = float(w_scale)
    params["model"]["v_scale"] = float(v_scale)
    params["model"]["f_scale"] = float(f_scale)

    return params


def save_params_to_yaml(params, filename):
    """
    Save the updated params dictionary to a YAML file.

    Args:
    - params (dict): Dictionary containing all parameters.
    - filename (str): Path to the YAML file to save.
    """
    with open(filename, "w") as file:
        yaml.dump(params, file, default_flow_style=False)


def create_disclinations(mesh, params, points=[0.0, 0.0, 0.0], signs=[1.0]):
    """
    Create disclinations based on the list of points and signs or the params dictionary.

    Args:
    - mesh: The mesh object, used to determine the data type for points.
    - points: A list of 3D coordinates (x, y, z) representing the disclination points.
    - signs: A list of signs (+1., -1.) associated with each point.
    - params: A dictionary containing model parameters, possibly including loading points and signs.

    Returns:
    - disclinations: A list of disclination points (coordinates).
    - signs: The same list of signs associated with each point.
    - params: Updated params dictionary with disclinations if not already present.
    """

    # Check if "loading" exists in the parameters and contains "points" and "signs"
    if (
        "loading" in params
        and params["loading"] is not None
        and "points" in params["loading"]
        and "signs" in params["loading"]
    ):
        # Use points and signs from the params dictionary
        points = params["loading"]["points"]
        signs = params["loading"]["signs"]
        print("Using points and signs from params dictionary.")
    else:
        # Otherwise, add the provided points and signs to the params dictionary
        print("Using provided points and signs, adding them to the params dictionary.")
        params["loading"] = {"points": points, "signs": signs}

    # Handle the case where rank is not 0 (for distributed computing)
    if mesh.comm.rank == 0:
        # Convert the points into a numpy array with the correct dtype from the mesh geometry
        disclinations = [
            np.array([point], dtype=mesh.geometry.x.dtype) for point in points
        ]
    else:
        # If not rank 0, return empty arrays for parallel processing
        disclinations = [np.zeros((0, 3), dtype=mesh.geometry.x.dtype) for _ in points]

    return disclinations, params


def compute_energy_terms(energy_components, comm):
    """Assemble and sum energy terms over all processes."""
    computed_energy_terms = {
        label: comm.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(energy_term)),
            op=MPI.SUM,
        )
        for label, energy_term in energy_components.items()
    }
    return computed_energy_terms


def print_energy_analysis(energy_terms, exact_energy_dipole):
    """Print computed energy vs exact energy analysis."""
    computed_membrane_energy = energy_terms["membrane"]
    error = np.abs(exact_energy_dipole - computed_membrane_energy)

    print(f"Exact energy: {exact_energy_dipole}")
    print(f"Computed energy: {computed_membrane_energy}")
    print(f"Abs error: {error:.3%}")
    print(f"Rel error: {error/exact_energy_dipole:.3%}")

    return error, error / exact_energy_dipole


def postprocess(state, model, mesh, params, exact_solution, prefix):
    with dolfinx.common.Timer(f"~Postprocessing and Vis") as timer:
        energy_components = {
            "bending": model.energy(state)[1],
            "membrane": model.energy(state)[2],
            "coupling": model.energy(state)[3],
            "external_work": -model.W_ext,
        }

        energy_terms = compute_energy_terms(energy_components, mesh.comm)
        distance = np.linalg.norm(params['loading']['points'][0] - params['loading']['points'][1])
        
        exact_energy_dipole = params["model"]["E"] * params["model"]["thickness"]**3 \
            * params["geometry"]["radius"]**2 / (8 * np.pi) *  distance**2 * \
                (np.log(4+distance**2) - np.log(4 * distance))

        print(yaml.dump(params["model"], default_flow_style=False))
        
        abs_error, rel_error = print_energy_analysis(
            energy_terms, exact_energy_dipole
        )

        if exact_solution is not None:
            _v_exact, _w_exact = exact_solution
        else:
            _v_exact, _w_exact = None, None
            
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
