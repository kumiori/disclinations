import pytest
import disclinations

import json
import logging
import os
import pdb
import sys
from pathlib import Path
from dolfinx.io import XDMFFile, gmshio
from disclinations.meshes.primitives import mesh_circle_gmshapi

import dolfinx
import numpy as np
import petsc4py
import ufl
import yaml
from mpi4py import MPI
comm = MPI.COMM_WORLD

models = ["variational", "brenner", "carstensen"]

@pytest.mark.parametrize("model", models)
def test_model_computation(model):
    """
    Parametric unit test for testing three different models:
    variational, brenner, and carstensen.
    """
    
    # 1. Load parameters from YML file
    params = load_parameters(f"parameters.yml")
    # params = load_parameters(f"{model}_params.yml")
    
    # 2. Construct or load mesh
    mesh = create_or_load_mesh(params)
    
    # 3. Construct FEM approximation
    fem = setup_fem(mesh, params["fem_degree"])
    
    # 4. Construct boundary conditions
    boundary_conditions = setup_boundary_conditions(mesh, params["boundary_conditions"])
    
    # 5. Initialize exact solutions (for comparison later)
    exact_solution = initialize_exact_solution(params["exact_solution"])
    
    # 6. Define variational form (depends on the model)
    if model == "variational":
        form = define_variational_form(fem, params)
    elif model == "brenner":
        form = define_brenner_form(fem, params)
    elif model == "carstensen":
        form = define_carstensen_form(fem, params)
    
    # 7. Set up the solver
    solver = setup_solver(form, fem, boundary_conditions)
    
    # 8. Solve
    solution = solver.solve()
    
    # 9. Postprocess (if any)
    postprocess(solution)
    
    # 10. Compute absolute and relative error with respect to the exact solution
    abs_error, rel_error = compute_error(solution, exact_solution)
    
    # 11. Display error results
    print(f"Model: {model}, Absolute Error: {abs_error}, Relative Error: {rel_error}")
    
    # 12. Assert that the relative error is within an acceptable range
    assert rel_error < params["error_tolerance"], f"Relative error too high for {model} model."


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

def create_or_load_mesh(parameters, comm, outdir):
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
    
    # Set up file prefix for mesh storage
    prefix = os.path.join(outdir, "plate_fvk_disclinations_monopole")
    mesh_file_path = f"{prefix}/fields.xdmf"
    
    # Check if the mesh file already exists
    if os.path.exists(mesh_file_path):
        print("Loading existing mesh...")
        with XDMFFile(comm, mesh_file_path, "r") as file:
            mesh = file.read_mesh()
            mts = file.read_meshtags(mesh, "facet")  # Assuming facet tags are needed
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
        with XDMFFile(comm, mesh_file_path, "w", encoding=XDMFFile.Encoding.HDF5) as file:
            file.write_mesh(mesh)
        
        return mesh, mts, fts 