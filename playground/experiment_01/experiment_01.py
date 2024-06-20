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
from disclinations.models import FVKAdimensional
from disclinations.solvers import SNESSolver
from disclinations.utils.la import compute_disclination_loads
from disclinations.utils.viz import plot_scalar, plot_profile, plot_mesh
from disclinations.utils import update_parameters, memory_usage

logging.basicConfig(level=logging.INFO)
comm = MPI.COMM_WORLD

def run_experiment(parameters: dict):
    # Setup, output and file handling
    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()[0:6]
    
    outdir = "output"
    prefix = os.path.join(outdir, "experiment_01", signature)

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
    # f0 = parameters["model"]["E"] * (parameters["model"]["thickness"] / parameters["geometry"]["radius"])**4 
    b0 = (radius / thickness)**2

    # Mesh
    model_rank = 0
    tdim = 2
    # order = 3
    with dolfinx.common.Timer("~Mesh Generation") as timer:
        gmsh_model, tdim = mesh_circle_gmshapi(
            parameters["geometry"]["geom_type"], 1, mesh_size, tdim
        )
        mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, model_rank, tdim)
        mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    h = CellDiameter(mesh)
    n = FacetNormal(mesh)

    # Functional setting

    X = basix.ufl.element("P", str(mesh.ufl_cell()), parameters["model"]["order"]) 
    Q = dolfinx.fem.functionspace(mesh, basix.ufl.mixed_element([X, X]))

    q = dolfinx.fem.Function(Q)
    f = dolfinx.fem.Function(Q.sub(TRANSVERSE).collapse()[0])
    
    v, w = ufl.split(q)
    state = {"v": v, "w": w}

    # Loading
    # Disclinations
    if mesh.comm.rank == 0:
        # point = np.array([[0.68, 0.36, 0]], dtype=mesh.geometry.x.dtype)
        points = [np.array([[-0.2, 0.0, 0]], dtype=mesh.geometry.x.dtype),
                np.array([[0.2, -0.0, 0]], dtype=mesh.geometry.x.dtype)]
        signs = [-1, 1]
    else:
        # point = np.zeros((0, 3), dtype=mesh.geometry.x.dtype)
        points = [np.zeros((0, 3), dtype=mesh.geometry.x.dtype),
                np.zeros((0, 3), dtype=mesh.geometry.x.dtype)]
        
    Q_v, Q_v_to_Q_dofs = Q.sub(AIRY).collapse()

    b = compute_disclination_loads(points, signs, Q, V_sub_to_V_dofs=Q_v_to_Q_dofs, V_sub=Q_v)
    
    # Transverse load (analytical solution)
    def transverse_load(x):
        _f = (40/3) * (1 - x[0]**2 - x[1]**2)**4 + (16/3) * (11 + x[0]**2 + x[1]**2)
        return _f * f_scale
        # return (11+ x[0]**2 + x[1]**2)

    f.interpolate(transverse_load)

    # Boundary conditions 
    bndry_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    dofs_v = dolfinx.fem.locate_dofs_topological(V=Q.sub(AIRY), entity_dim=1, entities=bndry_facets)
    dofs_w = dolfinx.fem.locate_dofs_topological(V=Q.sub(TRANSVERSE), entity_dim=1, entities=bndry_facets)
        
    bcs_w = dirichletbc(
        np.array(0, dtype=PETSc.ScalarType),
        dofs_w, Q.sub(TRANSVERSE)
    )
    bcs_v = dirichletbc(
        np.array(0, dtype=PETSc.ScalarType),
        dofs_v, Q.sub(AIRY)
    )
    bcs = list({AIRY: bcs_v, TRANSVERSE: bcs_w}.values())

    W_transv_coeff = (radius / thickness)**4 / E 
    W_ext = f * w * dx

    model = FVKAdimensional(mesh, nu = nu)
    energy = model.energy(state)[0]
    penalisation = model.penalisation(state)


    # Define the functional
    L = energy - W_ext + penalisation
    F = ufl.derivative(L, q, ufl.TestFunction(Q))

    solver = SNESSolver(
        F_form=F,
        u=q,
        bcs=bcs,
        bounds=None,
        petsc_options=parameters["solvers"]["elasticity"],
        prefix='plate_fvk',
        b0=b.vector,
    )
    solver.solve()
    
    del solver
    
    # Postprocessing and viz
    with dolfinx.common.Timer(f"~Postprocessing and Vis") as timer:
        pass

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
    parameters["solvers"]["elasticity"] = {
        "snes_type": "newtonls",      # Solver type: NGMRES (Nonlinear GMRES)
        "snes_max_it": 100,           # Maximum number of iterations
        "snes_rtol": 1e-6,            # Relative tolerance for convergence
        "snes_atol": 1e-15,           # Absolute tolerance for convergence
        "snes_stol": 1e-5,           # Tolerance for the change in solution norm
        "snes_monitor": None,         # Function for monitoring convergence (optional)
        "snes_linesearch_type": "basic",  # Type of line search
    }

    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()
    print(yaml.dump(parameters, default_flow_style=False))

    return parameters, signature

if __name__ == "__main__":
    from disclinations.utils import table_timing_data
    
    outdir = "output"
    parameters, signature = load_parameters("../../src/disclinations/test_branch/parameters.yml")
    experiment_dir = os.path.join(outdir, signature[0::6])
    max_memory = 0
    num_runs = 10
    if comm.rank == 0:
        Path(experiment_dir).mkdir(parents=True, exist_ok=True)
            
    logging.info(
        f"===================- {experiment_dir} -=================")
    
    
    with dolfinx.common.Timer(f"~Computation Experiment") as timer:
        for i, thickness in enumerate(np.linspace(0.1, 1, num_runs)):
            # Check memory usage before computation
            mem_before = memory_usage()
            
            # parameters["model"]["thickness"] = thickness
            update_parameters(parameters, "thickness", thickness)    
            run_experiment(parameters)
            
            # Check memory usage after computation
            mem_after = memory_usage()
            max_memory = max(max_memory, mem_after)
            
            # Log memory usage
            logging.info(f"Run {i}/{num_runs}: Memory Usage (MB) - Before: {mem_before}, After: {mem_after}")
            # Perform garbage collection
            gc.collect()
            
    timings = table_timing_data()
    list_timings(MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])
    