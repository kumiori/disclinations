# Solving a simple biharmonic problem with disclinations
# H^1_0 formulation
# Testing against closed form solutions:
#   - monopole
#   - dipole
# aligned on a symmetry axis

import pytest
import json
import logging
import os
import pdb
import sys
from pathlib import Path

import dolfinx
import dolfinx.plot
import numpy as np
import petsc4py
import ufl
import yaml
from disclinations.meshes import mesh_bounding_box
from disclinations.meshes.primitives import mesh_circle_gmshapi

# from damage.utils import ColorPrint
from disclinations.solvers import SNESSolver
from disclinations.models import Biharmonic
from disclinations.utils.la import compute_cell_contributions, compute_disclination_loads
from disclinations.utils.viz import plot_mesh

from dolfinx import log
from dolfinx.io import XDMFFile
from mpi4py import MPI
from petsc4py import PETSc


import dolfinx
import dolfinx.io
import dolfinx.mesh
import dolfinx.plot
import ufl
import yaml
from dolfinx.fem import Constant, dirichletbc
from dolfinx.io import XDMFFile, gmshio
from mpi4py import MPI

import gmsh

@pytest.fixture(scope="module", autouse=True)
def initialize_gmsh():
    """Initialize and finalize gmsh for the entire module."""
    gmsh.initialize()
    yield
    gmsh.finalize()
    

@pytest.fixture(scope="module", autouse=True)
def setup_environment():
    """Setup and teardown for the environment."""
    comm = MPI.COMM_WORLD
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parameters_file = os.path.join(script_dir, "parameters.yml")

    with open(parameters_file) as f:
        parameters = yaml.load(f, Loader=yaml.FullLoader)
        
    mesh_size = parameters["geometry"]["mesh_size"]
    parameters["geometry"]["radius"] = 1
    parameters["geometry"]["geom_type"] = "circle"
    tdim = 2

    gmsh_model, tdim = mesh_circle_gmshapi(
        parameters["geometry"]["geom_type"], 1, mesh_size, tdim
    )
    mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, 0, tdim)

    outdir = "output"
    prefix = os.path.join(outdir, "biharmonic")
    if comm.rank == 0:
        Path(prefix).mkdir(parents=True, exist_ok=True)

    return mesh, parameters, prefix, comm

def monitor(snes, it, norm):
    logging.info(f"Iteration {it}, residual {norm}")
    return PETSc.SNES.ConvergedReason.ITERATING



def test_bilaplacian_solution(setup_environment):
    mesh, parameters, prefix, comm = setup_environment
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", parameters["model"]["order"]))

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    bndry_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)

    dofs = dolfinx.fem.locate_dofs_topological(V=V, entity_dim=1, entities=bndry_facets)
    bcs = [dolfinx.fem.dirichletbc(value=np.array(0, dtype=PETSc.ScalarType), dofs=dofs, V=V)]

    # b = dolfinx.fem.Function(V)
    # b.x.array[:] = 0

    if mesh.comm.rank == 0:
        point = np.array([[0.0, 0.0, 0]], dtype=mesh.geometry.x.dtype)
        
        # points = [np.array([[-0.2, 0.1, 0]], dtype=mesh.geometry.x.dtype),
        #         np.array([[0.2, -0.1, 0]], dtype=mesh.geometry.x.dtype)]
        # signs = [-1, 1]
        sign = [1]
    else:
        point = np.zeros((0, 3), dtype=mesh.geometry.x.dtype)
        # points = [np.zeros((0, 3), dtype=mesh.geometry.x.dtype),
                # np.zeros((0, 3), dtype=mesh.geometry.x.dtype)]

    # _cells, _basis_values = compute_cell_contributions(V, point)

    b = compute_disclination_loads(point, sign, V)

    u = dolfinx.fem.Function(V)
    state = {"u": u}

    model = Biharmonic(mesh, parameters["model"])
    W_ext = u * ufl.dx
    L = model.energy(state) + model.penalisation(state) - W_ext

    F = ufl.derivative(L, u, ufl.TestFunction(V))

    solver = SNESSolver(
        F_form=F,
        u=u,
        bcs=bcs,
        bounds=None,
        petsc_options=parameters.get("solvers").get("elasticity").get("snes"),
        prefix='biharmonic',
        b0=b.vector
    )

    solver.solve()
    
    # Check the solution
    check_components = {
        "energy": model.energy(state),
        "penalisation": model.penalisation(state),
    }
    
    exact_energy = parameters["model"]["E"] * parameters["model"]["h"]**3 \
        * parameters["geometry"]["R"] / (32 * np.pi)
    
    print(exact_energy)
    
    
    check_terms = {label: comm.allreduce(
    dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(energy_term)),
    op=MPI.SUM,
    ) for label, energy_term in check_components.items()}

    print(check_terms)
    assert check_terms["penalisation"] < 1e-6, "Penalisation terms are _not_ small"
    assert np.isclose(check_terms["energy"], exact_energy, 1e-6)
    pdb.set_trace()

if __name__ == "__main__":
    pytest.main()