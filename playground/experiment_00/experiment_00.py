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
from disclinations.models import NonlinearPlateFVK
from disclinations.solvers import SNESSolver
from disclinations.utils.la import compute_disclination_loads
from disclinations.utils.viz import plot_scalar, plot_profile, plot_mesh
from disclinations.utils import update_parameters, memory_usage, save_parameters

logging.basicConfig(level=logging.INFO)
comm = MPI.COMM_WORLD

def run_experiment(mesh: None, parameters: dict, series: str):
    # Setup, output and file handling
    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()[0:6]
    data = {
        'membrane_energy': [],
        'bending_energy': [],
        'coupling_energy': [],
        'w_L2': [],
        'v_L2': [],
        'thickness': []
    }
    
    outdir = "output"
    prefix = os.path.join(outdir, series, signature)

    if comm.rank == 0:
        Path(prefix).mkdir(parents=True, exist_ok=True)

    save_parameters(parameters, prefix)

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
    f0 = parameters["model"]["E"] * (parameters["model"]["thickness"] / parameters["geometry"]["radius"])**4 
    b_scale = (radius / thickness)**2

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
    h = CellDiameter(mesh)
    n = FacetNormal(mesh)

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
    # Disclinations
    if mesh.comm.rank == 0:
        points = np.array([[0.0, 0.0, 0]], dtype=mesh.geometry.x.dtype)
        signs = [1]
    else:
        points = np.zeros((0, 3), dtype=mesh.geometry.x.dtype)
        
    Q_v, Q_v_to_Q_dofs = Q.sub(AIRY).collapse()

    b = compute_disclination_loads(points, signs, Q, V_sub_to_V_dofs=Q_v_to_Q_dofs, V_sub=Q_v)
    
    # Transverse load (analytical solution)
    def transverse_load(x):
        _f = (40/3) * (1 - x[0]**2 - x[1]**2)**4 + (16/3) * (11 + x[0]**2 + x[1]**2)
        return _f * f_scale

    def _v_exact(x):
        a1=-1/12
        a2=-1/18
        a3=-1/24
        _v = a1 * (1 - x[0]**2 - x[1]**2)**2 + a2 * (1 - x[0]**2 - x[1]**2)**3 + a3 * (1 - x[0]**2 - x[1]**2)**4
        _v = _v * v_scale
        return _v

    def _w_exact(x):
        _w = (1 - x[0]**2 - x[1]**2)**2
        return _w*w_scale
    
    f.interpolate(transverse_load)
    v_exact.interpolate(_v_exact)
    w_exact.interpolate(_w_exact)

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
    W_ext = dolfinx.fem.Constant(mesh, 0.) * f * w * dx
    # print(b_scale)

    # model = NonlinearPlateFVK(mesh, nu = nu)
    model = NonlinearPlateFVK(mesh, parameters["model"])
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
        b0=b_scale*b.vector,
    )
    solver.solve()


    V_P1 = dolfinx.fem.FunctionSpace(mesh, ("CG", 1))  # "CG" stands for continuous Galerkin (Lagrange)

    with XDMFFile(comm, f"{prefix}/fields-vs-thickness.xdmf", "w",
                encoding=XDMFFile.Encoding.HDF5) as file:
        file.write_mesh(mesh)
    
    
    with XDMFFile(comm, f"{prefix}/v-vs-thickness.xdmf", "w",
                    encoding=XDMFFile.Encoding.HDF5) as file:
        
        _v, _w = q.split()
        _v.name = "potential"
        _w.name = "displacement"
        
        file.write_mesh(mesh)
        interpolation = dolfinx.fem.Function(V_P1)

        interpolation.interpolate(_v)
        file.write_function(interpolation, thickness)  # Specify unique mesh_xpath for velocity

        # interpolation_w.interpolate(_w)
        # file.write_function(interpolation_w, mesh_xpath=f"/fields/{_w.name}")  # Specify unique mesh_xpath for velocity
    
    with XDMFFile(comm, f"{prefix}/w-vs-thickness.xdmf", "w",
                    encoding=XDMFFile.Encoding.HDF5) as file:
        
        _v, _w = q.split()
        _v.name = "potential"
        _w.name = "displacement"
        
        file.write_mesh(mesh)
        interpolation = dolfinx.fem.Function(V_P1)

        interpolation.interpolate(_w)
        file.write_function(interpolation, thickness)  # Specify unique mesh_xpath for velocity

    del solver
    
    # Postprocessing and viz
    with dolfinx.common.Timer(f"~Postprocessing and Vis") as timer:
        energy_components = {"bending": model.energy(state)[1],
                            "membrane": -model.energy(state)[2],
                            "coupling": model.energy(state)[3]}

        computed_energy_terms = {label: comm.allreduce(
            dolfinx.fem.assemble_scalar(
                dolfinx.fem.form(energy_term)),
            op=MPI.SUM,
        ) for label, energy_term in energy_components.items()}
        
        data["membrane_energy"] = computed_energy_terms["membrane"]
        data["bending_energy"] = computed_energy_terms["bending"]
        data["coupling_energy"] = computed_energy_terms["coupling"]
        data["thickness"] = thickness
        data["v_L2"] = comm.allreduce(dolfinx.fem.assemble_scalar(
            dolfinx.fem.form(ufl.inner(v, v) * dx)), op=MPI.SUM)
        data["w_L2"] = comm.allreduce(dolfinx.fem.assemble_scalar(
            dolfinx.fem.form(ufl.inner(w, w) * dx)), op=MPI.SUM)
        
        import pyvista
        from pyvista.plotting.utilities import xvfb

        xvfb.start_xvfb(wait=0.05)
        pyvista.OFF_SCREEN = True

        # import matplotlib.pyplot as plt

        # plt.figure()
        # ax = plot_mesh(mesh)
        # fig = ax.get_figure()
        # fig.savefig(f"{prefix}/mesh.png")
        
        κ = model.gaussian_curvature(w, order = parameters["model"]["order"]-2)
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

        return data

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
    from disclinations.utils import table_timing_data, Visualisation
    _experimental_data = []
    outdir = "output"
    parameters, signature = load_parameters("../../src/disclinations/test_branch/parameters.yml")
    series = signature[0::6]
    experiment_dir = os.path.join(outdir, series)
    max_memory = 0
    num_runs = 10
    if comm.rank == 0:
        Path(experiment_dir).mkdir(parents=True, exist_ok=True)
            
    logging.info(
        f"===================- {experiment_dir} -=================")
    postprocess = Visualisation(experiment_dir)
    
    with dolfinx.common.Timer(f"~Computation Experiment") as timer:
    
        # Mesh is fixed for all runs
        mesh = None
        mesh_size = parameters["geometry"]["mesh_size"]
        tdim = 2
        model_rank = 0
        
        with dolfinx.common.Timer("~Mesh Generation") as timer:
            gmsh_model, tdim = mesh_circle_gmshapi(
                parameters["geometry"]["geom_type"], 1, mesh_size, tdim
            )
            mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, model_rank, tdim)
    
        for i, thickness in enumerate(np.linspace(0.1, 1, num_runs)):
            # Check memory usage before computation
            mem_before = memory_usage()
            
            update_parameters(parameters, "thickness", thickness)  
            print(f"Running experiment for thickness: {thickness}")
            # print(parameters)

            data = run_experiment(mesh, parameters, series)

            _experimental_data.append(data)
            
            # Check memory usage after computation
            mem_after = memory_usage()
            max_memory = max(max_memory, mem_after)
            
            # Log memory usage
            logging.info(f"Run {i}/{num_runs}: Memory Usage (MB) - Before: {mem_before}, After: {mem_after}")
            # Perform garbage collection
            gc.collect()
    
    experimental_data = pd.DataFrame(_experimental_data)
    timings = table_timing_data()


    import matplotlib.pyplot as plt
    
    # Plot energy terms versus thickness
    __import__('pdb').set_trace()
    plt.figure(figsize=(10, 6))
    plt.plot(experimental_data["thickness"], experimental_data["membrane_energy"], label="Membrane Energy")
    plt.plot(experimental_data["thickness"], experimental_data["bending_energy"], label="Bending Energy")
    plt.plot(experimental_data["thickness"], experimental_data["coupling_energy"], label="Coupling Energy")
    plt.xlabel("Thickness")
    plt.ylabel("Energy")
    plt.title("Energy Terms vs Thickness")
    plt.legend()
    plt.savefig(f"{experiment_dir}/energy_terms.png")

    # Plot L2 norm terms versus thickness
    plt.figure(figsize=(10, 6))
    plt.plot(experimental_data["thickness"], experimental_data["w_L2"], label=r"$w_{L^2}$")
    plt.plot(experimental_data["thickness"], experimental_data["v_L2"], label=r"$v_{L^2}$")
    plt.xlabel("Thickness")
    plt.ylabel("L2 Norms")
    plt.title("L2 Norm Terms vs Thickness")
    plt.legend()

    plt.savefig(f"{experiment_dir}/norms_fields.png")

    list_timings(MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])
    postprocess.visualise_results(experimental_data)
    postprocess.save_table(timings, "timing_data")
    postprocess.save_table(experimental_data, "postprocessing_data")
    
    