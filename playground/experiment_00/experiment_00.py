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

from disclinations.models import (NonlinearPlateFVK, NonlinearPlateFVK_brenner,
                                  NonlinearPlateFVK_carstensen,
                                  calculate_rescaling_factors,
                                  compute_energy_terms,
                                  initialise_exact_solution_compatible_transverse)

from disclinations.solvers import SNESSolver
from disclinations.utils.la import compute_disclination_loads
from disclinations.utils.viz import plot_scalar, plot_profile, plot_mesh
from disclinations.utils import update_parameters, memory_usage, save_parameters, save_params_to_yaml
from disclinations.utils import create_or_load_circle_mesh, basic_postprocess
import importlib.resources as pkg_resources  # Python 3.7+ for accessing package files
import copy
from disclinations.utils import _logger, print_energy_analysis
import pytest

models = ["variational", "brenner", "carstensen"]

logging.basicConfig(level=logging.INFO)
comm = MPI.COMM_WORLD


def postprocess_and_visualize(model, state, exact_solution, data, prefix, parameters):
    with dolfinx.common.Timer(f"~Postprocessing and Vis") as timer:
        v, w = state["v"], state["w"]
        
        energy_components = {
            "bending": model.energy(state)[1],
            "membrane": -model.energy(state)[2],
            "coupling": model.energy(state)[3]
        }
        penalisation_components = {
            "dgv": model._dgv,
            "dgw": model._dgw,
            "dgc": model._dgc,
            "bcv": model._bcv,
            "bcw": model._bcw
        }
        computed_energy_terms = {
            label: comm.allreduce(
                dolfinx.fem.assemble_scalar(dolfinx.fem.form(energy_term)),
                op=MPI.SUM,
            ) for label, energy_term in energy_components.items()
        }

        computed_penalisation_terms = {
            label: comm.allreduce(
                dolfinx.fem.assemble_scalar(dolfinx.fem.form(penalisation_term)),
                op=MPI.SUM,
            ) for label, penalisation_term in penalisation_components.items()
        }
        
        exact_energy_transverse_load = comm.allreduce(
            dolfinx.fem.assemble_scalar(
                dolfinx.fem.form(model.energy(exact_solution)[0])),
            op=MPI.SUM)
        
        abs_error, rel_error = print_energy_analysis(
            computed_energy_terms, exact_energy_transverse_load
        )

        data["membrane_energy"] = computed_energy_terms["membrane"]
        data["bending_energy"] = computed_energy_terms["bending"]
        data["coupling_energy"] = computed_energy_terms["coupling"]
        
        data["v_L2"] = comm.allreduce(dolfinx.fem.assemble_scalar(
            dolfinx.fem.form(ufl.inner(v, v) * dx)), op=MPI.SUM)
        data["w_L2"] = comm.allreduce(dolfinx.fem.assemble_scalar(
            dolfinx.fem.form(ufl.inner(w, w) * dx)), op=MPI.SUM)

        data["∇²v_L2"] = comm.allreduce(dolfinx.fem.assemble_scalar(
            dolfinx.fem.form(ufl.inner(v, v) * dx)), op=MPI.SUM)
        data["∇²w_L2"] = comm.allreduce(dolfinx.fem.assemble_scalar(
            dolfinx.fem.form(ufl.inner(w, w) * dx)), op=MPI.SUM)

        import pyvista
        from pyvista.plotting.utilities import xvfb

        xvfb.start_xvfb(wait=0.05)
        pyvista.OFF_SCREEN = True

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
def run_experiment(mesh, parameters, experiment_dir, variant = "variational", initial_guess=None):

    # Setup, output and file handling
    _logger.info(f"Running experiment of the series {series} with parameter: {parameters['model']['a_adim']}")
    parameters = calculate_rescaling_factors(parameters)

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

    exact_solution = initialise_exact_solution_compatible_transverse(Q, parameters)

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
    # W_transv_coeff = (radius / thickness)**4 / E 
    W_ext = parameters["model"]["a_adim"]**4 * f * w * dx
    
    model = NonlinearPlateFVK(mesh, parameters["model"], adimensional=True)
    energy = model.energy(state)[0]
    model.W_ext = W_ext
    penalisation = model.penalisation(state)

    # Define the functional
    L = energy - W_ext + penalisation
    F = ufl.derivative(L, q, ufl.TestFunction(Q))
    
    save_params_to_yaml(parameters, os.path.join(prefix, "parameters.yml"))
    if MPI.COMM_WORLD.rank == 0:
        with open(f"{prefix}/signature.md5", "w") as f:
            f.write(signature)

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

    energy_terms = basic_postprocess(
        state, model, mesh, params=parameters,
        exact_solution=None, 
        prefix=prefix
    )

    convergence_reason = solver.solver.getConvergedReason()
    num_iterations = solver.solver.getIterationNumber()
    residual_norm = solver.solver.getFunctionNorm()
    num_function_evals = solver.solver.getFunctionEvaluations()

    v_L2 = comm.allreduce(dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(ufl.inner(v, v) * dx)), op=MPI.SUM)
    w_L2 = comm.allreduce(dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(ufl.inner(w, w) * dx)), op=MPI.SUM)

    _simulation_data = {
        "signature": signature,
        "series": series,
        "parameter": parameters["model"]["thickness"],
        "bending_energy": energy_terms['bending'],
        "membrane_energy": energy_terms['membrane'],
        "coupling_energy": energy_terms['coupling'],
        "external_work": energy_terms['external_work'],
        "v_L2": v_L2,
        "w_L2": w_L2,
        "convergence_reason": convergence_reason,
        "iterations": num_iterations,
        "residual_norm": residual_norm,
        "function_evaluations": num_function_evals,
    }

    V_P1 = dolfinx.fem.functionspace(mesh, ("CG", 1))  # "CG" stands for continuous Galerkin (Lagrange)
    
    with XDMFFile(comm, os.path.join(outdir, series, "fields-vs-thickness.xdmf"), "a",
                    encoding=XDMFFile.Encoding.HDF5) as file:
        
        _v, _w = q.split()
        _v.name = "potential"
        _w.name = "displacement"
        
        interpolation = dolfinx.fem.Function(V_P1)
        print(f"saving interpolation, {parameters['model']['a_adim']}")

        interpolation.interpolate(_v)
        interpolation.name = 'v'
        
        file.write_function(interpolation, parameters['model']['a_adim'])  # Specify unique mesh_xpath for velocity

        interpolation.interpolate(_w)
        interpolation.name = 'w'
        file.write_function(interpolation, parameters['model']['a_adim'])  # Specify unique mesh_xpath for velocity
    
    del solver

    # Call the new function in the main code
    postprocess_and_visualize(model, state, exact_solution, data, prefix, parameters)

    return energy_terms, q, _simulation_data

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
        "snes_max_it": 30,           # Maximum number of iterations
        "snes_rtol": 1e-6,            # Relative tolerance for convergence
        "snes_atol": 1e-6,           # Absolute tolerance for convergence
        "snes_stol": 1e-5,           # Tolerance for the change in solution norm
        "snes_monitor": None,         # Function for monitoring convergence (optional)
        "snes_linesearch_type": "basic",  # Type of line search
    }

    parameters["model"]["higher_regularity"] = True

    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()
    print(yaml.dump(parameters, default_flow_style=False))

    return parameters, signature

if __name__ == "__main__":
    from disclinations.utils import table_timing_data, Visualisation
    _experimental_data = []
    outdir = "output"
    prefix = outdir
    _simulation_data = []
    
    with pkg_resources.path('disclinations.test', 'parameters.yml') as f:
        # parameters = yaml.load(f, Loader=yaml.FullLoader)
        parameters, _ = load_parameters(f)

        base_parameters = copy.deepcopy(parameters)
        # Remove thickness from the parameters, to compute the parametric series signature
        if "model" in base_parameters and "thickness" in base_parameters["model"]:
            del base_parameters["model"]["thickness"]
        
        base_signature = hashlib.md5(str(base_parameters).encode('utf-8')).hexdigest()

    series = base_signature[0::6]

    experiment_dir = os.path.join(outdir, series)
    num_runs = 11
    
    if comm.rank == 0:
        Path(experiment_dir).mkdir(parents=True, exist_ok=True)
            
    logging.info(
        f"===================- {experiment_dir} -=================")
    _logger.info(f"Running {num_runs} experiments for series {series}")
    mesh, mts, fts = create_or_load_circle_mesh(parameters, prefix=prefix)


    with dolfinx.common.Timer("~Computation Experiment") as timer:
    
        # Mesh is fixed for all runs
        model_rank = 0
        initial_guess = None
        
        with XDMFFile(comm, os.path.join(outdir, series, "fields-vs-thickness.xdmf"), "w",
                    encoding=XDMFFile.Encoding.HDF5) as file:
            file.write_mesh(mesh)
        
        for i, a in enumerate(np.logspace(np.log10(10), np.log10(1000), num=30)):
            parameters['model']['a_adim'] = None
            parameters['model']['c_adim'] = 1

            # Check memory usage before computation
            mem_before = memory_usage()
            
            if changed := update_parameters(parameters, "a_adim", float(a)):
                signature = hashlib.md5(str(parameters).encode('utf-8')).hexdigest()
                energy_terms, initial_guess, simulation_data = run_experiment(mesh, parameters, experiment_dir,
                                                             initial_guess=None)
            else: 
                abs_error, rel_error = None, None
                raise ValueError("Failed to update parameters")
            
            _data = {
                "a_adim": a,
                "signature": signature,
            }

            _experimental_data.append(_data)
            _simulation_data.append(simulation_data)
            
            # Check memory usage after computation
            mem_after = memory_usage()
            
            # Log memory usage
            _logger.info(f"Run {i}/{num_runs}: Memory Usage (MB) - Before: {mem_before}, After: {mem_after}")
            # Perform garbage collection
            gc.collect()
    
    experimental_data = pd.DataFrame(_simulation_data)
    timings = table_timing_data()
    _logger.info(f"Saving experimental data to {experiment_dir}")


    import matplotlib.pyplot as plt
    
    # Plot energy terms versus thickness
    plt.figure(figsize=(10, 6))
    plt.plot(experimental_data["parameter"], experimental_data["membrane_energy"], label="Membrane Energy", marker='o')
    plt.plot(experimental_data["parameter"], experimental_data["bending_energy"], label="Bending Energy", marker='o')
    plt.plot(experimental_data["parameter"], experimental_data["coupling_energy"], label="Coupling Energy", marker='o')
    plt.xlabel("Thickness")
    plt.ylabel("Energy")
    plt.title("Energy Terms vs Thickness")
    plt.legend()
    plt.savefig(f"{experiment_dir}/energy_terms.png")

    # Plot L2 norm terms versus thickness
    plt.figure(figsize=(10, 6))
    plt.plot(experimental_data["parameter"], experimental_data["w_L2"], label=r"$w_{L^2}$", marker='o')
    plt.plot(experimental_data["parameter"], experimental_data["v_L2"], label=r"$v_{L^2}$", marker='o')
    plt.xlabel("Thickness")
    plt.ylabel("L2 Norms")
    plt.title("L2 Norm Terms vs Thickness")
    plt.legend()

    plt.savefig(f"{experiment_dir}/norms_fields.png")

    list_timings(MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])
    
    Visualisation(experiment_dir).visualise_results(experimental_data)
    Visualisation(experiment_dir).save_table(timings, "timing_data")
    Visualisation(experiment_dir).save_table(experimental_data, "postprocessing_data")
    
    