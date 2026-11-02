import hashlib
import json
import logging
import sys
from typing import List
import resource

import mpi4py
import numpy as np
import ufl
import yaml
from dolfinx.fem import assemble_scalar, form
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.io import XDMFFile
import dolfinx
import ufl
from dolfinx.fem import dirichletbc, locate_dofs_topological
import importlib.resources as pkg_resources  # Python 3.7+ for accessing package files

comm = MPI.COMM_WORLD

class ColorPrint:
    """
    Colored printing functions for strings that use universal ANSI escape
    sequences.
        - fail: bold red
        - pass: bold green,
        - warn: bold yellow,
        - info: bold blue
        - color: bold cyan
        - bold: bold white
    """

    @staticmethod
    def print_fail(message, end="\n"):
        if comm.rank == 0:
            message = str(message)
            sys.stderr.write("\x1b[1;31m" + message.strip() + "\x1b[0m" + end)
            sys.stderr.flush()

    @staticmethod
    def print_pass(message, end="\n"):
        if comm.rank == 0:
            message = str(message)
            sys.stdout.write("\x1b[1;32m" + message.strip() + "\x1b[0m" + end)
            sys.stdout.flush()

    @staticmethod
    def print_warn(message, end="\n"):
        if comm.rank == 0:
            message = str(message)
            sys.stderr.write("\x1b[1;33m" + message.strip() + "\x1b[0m" + end)
            sys.stderr.flush()

    @staticmethod
    def print_info(message, end="\n"):
        if comm.rank == 0:
            message = str(message)
            sys.stdout.write("\x1b[1;34m" + message.strip() + "\x1b[0m" + end)
            sys.stdout.flush()

    @staticmethod
    def print_color(message, end="\n"):
        if comm.rank == 0:
            message = str(message)
            sys.stdout.write("\x1b[1;36m" + message.strip() + "\x1b[0m" + end)
            sys.stdout.flush()

    @staticmethod
    def print_bold(message, end="\n"):
        if comm.rank == 0:
            message = str(message)
            sys.stdout.write("\x1b[1;37m" + message.strip() + "\x1b[0m" + end)
            sys.stdout.flush()

def setup_logger_mpi(root_priority: int = logging.INFO):
    import dolfinx
    from mpi4py import MPI
    class MPIFormatter(logging.Formatter):
        def format(self, record):
            record.rank = MPI.COMM_WORLD.Get_rank()
            record.size = MPI.COMM_WORLD.Get_size()
            return super(MPIFormatter, self).format(record)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Desired log level for the root process (rank 0)
    root_process_log_level = logging.INFO  # Adjust as needed

    # logger = logging.getLogger('E•volver')
    logger = logging.getLogger()
    logger.setLevel(root_process_log_level if rank == 0 else logging.WARNING)
    # Disable propagation to root logger for your logger
    logger.propagate = False
    # StreamHandler to log messages to the console
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('disclinations.log')

    # formatter = logging.Formatter('%(asctime)s - %(name)s - [%(levelname)s] - %(message)s')
    formatter = MPIFormatter('%(asctime)s  [Rank %(rank)d, Size %(size)d]  - %(name)s - [%(levelname)s] - %(message)s')

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # file_handler.setLevel(logging.INFO)
    file_handler.setLevel(root_process_log_level if rank == 0 else logging.CRITICAL)
    console_handler.setLevel(root_process_log_level if rank == 0 else logging.CRITICAL)


    # Disable propagation to root logger for both handlers
    console_handler.propagate = False
    file_handler.propagate = False
    
    
    # logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Log messages, and only the root process will log.
    logger.info("The root process spawning an evolution computation (rank 0)")
    logger.info(
    f"DOLFINx version: {dolfinx.__version__} based on GIT commit: {dolfinx.git_commit_hash} of https://github.com/FEniCS/dolfinx/")

    return logger

_logger = logging.getLogger(__name__)

from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import subprocess


def _code_info():
    try:
        root = Path(__file__).resolve().parents[3]
        def git(*args):
            return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL).decode().strip()
        return {"branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                "commit_hash": git("rev-parse", "HEAD")}
    except (OSError, subprocess.CalledProcessError):
        return {"branch": None, "commit_hash": None}


code_info = _code_info()
library_info = {"dolfinx_version": dolfinx.__version__}
for package in ("petsc4py", "slepc4py"):
    try:
        library_info[package+"_version"] = version(package)
    except PackageNotFoundError:
        library_info[package+"_version"] = None


simulation_info = {**library_info, **code_info}

def norm_L2(u):
    """
    Returns the L2 norm of the function u
    """
    comm = u.function_space.mesh.comm
    dx = ufl.Measure("dx", u.function_space.mesh)
    norm_form = form(ufl.inner(u, u) * dx)
    norm = np.sqrt(comm.allreduce(
        assemble_scalar(norm_form), op=mpi4py.MPI.SUM))
    return norm

def norm_H1(u):
    """
    Returns the H1 norm of the function u
    """
    comm = u.function_space.mesh.comm
    dx = ufl.Measure("dx", u.function_space.mesh)
    norm_form = form(
        (ufl.inner(u, u) + ufl.inner(ufl.grad(u), ufl.grad(u))) * dx)
    norm = np.sqrt(comm.allreduce(
        assemble_scalar(norm_form), op=mpi4py.MPI.SUM))
    return norm

def seminorm_H1(u):
    """
    Returns the H1 norm of the function u
    """
    comm = u.function_space.mesh.comm
    dx = ufl.Measure("dx", u.function_space.mesh)
    seminorm = form((ufl.inner(ufl.grad(u), ufl.grad(u))) * dx)
    seminorm = np.sqrt(comm.allreduce(
        assemble_scalar(seminorm), op=mpi4py.MPI.SUM))
    return seminorm

def set_vector_to_constant(x, value):
    with x.localForm() as local:
        local.set(value)
    x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

def table_timing_data(tasks=None):
    import pandas as pd
    from dolfinx.common import timing

    timing_data = []
    if tasks is None:
        tasks = [
            "~Mesh Generation",
            "~First Order: min-max equilibrium",
            "~Postprocessing and Vis",
            "~Computation Experiment"
            ]

    for task in tasks:
        timing_data.append(timing(task))
    
    df = pd.DataFrame(timing_data, columns=["reps", "wall tot", "usr", "sys"], index=tasks)

    return df


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

def save_parameters(parameters, prefix):
    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()
    if MPI.COMM_WORLD.rank == 0:
        with open(f"{prefix}/parameters.yaml", "w") as file:
            yaml.dump(parameters, file)
        with open(f"{prefix}/signature.md5", "w") as f:
            f.write(signature)

    return signature

from dolfinx.io import XDMFFile


class ResultsStorage:
    """
    Class for storing and saving simulation results.
    """

    def __init__(self, comm, prefix):
        self.comm = comm
        self.prefix = prefix

    def store_results(self, parameters, history_data, state):
        """
        Store simulation results in XDMF and JSON formats.

        Args:
            history_data (dict): Dictionary containing simulation data.
        """
        t = history_data["load"][-1]

        u = state["u"]
        alpha = state["alpha"]

        if self.comm.rank == 0:
            with open(f"{self.prefix}/parameters.yaml", 'w') as file:
                yaml.dump(parameters, file)

        with XDMFFile(self.comm, f"{self.prefix}/simulation_results.xdmf", "w", encoding=XDMFFile.Encoding.HDF5) as file:
            # for t, data in history_data.items():
                # file.write_scalar(data, t)
            file.write_mesh(u.function_space.mesh)

            file.write_function(u, t)
            file.write_function(alpha, t)

        if self.comm.rank == 0:
            with open(f"{self.prefix}/time_data.json", "w") as file:
                json.dump(history_data, file)

# Visualization functions/classes

class Visualisation:
    """
    Class for visualizing simulation results.
    """

    def __init__(self, prefix):
        self.prefix = prefix

    def visualise_results(self, df, drop=[]):
        """
        Visualise simulation results using appropriate visualization libraries.

        Args:
            df (dict): Pandas dataframe containing simulation data.
        """
        # Implement visualization code here
        print(df.drop(drop, axis=1))

    def save_table(self, data, name):
        """
        Save pandas table results using json.

        Args:
            data (dict): Pandas table containing simulation data.
            name (str): Filename.
        """

        if MPI.COMM_WORLD.rank == 0:
            data.to_json(f"{self.prefix}/{name}.json")
            # a_file = open(f"{self.prefix}/{name}.json", "w")
            # json.dump(data.to_json(), a_file)
            # a_file.close()

history_data = {
    "load": [],
    "elastic_energy": [],
    "fracture_energy": [],
    "total_energy": [],
    "equilibrium_data": [],
    "cone_data": [],
    "eigs_ball": [],
    "eigs_cone": [],
    "stable": [],
    "unique": [],
    "inertia": [],
}

def _write_history_data(equilibrium, bifurcation, stability, history_data, t, inertia, stable, energies: List):
    
    elastic_energy = energies[0]
    fracture_energy = energies[1]
    unique = True if inertia[0] == 0 and inertia[1] == 0 else False
    
    history_data["load"].append(t)
    history_data["fracture_energy"].append(fracture_energy)
    history_data["elastic_energy"].append(elastic_energy)
    history_data["total_energy"].append(elastic_energy+fracture_energy)
    history_data["equilibrium_data"].append(equilibrium.data)
    history_data["inertia"].append(inertia)
    history_data["unique"].append(unique)
    history_data["stable"].append(stable)
    history_data["eigs_ball"].append(bifurcation.data["eigs"])
    history_data["cone_data"].append(stability.data)
    history_data["eigs_cone"].append(stability.solution["lambda_t"])

    return 

def indicator_function(v):
    import dolfinx

    # Create the indicator function
    w = dolfinx.fem.Function(v.function_space)
    with w.x.petsc_vec.localForm() as w_loc, v.x.petsc_vec.localForm() as v_loc:
        w_loc[:] = np.where(v_loc[:] > 0, 1., 0.)

    w.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT,
        mode=PETSc.ScatterMode.FORWARD)
    
    return w

def update_parameters(parameters, key, value):
    """
    Recursively traverses the dictionary d to find and update the key's value.
    
    Args:
    d (dict): The dictionary to traverse.
    key (str): The key to find and update.
    value: The new value to set for the key.
    
    Returns:
    bool: True if the key was found and updated, False otherwise.
    """
    if key in parameters:
        parameters[key] = value
        return True

    for k, v in parameters.items():
        if isinstance(v, dict):
            if update_parameters(v, key, value):
                return True
    
    return False

def memory_usage():
    """Get the current memory usage of the Python process."""
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return mem / 1024  # Convert to MB

def monitor(snes, it, norm):
    logging.info(f"Iteration {it}, residual {norm}")
    print(f"Iteration {it}, residual {norm}")
    return PETSc.SNES.ConvergedReason.ITERATING

def write_to_output(prefix, q, extra_fields = {}):
    # Obtain the communicator from q's mesh
    import basix
    comm = q.function_space.mesh.comm
    
    # V_P1 = 
    # create function space based on mesh
    V_P1 = dolfinx.fem.functionspace(q.function_space.mesh, ("CG", 1))
    DG_e = basix.ufl.element("DG", str(q.function_space.mesh.ufl_cell()), order=1)
    V_DG = dolfinx.fem.functionspace(q.function_space.mesh, DG_e)  # 
    
    with XDMFFile(comm, f"{prefix}/fields.xdmf", "a", encoding=XDMFFile.Encoding.HDF5) as file:
        # Split q into components: potential (_v) and displacement (_w)
        _v, _w = q.split()
        _v.name, _w.name = "potential", "displacement"

        # Create an interpolation function
        interpolation = dolfinx.fem.Function(V_P1)

        # Interpolate and write potential (_v)
        interpolate_and_write(file, interpolation, _v, "v")

        # Interpolate and write displacement (_w)
        interpolate_and_write(file, interpolation, _w, "w")

        # Process extra fields
        for field_info in extra_fields:
            field = field_info['field']
            name = field_info['name']

            # Check if the field requires special handling (e.g., tensor components)
            if field_info.get('components') == 'tensor':
                interpolation = dolfinx.fem.Function(V_DG)
                
                write_tensor_components(file, interpolation, field, name)
            else:
                # Interpolate and write scalar or vector fields
                interpolate_and_write(file, interpolation, field, name)


def interpolate_and_write(file, interpolation, field, name):
    """Interpolate a field to V_P1 and write it to an XDMF file."""
    interpolation.interpolate(field)
    interpolation.name = name
    file.write_function(interpolation)

def write_tensor_components(file, interpolation, tensor_expr, tensor_name):
    """Interpolate and write specified components of a tensor expression."""
    components = {
        'xx': (0, 0),
        'xy': (0, 1),
        'yy': (1, 1)
    }
    for comp_name, (i, j) in components.items():
        # Extract the component expression
        component_expr = tensor_expr[i, j]
        # Create an expression for interpolation
        expr = dolfinx.fem.Expression(component_expr, interpolation.function_space.element.interpolation_points())
        # Create a function to hold the interpolated values
        interpolation.interpolate(expr)
        # Set the name accordingly
        interpolation.name = f"{tensor_name}{comp_name}"
        # Write to file
        file.write_function(interpolation)

AIRY = 0
TRANSVERSE = 1

def homogeneous_dirichlet_bc_H20(mesh, Q):
    """
    Impose zero value traces on both fields, on all exterior facets.

    This supplies only the H1 trace. Zero normal slope requires the model's
    separate Nitsche terms. On holes this fixes all affine Airy gauges to
    zero; it is not a general traction-free multiply connected formulation.

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

def save_params_to_yaml(params, filename):
    """
    Save the updated params dictionary to a YAML file.

    Args:
    - params (dict): Dictionary containing all parameters.
    - filename (str): Path to the YAML file to save.
    """
    with open(filename, "w") as file:
        yaml.dump(params, file, default_flow_style=False)

import hashlib
import yaml

def parameters_vs_thickness(parameters=None, thickness=1.):
    """
    Update the model parameters for a given value of 'ell'.

    This function modifies the 'thickness' parameter. 
    If no parameters are provided, it loads them from the default file.

    Args:
        parameters (dict, optional): Dictionary of parameters. 
                                      If None, load from "../test/parameters.yml".
        thickness (float, optional): The new 'thickness' value to set in the parameters. 
                               Default is 1.

    Returns:
        tuple: A tuple containing the updated parameters dictionary and 
               a unique hash (signature) based on the updated parameters.
    """
    if parameters is None:    
        # with open("../test/parameters.yml") as f:
            # parameters = yaml.load(f, Loader=yaml.FullLoader)
        with pkg_resources.path('disclinations.test', 'parameters.yml') as f:
            with open(f, 'r') as yaml_file:
                parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)
                
    parameters["model"]["thickness"] = thickness

    # Generate a unique signature using MD5 hash based on the updated parametersx
    signature = hashlib.md5(str(parameters).encode('utf-8')).hexdigest()

    return parameters, signature

from disclinations.meshes.primitives import mesh_circle_gmshapi
from dolfinx.io import XDMFFile, gmshio
from pathlib import Path
import os

def create_or_load_circle_mesh(parameters, prefix):
    """Create/cache a circle without altering radius or discarding mesh tags.

    Collective on MPI.COMM_WORLD. Geometry (including radius/mesh size) keys
    determine the cache identity. Both fresh and cached calls return mesh,
    cell tags and facet tags. Gmsh remains alive for the caller to finalise.
    """
    geometry = dict(parameters["geometry"])
    radius = geometry.setdefault("radius", 1.)
    geometry["geom_type"] = "circle"
    signature = hashlib.sha256(json.dumps({"cache_schema": "circle-tags/v1", "geometry": geometry}, sort_keys=True).encode()).hexdigest()
    path = f"{prefix}/mesh-{signature}.xdmf"
    if comm.rank == 0:
        Path(prefix).mkdir(parents=True, exist_ok=True)
    comm.barrier()
    exists = comm.bcast(os.path.exists(path) if comm.rank == 0 else None, root=0)
    if exists:
        with XDMFFile(comm, path, "r") as file:
            mesh = file.read_mesh()
            mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
            cells = file.read_meshtags(mesh, "cells")
            facets = file.read_meshtags(mesh, "facets")
        return mesh, cells, facets
    model, tdim = mesh_circle_gmshapi("circle", radius, geometry["mesh_size"], 2)
    mesh, cells, facets = gmshio.model_to_mesh(model, comm, 0, gdim=tdim)
    cells.name, facets.name = "cells", "facets"
    with XDMFFile(comm, path, "w") as file:
        file.write_mesh(mesh)
        file.write_meshtags(cells, mesh.geometry)
        file.write_meshtags(facets, mesh.geometry)
    return mesh, cells, facets
