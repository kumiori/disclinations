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
    file_handler = logging.FileHandler("disclinations.log")

    # formatter = logging.Formatter('%(asctime)s - %(name)s - [%(levelname)s] - %(message)s')
    formatter = MPIFormatter(
        "%(asctime)s  [Rank %(rank)d, Size %(size)d]  - %(name)s - [%(levelname)s] - %(message)s"
    )

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
        f"DOLFINx version: {dolfinx.__version__} based on GIT commit: {dolfinx.git_commit_hash} of https://github.com/FEniCS/dolfinx/"
    )

    return logger


_logger = setup_logger_mpi()

import subprocess

# Get the current Git branch
branch = (
    subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    .strip()
    .decode("utf-8")
)

# Get the current Git commit hash
commit_hash = (
    subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
)

code_info = {
    "branch": branch,
    "commit_hash": commit_hash,
}

from dolfinx import __version__ as dolfinx_version
from petsc4py import __version__ as petsc_version
from slepc4py import __version__ as slepc_version

library_info = {
    "dolfinx_version": dolfinx_version,
    "petsc4py_version": petsc_version,
    "slepc4py_version": slepc_version,
}

simulation_info = {
    **library_info,
    **code_info,
}


def norm_L2(u):
    """
    Returns the L2 norm of the function u
    """
    comm = u.function_space.mesh.comm
    dx = ufl.Measure("dx", u.function_space.mesh)
    norm_form = form(ufl.inner(u, u) * dx)
    norm = np.sqrt(comm.allreduce(assemble_scalar(norm_form), op=mpi4py.MPI.SUM))
    return norm


def norm_H1(u):
    """
    Returns the H1 norm of the function u
    """
    comm = u.function_space.mesh.comm
    dx = ufl.Measure("dx", u.function_space.mesh)
    norm_form = form((ufl.inner(u, u) + ufl.inner(ufl.grad(u), ufl.grad(u))) * dx)
    norm = np.sqrt(comm.allreduce(assemble_scalar(norm_form), op=mpi4py.MPI.SUM))
    return norm


def seminorm_H1(u):
    """
    Returns the H1 norm of the function u
    """
    comm = u.function_space.mesh.comm
    dx = ufl.Measure("dx", u.function_space.mesh)
    seminorm = form((ufl.inner(ufl.grad(u), ufl.grad(u))) * dx)
    seminorm = np.sqrt(comm.allreduce(assemble_scalar(seminorm), op=mpi4py.MPI.SUM))
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
            "~Computation Experiment",
        ]

    for task in tasks:
        timing_data.append(timing(task))

    df = pd.DataFrame(
        timing_data, columns=["reps", "wall tot", "usr", "sys"], index=tasks
    )

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
            with open(f"{self.prefix}/parameters.yaml", "w") as file:
                yaml.dump(parameters, file)

        with XDMFFile(
            self.comm,
            f"{self.prefix}/simulation_results.xdmf",
            "w",
            encoding=XDMFFile.Encoding.HDF5,
        ) as file:
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


def _write_history_data(
    equilibrium,
    bifurcation,
    stability,
    history_data,
    t,
    inertia,
    stable,
    energies: List,
):

    elastic_energy = energies[0]
    fracture_energy = energies[1]
    unique = True if inertia[0] == 0 and inertia[1] == 0 else False

    history_data["load"].append(t)
    history_data["fracture_energy"].append(fracture_energy)
    history_data["elastic_energy"].append(elastic_energy)
    history_data["total_energy"].append(elastic_energy + fracture_energy)
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
    with w.vector.localForm() as w_loc, v.vector.localForm() as v_loc:
        w_loc[:] = np.where(v_loc[:] > 0, 1.0, 0.0)

    w.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

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


def write_to_output(prefix, q, extra_fields={}):
    # Obtain the communicator from q's mesh
    import basix

    comm = q.function_space.mesh.comm

    # V_P1 =
    # create function space based on mesh
    V_P1 = dolfinx.fem.functionspace(q.function_space.mesh, ("CG", 1))
    DG_e = basix.ufl.element("DG", str(q.function_space.mesh.ufl_cell()), order=1)
    V_DG = dolfinx.fem.functionspace(q.function_space.mesh, DG_e)  #

    with XDMFFile(
        comm, f"{prefix}/fields.xdmf", "a", encoding=XDMFFile.Encoding.HDF5
    ) as file:
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
            field = field_info["field"]
            name = field_info["name"]

            # Check if the field requires special handling (e.g., tensor components)
            if field_info.get("components") == "tensor":
                interpolation = dolfinx.fem.Function(V_DG)

                write_tensor_components(file, interpolation, field, name)
            else:
                # Interpolate and write scalar or vector fields
                interpolate_and_write(file, interpolation, field, name)
        __import__("pdb").set_trace()


def interpolate_and_write(file, interpolation, field, name):
    """Interpolate a field to V_P1 and write it to an XDMF file."""
    interpolation.interpolate(field)
    interpolation.name = name
    file.write_function(interpolation)


def write_tensor_components(file, interpolation, tensor_expr, tensor_name):
    """Interpolate and write specified components of a tensor expression."""
    components = {"xx": (0, 0), "xy": (0, 1), "yy": (1, 1)}
    for comp_name, (i, j) in components.items():
        # Extract the component expression
        component_expr = tensor_expr[i, j]
        # Create an expression for interpolation
        expr = dolfinx.fem.Expression(
            component_expr, interpolation.function_space.element.interpolation_points()
        )
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


def parameters_vs_thickness(parameters=None, thickness=1.0):
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
        with pkg_resources.path("disclinations.test", "parameters.yml") as f:
            with open(f, "r") as yaml_file:
                parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)

    parameters["model"]["thickness"] = thickness

    # Generate a unique signature using MD5 hash based on the updated parametersx
    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()

    return parameters, signature


from disclinations.meshes.primitives import mesh_circle_gmshapi
from dolfinx.io import XDMFFile, gmshio
from pathlib import Path
import os


def create_or_load_circle_mesh(parameters, prefix):
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
    hash = hashlib.md5(str(parameters).encode('utf-8')).hexdigest()
    # Set up file prefix for mesh storage
    mesh_file_path = f"{prefix}/mesh-{hash}.xdmf"
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
            # os.makedirs(prefix, exist_ok=True)
            if comm.rank == 0:
                Path(prefix).mkdir(parents=True, exist_ok=True)

            with XDMFFile(
                comm, mesh_file_path, "w", encoding=XDMFFile.Encoding.HDF5
            ) as file:
                file.write_mesh(mesh)

        return mesh, mts, fts


def initialise_exact_solution_dipole(Q, params):
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
    v_scale = params["model"]["v_scale"]
    w_scale = params["model"]["w_scale"]

    q_exact = dolfinx.fem.Function(Q)
    v_exact, w_exact = q_exact.split()
    distance = np.linalg.norm(
        np.array(params['loading']['points'][0]) - np.array(params['loading']['points'][1])
        )

    # compute distance between disclinations

    def _v_exact(x):
        rq = (x[0]**2 + x[1]**2)
        _v = (1/(16*np.pi))*( ( x[1]**2 + (x[0]-distance/2)**2 )*( np.log(4.0) + np.log( x[1]**2 + (x[0]-distance/2)**2 ) - np.log( 4 - 4*x[0]*distance + rq*distance**2 ) ) - (1/4)*( 4*(x[1]**2) + ( 2*x[0]+distance)**2 ) * ( np.log(4) + np.log( x[1]**2 + (x[0]+distance/2)**2 ) - np.log( 4 + 4*x[0]*distance + rq*distance**2 ) ) )
        return _v * v_scale

    def _w_exact(x):
        _w = (1 - x[0]**2 - x[1]**2)**2
        _w = 0.0*_w
        return _w

    v_exact.interpolate(_v_exact)
    w_exact.interpolate(_w_exact)

    return v_exact, w_exact


def initialise_exact_solution_transverse(Q, params):
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
    v_scale = params["model"]["v_scale"]
    w_scale = params["model"]["w_scale"]

    q_exact = dolfinx.fem.Function(Q)
    v_exact, w_exact = q_exact.split()
    distance = np.linalg.norm(params['loading']['points'][0] - params['loading']['points'][1])

    # compute distance between disclinations

    def _v_exact(x):
        rq = x[0] ** 2 + x[1] ** 2

        a1 = -1 / 12
        a2 = -1 / 18
        a3 = -1 / 24

        _v = a1 * rq**2 + a2 * rq**3 + a3 * rq**4

        return _v * v_scale  # Apply scaling

    def _w_exact(x):
        w_scale = params["model"]["w_scale"]
        return (
            w_scale * (1 - x[0] ** 2 - x[1] ** 2) ** 2
        )  # Zero function as per your code

    v_exact.interpolate(_v_exact)
    w_exact.interpolate(_w_exact)

    return v_exact, w_exact

def exact_energy_dipole(parameters):
    # it should depend on the signs as well
    distance = np.linalg.norm(
        np.array(parameters['loading']['points'][0]) - np.array(parameters['loading']['points'][1]))
    
    return parameters["model"]["E"] * parameters["model"]["thickness"]**3 \
        * parameters["geometry"]["radius"]**2 / (8 * np.pi) *  distance**2 * \
            (np.log(4+distance**2) - np.log(4 * distance))


def _transverse_load_polynomial_analytic(x, params):
    f_scale = params["model"]["f_scale"]
    _p = (40/3) * (1 - x[0]**2 - x[1]**2)**4 + (16/3) * (11 + x[0]**2 + x[1]**2)
    return f_scale * _p


from disclinations.models import compute_energy_terms

def basic_postprocess(state, model, mesh, params, exact_solution, prefix):
    with dolfinx.common.Timer(f"~Postprocessing and Vis") as timer:
        energy_components = {
            "bending": model.energy(state)[1],
            "membrane": model.energy(state)[2],
            "coupling": model.energy(state)[3],
            "external_work": -model.W_ext,
        }

        energy_terms = compute_energy_terms(energy_components, mesh.comm)
        print(yaml.dump(params["model"], default_flow_style=False))

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
            {
                "field": model.gaussian_curvature(state["w"]),  # Tensor expression
                "name": "Kappa",
                "components": "tensor",
            },
        ]
        # write_to_output(prefix, q, extra_fields)
        return energy_terms