import preposterous
from preposterous import core
import dolfinx
import dolfinx
import numpy as np
from dolfinx.io import XDMFFile
from mpi4py import MPI
from petsc4py import PETSc
import ufl
import os
import h5py
import basix
import yaml
from dolfinx.fem import Constant
from dolfinx.fem import Function, functionspace

from disclinations.models import NonlinearPlateFVK

import _preposterous as future
from _preposterous import print_datasets, read_timestep_data


comm = MPI.COMM_WORLD
file_path = os.path.join(os.path.dirname(__file__), 'output/plate_fvk_disclinations_monopole/fields')
xdmf_file_path = file_path + '.xdmf'
h5_file_path = file_path + '.h5'

parameters_path = os.path.join(os.path.dirname(__file__), 'output/plate_fvk_disclinations_monopole/parameters.yml')

with open(parameters_path) as f:
    parameters = yaml.load(f, Loader=yaml.FullLoader)

outdir = "output"

with XDMFFile(comm, xdmf_file_path, "r") as xdmf_file:
    mesh = xdmf_file.read_mesh(name="mesh")

# element_v = basix.ufl.element("P", str(mesh.ufl_cell()), 1)
element_v = basix.ufl.element("P", str(mesh.ufl_cell()), degree=1)

V_v = dolfinx.fem.functionspace(mesh, element_v)
w = dolfinx.fem.Function(V_v, name="Transverse")
v = dolfinx.fem.Function(V_v, name="Airy")
# pxx = dolfinx.fem.Function(V_v, name="Pxx")

# DG_e = basix.ufl.element("DG", str(mesh.ufl_cell()), 
#                         parameters["model"]["order"]-2)
DG_e = basix.ufl.element("DG", str(mesh.ufl_cell()), 
                        parameters["model"]["order"]-2,
                        shape=(2,2))
DG = dolfinx.fem.functionspace(mesh, DG_e)

with h5py.File(h5_file_path, "r") as h5file:
    print_datasets(h5file)

# Read function data from HDF5
with h5py.File(h5_file_path, "r") as h5file:
    v_data = h5file["Function/v"]

    def print_attrs(name, obj):
        print(f"{name}: {obj}")

    h5file.visititems(print_attrs)

with h5py.File(h5_file_path, "r") as h5file:
    timesteps = preposterous.core.get_timesteps(h5file, "v")

print((len(timesteps), timesteps))

read_timestep_data(h5_file_path, w, "w", timesteps[0])
read_timestep_data(h5_file_path, v, "v", timesteps[0])
# read_timestep_data(h5_file_path, mxx, "Mxx", timesteps[0])
# read_timestep_data(h5_file_path, pxx, "Pxx", timesteps[0])


w_norm = w.vector.norm()
v_norm = v.vector.norm()

print(f"Norm of w field: {w_norm:.4e}")
print(f"Norm of v field: {v_norm:.4e}")
print("Plots")
print("Compute energy")
print("Compute curvature")

model = NonlinearPlateFVK(mesh, parameters["model"])
state = {"v": v, "w": w}
dx = ufl.Measure("dx")

W_ext = Constant(mesh, np.array(0., dtype=PETSc.ScalarType)) * w * dx

energy_components = {"bending": model.energy(state)[1], "membrane": -model.energy(state)[2], "coupling": model.energy(state)[3], "external_work": -W_ext}

z = dolfinx.fem.Function(V_v, name="Trial")

from ufl import (avg, ds, dS, outer, div, grad, inner, dot, jump)

form = dolfinx.fem.form(ufl.inner(grad(v), grad(v))*dx)
scalar_value = dolfinx.fem.assemble_scalar(form)

D = model.D
nu = model.nu
Eh = model.E*model.t
k_g = -D*(1-nu)

print(yaml.dump(parameters["model"], default_flow_style=False))

import dolfinx
from dolfinx.io import XDMFFile
from dolfinx.fem import Function, Expression
from ufl import as_tensor
from mpi4py import MPI

def load_and_reconstruct_tensor(mesh, file_path, tensor_name, components, V_scalar, V_tensor, timestep=0):
    """
    Load tensor components from file and reconstruct the tensor field.

    Parameters:
    mesh : dolfinx.mesh.Mesh
        The mesh on which the tensor is defined.
    file_path : str
        Path to the XDMF file containing the tensor components.
    tensor_name : str
        Base name of the tensor (e.g., 'M', 'P').
    components : list of tuples
        List of component names and their indices, e.g., [('xx', (0, 0)), ('xy', (0, 1)), ('yy', (1, 1))].
    V_scalar : dolfinx.fem.FunctionSpace
        Function space for scalar components.
    V_tensor : dolfinx.fem.FunctionSpace
        Function space for the reconstructed tensor field.
    timestep : int
        Timestep to load
        
    Returns:
    tensor_field : dolfinx.fem.Function
        The reconstructed tensor field.
    """
    # Dictionary to hold the scalar functions for tensor components
    components_dict = {}

    # Open the XDMF file for reading
    # with XDMFFile(mesh.comm, file_path, "r") as xdmf_file:
    #     # Loop over the components
    #     for comp_name, (i, j) in components:
    #         # Create a Function to hold the component
    #         comp_function = Function(V_scalar, name=f"{tensor_name}{comp_name}")
    #         # Read the function from the file
    #         __import__('pdb').set_trace()
    #         xdmf_file.read_function(comp_function, f"{tensor_name}{comp_name}")
    #         # Store the component in the dictionary
    #         components_dict[(i, j)] = comp_function
    
    with h5py.File(h5_file_path, "r") as h5file:
        # Loop over the components
        for comp_name, (i, j) in components:
            function_name = f"{tensor_name}{comp_name}"
            dataset_path = f"Function/{function_name}/{timestep}"
            if dataset_path in h5file:
                data = h5file[dataset_path][:]
                # Create a Function to hold the component
                comp_function = Function(V_scalar, name=function_name)
                # Get local ownership range
                local_range = comp_function.function_space.dofmap.index_map.local_range
                # Assign data to the function vector
                local_size = local_range[1] - local_range[0]
                # local_data 
                comp_function.x.array[:local_size] = data[local_range[0]:local_range[1]].flatten()
                # Update ghost values
                comp_function.x.scatter_forward()
                # Store the component in the dictionary
                components_dict[(i, j)] = comp_function
            else:
                print(f"Dataset {dataset_path} not found in the HDF5 file.")
    # Now reconstruct the tensor field
    # Create a Function to hold the tensor field
    tensor_field = Function(V_tensor, name=tensor_name)

    # Build the tensor expression
    tensor_expr_entries = [[None]*2 for _ in range(2)]
    for (i, j), comp_function in components_dict.items():
        tensor_expr_entries[i][j] = comp_function
        # If the tensor is symmetric, set the symmetric entry
        if i != j:
            tensor_expr_entries[j][i] = comp_function  # Assuming symmetry

    # Create the UFL tensor expression
    tensor_expr = as_tensor(tensor_expr_entries)

    # Interpolate the tensor expression into the tensor function
    tensor_interpolation = Expression(tensor_expr, V_tensor.element.interpolation_points())
    tensor_field.interpolate(tensor_interpolation)

    return tensor_field

components = [('xx', (0, 0)), ('xy', (0, 1)), ('yy', (1, 1))]

# Reconstruct tensor M
M = load_and_reconstruct_tensor(mesh, xdmf_file_path, 'M', components, V_v, DG)
__import__('pdb').set_trace()
# Reconstruct tensor P
P = load_and_reconstruct_tensor(mesh, xdmf_file_path, 'P', components, V_v, DG)
