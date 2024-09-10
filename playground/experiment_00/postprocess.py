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

comm = MPI.COMM_WORLD
file_path = os.path.join(os.path.dirname(__file__), 'output/fields-vs-thickness')
xdmf_file_path = file_path + '.xdmf'
h5_file_path = file_path + '.h5'

outdir = "output"

# Create a new postprocessor

def read_timestep_data(h5_file_path, function, function_name, timestep):
    """
    Read data for a specific timestep from the HDF5 file and update the function vector.

    Parameters:
    h5file (h5py.File): The HDF5 file object.
    function (dolfinx.fem.Function): The function to update.
    function_name (str): The name of the function in the HDF5 file.
    timestep (str): The timestep to read.
    """
    dataset_path = f"Function/{function_name}/{timestep}"
    with h5py.File(h5_file_path, "r") as h5file:
        if dataset_path in h5file:
            data = h5file[dataset_path][:]

            local_range = function.vector.getOwnershipRange()
            local_data = data[local_range[0]:local_range[1]]
            function.vector.setArray(local_data.flatten())
            function.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            # function.vector.setArray(data.flatten())
        else:
            print(f"Timestep {timestep} not found for function {function_name}.")

# Load the mesh and function spaces
with XDMFFile(comm, xdmf_file_path, "r") as xdmf_file:
    mesh = xdmf_file.read_mesh(name="mesh")

# Define the function spaces
element_v = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), degree=1)
V_v = dolfinx.fem.FunctionSpace(mesh, element_v)

def print_datasets(h5file, path='/'):
    """ Recursively prints the names and paths of all datasets in the HDF5 group provided. """
    for key in h5file[path]:
        item_path = f"{path}/{key}" if path != '/' else f"/{key}"
        if isinstance(h5file[item_path], h5py.Dataset):
            # It's a dataset, print its name and path
            print(f"Dataset Name: {key}")
            print(f"Path: {item_path}")
            print('-' * 40)
        elif isinstance(h5file[item_path], h5py.Group):
            # It's a group, recurse into it
            print_datasets(h5file, item_path)

with h5py.File(h5_file_path, "r") as h5file:
    print_datasets(h5file)
    
# Read function data from HDF5
with h5py.File(h5_file_path, "r") as h5file:
    v_data = h5file["Function/v"]

    def print_attrs(name, obj):
        print(f"{name}: {obj}")

    h5file.visititems(print_attrs)
    
    
with h5py.File(h5_file_path, "r") as h5file:
    displacement_timesteps = preposterous.core.get_timesteps(h5file, "v")

# Assign the read data to the functions
w = dolfinx.fem.Function(V_v, name="Transverse")

# Example usage: Read data for a specific timestep
timestep = "1"  # Adjust this to the desired timestep

read_timestep_data(h5_file_path, w, "w", timestep)


# Example postprocessing: Compute norms of the fields
u_norm = w.vector.norm()

print(f"Norm of displacement field u: {u_norm:.4e}")
# Additional postprocessing can be done here, such as visualization or further analysis.