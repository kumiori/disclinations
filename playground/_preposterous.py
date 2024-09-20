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

