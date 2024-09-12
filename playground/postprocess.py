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

with h5py.File(h5_file_path, "r") as h5file:
    print_datasets(h5file)

__import__('pdb').set_trace()

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

laplacian = lambda f : div(grad(f))
hessian = lambda f : grad(grad(f))


element_z = basix.ufl.element("P", str(mesh.ufl_cell()), degree=2)
V_z = functionspace(mesh, element_z)
z = Function(V_z)

z.interpolate(v)
__import__('pdb').set_trace()

bending = (D/2 * (ufl.inner(laplacian(w), laplacian(w)))) * dx
membrane = (1/(2*Eh) * inner(grad(v), grad(v))) * dx
membrane = (1/(2*Eh) * inner(laplacian(v), laplacian(v))) * dx

dolfinx.fem.form(bending)
dolfinx.fem.form(membrane)

energy_terms = {label: comm.allreduce(
    dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(energy_term)),
    op=MPI.SUM,
    ) for label, energy_term in energy_components.items()}

exact_energy_monopole = parameters["model"]["E"] * parameters["geometry"]["radius"]**2 / (32 * np.pi)

print(yaml.dump(parameters["model"], default_flow_style=False))

error = np.abs(exact_energy_monopole - energy_terms['membrane'])

print(f"Exact energy: {exact_energy_monopole}")
print(f"Computed energy: {energy_terms['membrane']}")
print(f"Abs error: {error}")
print(f"Rel error: {error/exact_energy_monopole:.3%}")
print(f"Error: {error/exact_energy_monopole:.1%}")

