import dolfinx
import ufl
from mpi4py import MPI
from dolfinx.mesh import create_unit_square
from dolfinx.fem import functionspace, form, assemble_scalar, Function
from ufl import div, grad, dx
import basix
from pathlib import Path
from dolfinx.io import VTXWriter
import numpy as np

# Step 1: Create a mesh (unit square domain)
mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)
filename = Path("output", "mesh_vtx.bp")

with VTXWriter(mesh.comm, filename, mesh) as f:
    f.write(0.0)
    mesh.geometry.x[:, 1] += 0.1
    f.write(0.1)

gdim = 2
dtype = np.float32
V = functionspace(mesh, ("DG", 2, (gdim,)))
W = functionspace(mesh, ("DG", 2))

v = Function(V)
w = Function(W)
bs = V.dofmap.index_map_bs

w.interpolate(lambda x: x[0] + x[1])

for c in [0, 1]:
    dofs = np.asarray(
        [V.dofmap.cell_dofs(c) * bs + b for b in range(bs)], dtype=np.int32
    )
    v.x.array[dofs] = 0
    w.x.array[W.dofmap.cell_dofs(c)] = 1
v.x.scatter_forward()
w.x.scatter_forward()

print(v.vector[:])

filename = Path("output", "fields_vtx.bp")

writer = VTXWriter(mesh.comm, filename, [v, w])

# Save twice and update geometry
for t in [0.1, 1]:
    mesh.geometry.x[:, :2] += 0.1
    writer.write(t)

writer.close()
