"""ADIOS output smoke test, explicitly skipped without compiled support."""
import pytest
import dolfinx
from dolfinx import fem, mesh, io
from mpi4py import MPI


def test_adios_output(tmp_path):
    if not dolfinx.has_adios2:
        pytest.skip("DOLFINx built without ADIOS2")
    msh = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    V = fem.functionspace(msh, ("Lagrange", 1))
    u = fem.Function(V)
    u.interpolate(lambda x: x[0] + x[1])
    # Every MPI rank must open the same collective output path.
    path = MPI.COMM_WORLD.bcast(
        str(tmp_path / "fields.bp") if MPI.COMM_WORLD.rank == 0 else None, root=0
    )
    with io.VTXWriter(msh.comm, path, [u]) as writer:
        writer.write(0.0)
