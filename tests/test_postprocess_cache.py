import numpy as np
import pytest
from dolfinx import fem, mesh
from mpi4py import MPI
from disclinations.utils.sample_function import sample_function
from disclinations.utils import create_or_load_circle_mesh


def test_sampling_nonunit_radius_and_missing_points():
    msh = mesh.create_rectangle(MPI.COMM_WORLD, [[-2.0, -2.0], [2.0, 2.0]], [4, 4])
    V = fem.functionspace(msh, ("Lagrange", 1))
    u = fem.Function(V)
    u.interpolate(lambda x: 2 * x[0] - x[1])
    x, y, z = map(np.asarray, sample_function(u, 2.0, num_points=9))
    assert len(x) == len(y) == len(z) == 81
    assert np.max(x * x + y * y) <= 4.0 + 1e-12
    assert np.max(abs(z - (2 * x - y))) < 1e-12
    # Sampling a larger disk drops locations outside the FE mesh consistently.
    x, y, z = map(np.asarray, sample_function(u, 3.0, num_points=9))
    assert 0 < len(x) < 81 and len(x) == len(y) == len(z)
    assert np.max(abs(z - (2 * x - y))) < 1e-12


def test_mesh_cache_preserves_radius_and_tags(tmp_path):
    import gmsh

    comm = MPI.COMM_WORLD
    folder = comm.bcast(str(tmp_path) if comm.rank == 0 else None, root=0)
    params = {"geometry": {"radius": 2.0, "mesh_size": 0.35, "geom_type": "circle"}}
    fresh, cells1, facets1 = create_or_load_circle_mesh(params, folder)
    cached, cells2, facets2 = create_or_load_circle_mesh(params, folder)
    assert params["geometry"]["radius"] == 2.0
    area = lambda m: comm.allreduce(
        fem.assemble_scalar(fem.form(fem.Constant(m, 1.0) * __import__("ufl").dx)),
        op=MPI.SUM,
    )
    assert area(fresh) == pytest.approx(4 * np.pi, rel=0.03)
    assert area(cached) == pytest.approx(area(fresh))
    assert cells1 is not None and cells2 is not None
    assert facets1 is not None and facets2 is not None
    assert comm.allreduce(len(facets1.find(6)), op=MPI.SUM) > 0

    def owned(m, tags):
        return int(np.count_nonzero(tags.find(6) < m.topology.index_map(1).size_local))

    assert comm.allreduce(owned(fresh, facets1), op=MPI.SUM) == comm.allreduce(
        owned(cached, facets2), op=MPI.SUM
    )
    import ufl

    def boundary_length(m, tags):
        ds = ufl.Measure("ds", domain=m, subdomain_data=tags)
        return comm.allreduce(
            fem.assemble_scalar(fem.form(fem.Constant(m, 1.0) * ds(6))), op=MPI.SUM
        )

    assert boundary_length(fresh, facets1) == pytest.approx(
        boundary_length(cached, facets2)
    )
    if comm.rank == 0:
        gmsh.finalize()
