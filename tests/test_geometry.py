import numpy as np
import pytest
import gmsh
from mpi4py import MPI
from dolfinx.io import gmshio
from dolfinx import fem
import ufl
from disclinations.meshes.primitives import (
    mesh_circle_with_holes_gmshapi,
    mesh_ep_gmshapi,
)


@pytest.fixture(autouse=True)
def lifecycle():
    if MPI.COMM_WORLD.rank == 0:
        if not gmsh.isInitialized():
            gmsh.initialize()
        gmsh.clear()
    yield
    if MPI.COMM_WORLD.rank == 0:
        gmsh.finalize()


def test_holes_model_alive_tagged_and_measurable():
    model, dim = mesh_circle_with_holes_gmshapi(
        "audit_holes", 1.0, 0.18, 2, 1, 0.1, [(0.0, 0.0)]
    )
    msh, cells, facets = gmshio.model_to_mesh(model, MPI.COMM_WORLD, 0, gdim=dim)
    assert facets is not None
    assert MPI.COMM_WORLD.allreduce(len(facets.find(100)), op=MPI.SUM) > 0
    ds = ufl.Measure("ds", domain=msh, subdomain_data=facets)
    one = fem.Constant(msh, 1.0)
    core = MPI.COMM_WORLD.allreduce(
        fem.assemble_scalar(fem.form(one * ds(100))), op=MPI.SUM
    )
    outer = MPI.COMM_WORLD.allreduce(
        fem.assemble_scalar(fem.form(one * ds(6))), op=MPI.SUM
    )
    assert core == pytest.approx(2 * np.pi * 0.1, rel=0.03)
    assert outer == pytest.approx(2 * np.pi, rel=0.03)


def test_invalid_hole_geometry_fails():
    with pytest.raises(ValueError):
        mesh_circle_with_holes_gmshapi("bad", 1.0, 0.2, 2, 1, 0.2, [(0.95, 0)])


def test_notch_tags_disjoint():
    model, _ = mesh_ep_gmshapi("notches", 2.0, 1.0, 0.3, 0.0, 0.15, 2, sep=0.05)
    if MPI.COMM_WORLD.rank == 0:
        groups = {
            tag: set(model.getEntitiesForPhysicalGroup(1, tag))
            for _, tag in model.getPhysicalGroups(1)
        }
        tags = list(groups)
        for i, a in enumerate(tags):
            for b in tags[i + 1 :]:
                assert not groups[a] & groups[b]
    msh, _, facets = gmshio.model_to_mesh(model, MPI.COMM_WORLD, 0, gdim=2)
    msh.topology.create_connectivity(1, 2)
    links = msh.topology.connectivity(1, 2)
    for tag in (14, 15, 16, 17):
        for facet in facets.find(tag):
            assert len(links.links(facet)) == 1
        ds = ufl.Measure("ds", domain=msh, subdomain_data=facets)
        dS = ufl.Measure("dS", domain=msh, subdomain_data=facets)
        one = fem.Constant(msh, 1.0)
        length = MPI.COMM_WORLD.allreduce(
            fem.assemble_scalar(fem.form(one * ds(tag))), op=MPI.SUM
        )
        interior_length = MPI.COMM_WORLD.allreduce(
            fem.assemble_scalar(fem.form(ufl.avg(one) * dS(tag))), op=MPI.SUM
        )
        assert length > 0 and abs(interior_length) < 1e-14
