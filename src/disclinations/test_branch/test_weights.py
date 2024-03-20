# https://github.com/mscroggs/dolfinx-point-sources/blob/main/point_sources.py 
# mscroggs Add weights

import numpy as np
from dolfinx.fem import FunctionSpace
from dolfinx.geometry import (BoundingBoxTree, compute_colliding_cells,
                              compute_collisions)
from dolfinx.mesh import create_unit_cube, create_unit_square
from mpi4py import MPI


def get_local_coordinates(vertices, point):
    """Get the local coordinates of the point in the cell."""
    origin = vertices[0]
    axes = [v - origin for v in vertices[1:]]
    tdim = 3
    if len(axes) == 2:
        axes.append(np.cross(axes[0], axes[1]))
        tdim = 2

    assert len(axes) == 3


    return np.linalg.solve(np.array(axes).T, point - origin)[:tdim]


def assemble_point_sources(space, points, weights):
    vector = np.zeros(space.dofmap.index_map.size_global)

    mesh = space.mesh
    for point, weight in zip(points, weights):
        # Get cell
        tree = BoundingBoxTree(mesh, mesh.geometry.dim)
        cell_candidates = compute_collisions(tree, point)
        # This gets the first cell. Would it be better to put a fraction of the delta function into each cell instead?
        cell = compute_colliding_cells(mesh, cell_candidates, point)[0]

        # Get local coordinates
        # Note: this currently only works for affine triangles and tetrahedra
        v = [mesh.geometry.x[i] for i in mesh.geometry.dofmap.links(cell)]
        local_coordinates = get_local_coordinates(v, point)

        # Note: this currently only works for scalar-valued elements that use an identity push forward map
        values = space.element.basix_element.tabulate(0, [local_coordinates])[0, 0, :, 0]
        dofs = space.dofmap.cell_dofs(cell)

        for d, v in zip(dofs, values):
            vector[d] += v * weight

    return vector


def test_point_sources_triangle():
    points = np.array([(1/4, 1/6, 0), (1/10, 6/10, 0)])

    mesh = create_unit_square(MPI.COMM_WORLD, 2, 2)
    space = FunctionSpace(mesh, ("Lagrange", 1))

    vector = assemble_point_sources(space, points, [1, 1])

    a = [i for i in vector]
    a.sort()
    assert np.isclose(a[-1], 4/5)
    assert np.isclose(a[-2], 1/2)
    assert np.isclose(a[-3], 1/3)
    assert np.isclose(a[-4], 1/5)
    assert np.isclose(a[-5], 1/6)
    assert np.isclose(a[-6], 0)
    assert np.isclose(a[0], 0)

    vector = assemble_point_sources(space, points, [2, 1])

    a = [i for i in vector]
    a.sort()
    assert np.isclose(a[-1], 1)
    assert np.isclose(a[-2], 4/5)
    assert np.isclose(a[-3], 2/3)
    assert np.isclose(a[-4], 1/3)
    assert np.isclose(a[-5], 1/5)
    assert np.isclose(a[-6], 0)
    assert np.isclose(a[0], 0)


def test_point_sources_tetrahedron():
    points = np.array([(1/4, 1/6, 0), (1/2, 1/2, 15/16)])

    mesh = create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
    space = FunctionSpace(mesh, ("Lagrange", 1))

    vector = assemble_point_sources(space, points, [1, 1])

    a = [i for i in vector]
    a.sort()
    assert np.isclose(a[-1], 7/8)
    assert np.isclose(a[-2], 1/2)
    assert np.isclose(a[-3], 1/3)
    assert np.isclose(a[-4], 1/6)
    assert np.isclose(a[-5], 1/8)
    assert np.isclose(a[-6], 0)
    assert np.isclose(a[0], 0)


    mesh = create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
    space = FunctionSpace(mesh, ("Lagrange", 1))

    vector = assemble_point_sources(space, points, [2, 1])

    a = [i for i in vector]
    a.sort()
    assert np.isclose(a[-1], 1)
    assert np.isclose(a[-2], 7/8)
    assert np.isclose(a[-3], 2/3)
    assert np.isclose(a[-4], 1/3)
    assert np.isclose(a[-5], 1/8)
    assert np.isclose(a[-6], 0)
    assert np.isclose(a[0], 0)
