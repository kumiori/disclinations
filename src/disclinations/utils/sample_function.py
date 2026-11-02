"""Collective scalar sampling with coordinates kept aligned with their values."""
import numpy as np
import dolfinx


def get_datapoints(u, points):
    """Return local owned sample coordinates/values for common (3,N) points.

    Collective on the function's mesh communicator. The common point list is
    routed from rank zero to avoid duplicate samples on partition interfaces.
    Points outside the mesh are omitted. Outputs have shapes (M,3), (M,1).
    """
    msh = u.function_space.mesh
    points = np.asarray(points, dtype=msh.geometry.x.dtype)
    if points.ndim != 2 or points.shape[0] != 3 or u.ufl_shape:
        raise ValueError("Require scalar u and points with shape (3,N)")
    source = (
        np.ascontiguousarray(points.T)
        if msh.comm.rank == 0
        else np.empty((0, 3), dtype=points.dtype)
    )
    data = dolfinx.cpp.geometry.determine_point_ownership(msh._cpp_object, source, 1e-8)
    owned = np.asarray(data.dest_points).reshape(-1, 3)
    cells = np.asarray(data.dest_cells, dtype=np.int32)
    values = u.eval(owned, cells).reshape(-1, 1)
    return owned, values


def sample_function(v, radius, num_points=1001):
    """Return global x,y,value lists sampled on an open disk of given radius.

    Samples outside the actual FE domain are omitted, with no coordinate/value
    misalignment. MPI results are combined and sorted on every rank. radius is
    in the mesh's coordinate units; v is scalar. num_points controls both axes.
    """
    if (
        not np.isfinite(radius)
        or radius <= 0
        or num_points < 2
        or int(num_points) != num_points
    ):
        raise ValueError("Require positive finite radius and integer num_points>=2")
    tol = min(0.01, 0.001 * radius)
    rows = []
    for y in np.linspace(-radius + tol, radius - tol, int(num_points)):
        half_chord = np.sqrt(radius**2 - y**2)
        inset = min(tol, 0.1 * half_chord)
        x = np.linspace(-half_chord + inset, half_chord - inset, int(num_points))
        rows.append(np.vstack([x, np.full_like(x, y), np.zeros_like(x)]))
    points = np.concatenate(rows, axis=1)
    local, values = get_datapoints(v, points)
    pieces = v.function_space.mesh.comm.allgather(
        np.column_stack([local[:, :2], values])
    )
    samples = np.concatenate(pieces, axis=0)
    samples = samples[np.lexsort((samples[:, 0], samples[:, 1]))]
    return tuple(samples[:, i].tolist() for i in range(3))


def interpolate_sample(x_list, y_list, z_list, radius):
    """Return cubic interpolation on a 500² Cartesian grid; outside is NaN."""
    from scipy.interpolate import griddata

    grid_x, grid_y = np.mgrid[-radius:radius:500j, -radius:radius:500j]
    grid_z = griddata((x_list, y_list), z_list, (grid_x, grid_y), method="cubic")
    return grid_x, grid_y, grid_z
