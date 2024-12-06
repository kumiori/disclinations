import numpy as np
import dolfinx
from dolfinx import mesh, fem
from mpi4py import MPI

def get_datapoints(u, points):
    import dolfinx.geometry
    mesh = u.function_space.mesh
    cells = []
    bb_tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
    points_on_proc = []
    cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(
        mesh, cell_candidates, points.T
    )
    for i, point in enumerate(points.T):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])

    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    u_values = u.eval(points_on_proc, cells)

    return points_on_proc, u_values

def sample_function(v, radius):
    """
    Identifier: sample_function
    Parametrs: dolfinx.fem.function.Function, number
    Return values: list, list, list
    Description: The function takes in input a Function object and samples it on the ball of radius radius
    """
    num_points0 = 1001
    x_list = []
    y_list = []
    z_list = []
    tol = 1e-2
    y_range = np.linspace(-radius+tol, radius-tol, num_points0)
    for y in y_range:
        choord = 2*radius - 2 * np.sqrt((radius)**2 - y**2)
        #num_points = np.floor( num_points0 * (chood / (2*radius)) )
        num_points = num_points0
        x_range = [-np.sqrt(radius-y**2)+tol, np.sqrt(radius-y**2)-tol]
        x_samples = np.linspace(x_range[0], x_range[1], num_points)
        y_samples = np.linspace(y, y, num_points)
        points = np.zeros((3, num_points))
        points[0] = x_samples
        points[1] = y_samples
        points_on_proc, v_values = get_datapoints(v, points)
        points[2] = v_values.flatten()
        x_list += points[0].ravel().tolist()
        y_list += points[1].ravel().tolist()
        z_list += points[2].ravel().tolist()
    return x_list, y_list, z_list

def interpolate_sample(x_list, y_list, z_list, radius):
    """
    Identifier: interpolate_sample
    Parametrs: list, list, list, number
    Return values: numpy array, numpy array, numpy array
    Description: The function takes in input three Pyhton lists representing the x, y coordinates and the heigth of a function v(x,y) and returns a cubic interpolation of the points
    """
    from scipy.interpolate import griddata
    grid_x, grid_y = np.mgrid[-radius:radius:500j, -radius:radius:500j]  # 500x500 grid for high resolution
    grid_z = griddata((x_list, y_list), z_list, (grid_x, grid_y), method='cubic')
    return grid_x, grid_y, grid_z
