import numpy as np
import dolfinx.geometry

def get_datapoints(u, points):
    """
    Retrieves data points for a given solution field evaluated at specific points.

    Args:
        u: Scalar or vector field.
        points: Points at which the field should be evaluated.

    Returns:
        tuple: Points on processor and evaluated values of u at the points.
    """

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


import numpy as np

def load_field_data(filepath):
    """
    Reads field data stored in an .npz file and reconstructs it into a usable format.

    Args:
        filepath (str): Path to the .npz file.

    Returns:
        dict: A dictionary containing mesh points, point values, and global values.
    """
    # Load the .npz file
    data = np.load(filepath, allow_pickle=True)
    
    # Extract and reconstruct the data
    fields_data = {
        "mesh": data["mesh"],
        "point_values": {},
        "global_values": {},
    }
    
    # Extract point values if they exist
    if "point_values" in data:
        point_values = data["point_values"].item()
        for key, value in point_values.items():
            fields_data["point_values"][key] = value

    # Extract global values if they exist
    if "global_values" in data:
        global_values = data["global_values"].item()
        for key, value in global_values.items():
            fields_data["global_values"][key] = value

    return fields_data