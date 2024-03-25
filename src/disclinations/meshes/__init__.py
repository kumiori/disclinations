
def mesh_bounding_box(mesh, i): return (
    min(mesh.geometry.x[:, i]), max(mesh.geometry.x[:, i]))
