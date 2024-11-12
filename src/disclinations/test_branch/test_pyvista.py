import pyvista as pv
from pyvista import examples
from pyvista.plotting.utilities import xvfb

xvfb.start_xvfb(wait=0.05)
pv.OFF_SCREEN = True

# Create a plotter object with 3 subplots in a single row
plotter = pv.Plotter(shape=(1, 3))

# Example meshes
sphere = pv.Sphere()
cone = pv.Cone()
cube = pv.Cube()

# First subplot: Add the sphere
plotter.subplot(0, 0)
plotter.add_text("Sphere", font_size=12)
plotter.add_mesh(sphere, color='blue')

# Second subplot: Add the cone
plotter.subplot(0, 1)
plotter.add_text("Cone", font_size=12)
plotter.add_mesh(cone, color='green')

# Third subplot: Add the cube
plotter.subplot(0, 2)
plotter.add_text("Cube", font_size=12)
plotter.add_mesh(cube, color='red')

# Display the plotter with subplots
from pathlib import Path
import os
from mpi4py import MPI
comm = MPI.COMM_WORLD

full_path = os.path.abspath(__file__)
prefix = os.path.basename(full_path).split('.')[0]

if comm.rank == 0:
    Path(prefix).mkdir(parents=True, exist_ok=True)

plotter.screenshot(f"{prefix}/test_pyvista.png")
# plotter.show()
