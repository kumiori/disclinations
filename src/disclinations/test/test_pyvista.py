import pyvista as pv
pv.OFF_SCREEN = True
sphere = pv.Sphere()

# Create a PyVista plotter object
plotter = pv.Plotter()

# Add the sphere mesh to the plotter
plotter.add_mesh(sphere, color="cyan", show_edges=True)

# Display the plot in an interactive window
plotter.show()

# Save the plot to an image file (PNG format)
plotter.screenshot("sphere_plot.png")