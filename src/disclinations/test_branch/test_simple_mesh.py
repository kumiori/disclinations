import gmsh

# Initialize gmsh and the OCC kernel
gmsh.initialize()
gmsh.model.add("Circle with Hole")

# Parameters for the main circle and the hole
R = 1.0      # Radius of the main circle
r = 0.1      # Radius of the hole

# Create the main circle
main_circle = gmsh.model.occ.addCircle(0, 0, 0, R)

# Create the hole
hole_circle = gmsh.model.occ.addCircle(0, 0, 0, r)

# Synchronize the CAD kernel to reflect the added geometry
gmsh.model.occ.synchronize()

# Create curve loops from the circles
main_loop = gmsh.model.occ.addCurveLoop([main_circle])
hole_loop = gmsh.model.occ.addCurveLoop([hole_circle])

# Create plane surfaces from the curve loops
main_surface = gmsh.model.occ.addPlaneSurface([main_loop])
hole_surface = gmsh.model.occ.addPlaneSurface([hole_loop])

# Perform the cut operation to create the domain with a hole
gmsh.model.occ.cut([(2, main_surface)], [(2, hole_surface)], tag=100)

# Synchronize the CAD kernel again after the cut operation
gmsh.model.occ.synchronize()

# Generate the mesh
gmsh.model.mesh.generate(2)

# Save the mesh to a file
gmsh.write("circle_with_hole.msh")

# Finalize gmsh
gmsh.finalize()
