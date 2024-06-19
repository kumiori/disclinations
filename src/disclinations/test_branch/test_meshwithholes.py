import gmsh
import random
from mpi4py import MPI


class SimpleMeshWithHoles:
    def __init__(self, R=1.0, hole_radii=None, hole_positions=None, num_holes=0, refinement_radius=0.1, mesh_size=0.1, comm=MPI.COMM_WORLD):
        self.R = R
        self.hole_radii = hole_radii
        self.hole_positions = hole_positions
        self.num_holes = num_holes
        self.refinement_radius = refinement_radius
        self.mesh_size = mesh_size
        self.comm = comm
        self.model = None
        self.create_mesh_with_holes()

    def create_mesh_with_holes(self):
        if self.comm.rank == 0:
            gmsh.initialize()
            gmsh.model.add("Mesh with Holes")
            hole_entities = []
            self.hole_positions = [(random.uniform(0, self.R), random.uniform(0, self.R)) for _ in range(self.num_holes)]
            self.hole_radii = [.05] * 3
            
            domain = gmsh.model.occ.addCircle(0, 0, 0., self.R, tag=100)
            hole = gmsh.model.occ.addCircle(0, 0, 0, .1, tag=2)
            gmsh.model.occ.synchronize()


            # Create curve loops from the circles
            main_loop = gmsh.model.occ.addCurveLoop([domain])
            hole_loop = gmsh.model.occ.addCurveLoop([hole])

            # Create plane surfaces from the curve loops
            main_surface = gmsh.model.occ.addPlaneSurface([main_loop])
            hole_surface = gmsh.model.occ.addPlaneSurface([hole_loop])
            gmsh.model.occ.synchronize()

            gmsh.model.occ.cut([(2, domain)], [(2, hole)])
            print(gmsh.model.getEntities(2))


class MeshWithHoles:
    def __init__(self, R=1.0, hole_radii=None, hole_positions=None, num_holes=0, refinement_radius=0.1, mesh_size=0.1, comm=MPI.COMM_WORLD):
        self.R = R
        self.hole_radii = hole_radii
        self.hole_positions = hole_positions
        self.num_holes = num_holes
        self.refinement_radius = refinement_radius
        self.mesh_size = mesh_size
        self.comm = comm
        self.model = None
        self.create_mesh_with_holes()


    def generate_hole_positions(self):
        positions = []
        min_distance = self.R / 10  # Minimum distance from the boundary (can be adjusted)

        for _ in range(self.num_holes):
            while True:
                x = random.uniform(-self.R + min_distance, self.R - min_distance)
                y = random.uniform(-self.R + min_distance, self.R - min_distance)
                if (x**2 + y**2 <= (self.R - min_distance)**2):
                    # Check if the new hole position is sufficiently far from all existing holes
                    valid_position = True
                    for (px, py) in positions:
                        if ((x - px)**2 + (y - py)**2) < (2 * min_distance)**2:
                            valid_position = False
                            break
                    if valid_position:
                        positions.append((x, y))
                        break

        return positions

    def create_mesh_with_holes(self):
        if self.comm.rank == 0:
            gmsh.initialize()
            gmsh.model.add("Mesh with Holes")
            hole_entities = []
            hole_loops = []
            hole_surfaces = []

            # Generate hole positions and radii if not provided
            if self.hole_positions is None or self.hole_radii is None:
                # self.hole_positions = [(random.uniform(-self.R, self.R), random.uniform(-self.R, self.R)) for _ in range(self.num_holes)]
                self.hole_positions = self.generate_hole_positions()
                # self.hole_radii = [random.uniform(0.05, 0.15) for _ in range(self.num_holes)]

            self.hole_radii = [.05] * self.num_holes

            # Add holes to the model
            for i, (pos, radius) in enumerate(zip(self.hole_positions, self.hole_radii)):
                hole_entities.append(gmsh.model.occ.addCircle(pos[0], pos[1], 0, radius, tag=i+1))
                hole_loops.append(gmsh.model.occ.addCurveLoop([hole_entities[-1]]))
                hole_surfaces.append(gmsh.model.occ.addPlaneSurface([hole_loops[-1]]))

            domain = gmsh.model.occ.addCircle(0, 0, 0., self.R, tag=100)
            main_loop = gmsh.model.occ.addCurveLoop([domain], tag=100)
            domain_surface = gmsh.model.occ.addPlaneSurface([main_loop], tag=100)
            
            # Add domain rectangle
            # domain = gmsh.model.occ.addRectangle(0, 0, 0, self.L, self.H, tag=len(hole_entities) + 1)

            gmsh.model.occ.synchronize()
            print(gmsh.model.getEntities(2))

            # Cut the holes out of the domain
            holes = [(2, tag) for tag in range(1, len(hole_entities) + 1)]
            domain_with_holes = gmsh.model.occ.cut([(2, domain)], holes, tag=len(hole_entities) + 200)

            gmsh.model.occ.synchronize()
            print(gmsh.model.getEntities(2))

            # Refine around the holes
            field_tag = 1111
            for pos, radius in zip(self.hole_positions, self.hole_radii):
                gmsh.model.mesh.field.add("Ball", tag=field_tag)
                gmsh.model.mesh.field.setNumber(field_tag, "Radius", self.refinement_radius)
                gmsh.model.mesh.field.setNumber(field_tag, "VIn", self.mesh_size)
                gmsh.model.mesh.field.setNumber(field_tag, "VOut", self.mesh_size)
                gmsh.model.mesh.field.setNumber(field_tag, "XCenter", pos[0])
                gmsh.model.mesh.field.setNumber(field_tag, "YCenter", pos[1])
                gmsh.model.mesh.field.setNumber(field_tag, "Thickness", radius)
                # gmsh.model.mesh.field.setAsBackgroundMesh(field_tag)
                field_tag +=1 

            # Generate the mesh
            print("Generating mesh...")
            print(self.mesh_size)
            # gmsh.option.setNumber("Mesh.MeshSizeMin", self.mesh_size)
            gmsh.option.setNumber("Mesh.MeshSizeMax", self.mesh_size)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 20)

            gmsh.model.mesh.generate(2)

            # Save the mesh
            gmsh.write("output/mesh_with_holes.msh")

            # Finalize gmsh
            gmsh.finalize()

        self.comm.Barrier()

    def load_mesh(self, filename="output/mesh_with_holes.msh"):
        from dolfinx.io import gmshio
        from dolfinx.mesh import create_mesh
        self.model = gmshio.read_from_msh(filename, MPI.COMM_WORLD)
        return create_mesh(MPI.COMM_WORLD, self.model.points, self.model.cells)

# Example usage
if __name__ == "__main__":
    # mesh_with_holes = SimpleMeshWithHoles(num_holes=3, mesh_size=0.1)
    mesh_with_holes = MeshWithHoles(num_holes=3, mesh_size=0.1)
    # mesh = mesh_with_holes.load_mesh()
