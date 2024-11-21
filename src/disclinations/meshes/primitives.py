#!/usr/bin/env python3

from mpi4py import MPI

def mesh_circle_gmshapi(name,
                        R,
                        lc,
                        tdim,
                        order=1,
                        msh_file=None,
                        comm=MPI.COMM_WORLD):
    """
    Create 2d circle mesh using the Python API of Gmsh.
    """
    # Perform Gmsh work only on rank = 0

    if comm.rank == 0:
        import gmsh

        # Initialise gmsh and set options
        if not gmsh.is_initialized():
            gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)

        gmsh.option.setNumber("Mesh.Algorithm", 6)
        model = gmsh.model()
        model.add("Circle")
        model.setCurrent("Circle")
        p0 = model.geo.addPoint(0.0, 0.0, 0, lc, tag=0)
        p1 = model.geo.addPoint(R, 0.0, 0, lc, tag=1)
        p2 = model.geo.addPoint(0.0, R, 0.0, lc, tag=2)
        p3 = model.geo.addPoint(-R, 0, 0, lc, tag=3)
        p4 = model.geo.addPoint(0, -R, 0, lc, tag=4)
        # points = [p0, p1, p2, p3]
        c1 = gmsh.model.geo.addCircleArc(p1, p0, p2)
        c2 = gmsh.model.geo.addCircleArc(p2, p0, p3)
        c3 = gmsh.model.geo.addCircleArc(p3, p0, p4)
        c4 = gmsh.model.geo.addCircleArc(p4, p0, p1)

        circle = model.geo.addCurveLoop([c1, c2, c3, c4])
        model.geo.addPlaneSurface([circle])

        model.geo.synchronize()
        surface_entities = [model[1] for model in model.getEntities(tdim)]
        model.addPhysicalGroup(tdim, surface_entities, tag=5)
        model.setPhysicalName(tdim, 5, "Film surface")

        gmsh.model.mesh.setOrder(order)

        model.mesh.generate(tdim)

        # Optional: Write msh file
        if msh_file is not None:
            gmsh.write(msh_file)

        # gmsh.finalize()

    return gmsh.model if comm.rank == 0 else None, tdim

from mpi4py import MPI
import random

def mesh_circle_with_holes_gmshapi(name,
                                   R,
                                   lc,
                                   tdim,
                                   num_holes,
                                   hole_radius,
                                   hole_positions=None,
                                   refinement_factor=0.1,
                                   order=1,
                                   msh_file=None,
                                   comm=MPI.COMM_WORLD):
    """
    Create 2d circle mesh with holes using the Python API of Gmsh.
    """
    # Perform Gmsh work only on rank = 0
    if comm.rank == 0:
        import gmsh

        # Initialise gmsh and set options
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 6)

        model = gmsh.model()
        model.add("CircleWithHoles")
        model.setCurrent("CircleWithHoles")

        # Define main circle
        p0 = model.geo.addPoint(0.0, 0.0, 0, lc)
        p1 = model.geo.addPoint(R, 0.0, 0, lc)
        p2 = model.geo.addPoint(0.0, R, 0.0, lc)
        p3 = model.geo.addPoint(-R, 0, 0, lc)
        p4 = model.geo.addPoint(0, -R, 0, lc)
        c1 = model.geo.addCircleArc(p1, p0, p2)
        c2 = model.geo.addCircleArc(p2, p0, p3)
        c3 = model.geo.addCircleArc(p3, p0, p4)
        c4 = model.geo.addCircleArc(p4, p0, p1)
        outer_circle = model.geo.addCurveLoop([c1, c2, c3, c4])
        # Define holes
        hole_points = []
        hole_loops = []
        
        if hole_positions is None:
            # Generate random positions for the holes
            hole_positions = [(random.uniform(-R + hole_radius, R - hole_radius), 
                               random.uniform(-R + hole_radius, R - hole_radius)) 
                              for _ in range(num_holes)]
        
        for i, (hx, hy) in enumerate(hole_positions):
            h0 = model.geo.addPoint(hx, hy, 0, lc * refinement_factor)
            h1 = model.geo.addPoint(hx + hole_radius, hy, 0, lc * refinement_factor)
            h2 = model.geo.addPoint(hx, hy + hole_radius, 0, lc * refinement_factor)
            h3 = model.geo.addPoint(hx - hole_radius, hy, 0, lc * refinement_factor)
            h4 = model.geo.addPoint(hx, hy - hole_radius, 0, lc * refinement_factor)
            hc1 = model.geo.addCircleArc(h1, h0, h2)
            hc2 = model.geo.addCircleArc(h2, h0, h3)
            hc3 = model.geo.addCircleArc(h3, h0, h4)
            hc4 = model.geo.addCircleArc(h4, h0, h1)
            hole_loop = model.geo.addCurveLoop([hc1, hc2, hc3, hc4])
            hole_loops.append(hole_loop)
        
        # Define the surface with holes
        plane_surface = model.geo.addPlaneSurface([outer_circle] + hole_loops)
        
        model.geo.synchronize()
        surface_entities = [model[1] for model in model.getEntities(tdim)]
        model.addPhysicalGroup(tdim, surface_entities, tag=5)
        model.setPhysicalName(tdim, 5, "Film surface")
        
        gmsh.model.mesh.setOrder(order)
        model.mesh.generate(tdim)
        
        # Optional: Write msh file
        if msh_file is not None:
            gmsh.write(msh_file)
        
        gmsh.finalize()
    
    return gmsh.model if comm.rank == 0 else None, tdim

# Example usage:

if __name__ == "__main__":
    import sys

    sys.path.append("../../damage")
    from xdmf import XDMFFile
    from mesh import gmsh_to_dolfin

    # , merge_meshtags, locate_dofs_topological
    from mpi4py import MPI
    from pathlib import Path
    import dolfinx.plot

    gmsh_model, tdim = mesh_bar_gmshapi("bar",
                                        1,
                                        0.1,
                                        0.01,
                                        2,
                                        msh_file="output/bar.msh")
    mesh, mts = gmsh_to_dolfin(gmsh_model, tdim, prune_z=True)
    Path("output").mkdir(parents=True, exist_ok=True)
    with XDMFFile(MPI.COMM_WORLD, "output/bar.xdmf", "w") as ofile:
        ofile.write_mesh_meshtags(mesh, mts)

    import pyvista
    from pyvista.utilities import xvfb

    xvfb.start_xvfb(wait=0.05)
    pyvista.OFF_SCREEN = True
    plotter = pyvista.Plotter(title="Bar mesh")
    topology, cell_types = dolfinx.plot.create_vtk_topology(
        mesh, mesh.topology.dim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, mesh.geometry.x)
    # plotter.subplot(0, 0)
    actor_1 = plotter.add_mesh(grid, show_edges=True)

    plotter.view_xy()
    if not pyvista.OFF_SCREEN:
        plotter.show()
    figure = plotter.screenshot("output/bar.png")
