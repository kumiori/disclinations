"""Run outside the checkout, with PYTHONPATH set to a temporary wheel target."""
import importlib.resources
from pathlib import Path
import sys

import disclinations
from disclinations.models import NonlinearPlateFVK
from disclinations.solvers import SNESSolver
from disclinations.utils import code_info
from dolfinx import fem, mesh
from mpi4py import MPI
import numpy as np
import ufl


def main(target):
    location = Path(disclinations.__file__).resolve()
    assert location.is_relative_to(Path(target).resolve()), location
    assert (
        importlib.resources.files("disclinations.models")
        .joinpath("default_parameters.yml")
        .is_file()
    )
    msh = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    u = fem.Function(fem.functionspace(msh, ("Lagrange", 2)))
    solver = SNESSolver(
        (u - 3) * ufl.TestFunction(u.function_space) * ufl.dx, u, monitor=None
    )
    _, reason = solver.solve()
    assert np.max(abs(u.x.array - 3)) < 1e-10 and reason > 0
    assert NonlinearPlateFVK(msh).D > 0
    print(
        "INSTALLED WHEEL: bundled YAML, default material and numerical solver verified outside checkout"
    )
    print("Package location:", location)
    print("Safe provenance fallback:", code_info)


if __name__ == "__main__":
    main(sys.argv[1])
