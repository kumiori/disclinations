"""Independent polynomial patch tests for the *implemented* variational energies."""
import basix.ufl
import dolfinx
import numpy as np
import pytest
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI
from disclinations.models import (
    Biharmonic,
    NonlinearPlateFVK,
    NonlinearPlateFVK_brenner,
    NonlinearPlateFVK_carstensen,
)
from disclinations.models.adimensional import (
    A_NonlinearPlateFVK,
    A_NonlinearPlateFVK_brenner,
    A_NonlinearPlateFVK_carstensen,
)


def scalar(expr, msh):
    return msh.comm.allreduce(fem.assemble_scalar(fem.form(expr)), op=MPI.SUM)


@pytest.mark.parametrize(
    "kind",
    [
        "biharmonic",
        "dimensional",
        "dimensional_brenner",
        "adimensional",
        "adimensional_brenner",
        "dimensional_carstensen",
        "adimensional_carstensen",
    ],
)
@pytest.mark.parametrize("field", ["v", "w"])
@pytest.mark.parametrize("coupled", [False, True])
def test_clamped_polynomial_consistency(kind, field, coupled):
    """An exact smooth clamped polynomial must satisfy the broken weak equation.

    The direction is a random C0 P2 function with zero trace, hence its slope
    jumps detect missing interior fluxes independently of any numerical solve.
    """
    if kind == "biharmonic" and (field == "w" or coupled):
        pytest.skip("scalar model has only one field")
    msh = mesh.create_unit_square(
        MPI.COMM_WORLD, 3, 3, ghost_mode=mesh.GhostMode.shared_facet
    )
    V = fem.functionspace(msh, ("Lagrange", 2))
    u, z = fem.Function(V), fem.Function(V)
    z.interpolate(lambda x: np.sin(3 * x[0] + 7 * x[1]))
    msh.topology.create_connectivity(1, 2)
    dofs = fem.locate_dofs_topological(V, 1, mesh.exterior_facet_indices(msh.topology))
    z.x.array[dofs] = 0
    z.x.scatter_forward()
    zero = fem.Function(V)
    params = {"order": 2, "E": 7.0, "nu": 0.23, "thickness": 0.4, "alpha_penalty": 30.0}
    classes = {
        "biharmonic": Biharmonic,
        "dimensional": NonlinearPlateFVK,
        "dimensional_brenner": NonlinearPlateFVK_brenner,
        "adimensional": A_NonlinearPlateFVK,
        "adimensional_brenner": A_NonlinearPlateFVK_brenner,
        "dimensional_carstensen": NonlinearPlateFVK_carstensen,
        "adimensional_carstensen": A_NonlinearPlateFVK_carstensen,
    }
    model = classes[kind](msh, params)
    state = (
        {"u": u}
        if kind == "biharmonic"
        else {"v": u if field == "v" else zero, "w": u if field == "w" else zero}
    )
    energy = model.energy(state) if kind == "biharmonic" else model.energy(state)[0]
    residual = ufl.derivative(energy + model.penalisation(state), u, z)
    if kind.endswith(("brenner", "carstensen")):
        residual += model.coupling_term(
            state, z if field == "v" else 0 * z, z if field == "w" else 0 * z
        )
    x = ufl.SpatialCoordinate(msh)
    exact = x[0] ** 2 * (1 - x[0]) ** 2 * x[1] ** 2 * (1 - x[1]) ** 2
    residual = ufl.replace(
        ufl.algorithms.expand_derivatives(residual),
        {u: exact, zero: 2 * exact if coupled else 0 * exact},
    )
    bih = ufl.div(ufl.grad(ufl.div(ufl.grad(exact))))
    coeff = (
        1.0
        if kind == "biharmonic"
        else (-1.0 if field == "v" else model.c_nu)
        if kind.startswith("adimensional")
        else (-1 / (model.E * model.t) if field == "v" else model.D)
    )
    source = coeff * bih
    if coupled:
        source -= (
            0.5 * model.bracket(2 * exact, 2 * exact)
            if field == "v"
            else model.bracket(2 * exact, exact)
        )
    error = scalar(residual - source * z * ufl.dx, msh)
    assert abs(error) < 2e-10, (kind, field, error)
