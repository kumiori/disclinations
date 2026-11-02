"""Constitutive, gauge, traction and discrete symmetry checks."""
import numpy as np
import pytest
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import assemble_matrix
from mpi4py import MPI
from petsc4py import PETSc
from disclinations.models import NonlinearPlateFVK, create_disclinations
from disclinations.models.forms import airy_stress, hessian, laplacian
from disclinations.models.material import PlateMaterial
from disclinations.solvers import (
    calculate_condition_number_1,
    calculate_condition_number_2,
)


def integral(expr):
    return MPI.COMM_WORLD.allreduce(fem.assemble_scalar(fem.form(expr)), op=MPI.SUM)


def test_material_from_plane_stress_compliance():
    E, nu, t = 17.0, 0.27, 0.3
    m = PlateMaterial(E, nu, t)
    H = np.array([[3.0, -2.0], [-2.0, 7.0]])
    N = np.trace(H) * np.eye(2) - H
    # Independent inverse Hooke law for stress RESULTANTS.
    strain = ((1 + nu) * N - nu * np.trace(N) * np.eye(2)) / (E * t)
    expected = 0.5 * np.sum(N * strain)
    assert expected == pytest.approx(
        0.5
        * m.airy_compliance
        * (np.sum(H * H) - nu * (np.trace(H) ** 2 - np.sum(H * H)))
    )
    assert PlateMaterial(2 * E, nu, t).bending_stiffness == pytest.approx(
        2 * m.bending_stiffness
    )
    assert PlateMaterial(E, nu, 2 * t).bending_stiffness == pytest.approx(
        8 * m.bending_stiffness
    )
    assert PlateMaterial(2 * E, nu, t).airy_compliance == pytest.approx(
        0.5 * m.airy_compliance
    )


@pytest.mark.parametrize(
    "params",
    [(0.0, 0.2, 1.0), (1.0, 0.5, 1.0), (1.0, 0.2, -1.0), (float("nan"), 0.2, 1.0)],
)
def test_invalid_material(params):
    with pytest.raises(ValueError):
        PlateMaterial(*params)


def test_stress_gauge_divergence_and_traction():
    msh = mesh.create_unit_square(MPI.COMM_WORLD, 3, 3)
    x = ufl.SpatialCoordinate(msh)
    u = x[0] ** 3 * x[1] ** 2
    sigma = airy_stress(u)
    affine = 3 + 2 * x[0] - 7 * x[1]
    assert (
        abs(
            integral(
                ufl.inner(
                    airy_stress(u + affine) - sigma, airy_stress(u + affine) - sigma
                )
                * ufl.dx
            )
        )
        < 1e-20
    )
    assert abs(integral(ufl.inner(ufl.div(sigma), ufl.div(sigma)) * ufl.dx)) < 1e-20
    # clamped Airy trace on all sides implies zero physical traction.
    phi = x[0] ** 2 * (1 - x[0]) ** 2 * x[1] ** 2 * (1 - x[1]) ** 2
    traction = airy_stress(phi) * ufl.FacetNormal(msh)
    assert abs(integral(ufl.inner(traction, traction) * ufl.ds)) < 1e-20
    assert integral(sigma[0, 1] * ufl.dx) == pytest.approx(-1.0)


def test_quadratic_form_symmetry_and_energy_scaling():
    msh = mesh.create_unit_square(
        MPI.COMM_WORLD, 3, 3, ghost_mode=mesh.GhostMode.shared_facet
    )
    V = fem.functionspace(msh, ("Lagrange", 2))
    u, z = fem.Function(V), fem.Function(V)
    u.interpolate(lambda x: x[0] ** 2 + x[1] ** 2)
    model = NonlinearPlateFVK(
        msh, {"order": 2, "E": 7.0, "nu": 0.23, "thickness": 0.4, "alpha_penalty": 40.0}
    )
    state = {"v": u, "w": z}
    physical = model.energy(state)[2]
    q = physical - model.penalisation(state)
    F = ufl.derivative(q, u, ufl.TestFunction(V))
    J = ufl.derivative(F, u, ufl.TrialFunction(V))
    assert (
        len(q.arguments()) == 0 and len(F.arguments()) == 1 and len(J.arguments()) == 2
    )
    A = assemble_matrix(fem.form(J))
    A.assemble()
    assert A.isSymmetric(tol=1e-10)
    e = integral(physical)
    u.x.array[:] *= -3
    assert integral(physical) == pytest.approx(9 * e)
    A.destroy()


def test_flat_defect_input_normalised():
    msh = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    points, params = create_disclinations(msh, {}, points=[0.2, 0.3, 0], signs=[-2])
    assert params["loading"]["points"] == [[0.2, 0.3, 0.0]]
    assert len(points) == 1
    if msh.comm.rank == 0:
        assert points[0].shape == (1, 3)


def test_condition_number_uses_singular_values():
    if MPI.COMM_WORLD.size != 1:
        pytest.skip("serial dense oracle")
    A = PETSc.Mat().createAIJ([2, 2], comm=PETSc.COMM_SELF)
    A.setUp()
    values = np.array([[1.0, 5.0], [0.0, 1.0]])
    A.setValues([0, 1], [0, 1], values)
    A.assemble()
    assert calculate_condition_number_1(A) == pytest.approx(
        np.linalg.cond(values), rel=1e-7
    )
    assert calculate_condition_number_2(A) == pytest.approx(np.linalg.cond(values))
    A.destroy()


def test_gaussian_curvature_is_determinant():
    msh = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    V = fem.functionspace(msh, ("Lagrange", 2))
    w = fem.Function(V)
    w.interpolate(lambda x: 0.5 * (x[0] ** 2 + x[1] ** 2))
    model = NonlinearPlateFVK(msh, {"order": 2})
    curvature = model.gaussian_curvature(w, order=0)
    assert integral((curvature - 1) ** 2 * ufl.dx) < 1e-20


def test_dimensionless_energy_scale():
    m = PlateMaterial(17.0, 0.27, 0.3)
    assert m.energy_scale(2.0) == pytest.approx(17.0 * 0.3**5 / 4)
    assert PlateMaterial(17.0, 0.27, 0.6).energy_scale(2.0) == pytest.approx(
        32 * m.energy_scale(2.0)
    )
    with pytest.raises(ValueError):
        m.energy_scale(0.0)


def test_operator_base_rejects_undefined_model():
    from disclinations.models import ToyPlateFVK

    with pytest.raises(NotImplementedError):
        ToyPlateFVK(None).penalisation({})


def test_model_default_order_and_invalid_degree():
    msh = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    model = NonlinearPlateFVK(msh)
    assert model.D > 0
    with pytest.raises(ValueError):
        NonlinearPlateFVK(msh, {"order": 1})


@pytest.mark.parametrize("brenner", [False, True])
def test_adimensional_penalty_decomposition(brenner):
    from disclinations.models.adimensional import (
        A_NonlinearPlateFVK,
        A_NonlinearPlateFVK_brenner,
    )

    msh = mesh.create_unit_square(
        MPI.COMM_WORLD, 3, 3, ghost_mode=mesh.GhostMode.shared_facet
    )
    V = fem.functionspace(msh, ("Lagrange", 2))
    v, w = fem.Function(V), fem.Function(V)
    v.interpolate(lambda x: np.sin(2 * x[0] + x[1]))
    w.interpolate(lambda x: np.cos(3 * x[0] - x[1]))
    cls = A_NonlinearPlateFVK_brenner if brenner else A_NonlinearPlateFVK
    model = cls(msh, {"order": 2, "alpha_penalty": 30.0})
    state = {"v": v, "w": w}
    before = (model.dg1, model.bc1)
    model.energy(state)
    assert before == (model.dg1, model.bc1)
    terms = model.compute_penalisation_terms_w(state, MPI.COMM_WORLD)
    assert terms[-1] == 0.0
    total = sum(terms) - sum(model.compute_penalisation_terms_v(state, MPI.COMM_WORLD))
    if not brenner:
        total += model.compute_penalisation_coupling(state, MPI.COMM_WORLD)
    assert total == pytest.approx(
        model.compute_penalisation(state, MPI.COMM_WORLD), rel=1e-12
    )
