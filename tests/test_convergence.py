import numpy as np
import pytest
from mpi4py import MPI
from examples.verify_clamped import solve_case


def test_physical_energy_and_mesh_order_convergence():
    coarse, _ = solve_case(4, 2)
    fine, _ = solve_case(8, 2)
    cubic, _ = solve_case(8, 3)
    exact = fine["exact_membrane_energy"]
    assert abs(fine["physical_membrane_energy"] - exact) < abs(
        coarse["physical_membrane_energy"] - exact
    )
    assert abs(cubic["physical_membrane_energy"] - exact) < abs(
        fine["physical_membrane_energy"] - exact
    )
    assert cubic["l2_error"] < fine["l2_error"] < coarse["l2_error"]
    assert cubic["load_work"] == pytest.approx(
        2 * cubic["discrete_quadratic"], rel=1e-8
    )


def test_linearity_reversal_and_quadratic_energy():
    base, u = solve_case(4, 2, intensity=1.0)
    rev, v = solve_case(4, 2, intensity=-2.0)
    assert np.max(abs(v.x.array + 2 * u.x.array)) < 1e-10
    assert rev["physical_membrane_energy"] == pytest.approx(
        4 * base["physical_membrane_energy"], rel=1e-8
    )


def test_penalty_sensitivity_and_repeatability():
    first, u = solve_case(4, 3, alpha=40.0)
    second, v = solve_case(4, 3, alpha=40.0)
    high, _ = solve_case(4, 3, alpha=80.0)
    assert np.max(abs(u.x.array - v.x.array)) < 1e-12
    assert second["physical_membrane_energy"] == pytest.approx(
        first["physical_membrane_energy"], abs=1e-14
    )
    assert high["physical_membrane_energy"] > 0
    assert high["l2_error"] < 1e-3
