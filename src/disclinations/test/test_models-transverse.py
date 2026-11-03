import pytest
from disclinations.test.test_models import run_model


@pytest.mark.parametrize("variant", [0, 1, 2])
def test_coupled_solver(variant):
    run_model(variant, transverse=1e-3, defect=-1e-3)
