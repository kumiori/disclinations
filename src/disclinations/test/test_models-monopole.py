import pytest
from disclinations.test.test_models import run_model


@pytest.mark.parametrize("variant", [0, 1, 2])
def test_monopole_solver(variant):
    run_model(variant, transverse=0.0, defect=1e-3)
