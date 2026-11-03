# Verification record — 2 November 2026

Reference checkpoint: `71e11e4d43174d26149130e97bd98b2f9635e09e`.
Final source is on `codex/scientific-audit`; the repair commit is the commit
containing this record. Do not interpret older red/intermediate logs as final
results. Tests are local, on the native runtime in `runtime.json`.
Text logs have trailing whitespace trimmed; numerical and diagnostic content is retained.

| Check | Result | Evidence |
|---|---|---|
| Complete unattended serial suite | **67 passed, 3 skipped**, exit0 | `full_suite.txt` |
| Two-rank MPI subset, including mixed nonlinear/source, geometry/cache and sampling | **44 passed, 3 skipped per rank**, exit0 | `mpi_tests.txt` |
| Independent smooth polynomial consistency | All applicable scalar, dimensional/reduced, coupled/uncoupled cases pass | `full_suite.txt`, `tests/test_forms.py` |
| Manufactured h/p study | Physical membrane energy approaches the exact value; cubic n16 L2 error≈6.55e-7 | `convergence.json` |
| Checkpoint versus repaired forms, common solver/mesh/source | Cubic n16 L2 error≈1.47e-4→6.55e-7 | `before_after.json` |
| Penalty decomposition, material scaling, stress/gauge/traction, curvature, source pairing, reversal/scaling, failure detection | Passed | Complete suite |
| Build wheel without dependency downloads | Passed | `wheel_build.txt` |
| Install wheel in temporary target and solve outside the repository | Passed | `installed_smoke.txt`, `check_installed.py` |
| Static syntax / whitespace | Source/test/example/audit files parse; `git diff --check` clean | Final worktree verification |
| New Conda environment / other platforms / MUMPS / GUI acceptance | **Not verified** | Explicit limitation |
| Core-radius/ideal-crack/dislocation scattering convergence | **Outside the user-confirmed repository scope** | Audit U01–U04 |

The three skips are redundant combinations for the scalar biharmonic model,
which has neither a second field nor a coupled variant. MPI output is repeated
once per rank; it represents44 distinct test cases, not88 independent cases.

## Reproduction on the recorded runtime

Activate the matching environment and run from the repository root:

```bash
PYTHONPATH=src:. python -m pytest -q
mpiexec -n 2 python -m pytest tests/test_postprocess_cache.py tests/test_loads_solver.py tests/test_forms.py tests/test_geometry.py src/disclinations/test/test_models.py src/disclinations/test/test_models-monopole.py src/disclinations/test/test_models-transverse.py -q
PYTHONPATH=src:. python examples/verify_clamped.py
PYTHONPATH=src:. python audit/compare_forms.py
python audit/inventory.py
python -m pip wheel --no-deps --no-build-isolation --wheel-dir /tmp/disclinations-audit-wheel .
python -m pip install --no-deps --target /tmp/disclinations-audit-installed /tmp/disclinations-audit-wheel/disclinations-2024.0.2-py3-none-any.whl
```

To reproduce installed-code verification, change to a directory outside this
checkout, set `PYTHONPATH` to the temporary target, and invoke the absolute path
to `audit/check_installed.py` with the target directory as its sole argument.
The script asserts the import actually came from that installation.

## Original errors and intermediate observations

- `red_forms.txt`: original nine applicable polynomial block cases fail.
- `red_loads_solver.txt`: ownership-object and removed Function-vector APIs fail;
  the early tolerance assertion also used the wrong tuple index and was corrected
  to PETSc's documented rtol-first ordering. This assertion is not evidence that
  the original rtol option was ignored.
- `red_curvature.txt`: original curvature output is2 for a determinant1 field.
- `reference_geometry_lifecycle.txt`: original hole helper returns after Gmsh has
  been finalised. The initial conversion reproduction stalled and was terminated;
  `red_geometry.txt` contains only partial progress, not a test result.
- `reference_sampler.txt`: original radius.5 sampling fails on mismatched returned
  and requested coordinate counts. `red_cache.txt`: requested radius2 is mutated
  to1 by the original helper.
- `baseline_collection.txt`: original suite collection aborted; debugger and GUI
  side effects were present. A Python3.13 native environment also crashed while
  importing readline/pdb; final verification uses the working Python3.11 stack.
- During new-test development a temporary DOLFINx vector owner was released
  before copying its PETSc view; retaining the owner fixed that test-harness
  crash. MPI cache assertions were corrected to count **owned** facets and
  compare measures, since cached reads can repartition ghost entities. Neither
  harness issue is represented as an original-code scientific defect.

The benchmark is smooth and clamped, with regular volume incompatibility.
The separate core/crack/dislocation implementation is outside the user-confirmed
scope. These checks do not validate historical publication plots. See the main report for unresolved
scientific decisions and exact original source locations.
