# Disclinations: audited plate research code

This checkout implements Airy/transverse Föppl–von Kármán plate models with
continuous interior-penalty finite elements. The scientific audit and its
remaining limitations are in [SCIENTIFIC_AUDIT_AND_REPAIRS.md](SCIENTIFIC_AUDIT_AND_REPAIRS.md).
The reconstructed equations and implementation map are in
[docs/formulation.md](docs/formulation.md).

The confirmed scope is this repository's plate code. The core-regularised
linear defect/crack implementation lives elsewhere and is outside this audit.
The reported tests and energies apply to the plate formulations documented here.

## Installation

Use a matched FEniCSx stack. The tested API is DOLFINx/Basix 0.9.0,
UFL 2024.2.0, real-valued PETSc/petsc4py 3.22 and Python 3.11.

```bash
conda env create -f environment.yml
conda activate disclinations-audit
python -m pip install --no-deps -e .
python -m pytest -q
mpiexec -n 2 python -m pytest tests/test_postprocess_cache.py tests/test_loads_solver.py tests/test_forms.py tests/test_geometry.py -q
python examples/verify_clamped.py
```

Without installing the project, prefix Python commands with `PYTHONPATH=src:.`.
The Conda specification pins the ABI-sensitive versions used for verification;
the resolved package builds are recorded in `audit/runtime.json`. Pull requests
create this environment and run both serial and two-rank MPI validation. The
`requirements_pipreq.txt` file is a historical scanner export, contains unrelated
package names, and is not an installation lockfile.

The example solves a smooth manufactured Airy problem and prints separate
physical membrane energy, discrete corrections, load work, functional and
residual. `audit/convergence.json` records mesh/order comparisons;
`audit/before_after.json` isolates the repaired forms using a common solver.

## Research interfaces and limitations

Existing model class names and the energy tuple `(saddle, bending, membrane,
coupling)` are retained. The first entry is a saddle functional, not total
physical elastic energy. Assembly of a zero value trace is separate from weak
zero-slope enforcement. Hole meshing now retains a live Gmsh model and tags
outer/core boundaries as 6/100+i; callers must finalise Gmsh after conversion.

Point loads require one global ordered strength list on every MPI rank. Each
point is supplied once, with an empty `(0,3)` array on other ranks. The returned
vector is already accumulated. `SNESSolver` adds this vector with a **plus**
sign to the residual, and raises on divergence. SNES tolerances use PETSc
defaults unless supplied explicitly. See the source docstrings.

Archived plotting and parameter-sweep scripts remain research prototypes.
Several still use pre-audit APIs and are listed explicitly in `conftest.py` so
pytest does not mistake them for unattended validation. Their historical outputs
have not been regenerated or accepted as numerical evidence. Manual PyVista/Xvfb
examples run only when invoked directly. Maintained smoke, patch and
manufactured-solution tests are collected by the default pytest command.

Disk sampling now preserves coordinates and values for arbitrary positive radii,
with single MPI ownership and an optional resolution argument. Mesh caches retain
the requested radius and persist both cell and facet tags; their versioned
identity leaves old incomplete caches untouched.

The original supplied working tree is committed at
`71e11e4d43174d26149130e97bd98b2f9635e09e`; the incoming branch was
`final-figures`, HEAD `6e64d61`. All pre-existing tracked edits and non-ignored
untracked files are recoverable at that checkpoint. Ignored environment/output
files remain in place and are outside the checkpoint. To compare:

```bash
git diff 71e11e4 -- src tests test docs pyproject.toml
```

## Authors and licence

See Git history for contributors. GPLv3; see [LICENSE](LICENSE).
