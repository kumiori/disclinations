# Scientific audit and repairs

Audit date: 2 November 2026. Scope: the supplied local repository and its non-ignored
research files, not an assumed implementation matching the mission description.

## Deliverable status and recoverability

The two outputs are this report and the versioned repaired project containing
source, configurations, tests, mathematical documentation, reproducible example
and evidence. The available **plate-code repairs are verified in their stated
cases**. The user confirmed that the core-regularised implementation lives
elsewhere and instructed this audit to cover the current repository only.
That separate implementation and its scattering study are outside the agreed
scope, not blockers for delivery. Scientific and reproducibility limits of
this repository remain explicit below; no replacement model is introduced.

Incoming branch `final-figures`, HEAD `6e64d61`; pre-audit checkpoint
`71e11e4d43174d26149130e97bd98b2f9635e09e` contains all supplied tracked edits and
non-ignored untracked files. Repairs are on `codex/scientific-audit`.
Ignored environments/output directories remain in place, outside this Git
checkpoint. Compare `git diff 71e11e4 -- <path>`; do not reset the user's checkout
to inspect the original.

## Coverage and scientific reconstruction

The inventory scans the tracked/non-ignored production and historical
Python/configuration/text sources, notebook source cells, and TeX members of
the paper archive. New tests, examples and audit tools are verified separately. Binary plots/PDFs and
ignored generated output are not treated as executable mathematical evidence.
See `audit/source_inventory.json` (paths/line counts/hashes) and
`audit/constants_and_scope_inventory.json` (exact expression/scope occurrences).
The archive revealed the plate model's nondimensionalisation and an erroneous
half-weight consistency expression also present in code. Historical plotting
scripts were inspected for interfaces/scales but not all executed.

The full derivation, dimensional checks, continuous/discrete equations,
external/core/crack boundary analysis and term-to-function map are in
[docs/formulation.md](docs/formulation.md). That document is part of the repaired
codebase, not a third final deliverable.

### Priority C1/C2 conclusion

No constitutive uppercase `C1`/`C2` identifiers occur in this checkout. Their
uppercase string occurrences in `validation_fem*.py` are Matplotlib colour
names. Circle-mesher `c1/c2` variables are Gmsh arc identifiers, not material
constants. At the checkpoint the relevant lower-case constants were defined twice in
dimensional plate penalisation: **c1=D=Et³/[12(1−ν²)] [N m]**, **c2=1/(Et) [m/N]**. Their
definitions are correct for the implemented plane-stress **stress-resultant**
Airy potential; their boundary placement was wrong (A01). They are not the
plane-strain compatibility coefficient (1−ν²)/E. `c_nu` is dimensionless and
belongs to a different nondimensional formulation. No unsupported plane-strain
switch or source rescaling has been introduced.

### Priority boundary and energy conclusion

Repaired C0-IP fluxes follow the derivative of each actual bulk density; the
quadratic consistency term has unit weight, and its derivative supplies both
symmetric terms (A02/A03). Strong zero value traces plus slope Nitsche terms
implement the selected homogeneous clamped Airy/bending space. These terms do
not by themselves establish arbitrary traction-free holes or free crack lips.
The available “cracks” are finite-width exterior notches. Core meshing exists,
but the solved point-defect forms do not implement excluded-core period loads.

The energy tuple starts with a **saddle functional**. Physical positive bulk
energy, consistency/penalty correction, ½a_h(u,u), linear work and complete
discrete functional are distinguished in the derivation and example. The
mission's `stats['hess_l2_norm']` energy assignment and internal crack bilinear
form are absent from the scanned source. No conclusion about their correctness
in another implementation can be made.

## Confirmed errors and implemented repairs

All original line ranges below refer to checkpoint `71e11e4`. Repaired functions
are named so locations remain traceable after refactoring. Verification applies
to the specific test/evidence stated, not to every historic numerical output.

### A01 — critical

| Required field | Evidence and action |
|---|---|
| Issue ID / classification | A01; confirmed error / verified repair within the tests below |
| Severity | critical |
| Original file and exact location | `src/disclinations/models/__init__.py:154-160,259-265` |
| Original implementation | c2*bc1(w), -c1*bc3(v), with c1=D and c2=1/(Et). |
| Mathematical/computational problem | The bending consistency term uses membrane compliance and the membrane penalty uses bending stiffness. These have incompatible physical dimensions. |
| Evidence | Inverse plane-stress Hooke law gives the independent density in test_material_from_plane_stress_compliance; polynomial patch residuals failed for both dimensional fields. |
| Correct formulation | Every bending term carries D; every membrane term carries 1/(Et), with the saddle sign retained. |
| Repair implemented | PlateMaterial centralises both constants; dimensional penalisation uses D Qcorr(w;M) − (Et)^−1 Qcorr(v;P). |
| Associated test | tests/test_mechanics.py::test_material_from_plane_stress_compliance and tests/test_forms.py::test_clamped_polynomial_consistency. |
| Numerical consequences | The fixed-solver cubic Airy comparison changes physical energy at n=16 from 5.3598725151e-4 to 5.8284329687e-4; exact 5.8309037901e-4. This comparison includes A02/A03 together; it does not identify each contribution separately. |

### A02 — critical

| Required field | Evidence and action |
|---|---|
| Issue ID / classification | A02; confirmed error / verified repair within the tests below |
| Severity | critical |
| Original file and exact location | `src/disclinations/models/__init__.py:34-47,127-160,230-265; models/adimensional.py:117-128,237-243` |
| Original implementation | The quadratic correction uses −1/2 times the moment/slope product, while bulk energy already has 1/2. |
| Mathematical/computational problem | Differentiation leaves half-strength consistency terms; symmetry alone does not repair consistency. Averaging Hn with opposite outward normals also destroys the intended scalar normal-moment flux in the dimensional variational branch. |
| Evidence | audit/red_forms.txt records 9 failed patch cases. Exact smooth polynomial fields and random C0 zero-trace test directions expose nonzero residuals without a numerical solve. The archive discrete.tex repeats the half-weight error. |
| Correct formulation | Qcorr contains −{Tnn(u)} j(u), not −1/2 {Tnn(u)} j(u). The derivative has both symmetric terms with coefficient −1. Boundary product has the same coefficient. |
| Repair implemented | Shared c0ip_quadratic_terms uses scalar normal moments and outward-normal slope jumps in all concrete models; the duplicated adimensional Biharmonic now imports the repaired class. |
| Associated test | tests/test_forms.py (uncoupled and coupled cases, all concrete variants), test_quadratic_form_symmetry_and_energy_scaling. |
| Numerical consequences | At n=16,p=3 the smooth Airy L2 error falls from 1.4697894951e-4 (checkpoint forms) to 6.5546635038e-7 (repaired); the comparison also includes A01/A03. Existing research plots must be recomputed. |

### A03 — major

| Required field | Evidence and action |
|---|---|
| Issue ID / classification | A03; confirmed error / verified repair within the tests below |
| Severity | major |
| Original file and exact location | `src/disclinations/models/__init__.py:28-47,99-125,127-160,230-265; models/adimensional.py:30-49` |
| Original implementation | Biharmonic bulk is (Δu)^2/2 but the facet flux uses Hess(u)n; the exterior has a penalty and no consistency term. Dimensional plate fluxes use Hnn instead of their constitutive moments. |
| Mathematical/computational problem | Boundary/facet fluxes must be derivatives of the actual bulk density. Sharing the same differential operator Δ² in the interior does not make distinct boundary variational forms interchangeable. |
| Evidence | Direct integration by parts in docs/formulation.md and the failing polynomial residual tests; scalar patch residual also failed on the original source. |
| Correct formulation | T=(Δu)I for the scalar Laplacian energy, T=M(u) for bending, T=P(u) for membrane. Homogeneous clamping needs both symmetric boundary consistency terms after differentiation and α/h penalty. |
| Repair implemented | Correct moments are passed to the shared energy helper; exterior consistency is added for Biharmonic. Dimensionless Hessian bulk retains its matching Hessian moment. |
| Associated test | tests/test_forms.py; src/disclinations/test/test_bilaplacian.py::test_bilaplacian_solution; tests/test_convergence.py. |
| Numerical consequences | Smooth manufactured physical membrane energies converge under h/p refinement. No inference is made about core or free-crack mechanics. |

### A04 — major

| Required field | Evidence and action |
|---|---|
| Issue ID / classification | A04; confirmed error / verified repair within the tests below |
| Severity | major |
| Original file and exact location | `src/disclinations/utils/la.py:6-41,43-60,76-99` |
| Original implementation | Ownership data are unpacked as a tuple, basis_values=values[:num_dofs:num_dofs*len(cells)], then local concatenated cells are zipped with the original global signs. |
| Mathematical/computational problem | DOLFINx 0.9 ownership is an object. The slice is not the per-point basis diagonal. Dropping unowned points before zipping can assign the wrong strength to subsequent points on an MPI rank. Outside points silently remove a defect. |
| Evidence | audit/red_loads_solver.txt records TypeError on the real ownership API. Partition-of-unity/affine pairing is checked against sum s_k z(x_k), with distinct strengths and source ranks. |
| Correct formulation | Route each defect separately while retaining its strength; evaluate its scalar Lagrange basis at the owning cell; sum owned/ghost contributions exactly once. |
| Repair implemented | Ownership-object API, scalar Basix basis evaluation, mixed collapse map, collective shape/count/outside checks, reverse then forward scatter. |
| Associated test | tests/test_loads_solver.py::test_point_load_strengths_and_pairing/test_point_load_outside_fails and serial/MPI mixed monopole tests. |
| Numerical consequences | Numerical dual pairing matches the independently evaluated affine test function to 1e-12 in serial and on two ranks. Historical point-load outputs are not trusted until rerun. |

### A05 — major

| Required field | Evidence and action |
|---|---|
| Issue ID / classification | A05; confirmed error / verified repair within the tests below |
| Severity | major |
| Original file and exact location | `src/disclinations/models/__init__.py:294-336; models/adimensional.py:299-341` |
| Original implementation | Default points=[0,0,0] is iterated as three separate point coordinates; defaults are mutable and input lengths are unchecked. |
| Mathematical/computational problem | A single flat coordinate creates malformed (1,) arrays rather than one (1,3) point; mismatch can truncate source assignment. |
| Evidence | Original return construction wraps each scalar of the flat input. The regression requires exactly one global defect and valid per-rank arrays. |
| Correct formulation | Normalise a flat triple to a single point, require one finite strength per finite 3D point, retain the existing (points,params) return interface. |
| Repair implemented | One shared validated create_disclinations function; adimensional module re-exports it. |
| Associated test | tests/test_mechanics.py::test_flat_defect_input_normalised plus load/monopole tests. |
| Numerical consequences | Flat single-point input now loads one intended source. No physical Frank-angle convention is inferred from its sign. |

### A06 — critical

| Required field | Evidence and action |
|---|---|
| Issue ID / classification | A06; confirmed error / verified repair within the tests below |
| Severity | critical |
| Original file and exact location | `src/disclinations/solvers/__init__.py:173-217,307-343,345-364,383-437` |
| Original implementation | b0 defaults to a type union; Function.vector and scale= are used; residual ghosts receive ADD/FORWARD; J does not refresh u; solve prints “converged” and returns negative reasons. |
| Mathematical/computational problem | Removed APIs prevent execution on the declared stack. Forward-add can contaminate ghost residuals. A negative SNES convergence reason is a failed solve, not a valid result. |
| Evidence | audit/red_loads_solver.txt exposes removed vector access. A forced max_it=0 test checks divergence; two-rank residual tests exercise assembly and source accumulation. |
| Correct formulation | Use Function.x.petsc_vec, alpha= for DOLFINx 0.9 lifting/BC calls, reverse-add local assembly once, add already assembled b0 via owned Vec.axpy, refresh coefficients for both callbacks, require positive reason and finite state/norm. |
| Repair implemented | Rewritten shared SNES assembly; b0=None by default; explicit rank/type checks; SNESProblem delegates to the same implementation. |
| Associated test | tests/test_loads_solver.py::test_solver_configuration_and_convergence/test_solver_nonconvergence_raises; all nonlinear and MPI solver tests. |
| Numerical consequences | Divergence now raises RuntimeError instead of permitting downstream energies. Complete scalar/mixed test solves execute. Existing ignored failed-run output files are not rewritten. |

### A07 — moderate

| Required field | Evidence and action |
|---|---|
| Issue ID / classification | A07; confirmed error / verified repair within the tests below |
| Severity | moderate |
| Original file and exact location | `src/disclinations/solvers/__init__.py:194-196,227-245,263-305` |
| Original implementation | Default prefix is based on the first four decimal characters of id(self); options remain in the global PETSc database; compiler/JIT arguments are accepted but ignored. |
| Mathematical/computational problem | Prefix collisions cause configuration cross-talk and are not collective-safe; stale options can affect later runs. Accepting unused configuration is misleading. |
| Evidence | The old prefix expression has only a small leading-digit namespace. Effective SNES tolerance and KSP/PC types are inspected programmatically in tests. |
| Correct formulation | Broadcast a unique default prefix; scope and restore options; pass compiler/JIT options to fem.form. |
| Repair implemented | Collective UUID default prefix, temporary scoped option insertion, effective runtime configuration available through .solver. |
| Associated test | test_solver_configuration_and_convergence and test_solver_prefixes_are_isolated. |
| Numerical consequences | Configured relative tolerance and LU/preonly choices are verified. No MUMPS availability or clean-environment installation is asserted. |

### A08 — major

| Required field | Evidence and action |
|---|---|
| Issue ID / classification | A08; confirmed error / verified repair within the tests below |
| Severity | major |
| Original file and exact location | `src/disclinations/models/__init__.py:171; models/adimensional.py:211` |
| Original implementation | gaussian_curvature returns bracket(w,w). |
| Mathematical/computational problem | The bracket is twice det Hess(w), so the labelled small-slope Gaussian curvature is doubled. |
| Evidence | For w=(x²+y²)/2 the exact determinant is 1; audit/red_curvature.txt records squared error integral ≈1, i.e. output 2. |
| Correct formulation | K=det Hess(w)=bracket(w,w)/2 in the small-slope plate model. |
| Repair implemented | Both dimensional and adimensional methods include the missing 1/2. |
| Associated test | tests/test_mechanics.py::test_gaussian_curvature_is_determinant. |
| Numerical consequences | Reported small-slope curvature is halved. Gaussian-curvature plots must be regenerated; this diagnostic change does not alter the solved residual. |

### A09 — major

| Required field | Evidence and action |
|---|---|
| Issue ID / classification | A09; confirmed error / verified repair within the tests below |
| Severity | major |
| Original file and exact location | `src/disclinations/test_branch/validation_test1_adim.py:221; validation_test1_BR_adim.py:203; validation_test1_CA_adim.py:204` |
| Original implementation | energy_dimension = E*thickness**3/a**2. |
| Mathematical/computational problem | For x=Rξ,w=t w~,v=Et³ v~, the factor has units J/m² rather than J. It omits t². |
| Evidence | Independent substitution into all three densities yields Et⁵/R². paper/Archive.zip!numerics.tex:48 states the same global scale; validation_test2 variants already use it. |
| Correct formulation | energy_dimension = E*t**5/R**2. |
| Repair implemented | PlateMaterial.energy_scale(R) supplies the factor to all three affected validation scripts. Alternative analytic-benchmark scaling is retained separately. |
| Associated test | tests/test_mechanics.py::test_dimensionless_energy_scale; constitutive substitution in docs/formulation.md. |
| Numerical consequences | For unchanged dimensionless fields these three reported dimensional energy components are multiplied by t² relative to the old output (e.g. t=.4 gives .16). Historical graphical validation scripts were not run end-to-end; their full output validity is not claimed. |

### A10 — moderate

| Required field | Evidence and action |
|---|---|
| Issue ID / classification | A10; confirmed error / verified repair within the tests below |
| Severity | moderate |
| Original file and exact location | `src/disclinations/models/adimensional.py:174-181,191-194,220-246` |
| Original implementation | The w decomposition includes dg3 although penalisation does not; energy() mutates the Brenner facet callbacks. |
| Mathematical/computational problem | The reported decomposition is not the assembled functional, and postprocessing has side effects on operators. The smooth flag never activates dg3. |
| Evidence | Source-level comparison of penalisation() and the summed terms. A non-C1 polynomial FE field makes dormant dg3 nonzero. |
| Correct formulation | A decomposition must sum only contributions actually present. Diagnostic evaluation must not change assembly functions. |
| Repair implemented | Fifth legacy w diagnostic is 0 for the inactive term (tuple length retained); duplicate mutation is removed. Signed membrane/coupling bookkeeping is documented. |
| Associated test | tests/test_mechanics.py::test_adimensional_penalty_decomposition. |
| Numerical consequences | Reported w total no longer includes an unassembled Hessian-jump quantity. This bookkeeping repair alone does not change the mathematical residual. |

### A11 — major

| Required field | Evidence and action |
|---|---|
| Issue ID / classification | A11; confirmed error / verified repair within the tests below |
| Severity | major |
| Original file and exact location | `src/disclinations/meshes/primitives.py:409-487` |
| Original implementation | The holes helper finalises Gmsh before returning gmsh.model; core/outer curves have no physical groups; random centres are unchecked. |
| Mathematical/computational problem | The returned model cannot be converted reliably, core conditions cannot be selected by tags, and holes may intersect/leave the outer disk. Unseeded remeshing is irreproducible. |
| Evidence | The safe checkpoint replay in audit/reference_geometry_lifecycle.txt records Gmsh.isInitialized()==0 immediately after the original helper returns. The repaired test converts the returned model and integrates each tagged circumference against 2πr. |
| Correct formulation | Keep the model live until caller conversion; use disjoint facet tags and admissible geometry; make default random placement reproducible. |
| Repair implemented | Caller-owned finalisation; outer tag6/core100+i; validated containment/non-overlap/count; seeded local rejection placement; generated high-order mesh set after generation. |
| Associated test | tests/test_geometry.py::test_holes_model_alive_tagged_and_measurable/test_invalid_hole_geometry_fails. |
| Numerical consequences | Core/outer boundary measures are nonempty and match circumference within polygonal-mesh tolerance. This establishes geometry, not mechanically correct core boundary constraints. |

### A12 — major

| Required field | Evidence and action |
|---|---|
| Issue ID / classification | A12; confirmed error / verified repair within the tests below |
| Severity | major |
| Original file and exact location | `src/disclinations/meshes/primitives.py:81-98` |
| Original implementation | Left/right physical groups include notch lip curves that also have their own physical groups; Rliptop/Rlipbot names are reversed. |
| Mathematical/computational problem | Overlapping curve membership makes one facet carry competing physical tags in conversion. Selecting a side may unintentionally include a lip. |
| Evidence | The original left set [6,7,8,9] overlaps lips [7],[8], and right [1,2,3,4] overlaps [2],[3]. Tests check all pairs and the converted facet-to-cell adjacency. |
| Correct formulation | Ordinary left/right side groups exclude lip curves; upper/lower names follow their coordinates. |
| Repair implemented | Left=[6,9], right=[1,4]; corrected right-lip names. Existing numeric lip tags are preserved. |
| Associated test | tests/test_geometry.py::test_notch_tags_disjoint. |
| Numerical consequences | All four lips are measurable exterior facets with one adjacent cell; their tagged dS integrals are zero. Selecting a side no longer accidentally selects a notch lip. |

### A13 — moderate

| Required field | Evidence and action |
|---|---|
| Issue ID / classification | A13; confirmed error / verified repair within the tests below |
| Severity | moderate |
| Original file and exact location | `src/disclinations/solvers/__init__.py:82-122` |
| Original implementation | A non-Hermitian eigenvalue ratio is called a condition number; a local dense conversion is used without an MPI restriction. |
| Mathematical/computational problem | Eigenvalues do not determine the 2-norm condition number of nonnormal matrices. A local dense array is not a distributed global matrix. |
| Evidence | For [[1,5],[0,1]], both eigenvalues are1, whereas cond₂≈26.96. This independent NumPy oracle detects the wrong diagnostic. |
| Correct formulation | Use σmax/σmin and check extremal solve convergence; restrict dense oracle to serial. |
| Repair implemented | SLEPc extremal singular-value solves; checked failure; dense serial-only path and cleanup. |
| Associated test | tests/test_mechanics.py::test_condition_number_uses_singular_values. |
| Numerical consequences | The diagnostic now agrees with np.linalg.cond for the nonnormal example. This does not establish conditioning of historic sweep matrices. |

### A14 — major

| Required field | Evidence and action |
|---|---|
| Issue ID / classification | A14; confirmed error / verified repair within the tests below |
| Severity | major |
| Original file and exact location | `test/test_assemble.py:66; test/test_adios.py:52; src/disclinations/test/test_bilaplacian.py:149-162; test_models.py:56,128-140; test_models-monopole.py:134-145; test_models-transverse.py:129-140` |
| Original implementation | Collection executes writers/GUI/debuggers; some tests use missing keys/files, undefined L or missing coupling arguments, or commented-out assertions. |
| Mathematical/computational problem | The original suite cannot serve as unattended or mathematical validation. “Penalty is small” is not an independent correctness criterion. |
| Evidence | audit/baseline_collection.txt records an abort during original collection; source locations show breakpoints and invalid branches. Earlier full-suite attempts are not claimed as passing. |
| Correct formulation | Tests must run without interactive side effects and assert independent numerical properties. |
| Repair implemented | Replaced invalid demonstrations with assembly/output, scalar manufactured, and nine mixed-solver smoke cases; main-guarded manual visualisation; independent new mechanics/patch/geometry/convergence tests; pytest collects all three test directories. |
| Associated test | audit/full_suite.txt; audit/mpi_tests.txt. |
| Numerical consequences | The final recorded suite passes. Smoke tests establish executability only; mathematical patch/convergence evidence is separately identified. The original test scripts remain recoverable at the checkpoint. |

### A15 — moderate

| Required field | Evidence and action |
|---|---|
| Issue ID / classification | A15; confirmed error / verified repair within the tests below |
| Severity | moderate |
| Original file and exact location | `pyproject.toml:24-32; setup.py:14-20; requirements_pipreq.txt:2,47; MANIFEST.in:1; utils/__init__.py:128-161` |
| Original implementation | Two dependency definitions disagree, an unpinned visuals Git URL is mandatory, historical pipreqs names ufl/basix refer to unrelated distributions, and numerical utility imports create a log/read Git from the current directory. |
| Mathematical/computational problem | Installation and imports outside the repository are unreliable; matching native finite-element dependencies are unspecified. |
| Evidence | Package metadata differs between the two files. The historical scanner lists ufl2017 alongside the 0.9 stack; baseline runtime used removed APIs. |
| Correct formulation | One metadata source, matched native stack, bundled parameter files, optional postprocessing, safe provenance lookup. |
| Repair implemented | Consolidated pyproject/setup, environment.yml, fixed MANIFEST, runtime evidence, no import-time file logger, optional SLEPc metadata and safe repository-relative provenance fallback. Function-vector accesses migrated in existing scripts. |
| Associated test | audit/wheel_build.txt and audit/installed_smoke.txt; full suite. |
| Numerical consequences | Wheel build/install and numerical import/solve outside the checkout are checked. Creating a new Conda environment and every historical plotting workflow remain unverified. |

### A16 — moderate

| Required field | Evidence and action |
|---|---|
| Issue ID / classification | A16; confirmed error / verified repair within the tests below |
| Severity | moderate |
| Original file and exact location | `models/__init__.py:63-81,84-96,200-208,272-277; models/adimensional.py:87-109,248-255,280-285` |
| Original implementation | ToyPlateFVK has a standalone penalty without a defined energy/mesh; missing order is indexed directly; residual helpers create unused tests and assume mixed Indexed inputs. |
| Mathematical/computational problem | The toy method has no defensible standalone model; default constructors fail; scalar coupling tests fail despite callers supplying explicit test functions. |
| Evidence | Coupled scalar patch tests exposed IndexError/missing-domain errors in unused space handling. Tests require default construction and explicit rejection of undefined toy behaviour. |
| Correct formulation | Concrete models define the mechanical form; fourth-order C0 elements require p≥2; residual helpers use caller-supplied tests with explicit domains. |
| Repair implemented | Toy penalisation raises NotImplementedError; validated defaults/materials/order; shared fvk_coupling_residual; removed duplicate operator/defect definitions. |
| Associated test | test_operator_base_rejects_undefined_model/test_model_default_order_and_invalid_degree and all coupled patch tests. |
| Numerical consequences | Concrete public model names and energy tuple shapes are preserved. Direct use of ToyPlateFVK.penalisation is an intentional breaking fail-fast change; no physical toy model is invented. |

### A17 — major

| Required field | Evidence and action |
|---|---|
| Issue ID / classification | A17; confirmed postprocessing error / verified repair |
| Severity | major |
| Original file and exact location | `src/disclinations/utils/sample_function.py:6-24,26-54`, especially line44 |
| Original implementation | x endpoints use ±√(radius−y²); returned values are assigned to the full requested coordinate array regardless of missed points. |
| Mathematical/computational problem | The disk chord has incorrect dimensions unless radius=1; missing samples misalign values and coordinates; MPI sampling may duplicate interface points. |
| Evidence | `audit/reference_sampler.txt` records the actual original radius=.5 failure: 999 returned values cannot fill1001 requested entries. Nonunit-radius and out-of-domain tests exercise the repaired numerical path. |
| Correct formulation | x²+y²≤R² implies half-chord √(R²−y²); evaluate only owned valid coordinates, keeping each value paired with its actual coordinate. |
| Repair implemented | Correct chord, optional sampling resolution, collective single ownership, aligned coordinates, omitted outside points, global deterministic ordering on all ranks. |
| Associated test | `tests/test_postprocess_cache.py::test_sampling_nonunit_radius_and_missing_points`, serial and MPI. |
| Numerical consequences | Radius2 samples satisfy x²+y²≤4 and reproduce a known affine field to1e-12. Sampling beyond the mesh drops points without failure or misalignment. The old million-point default is retained; the new resolution argument permits inexpensive diagnostics. |

### A18 — major

| Required field | Evidence and action |
|---|---|
| Issue ID / classification | A18; confirmed geometry/cache error / verified repair |
| Severity | major |
| Original file and exact location | `src/disclinations/utils/__init__.py:552-610`; `meshes/primitives.py:321-404` |
| Original implementation | The cache helper sets radius=1 in the user's dictionary and generates a unit disk; cached reads return `mts=None,fts=None`; only the mesh is saved. The circle mesher has no outer-boundary physical tag. |
| Mathematical/computational problem | The requested physical domain is silently changed; fresh/cached simulations receive different metadata and cannot consistently select boundary measures. |
| Evidence | `audit/red_cache.txt` records requested radius2 being mutated to1. The repaired regression compares fresh/cached area with4π and checks cell/facet tag persistence. |
| Correct formulation | Respect input geometry; include its complete specification and cache schema in the identity; preserve required cell/facet tags on save/read. |
| Repair implemented | Copy geometry parameters, retain radius, tag outer circle6, persist/reload both MeshTags, version cache identity `circle-tags/v1`, collective directory/read decision. Older incompatible cache files remain untouched. |
| Associated test | `tests/test_postprocess_cache.py::test_mesh_cache_preserves_radius_and_tags`, serial and MPI. |
| Numerical consequences | A radius2 request now generates area≈4π rather than≈π. Cached and freshly generated meshes preserve equivalent measurable tags. This repair changes legacy runs that unintentionally relied on the forced unit radius. |

## Scope decisions, modelling choices and numerical limits

### U01 — confirmed scope exclusion: separate linear defect/crack model

**Location/evidence:** the current-tree inventory contains FvK plates, meshing,
utilities and SNES, without the mission's Burgers-vector representation,
excluded-core defect loading, fundamental-cell solver, distance sweep, or
internal crack bilinear form. The user confirmed that the core-regularised
implementation is elsewhere and requested work on this repository only.
**Classification:** scope decision, not a defect or delivery blocker.
**Action:** retain the current plate model and its verified repairs; do not
import a different branch or invent replacement defect physics.
**Associated tests:** current-repository verification in `audit/RESULTS.md`.
**Numerical consequences:** the delivered results apply to this plate code.
No repository path or external dataset is required to deliver the agreed scope;
results for the separate solver are not claimed. The U01 identifier is retained
so existing evidence references remain stable.

### U02 — critical scientific decision: independent core Airy constants/periods

**Original location:** `utils/__init__.py:469-500` at checkpoint (zero-value BC
helper), `models/*:penalisation` (unrestricted `ds`), and
`meshes/primitives.py:409-487` (hole geometry). **Original implementation:**
all exterior values are set to zero and all exterior slopes are penalised.
**Problem:** each traction-free boundary component admits its own affine Airy
data. One global gauge cannot generally eliminate all of them. Defect periods
and force/moment constraints are not specified. **Evidence:** the traction
identity σn=Jᵀ∂ₛ∇v and missing period-loading code; geometry tags alone do not
provide these constraints. **Correct formulation:** affine traces plus the
appropriate independent period/resultant equations, once the physical problem
is specified. **Repair:** the helper's actual H1 role is documented; tags and
geometry repaired, mechanical constraints retained rather than guessed.
**Test:** stress/gauge/traction identity and nonempty core measures only.
**Numerical consequence:** hole solves with homogeneous data may impose extra
constraints. They are not certified as general traction-free core models.

### U03 — critical scientific decision: ideal free slit versus exterior notch

**Original location:** `meshes/primitives.py:6-229`; there is no crack-specific
bilinear form in this checkout. **Original implementation:** polygonal notches
with finite opening `sep`; `mesh_rightCrack_gmshapi` actually cuts the left side
of its polygon despite its name. **Problem:** this geometry is not a
codimension-one internal crack with two independent traces. **Evidence:** the
repaired converted lip facets each have one adjacent cell and zero `dS(tag)`
measure. **Correct formulation:** depends on whether the unknown is primal
bending or dual Airy, on boundary affine data/gauges, and on a mesh/space with
independent lips. A free primal fourth-order boundary does not need a slope
penalty; Airy traction-free data can instead require affine trace constraints.
**Repair:** conflicting tags/names fixed; no geometry/topology or mechanical
model silently changed. **Test:** `test_notch_tags_disjoint`, including converted
adjacency and measures. **Numerical consequence:** these tests cannot certify
ideal crack-tip mechanics. Merely removing an interior penalty cannot remove
shared C0 value DOFs across a genuine internal slit.

### U04 — scope exclusion: external dislocation-distance scattering study

**Location/scope:** the dislocation-distance study belongs to the separate
implementation excluded by the user. Current sweep sources vary plate
thickness/aspect ratio/load parameters. This exclusion is not a delivery blocker.
**Problem:** no traceable dataset, geometry, intensity/orientation convention,
mesh sequence or solver diagnostics identifies the outlier runs. **Evidence:**
source inventory and archived numerical discussion describe a different plate
study. **Correct investigation:** compare fixed meshes and remeshed controls;
record core/crack clearance, core facets/DOFs, quadrature degree, conditioning,
KSP/SNES reason, residual, physical energy and separate stabilisation for every
sample; then refine h and p and vary core radius and penalty independently.
**Repair:** confirmed formulation/configuration defects were repaired, but no
outlier cause is assigned. **Test:** controlled smooth mesh/order/penalty and
intensity studies only. **Numerical consequence:** smooth benchmark convergence
does not establish convergence near a dislocation/crack tip; “mesh sensitivity”
is a hypothesis, not this audit's finding.

### U05 — major numerical limitation: global nonlinear physics and stabilisation

**Location/original:** all FvK variants and legacy beta/gamma experiments.
**Problem:** the complete nonlinear system is a saddle problem with possible
multiple equilibria and buckling; a local SNES solve does not select a uniquely
physical branch. α/h has no automatic degree/anisotropy adaptation. **Evidence:**
the variational functional and observed test scope; no new free-core or
near-tip data are available. **Correct formulation:** identify equilibria and
stability separately, choose penalties from appropriate trace bounds, and run
nonlinear manufactured/analytical h/p studies at the intended parameter ranges.
**Repair:** smooth coupled patch consistency and checked solves added; historical
coupling distinctions retained. **Tests:** coupled patch, symmetry of the
variational quadratic block, nine scalar/mixed load smoke solves. **Numerical
consequences:** all these pass; they do not establish nonlinear global
convergence, an hp coercivity theorem, or equivalence of all three schemes.

### U06 — moderate suspected inconsistency: historical benchmark/plotting workflows

**Location/original:** `test_branch/validation_test1*.py`,
`test_branch/validation_test2*.py`, `playground/*`, archive `numerics.tex`.
**Problem/evidence:** several different Airy/pressure scales occur; archive
statements use both D and D/t and contradictory pressure powers, whereas the
implemented constitutive/energy law fixes the resultants convention. Several
legacy plotting scripts require missing parameters or external data/packages.
These peripheral workflows were not all given independent end-to-end numerical
validation. **Correct formulation:** trace exact field/length units and source
conventions for each experiment. **Repair:** established dimensional energy
factors and API accesses were corrected; sampler and cache defects were
subsequently independently reproduced and repaired as A17/A18. Remaining
analytic-source conventions and historical plotted comparisons are not claimed
as verified repairs. **Associated tests:** constitutive and scale tests, but no
end-to-end acceptance of every historical plot. **Numerical consequences:**
historical figures and absolute-energy comparisons require rerunning a
resolved experiment-specific benchmark; they are not publication evidence from
this audit.

### U07 — moderate reproducibility limitation: fresh environment/platform

**Location/original:** dependency specifications and local native installations.
**Problem:** native solver/renderer compatibility differs by platform; the local
Python3.13 environment crashed in readline/pdb, while the Python3.11 stack ran
the tests. Docker was unavailable because its daemon was stopped. The existing
Conda registry also contains stale package records. **Evidence:** baseline
logs, `audit/runtime.json`, and successful tests in `fenicsx-env`.
**Correct workflow:** provision a clean matched native stack and rerun the
recorded commands. **Repair:** one metadata source, environment specification,
working package build and installation smoke test; no system environment was
repaired/replaced. **Tests:** final serial/MPI suite and wheel smoke. **Numerical
consequence:** the supplied code is executable on the recorded runtime; fresh
Conda resolution, other platforms, GPU/display rendering and MUMPS are not
certified. `environment.yml` is a specification, not a clean-install attestation.

## Numerical record

All energies below refer to the smooth unit-square **plane-stress Airy block**,
E=7, ν=.23, t=.4, α=40 and numerical exact potential
x²(1−x)²y²(1−y)². They are not crack/dislocation results. Exact membrane energy
is approximately **0.00058309037900875** in the benchmark's consistent units.

| Degree | Subdivisions n | Physical membrane energy | L2 error |
|---|---:|---:|---:|
| 2 | 4 | 0.000114318845025 | 9.27928476071e-4 |
| 2 | 8 | 0.000302340111415 | 4.78993061343e-4 |
| 2 | 16 | 0.000473965545750 | 1.69191666394e-4 |
| 3 | 4 | 0.000515319441177 | 1.01977155491e-4 |
| 3 | 8 | 0.000577491140996 | 9.16351517573e-6 |
| 3 | 16 | 0.000582843296867 | 6.55466350383e-7 |

Residual norms are approximately 1.6e-14 to 2.1e-12. The cubic n=16 physical
energy error is about 0.0424%; quadratic meshes here are visibly underresolved.
No asymptotic convergence rate is inferred from just these three meshes.
`audit/convergence.json` also records correction, discrete quadratic energy,
work, complete functional, solver reason and effective KSP/PC. Tests verify
work=2×discrete quadratic, reversal/linearity, quadratic physical-energy
scaling, repeatability, and a second penalty value.

The checkpoint comparison in `audit/before_after.json` uses the **same repaired
solver, source, mesh, degree and quadrature policy** for both sets of forms.
It isolates A01–A03 together; it is not an execution claim for the original
API-incompatible solver. At cubic n=16, the checkpoint physical energy is
0.000535987251507 and its L2 error is 0.000146978949512; the repaired values are
shown above. `audit/compare_forms.py` reproduces this comparison using the Git
checkpoint and a temporary import of its model source.

## Verification record and reproducible commands

Commands are run at the repository root with the recorded DOLFINx0.9/Python3.11
runtime. `python` below means that activated interpreter, not the host's
unconfigured Homebrew Python.

```bash
PYTHONPATH=src:. python -m pytest -q
mpiexec -n 2 python -m pytest tests/test_postprocess_cache.py tests/test_loads_solver.py tests/test_forms.py tests/test_geometry.py src/disclinations/test/test_models.py src/disclinations/test/test_models-monopole.py src/disclinations/test/test_models-transverse.py -q
PYTHONPATH=src:. python examples/verify_clamped.py
PYTHONPATH=src:. python audit/compare_forms.py
python audit/inventory.py
python -m pip wheel --no-deps --no-build-isolation --wheel-dir /tmp/disclinations-audit-wheel .
```

Final counts and installed-wheel results are in `audit/RESULTS.md` and the
referenced test logs. Earlier red/intermediate logs are retained as labelled
evidence and do not override the final results. Three skips are redundant
scalar-model combinations: the scalar biharmonic has no transverse field or
second coupled field. They are not skipped failed numerical checks. Manual
visualisation demos are main-guarded, not hidden numerical failures.

## External primary references

The reconstructed law and boundary identities are derived explicitly above
and in `docs/formulation.md`. The following references support API and generic
assembly conventions; their boundary conditions must not be substituted for
this project's clamped problem:

- [DOLFINx0.9 biharmonic demo](https://docs.fenicsproject.org/dolfinx/v0.9.0/python/demos/demo_biharmonic.html): interior normal jumps, two symmetric consistency terms, α/h. Its displayed exterior conditions differ from the clamped examples here.
- [DOLFINx0.9 PETSc assembly API](https://docs.fenicsproject.org/dolfinx/v0.9.0/python/generated/dolfinx.fem.petsc.html): lifting/BC alpha argument and nonfinalised vector assembly.
- [FEniCS0.9 release notes](https://fenicsproject.org/blog/v0.9.0/): Function.vector moved to Function.x.petsc_vec.
- [PETSc SNESGetConvergedReason](https://petsc.org/main/manualpages/SNES/SNESGetConvergedReason/): positive/negative reasons distinguish convergence from divergence.

## Intentional compatibility changes

Concrete model names, σ/bracket methods, energy tuple order and defect-helper
return interface are retained. Changes in computed values are expected from
the repaired forms, scales and curvature. Direct `ToyPlateFVK.penalisation`
now fails explicitly; use a concrete model. `compute_disclination_loads` rejects
outside/duplicated/mismatched points and needs the global ordered strength list
on all ranks. Its assembled-vector contract is documented. SNES divergence now
raises instead of returning a usable-looking state. Unspecified SNES tolerances
follow PETSc defaults instead of the old hard-coded1e-16 tolerances. Gmsh hole-model finalisation
belongs to the caller. Physical facet tags are disjoint; numeric lip tags are
retained. The sampling helper now has an explicitly collective common-point contract; `sample_function` returns globally ordered samples on all ranks and preserves its three-sequence return shape. Mesh caching respects the requested radius and uses a new schema identity to preserve tags without reading incompatible old caches. The dormant dg3 diagnostic remains a fifth tuple entry with zero.
The supported native API is pinned to0.9; later DOLFINx APIs require a separate
migration/verification. Old test/plotting sources and all pre-audit user changes
remain recoverable at `71e11e4`.
