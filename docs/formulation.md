# Reconstructed formulation and implementation map

## Scope and evidence

This document describes the supplied `final-figures` implementation and its
verified repairs. The user confirmed that the core-regularised linear
elasticity/crack implementation lives elsewhere and excluded it from this audit. The original working tree is checkpoint `71e11e4`.
`paper/Archive.zip!numerics.tex` identifies a thin-plate formulation and the
scales below; `!discrete.tex` contains an inconsistent half-weight consistency
functional. Those draft files are evidence of intent, not mathematical authority.

## Kinematics and material law

The dimensional unknowns are an Airy potential **v for membrane stress
resultants**, and transverse displacement w. For a smooth domain the implemented
homogeneous admissible space is H²₀(Ω) × H²₀(Ω). The finite element fields are
continuous Lagrange polynomials of degree p≥2, with zero value traces imposed
strongly; their normal derivatives are imposed weakly. Curved discrete
boundaries approximate the continuum domain.

Let H(v)=∇²v, Δv=tr H(v), σ(v)=cof H(v)=Δv I−H(v), and
[f,g]=f_xx g_yy+f_yy g_xx−2f_xy g_xy. Then div σ(v)=0 is an identity,
not an imposed equation. Small-slope Gaussian curvature is det H(w)=[w,w]/2.
`models/forms.py` defines these shared operators; the public σ/bracket methods
are retained. A global affine addition to v leaves stresses unchanged; clamped
traces fix this gauge. The nonlinear plate does not obey linear superposition.

The model assumes isotropic plane-stress membrane response and
Kirchhoff–Love bending, with thickness t, Young's modulus E and Poisson ratio ν.
Write N=σ(v) [N/m], rather than interpreting it directly as Cauchy stress [Pa].
The thickness-averaged in-plane Cauchy stress would be N/t. With E [Pa], t [m],
v [N m], w [m], the plane-stress compliance is

$$\epsilon=\frac{(1+\nu)N-\nu\,\mathrm{tr}(N)I}{Et}.$$

Consequently

$$C_1=D=\frac{Et^3}{12(1-\nu^2)}\quad[\mathrm{N\,m}],\qquad
C_2=\frac1{Et}\quad[\mathrm{m/N}].$$

These are the original **lower-case c1/c2**, not arbitrary normalisations.
`PlateMaterial` defines them once. The in-plane displacement and strains are
not primary finite element unknowns in this code.

For comparison, a truly two-dimensional **plane-strain Cauchy-stress** problem
has compliance ε=(1+ν)(σ−ν tr(σ)I)/E and Airy compatibility coefficient
(1−ν²)/E. Plane stress has coefficient 1/E. These relations cannot be
interchanged with the plate-resultant coefficient 1/(Et). For a Cauchy Airy
potential χ [N], the corresponding plane-strain complementary density is
(1+ν)(|Hχ|²−ν(Δχ)²)/(2E). No plane-strain mode has been added.

## Dimensional energy and continuous variational equations

The source-free bulk densities in `NonlinearPlateFVK.energy` are

$$U_b=\frac D2\int_\Omega\{(1-\nu)|H(w)|^2+\nu(\Delta w)^2\},$$
$$U_m=\frac1{2Et}\int_\Omega\{|H(v)|^2-\nu[v,v]\},\qquad
C=\frac12\int_\Omega \sigma(v):(\nabla w\otimes\nabla w).$$

The returned first component is **S=U_b−U_m+C**, a saddle functional.
It is not U_b+U_m. Pressure work is W=∫fw. A point incompatibility vector
adds +ℓ(v) to the functional/residual in the existing calling convention.
The variational class finds a stationary point of S−W+ℓ. Define

$$M(w)=H(w)+\nu\sigma(w),\quad P(v)=H(v)-\nu\sigma(v).$$

For test pair (η,z)∈H²₀×H²₀, its equations are

$$-C_2\int P(v):H(\eta)
 +\frac12\int\sigma(\eta):(\nabla w\otimes\nabla w)+\ell(\eta)=0,$$
$$D\int M(w):H(z)+\int\sigma(v)\nabla w\cdot\nabla z=\int fz.$$

For smooth fields, with the sign of ℓ defined as above, this gives

$$C_2\Delta^2v=-\tfrac12[w,w]+\theta,\qquad
D\Delta^2w=[v,w]+f.$$

Here θ denotes the numerical incompatibility distribution used in ℓ. A
physical Frank-angle sign requires a declared orientation/convention for the
underlying incompatibility operator. The code supplies signed Dirac weights,
not a complete derivation linking them to Frank angles on excluded cores.
There is no Burgers-vector data structure, dislocation derivative source,
period constraint, or core-radius regularisation in the solved forms.

`NonlinearPlateFVK_brenner` and `_carstensen` omit C from the differentiated
energy and add an explicit bracket residual. They retain different facet
couplings. Their energy tuple still reports C diagnostically. The shared
`fvk_coupling_residual` represents those existing expressions. Smooth coupled
patch tests verify consistency; nonlinear equivalence, branch selection and
publication-quality nonlinear convergence remain unproved.

## Nondimensionalisation

`A_NonlinearPlateFVK` uses x=Rξ, w=t w~, v=Et³ v~. Thus the energy scale is
Et⁵/R² [J] and cν=1/[12(1−ν²)]. Its bulk is

$$\widetilde S=\frac{c_\nu}{2}\int|H_\xi w|^2
-\frac12\int|H_\xi v|^2
+\frac12\int\sigma_\xi(v):(\nabla_\xi w\otimes\nabla_\xi w).$$

Under homogeneous clamped conditions, ∫[u,u]=0 and ∫|H(u)|²=∫(Δu)²;
these are continuum null-Lagrangian identities. They do **not** hold
cell-by-cell for C0 finite element fields. The nondimensional model deliberately
uses its Hessian-based bulk and matching Hessian flux; it has not silently been
changed to the dimensional bulk on broken spaces. Pressure scale becomes
(p₀/E)(R/t)⁴ and Dirac strength scale (R/t)². The archive contains conflicting
powers in later displayed equations; the rescaling algebra fixes the result.

`calculate_rescaling_factors` is a separate, historical analytic-benchmark
scaling (v₀=D, w₀=√(2D/Et)); it must not be confused with Et³/t scaling.
Its f coefficient √(2D³/Et) has units N m² and needs R⁻⁴ to be a pressure on
a general physical radius. Historic examples frequently set numerical R=1.

## Integration by parts, facets and slope conditions

For any smooth symmetric tensor T on a cell K,

$$\int_K T:H(z)=\int_K\mathrm{div\,div}(T)z+
\int_{\partial K}(Tn)\cdot\nabla z-
\int_{\partial K}(\mathrm{div}T\cdot n)z.$$

For continuous z, tangential derivatives agree across an ordinary straight
facet. The surviving normal term is {T_nn} j(z), with
j(z)=∇z⁺·n⁺+∇z⁻·n⁻ and {T_nn}=(T⁺_nn+T⁻_nn)/2. T_nn is even under normal
reversal. Averaging **Hn** with the two outward normals is not this scalar flux.
The scalar Laplacian energy requires T=(Δu)I; the dimensional plate blocks
require T=M(u) or P(u), and the reduced dimensionless block requires T=H(u).

For each block with positive bulk coefficient c, `c0ip_quadratic_terms` returns

$$Q_{\rm corr}(u)=c\left[-\int_{\mathcal E_i}\{T_{nn}(u)\}j(u)
+\frac12\int_{\mathcal E_i}\frac\alpha{\{h\}}j(u)^2
-\int_{\Gamma_0}T_{nn}(u)\partial_nu
+\frac12\int_{\Gamma_0}\frac\alpha h(\partial_nu)^2\right].$$

Here T is linear in u. Its derivative has **both** consistency terms with
coefficient −1, and the penalty α/h times the slope product. The original
−1/2 consistency energy yielded −1/2 in each term and failed the patch test.
The dimensional signed saddle correction is Q_b−Q_m; it therefore uses D on
all bending terms and 1/(Et) on all membrane terms. The variational class also
retains its nonlinear facet correction

$$\int_{\mathcal E_i}\{W(w)_{nn}\}j(v),\quad
W(w)=-\tfrac12 J^T(\nabla w\otimes\nabla w)J.$$

This cancels the broken Hessian-test flux of the coupling in the Airy equation;
it vanishes on smooth fields, but its variation need not vanish. It was not
removed because its value is small. Coupled polynomial tests cover this term.

h is `CellDiameter`; the interior denominator is its arithmetic average. α is
positive, dimensionless, and must exceed a mesh/degree-dependent trace-inverse
bound for scalar-block coercivity. α=40 is tested for p=2,3 on the reported
meshes; arbitrary p/anisotropy/coercivity is not certified. There is no automatic
p scaling. The nondimensional scheme historically uses α independent of cν;
this is an explicit effective penalty choice, not a missing dimensional factor.
The complete FvK Hessian is a saddle operator, so global coercivity is not an
appropriate claim. UFL estimates quadrature order automatically; the independent
polynomial identities exercise high-order quadrature. Singular-source and
curved-mesh quadrature convergence are not established.

## Mechanical boundaries: Airy traction versus plate slope

Let t=Jn be the oriented tangent and σ(v)=JᵀH(v)J. Then
σ(v)n=Jᵀ ∂ₛ∇v. Hence zero traction is equivalent, on a smooth connected
boundary component, to ∇v=a (a constant vector), and v=a·x+b. Its normal
derivative is a·n, not an arbitrary independently chosen zero derivative.
One global affine gauge sets a=b=0 on **one** connected boundary component.
Separate holes may carry distinct constants and require period/resultant
conditions. Setting all hole values/slopes to zero can therefore constrain
additional physical degrees of freedom. No missing period conditions have been
invented in this repair.

By contrast, for a primal bending field with free value and free slope, splitting
Tn into normal/tangential components and integrating tangentially gives natural
conditions T_nn=0 and div T·n+∂ₛT_nt=0, with endpoint/corner forces where relevant.
Those free conditions do not require symmetric slope Nitsche terms or a slope
penalty. Such terms enforce a selected slope, rather than simply making a free
boundary smoother. They can be appropriate for an Airy field **only when** its
affine boundary data/gauge and topology justify that selected slope.

The schematic crack form in the request is a slope-enforcing Nitsche form for
Hessian bulk, with γ≈α/h (and the block's material coefficient). It is not, by
itself, proof of two free mechanical crack lips. If a true internal slit is
represented by two independent traces, ordinary C0-IP facet terms must not
couple the lips. Merely excluding crack facets from `dS` still leaves shared
C0 value DOFs unless the space/mesh is split. `ds` integrates exterior facets;
`dS` integrates facets with two adjacent cells. The present notch mesher makes
exterior finite-width lips, not an internal `dS(crack)` set. Its tags are now
disjoint; the mechanical conditions are still the caller's responsibility.

## Sources, energies, solvers and verification boundary

`utils/la.py` pairs each point with its strength before collective ownership
routing, uses scalar Lagrange basis evaluations, reverse-accumulates the dual
vector once, then synchronises ghosts. Out-of-domain and duplicated input
points are errors. `SNESSolver.F` adds its b0 **owned** entries after reverse
assembly; ghosts must not be accumulated twice. Strong BC lifting and zeroed
residual DOFs are applied as part of assembly. The Jacobian refreshes the
coefficient state even if no residual callback ran first.

Only SNES is used in the production package; there was no `LinearProblem`
call to which SNES YAML options were misapplied. Default linear subproblems use
KSP `preonly`/PC `lu`. A caller may request MUMPS using
`pc_factor_mat_solver_type=mumps` if the build supports it; this backend was
not assumed. Effective SNES/KSP/PC options and divergence handling are tested.
Options have a collective unique prefix and are removed/restored after parsing.
A converged SNES reason alone is not evidence of a correct PDE. No affine
nullspace is attached for the strongly clamped examples; free or multiply
connected Airy gauges need separate constraints before such solves are valid.

The example reports U_m (physical complementary membrane energy for the flat
Airy benchmark), Q_corr (numerical consistency+stabilisation, not physical
energy), Q_h=U_m+Q_corr=½a_h(u,u), load work ℓ_h(u), and F_h=Q_h−ℓ_h(u).
At a linear stationary solution ℓ_h(u)=2Q_h. Norms, squared integrals, and
energy are not interchangeable. In particular neither `E/(1−nu²)` times an
unspecified Hessian norm nor the saddle's first tuple component is a justified
physical energy. The quoted `stats['hess_l2_norm']` expression does not occur
in the available source/notebook code.

`utils/sample_function.py` routes a common sample list from rank zero, omits
outside points, and returns globally ordered disk samples without duplicating
partition-interface values. The chord is sqrt(R²−y²). The mesh cache respects
the input radius and preserves cell/facet tags under a versioned schema.

Physical stress/curvature tests, polynomial residuals, mesh/order convergence,
intensity reversal/scaling, solver failure tests, and two-rank assembly tests
are evidence for their stated cases. They do not establish core-radius,
crack-tip, remeshing, near-contact, nonlinear bifurcation, or dislocation-sweep
convergence. The separate core/crack/dislocation study is outside the agreed
repository scope; nonlinear bifurcation and historical plate sweeps retain the
verification limits stated in the audit.
