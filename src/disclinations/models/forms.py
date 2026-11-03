"""Shared differential operators and homogeneous clamped C0-IP energies.

These are quadratic functionals, not bilinear evaluations. Differentiating
the consistency product supplies both symmetric terms, each with unit weight.
Boundary measures must contain only facets where zero slope is intended.
"""
import ufl


def hessian(u):
    return ufl.grad(ufl.grad(u))


def laplacian(u):
    return ufl.div(ufl.grad(u))


def airy_stress(u):
    """cof(Hess u), a stress resultant [N/m] when u has units [N m]."""
    return laplacian(u) * ufl.Identity(2) - hessian(u)


def bracket(u, v):
    """von Kármán bracket: u_xx v_yy + u_yy v_xx - 2 u_xy v_xy."""
    return ufl.inner(hessian(u), airy_stress(v))


def fvk_coupling_residual(state, v_test, w_test, msh, facet_fluxes=True):
    """Bracket residual, with optional symmetric broken gradient fluxes.

    The two tests are supplied by the caller; no assumption about whether
    state entries are scalar Functions or mixed-space components is needed.
    """
    v, w = state["v"], state["w"]
    dx = ufl.Measure("dx", domain=msh)
    residual = -bracket(w, v) * w_test * dx - 0.5 * bracket(w, w) * v_test * dx
    if not facet_fluxes:
        return residual
    n = ufl.FacetNormal(msh)
    dS, ds = ufl.Measure("dS", domain=msh), ufl.Measure("ds", domain=msh)

    def edge(f, g, test):
        flux = ufl.dot(ufl.avg(airy_stress(f)) * ufl.grad(g("+")), n("+")) + ufl.dot(
            ufl.avg(airy_stress(f)) * ufl.grad(g("-")), n("-")
        )
        return (
            flux * ufl.avg(test) * dS
            + ufl.dot(airy_stress(f) * ufl.grad(g), n) * test * ds
        )

    return (
        residual
        + 0.5 * edge(v, w, w_test)
        + 0.5 * edge(w, v, w_test)
        + 0.5 * edge(w, w, v_test)
    )


def c0ip_quadratic_terms(u, msh, alpha, moment=None, interior=None, boundary=None):
    """Return unscaled consistency/penalty energies for a scalar C0 field.

    moment is the derivative of the unit bulk density with respect to Hess u.
    Defaults to Hess u; Laplacian bulk requires (Delta u) I. alpha is a
    positive dimensionless parameter; degree dependence is chosen by caller.
    No coercivity claim is made for every positive alpha or arbitrary mesh.
    """
    if alpha <= 0:
        raise ValueError("alpha_penalty must be positive")
    n, h = ufl.FacetNormal(msh), ufl.CellDiameter(msh)
    interior = ufl.Measure("dS", domain=msh) if interior is None else interior
    boundary = ufl.Measure("ds", domain=msh) if boundary is None else boundary
    moment = hessian(u) if moment is None else moment
    normal_moment = ufl.inner(moment, ufl.outer(n, n))
    slope = ufl.dot(ufl.grad(u), n)
    slope_jump = ufl.jump(ufl.grad(u), n)
    return {
        "interior_consistency": -ufl.avg(normal_moment) * slope_jump * interior,
        "interior_penalty": 0.5 * alpha / ufl.avg(h) * slope_jump**2 * interior,
        "boundary_consistency": -normal_moment * slope * boundary,
        "boundary_penalty": 0.5 * alpha / h * slope**2 * boundary,
    }
