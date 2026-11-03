import ufl
from ufl import avg, ds, dS, outer, div, grad, inner, dot, jump
import basix
import dolfinx
import yaml
import os
import numpy as np
from mpi4py import MPI

from . import (
    AIRY,
    TRANSVERSE,
    default_parameters,
    default_model_parameters,
    Biharmonic,
    ToyPlateFVK,
    create_disclinations,
    compute_energy_terms,
)
from .forms import laplacian, hessian, c0ip_quadratic_terms, fvk_coupling_residual
from .material import PlateMaterial


class A_NonlinearPlateFVK(ToyPlateFVK):
    """Dimensionless FvK model: x=R xi, w=t w~, v=E t³ v~."""

    def __init__(self, mesh, model_parameters=None, smooth=False) -> None:
        model_parameters = {} if model_parameters is None else model_parameters
        self.alpha_penalty = model_parameters.get(
            "alpha_penalty", default_model_parameters["alpha_penalty"]
        )
        self.nu = model_parameters.get("nu", default_model_parameters["nu"])
        self.E = model_parameters.get("E", default_model_parameters["E"])
        self.t = model_parameters.get(
            "thickness", default_model_parameters["thickness"]
        )
        self.material = PlateMaterial(self.E, self.nu, self.t)
        self.c_nu = self.material.dimensionless_bending
        self.smooth = smooth

        # Scaling parameters to get dimensional physical values starting from dimensionless values
        self.v_scale = self.E * (self.t**3)
        self.w_scale = self.t

        self.mesh = mesh

        if model_parameters.get("order", default_model_parameters["order"]) < 2:
            raise ValueError("Fourth-order C0-IP models require polynomial order >= 2")
        X = basix.ufl.element(
            "P",
            str(mesh.ufl_cell()),
            model_parameters.get("order", default_model_parameters["order"]),
        )
        self.Q = dolfinx.fem.functionspace(mesh, basix.ufl.mixed_element([X, X]))

        self.dx = ufl.Measure("dx")
        self.dS = ufl.Measure("dS")
        self.ds = ufl.Measure("ds")

        self._configure_penalty_terms()

    def _configure_penalty_terms(self):
        def term(u, name):
            return c0ip_quadratic_terms(
                u, self.mesh, self.alpha_penalty, interior=self.dS, boundary=self.ds
            )[name]

        self.dg1 = lambda u: term(u, "interior_consistency")
        self.dg2 = lambda u: term(u, "interior_penalty")
        self.bc1 = lambda u: term(u, "boundary_consistency")
        self.bc3 = lambda u: term(u, "boundary_penalty")
        n, h = ufl.FacetNormal(self.mesh), ufl.CellDiameter(self.mesh)
        self.dg3 = (
            lambda u: 0.5
            * self.alpha_penalty
            / avg(h)
            * inner(jump(hessian(u), n), jump(hessian(u), n))
            * self.dS
        )
        self.dgc = (
            lambda w, v: avg(inner(self.W(w), outer(n, n))) * jump(grad(v), n) * self.dS
        )

    def M(self, f):
        return grad(grad(f)) + self.nu * self.σ(f)

    def P(self, f):
        return grad(grad(f)) - self.nu * self.σ(f)

    def W(self, f):
        J = ufl.as_matrix([[0, -1], [1, 0]])
        return -0.5 * J.T * (outer(grad(f), grad(f))) * J

    def energy(self, state):
        v = state["v"]
        w = state["w"]
        dx = self.dx

        membrane = 1 / 2 * inner(hessian(v), hessian(v)) * dx
        bending = self.c_nu / 2 * inner(hessian(w), hessian(w)) * dx
        coupling = 1 / 2 * inner(self.σ(v), outer(grad(w), grad(w))) * dx
        energy = bending - membrane + coupling
        return energy, bending, membrane, coupling

    def penalisation(self, state):
        v = state["v"]
        w = state["w"]
        return_value = (
            self.c_nu * self.dg1(w)
            + self.dg2(w)
            - self.dg1(v)
            - self.dg2(v)
            + self.c_nu * self.bc1(w)
            - self.bc1(v)
            + self.bc3(w)
            - self.bc3(v)
            + self.dgc(w, v)
        )
        return return_value

    def compute_bending_energy(self, state, COMM):
        return COMM.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(self.energy(state)[1])),
            op=MPI.SUM,
        )

    def compute_membrane_energy(self, state, COMM):
        return COMM.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(self.energy(state)[2])),
            op=MPI.SUM,
        )

    def compute_coupling_energy(self, state, COMM):
        return COMM.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(self.energy(state)[3])),
            op=MPI.SUM,
        )

    def compute_penalisation(self, state, COMM):
        return COMM.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(self.penalisation(state))),
            op=MPI.SUM,
        )

    def compute_penalisation_terms_w(self, state, COMM):
        """Weighted w corrections; final dormant dg3 entry is zero."""
        w = state["w"]
        dg1_computed = COMM.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(self.dg1(w))), op=MPI.SUM
        )
        dg2_computed = COMM.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(self.dg2(w))), op=MPI.SUM
        )
        bc1_computed = COMM.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(self.bc1(w))), op=MPI.SUM
        )
        bc3_computed = COMM.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(self.bc3(w))), op=MPI.SUM
        )
        dg3_computed = 0.0  # dg3 is not part of the assembled functional
        return (
            self.c_nu * dg1_computed,
            dg2_computed,
            self.c_nu * bc1_computed,
            bc3_computed,
            dg3_computed,
        )

    def compute_penalisation_terms_v(self, state, COMM):
        """Unsigned v-block corrections; subtract their sum in the saddle."""
        v = state["v"]
        dg1_computed = COMM.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(self.dg1(v))), op=MPI.SUM
        )
        dg2_computed = COMM.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(self.dg2(v))), op=MPI.SUM
        )
        bc1_computed = COMM.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(self.bc1(v))), op=MPI.SUM
        )
        bc3_computed = COMM.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(self.bc3(v))), op=MPI.SUM
        )
        return dg1_computed, dg2_computed, bc1_computed, bc3_computed

    def compute_total_penalisation_w(self, state, COMM):
        return_value = 0
        for element in self.compute_penalisation_terms_w(state, COMM):
            return_value += element
        return return_value

    def compute_total_penalisation_v(self, state, COMM):
        return_value = 0
        for element in self.compute_penalisation_terms_v(state, COMM):
            return_value += element
        return return_value

    def compute_penalisation_coupling(self, state, COMM):
        v = state["v"]
        w = state["w"]
        return COMM.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(self.dgc(w, v))), op=MPI.SUM
        )

    def gaussian_curvature(self, w, order=1):
        mesh = self.mesh
        DG_e = basix.ufl.element("DG", str(mesh.ufl_cell()), order)
        DG = dolfinx.fem.functionspace(mesh, DG_e)
        kappa = 0.5 * self.bracket(w, w)  # det Hess(w), small-slope Gaussian curvature
        kappa_expr = dolfinx.fem.Expression(kappa, DG.element.interpolation_points())
        Kappa = dolfinx.fem.Function(DG)
        Kappa.interpolate(kappa_expr)

        return Kappa


class A_NonlinearPlateFVK_brenner(A_NonlinearPlateFVK):
    def energy(self, state):
        """Return uncoupled dimensionless saddle and bulk diagnostics."""
        _, bending, membrane, coupling = super().energy(state)
        return bending - membrane, bending, membrane, coupling

    def coupling_term(self, state, v_test, w_test):
        """Nonlinear bracket residual with broken facet fluxes."""
        return fvk_coupling_residual(
            state, v_test, w_test, self.mesh, facet_fluxes=True
        )

    def penalisation(self, state):
        v = state["v"]
        w = state["w"]

        return (
            self.c_nu * self.dg1(w)
            + self.dg2(w)
            - self.dg1(v)
            - self.dg2(v)
            + self.c_nu * self.bc1(w)
            - self.bc1(v)
            + self.bc3(w)
            - self.bc3(v)
        )


class A_NonlinearPlateFVK_carstensen(A_NonlinearPlateFVK_brenner):
    def coupling_term(self, state, v_test, w_test):
        """Nonlinear bracket residual with volume terms only."""
        return fvk_coupling_residual(
            state, v_test, w_test, self.mesh, facet_fluxes=False
        )
