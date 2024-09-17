import ufl
from ufl import (avg, ds, dS, outer, div, grad, inner, dot, jump)
import basix 
import dolfinx
import yaml
import os

AIRY = 0
TRANSVERSE = 1

dir_path = os.path.dirname(os.path.realpath(__file__))
with open(f"{dir_path}/default_parameters.yml") as f:
    default_parameters = yaml.load(f, Loader=yaml.FullLoader)

default_model_parameters = default_parameters["model"]

laplacian = lambda f : div(grad(f))
hessian = lambda f : grad(grad(f))

class Biharmonic:
    def __init__(self, mesh, model_parameters = {}) -> None:
        self.alpha_penalty = model_parameters.get("alpha_penalty",
                                                       default_model_parameters["alpha_penalty"])
        self.mesh = mesh
    
    def energy(self, state):
        u = state["u"]
        dx = ufl.Measure("dx")

        return (1/2 * (inner(div(grad(u)), div(grad(u))))) * dx
    
    def penalisation(self, state):
        u = state["u"]
        α = self.alpha_penalty
        h = ufl.CellDiameter(self.mesh)
        n = ufl.FacetNormal(self.mesh)

        dS = ufl.Measure("dS")
        ds = ufl.Measure("ds")
        
        dg1 = lambda u: 1/2 * dot(jump(grad(u)), avg(grad(grad(u)) * n)) * dS
        dg2 = lambda u: 1/2 * α/avg(h) * inner(jump(grad(u)), jump(grad(u))) * dS
        bc1 = lambda u: 1/2 * α/h * inner(grad(u), grad(u)) * ds
        
        return - dg1(u) + dg2(u) + bc1(u)
    
class ToyPlateFVK:
    def __init__(self, mesh, model_parameters = {}) -> None:
        self.alpha_penalty = model_parameters.get("alpha_penalty",
                                                       default_model_parameters["alpha_penalty"])
    
    def σ(self, v):
        # J = ufl.as_matrix([[0, -1], [1, 0]])
        # return inner(grad(grad(v)), J.T * grad(grad(v)) * J)
        return div(grad(v)) * ufl.Identity(2) - grad(grad(v))

    def bracket(self, f, g):
        J = ufl.as_matrix([[0, -1], [1, 0]])
        return inner(grad(grad(f)), J.T * grad(grad(g)) * J)
    
    def penalisation(self, state):
        v = state["v"]
        w = state["w"]
        α = self.alpha_penalty
        h = ufl.CellDiameter(self.mesh)
        n = ufl.FacetNormal(self.mesh)

        dS = ufl.Measure("dS")
        ds = ufl.Measure("ds")
        
        dg1 = lambda u: 1/2 * dot(jump(grad(u)), avg(grad(grad(u)) * n)) * dS
        dg2 = lambda u: 1/2 * α/avg(h) * inner(jump(grad(u)), jump(grad(u))) * dS
        bc1 = lambda u: 1/2 * inner(grad(u), grad(grad(u)) * n) * ds
        bc2 = lambda u: 1/2 * α/h * inner(grad(u), grad(u)) * ds
        
        return - dg1(w) + dg2(w) \
                - dg1(v) + dg2(v) \
                - bc1(w) - bc2(w) \
                - bc1(v) + bc2(v)

class NonlinearPlateFVK(ToyPlateFVK):
    def __init__(self, mesh, model_parameters = {}) -> None:
        self.alpha_penalty = model_parameters.get(
            "alpha_penalty",
            default_model_parameters["alpha_penalty"])
        self.nu = model_parameters.get("nu", default_model_parameters["nu"])
        self.E = model_parameters.get("E", default_model_parameters["E"])
        self.t = model_parameters.get("thickness", default_model_parameters["thickness"])
        self.D = self.E * self.t**3 / (12*(1-self.nu**2))
        
        self.mesh = mesh
    
    def energy(self, state):
        v = state["v"]
        w = state["w"]
        dx = ufl.Measure("dx")

        D = self.D
        nu = self.nu
        Eh = self.E*self.t
        k_g = -D*(1-nu)


        membrane = (1/(2*Eh) * inner(hessian(v), hessian(v)) + nu/(2*Eh) * self.bracket(v, v)) * dx
        bending = (D/2 * (inner(laplacian(w), laplacian(w))) + k_g/2 * self.bracket(w, w)) * dx
        coupling = 1/2 * inner(self.σ(v), outer(grad(w), grad(w))) * dx # compatibility coupling term
        energy = bending - membrane + coupling

        return energy, bending, membrane, coupling

    def M(self, f):
        return grad(grad(f)) + self.nu*self.σ(f)
    
    def P(self, f):
        c_nu = 1. #12*(1-self.nu**2)
        return (1.0/c_nu)*grad(grad(f)) - self.nu*self.σ(f)

    def W(self, f):
        J = ufl.as_matrix([[0, -1], [1, 0]])
        return -0.5*J.T*(outer(grad(f), grad(f))) * J
    
    def penalisation(self, state):
        v = state["v"]
        w = state["w"]
        α = self.alpha_penalty
        h = ufl.CellDiameter(self.mesh)
        n = ufl.FacetNormal(self.mesh)
        nu = self.nu

        dS = ufl.Measure("dS")
        ds = ufl.Measure("ds")
        c_nu = 12*(1-nu**2)
        
        # M = lambda f : grad(grad(f)) + nu*self.σ(f)       
        M = lambda f : self.M(f)       
        # P = lambda f : (1.0/c_nu)*grad(grad(f)) - nu*self.σ(f)
        P = lambda f : self.P(f)
        W = lambda f : self.W(f)
        
        dg1 = lambda u: - 1/2 * dot(jump(grad(u)), avg(grad(grad(u)) * n)) * dS
        dg2 = lambda u: + 1/2 * α/avg(h) * inner(jump(grad(u)), jump(grad(u))) * dS
        dgc = lambda f, g: avg(inner(W(f), outer(n, n)))*jump(grad(g), n)*dS

        # bc1 = lambda u: - 1/2 * inner(grad(u), n) * inner(M(u), outer(n, n)) * ds
        bc1 = lambda u: - 1/2 * inner(grad(u), n) * inner(grad(grad(u)), outer(n, n)) * ds
        # bc2 = lambda u: - 1/2 * inner(grad(u), n) * inner(P(u), outer(n, n)) * ds
        bc2 = lambda u: - 1/2 * inner(grad(u), n) * inner(grad(grad(u)), outer(n, n)) * ds
        bc3 = lambda u: 1/2 * α/h * inner(grad(u), grad(u)) * ds
        
        c1 = self.D
        c2 = (1/(self.E*self.t))
        
        return   (c1*dg1(w) + c1*dg2(w)) \
                - c2*dg1(v) - c2*dg2(v) \
                + c2*bc1(w) - c2*bc2(v) \
                + c1*bc3(w) - c1*bc3(v) 
                # + dgc(w, v)

    def gaussian_curvature(self, w, order = 1):
        mesh = self.mesh
        DG_e = basix.ufl.element("DG", str(mesh.ufl_cell()), order)
        DG = dolfinx.fem.functionspace(mesh, DG_e)
        kappa = self.bracket(w, w)
        kappa_expr = dolfinx.fem.Expression(kappa, DG.element.interpolation_points())
        Kappa = dolfinx.fem.Function(DG)
        Kappa.interpolate(kappa_expr)

        return Kappa
    
class NonlinearPlateFVK_brenner(NonlinearPlateFVK):
    def coupling_term(self, state):
        v = state["v"]
        w = state["w"]
        # w_function_space = w.ufl_operands[0].ufl_function_space()
        # v_function_space = v.ufl_operands[0].ufl_function_space()
        _function_space = v.ufl_operands[0].ufl_function_space()
        _test = ufl.TestFunction(_function_space)

        # Split the test function into subspaces for _w_test and _v_test
        _w_test, _v_test = ufl.split(_test)
        
        dx = ufl.Measure("dx")
        dS = ufl.Measure("dS")
        ds = ufl.Measure("ds")
        
        n = ufl.FacetNormal(self.mesh)

        coupling_in_edge = lambda f1, f2, test: ( dot(dot(avg(self.σ(f1)),grad(f2('+'))), n('+')) + dot(dot(avg(self.σ(f1)),grad(f2('-'))), n('-')) )* avg(test) * dS
        coupling_bnd_edge = lambda f1, f2, test: dot(dot(self.σ(f1),grad(f2)), n) * test * ds

        cw_bulk = - self.bracket(w,v) * _w_test * dx
        cw_in_edges = 0.5*coupling_in_edge(v, w, _w_test) + 0.5*coupling_in_edge(w, v, _w_test)
        cw_bnd_edges = 0.5*coupling_bnd_edge(v, w, _w_test) + 0.5*coupling_bnd_edge(w, v, _w_test)

        cv_bulk = - 0.5*self.bracket(w,w) * _v_test * dx
        cv_in_edges = 0.5*coupling_in_edge(w, w, _v_test)
        cv_bnd_edges = 0.5*coupling_bnd_edge(w, w, _v_test)
        
        return cw_bulk + cv_bulk + cw_bnd_edges + cv_bnd_edges + cv_in_edges + cw_in_edges
    
class NonlinearPlateFVK_carstensen(NonlinearPlateFVK):
    def coupling_term(self, state):
        v = state["v"]
        w = state["w"]
        # w_function_space = w.ufl_operands[0].ufl_function_space()
        # v_function_space = v.ufl_operands[0].ufl_function_space()
        _function_space = v.ufl_operands[0].ufl_function_space()
        _test = ufl.TestFunction(_function_space)

        # Split the test function into subspaces for _w_test and _v_test
        _w_test, _v_test = ufl.split(_test)
        
        dx = ufl.Measure("dx")
        dS = ufl.Measure("dS")
        ds = ufl.Measure("ds")
        
        n = ufl.FacetNormal(self.mesh)

        coupling_bnd_edge = lambda f1, f2, test: dot(dot(self.σ(f1),grad(f2)), n) * test * ds

        cw_bulk = - self.bracket(w,v) * _w_test * dx
        cw_bnd_edges = 0.5*coupling_bnd_edge(v, w, _w_test) + 0.5*coupling_bnd_edge(w, v, _w_test)

        cv_bulk = - 0.5*self.bracket(w,w) * _v_test * dx
        cv_bnd_edges = 0.5*coupling_bnd_edge(w, w, _v_test)
        
        return cw_bulk + cv_bulk + cw_bnd_edges + cv_bnd_edges
    