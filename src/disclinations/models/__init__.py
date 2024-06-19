import ufl
from ufl import (
    avg,
    ds,
    dS,
    outer,
    div,
    grad,
    inner,
    dot,
    jump,
)

import yaml
import os
from dolfinx.fem import Constant

dir_path = os.path.dirname(os.path.realpath(__file__))
with open(f"{dir_path}/default_parameters.yml") as f:
    default_parameters = yaml.load(f, Loader=yaml.FullLoader)

default_model_parameters = default_parameters["model"]


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
        
        return - (dg1(w) + dg2(w)) \
                - dg1(v) + dg2(v) \
                - bc1(w) - bc2(w) \
                - bc1(v) - bc2(v) 

class NonlinearPlateFVK(ToyPlateFVK):
    def __init__(self, mesh, model_parameters = {}) -> None:
        self.alpha_penalty = model_parameters.get("alpha_penalty",
                                                       default_model_parameters["alpha_penalty"])
        self.nu = model_parameters.get("nu",
                                       default_model_parameters["nu"])
        self.E = model_parameters.get("E",
                                      default_model_parameters["E"])
        self.h = model_parameters.get("h",
                                        default_model_parameters["h"])
        self.D = self.E * self.h**3 / (12*(1-self.nu**2))
        
        self.mesh = mesh
    
    def energy(self, state):
        v = state["v"]
        w = state["w"]
        dx = ufl.Measure("dx")

        D = self.D
        nu = self.nu
        Eh = self.E
        k_g = -D*(1-nu)

        laplacian = lambda f : div(grad(f))
        hessian = lambda f : grad(grad(f))

        membrane = (-1/(2*Eh) * inner(hessian(v), hessian(v)) + nu/(2*Eh) * self.bracket(v, v)) * dx 
        bending = (D/2 * (inner(laplacian(w), laplacian(w))) + k_g/2 * self.bracket(w, w)) * dx 
        coupling = 1/2 * inner(self.σ(v), outer(grad(w), grad(w))) * dx # compatibility coupling term
        energy = bending + membrane + coupling

        return energy, bending, membrane, coupling

    def M(self, f):
        return grad(grad(f)) + self.nu*self.σ(f)
    
    def P(self, f):
        c_nu = 12*(1-self.nu**2)
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
        dgc   = lambda f, g: avg(inner(W(f), outer(n, n)))*jump(grad(g), n)*dS

        bc1 = lambda u: 1/2 * inner(grad(u), n) * inner(M(u), outer(n, n)) * ds
        bc2 = lambda u: 1/2 * inner(grad(u), n) * inner(P(u), outer(n, n)) * ds
        bc3 = lambda u: 1/2 * α/h * inner(grad(u), grad(u)) * ds
        
        return   (dg1(w) + dg2(w)) \
                + dg1(v) + dg2(v) \
                - bc1(w) - bc2(v) \
                + bc3(w) + bc3(v) \
                + dgc(w, v) 
                
# FVKAdimensional class
class FVKAdimensional(NonlinearPlateFVK):
    def __init__(self, mesh, nu = 0, alpha_penalty = 100) -> None:
        self.alpha_penalty = alpha_penalty
        self.nu = nu
        self.mesh = mesh
        
    def energy(self, state):
        v = state["v"]
        w = state["w"]
        dx = ufl.Measure("dx")

        nu = self.nu
        
        k_g = -(1-nu)
        k_g = Constant(self.mesh, -(1-self.nu))
        
        laplacian = lambda f : div(grad(f))
        hessian = lambda f : grad(grad(f))

        membrane = (-1/(2) * inner(hessian(v), hessian(v)) + nu/2 * self.bracket(v, v)) * dx 
        bending = (1/2 * (inner(laplacian(w), laplacian(w))) + k_g/2 * self.bracket(w, w)) * dx 
        coupling = 1/2 * inner(self.σ(v), outer(grad(w), grad(w))) * dx # compatibility coupling term
        energy = bending + membrane + coupling

        return energy, bending, membrane, coupling