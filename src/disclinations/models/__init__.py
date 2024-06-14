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
    

class NonlinearPlateFVK:
    def __init__(self, mesh, model_parameters = {}) -> None:
        self.alpha_penalty = model_parameters.get("alpha_penalty",
                                                       default_model_parameters["alpha_penalty"])
        self.D = model_parameters.get("D",
                                      default_model_parameters["D"])
        self.nu = model_parameters.get("nu",
                                       default_model_parameters["nu"])
        self.E = model_parameters.get("E",
                                      default_model_parameters["E"])
        self.mesh = mesh
    
    def σ(self, v):
        # J = ufl.as_matrix([[0, -1], [1, 0]])
        # return inner(grad(grad(v)), J.T * grad(grad(v)) * J)
        return div(grad(v)) * ufl.Identity(2) - grad(grad(v))

    def bracket(self, f, g):
        J = ufl.as_matrix([[0, -1], [1, 0]])
        return inner(grad(grad(f)), J.T * grad(grad(g)) * J)
    
    def energy(self, state):
        v = state["v"]
        w = state["w"]
        dx = ufl.Measure("dx")

        D = self.D
        nu = self.nu
        Eh = self.E
        k_g = -D*(1-nu)
        
        bending = (D/2 * (inner(div(grad(w)), div(grad(w)))) + k_g * self.bracket(w, w)) * dx 
        membrane = (-1/(2*Eh) * inner(grad(grad(v)), grad(grad(v))) + nu/(2*Eh) * self.bracket(v, v)) * dx 
        # membrane = 1/2 * inner(Ph(ph_), grad(grad(ph_)))*dx 
        coupling = 1/2 * inner(self.σ(v), outer(grad(w), grad(w))) * dx # compatibility coupling term
        energy = bending + membrane + coupling

        return energy
    
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