import ufl
from ufl import (avg, ds, dS, outer, div, grad, inner, dot, jump)
import basix 
import dolfinx
import yaml
import os
import numpy as np
from mpi4py import MPI

AIRY = 0
TRANSVERSE = 1

dir_path = os.path.dirname(os.path.realpath(__file__))
with open(f"{dir_path}/default_parameters.yml") as f:
    default_parameters = yaml.load(f, Loader=yaml.FullLoader)

default_model_parameters = default_parameters["model"]
#default_geo_parameters = default_parameters["geometry"]

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

class A_NonlinearPlateFVK(ToyPlateFVK):
    def __init__(self, mesh, model_parameters = {}) -> None:
        self.alpha_penalty = model_parameters.get("alpha_penalty", default_model_parameters["alpha_penalty"])
        self.nu = model_parameters.get("nu", default_model_parameters["nu"])
        self.E = model_parameters.get("E", default_model_parameters["E"])
        self.t = model_parameters.get("thickness", default_model_parameters["thickness"])
        #self.R = model_parameters.get("radius", default_geo_parameters["radius"])
        #self.D = self.E * self.t**3 / (12*(1-self.nu**2))
        self.c_nu = 1/(12*(1-self.nu**2))

        # Scaling parameters to get dimensional physical values starting from dimensionless values
        self.v_scale = self.E * (self.t**3)
        self.w_scale = self.t
        #self.energy_scale =  self.E * (self.t**3) * ( self.t / self.R )**2
        
        self.mesh = mesh
        
        X = basix.ufl.element("P", str(mesh.ufl_cell()), model_parameters["order"]) 
        self.Q = dolfinx.fem.functionspace(mesh, basix.ufl.mixed_element([X, X]))

    
    def energy(self, state):
        v = state["v"]
        w = state["w"]
        dx = ufl.Measure("dx")

        k_g = - self.c_nu*(1-self.nu)

        #membrane = ( 1/2 * inner(hessian(v), hessian(v)) - self.nu/2 * self.bracket(v, v) ) * dx
        membrane = 1/2 * laplacian(v) * laplacian(v) * dx
        #bending = ( self.c_nu/2 * (inner(laplacian(w), laplacian(w))) + k_g/2 * self.bracket(w, w) ) * dx
        bending = self.c_nu/2 * laplacian(w) * laplacian(w) * dx
        coupling = 1/2 * inner(self.σ(v), outer(grad(w), grad(w))) * dx
        #coupling = -1/2 * self.bracket(w,w) * v * dx
        energy = bending - membrane + coupling

        return energy, bending, membrane, coupling

    def M(self, f):
        return grad(grad(f)) + self.nu*self.σ(f)
    
    def P(self, f):
        return grad(grad(f)) - self.nu*self.σ(f)

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

        M = lambda f : self.M(f)
        P = lambda f : self.P(f)
        W = lambda f : self.W(f)
        
        dg1 = lambda u: - 1/2 * dot(jump(grad(u)), avg(grad(grad(u)) * n)) * dS
        dg2 = lambda u: + 1/2 * α/avg(h) * inner(jump(grad(u)), jump(grad(u))) * dS
        dgc = lambda w, g: avg(inner(W(w), outer(n, n)))*jump(grad(g), n)*dS

        bc1 = lambda u: - 1/2 * inner(grad(u), n) * inner(grad(grad(u)), outer(n, n)) * ds
        bc2 = lambda u: - 1/2 * inner(grad(u), n) * inner(grad(grad(u)), outer(n, n)) * ds
        bc3 = lambda u: 1/2 * α/h * inner(grad(u), grad(u)) * ds
        
        return   self.c_nu*dg1(w) + dg2(w) - dg1(v) - dg2(v) + self.c_nu*bc1(w) - bc2(v) + bc3(w) - bc3(v) + dgc(w, v)

    def gaussian_curvature(self, w, order = 1):
        mesh = self.mesh
        DG_e = basix.ufl.element("DG", str(mesh.ufl_cell()), order)
        DG = dolfinx.fem.functionspace(mesh, DG_e)
        kappa = self.bracket(w, w)
        kappa_expr = dolfinx.fem.Expression(kappa, DG.element.interpolation_points())
        Kappa = dolfinx.fem.Function(DG)
        Kappa.interpolate(kappa_expr)

        return Kappa
    
class A_NonlinearPlateFVK_brenner(A_NonlinearPlateFVK):

    def energy(self, state):
        v = state["v"]
        w = state["w"]
        dx = ufl.Measure("dx")
        k_g = - self.c_nu*(1-self.nu)

        #membrane = ( 1/2 * inner(hessian(v), hessian(v)) - self.nu/2 * self.bracket(v, v) ) * dx
        membrane = 1/2 * laplacian(v) * laplacian(v) * dx
        #bending = ( self.c_nu/2 * (inner(laplacian(w), laplacian(w))) + k_g/2 * self.bracket(w, w) ) * dx
        bending = self.c_nu/2 * laplacian(w) * laplacian(w) * dx
        coupling = 1/2 * inner(self.σ(v), outer(grad(w), grad(w))) * dx
        energy = bending - membrane

        return energy, bending, membrane, coupling

    def coupling_term(self, state, v_test, w_test):
        v = state["v"]
        w = state["w"]

        _function_space = self.Q
        _test = ufl.TestFunction(_function_space)

        dx = ufl.Measure("dx")
        dS = ufl.Measure("dS")
        ds = ufl.Measure("ds")

        n = ufl.FacetNormal(self.mesh)

        coupling_in_edge = lambda f1, f2, test: ( dot(dot(avg(self.σ(f1)),grad(f2('+'))), n('+')) + dot(dot(avg(self.σ(f1)),grad(f2('-'))), n('-')) )* avg(test) * dS
        coupling_bnd_edge = lambda f1, f2, test: dot(dot(self.σ(f1),grad(f2)), n) * test * ds

        cw_bulk = - self.bracket(w,v) * w_test * dx
        cw_in_edges = 0.5*coupling_in_edge(v, w, w_test) + 0.5*coupling_in_edge(w, v, w_test)
        cw_bnd_edges = 0.5*coupling_bnd_edge(v, w, w_test) + 0.5*coupling_bnd_edge(w, v, w_test)

        cv_bulk = - 0.5*self.bracket(w,w) * v_test * dx
        cv_in_edges = 0.5*coupling_in_edge(w, w, v_test)
        cv_bnd_edges = 0.5*coupling_bnd_edge(w, w, v_test)
        return cw_bulk + cv_bulk + cw_bnd_edges + cv_bnd_edges + cv_in_edges + cw_in_edges

    def penalisation(self, state):
        v = state["v"]
        w = state["w"]
        α = self.alpha_penalty
        h = ufl.CellDiameter(self.mesh)
        n = ufl.FacetNormal(self.mesh)
        nu = self.nu

        dS = ufl.Measure("dS")
        ds = ufl.Measure("ds")

        M = lambda f : self.M(f)
        P = lambda f : self.P(f)
        W = lambda f : self.W(f)

        dg1 = lambda u: - 1/2 * jump(grad(u),n)*avg(dot(grad(grad(u)) * n, n)) * dS
        dg2 = lambda u: + 1/2 * α/avg(h) * jump(grad(u),n) * jump(grad(u),n) * dS

        bc1 = lambda u: - 1/2 * inner(grad(u), n) * inner(grad(grad(u)), outer(n, n)) * ds
        bc2 = lambda u: - 1/2 * inner(grad(u), n) * inner(grad(grad(u)), outer(n, n)) * ds
        bc3 = lambda u: 1/2 * α/h * dot(grad(u),n) * dot(grad(u),n) * ds

        return  self.c_nu*dg1(w) + dg2(w) - dg1(v) - dg2(v) + self.c_nu*bc1(w) - bc2(v) + bc3(w) - bc3(v)
    
class A_NonlinearPlateFVK_carstensen(A_NonlinearPlateFVK_brenner):
    def coupling_term(self, state, v_test, w_test):
        v = state["v"]
        w = state["w"]

        _function_space = v.ufl_operands[0].ufl_function_space()
        _test = ufl.TestFunction(_function_space)
        
        dx = ufl.Measure("dx")
        dS = ufl.Measure("dS")
        ds = ufl.Measure("ds")
        
        n = ufl.FacetNormal(self.mesh)

        cw_bulk = - self.bracket(w,v) * w_test * dx
        cv_bulk = - 0.5*self.bracket(w,w) * v_test * dx
        
        return cw_bulk + cv_bulk


def create_disclinations(mesh, params, points=[0.0, 0.0, 0.0], signs=[1.0]):
    """
    Create disclinations based on the list of points and signs or the params dictionary.

    Args:
    - mesh: The mesh object, used to determine the data type for points.
    - points: A list of 3D coordinates (x, y, z) representing the disclination points.
    - signs: A list of signs (+1., -1.) associated with each point.
    - params: A dictionary containing model parameters, possibly including loading points and signs.

    Returns:
    - disclinations: A list of disclination points (coordinates).
    - signs: The same list of signs associated with each point.
    - params: Updated params dictionary with disclinations if not already present.
    """

    # Check if "loading" exists in the parameters and contains "points" and "signs"
    if (
        "loading" in params
        and params["loading"] is not None
        and "points" in params["loading"]
        and "signs" in params["loading"]
    ):
        # Use points and signs from the params dictionary
        points = params["loading"]["points"]
        signs = params["loading"]["signs"]
        print("Using points and signs from params dictionary.")
    else:
        # Otherwise, add the provided points and signs to the params dictionary
        print("Using provided points and signs, adding them to the params dictionary.")
        params["loading"] = {"points": points, "signs": signs}

    # Handle the case where rank is not 0 (for distributed computing)
    if mesh.comm.rank == 0:
        # Convert the points into a numpy array with the correct dtype from the mesh geometry
        disclinations = [
            np.array([point], dtype=mesh.geometry.x.dtype) for point in points
        ]
    else:
        # If not rank 0, return empty arrays for parallel processing
        disclinations = [np.zeros((0, 3), dtype=mesh.geometry.x.dtype) for _ in points]

    return disclinations, params


def compute_energy_terms(energy_components, comm):
    """Assemble and sum energy terms over all processes."""
    computed_energy_terms = {
        label: comm.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(energy_term)),
            op=MPI.SUM,
        )
        for label, energy_term in energy_components.items()
    }
    return computed_energy_terms

