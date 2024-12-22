import ufl
from ufl import (avg, ds, dS, outer, div, grad, inner, dot, jump)
import basix 
import dolfinx
import yaml
import os
import numpy as np
from mpi4py import MPI
import pdb

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

    def __init__(self, mesh, model_parameters = {}, smooth=False) -> None:
        self.alpha_penalty = model_parameters.get("alpha_penalty", default_model_parameters["alpha_penalty"])
        self.nu = model_parameters.get("nu", default_model_parameters["nu"])
        self.E = model_parameters.get("E", default_model_parameters["E"])
        self.t = model_parameters.get("thickness", default_model_parameters["thickness"])
        #self.R = model_parameters.get("radius", default_geo_parameters["radius"])
        #self.D = self.E * self.t**3 / (12*(1-self.nu**2))
        self.c_nu = 1/(12*(1-self.nu**2)) #np.round(1/(12*(1-self.nu**2)), 3)
        self.smooth = smooth

        # Scaling parameters to get dimensional physical values starting from dimensionless values
        self.v_scale = self.E * (self.t**3)
        self.w_scale = self.t
        #self.energy_scale =  self.E * (self.t**3) * ( self.t / self.R )**2
        
        self.mesh = mesh
        
        X = basix.ufl.element("P", str(mesh.ufl_cell()), model_parameters["order"]) 
        self.Q = dolfinx.fem.functionspace(mesh, basix.ufl.mixed_element([X, X]))

        self.dx = ufl.Measure("dx")
        self.dS = ufl.Measure("dS")
        self.ds = ufl.Measure("ds")
        #self.dx = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": 3})
        #self.dS = ufl.Measure("dS", metadata={"quadrature_degree": 5})
        #self.ds = ufl.Measure("ds", metadata={"quadrature_degree": 3})

        W = lambda f : self.W(f)
        h = ufl.CellDiameter(self.mesh)
        n = ufl.FacetNormal(self.mesh)
        α2 = 1*self.alpha_penalty
        #self.dg1 = lambda u: - 1/2 * dot(jump(grad(u)), avg(grad(grad(u)) * n)) * dS
        self.dg1 = lambda u: - 1/2 * ( jump(grad(u),n)*avg(dot(grad(grad(u)) * n, n)) ) * dS
        self.dg2 = lambda u: + 1/2 * self.alpha_penalty/avg(h) * inner(jump(grad(u),n), jump(grad(u),n)) * dS
        self.dg3 = lambda u: + 1/2 * α2/avg(h) * inner(jump(hessian(u), n), jump(hessian(u), n)) * dS

        self.dgc = lambda w, g: ( avg(inner(W(w), outer(n, n)))*jump(grad(g), n) ) *dS

        self.bc1 = lambda u: - 1/2 * ( inner(grad(u), n) * inner(grad(grad(u)), outer(n, n)) ) * ds
        #self.bc1 = lambda u: - 1/2 * ( inner(grad(u), n) * div( grad(u) ) ) * ds
        #self.bc2 = lambda u: - 1/2 * ( inner(grad(u), n) * inner(grad(grad(u)), outer(n, n)) ) * ds
        self.bc3 = lambda u: 1/2 * self.alpha_penalty/h * ( dot(grad(u),n) * dot(grad(u),n) ) * ds
    
    def M(self, f):
        return grad(grad(f)) + self.nu*self.σ(f)

    def P(self, f):
        return grad(grad(f)) - self.nu*self.σ(f)

    def W(self, f):
        J = ufl.as_matrix([[0, -1], [1, 0]])
        return -0.5*J.T*(outer(grad(f), grad(f))) * J

    def energy(self, state):
        v = state["v"]
        w = state["w"]
        dx = self.dx

        k_g = - self.c_nu*(1-self.nu)

        #membrane = ( 1/2 * inner(hessian(v), hessian(v)) - self.nu/2 * self.bracket(v, v) ) * dx
        #bending = ( self.c_nu/2 * (inner(laplacian(w), laplacian(w))) + k_g/2 * self.bracket(w, w) ) * dx
        membrane = 1/2 * inner( hessian(v), hessian(v) ) * dx
        bending = self.c_nu/2 * inner( hessian(w), hessian(w) ) * dx
        coupling = 1/2 * inner(self.σ(v), outer(grad(w), grad(w))) * dx
        energy = bending - membrane + coupling
        return energy, bending, membrane, coupling

    def penalisation(self, state):
        v = state["v"]
        w = state["w"]
        return_value = self.c_nu*self.dg1(w) + self.dg2(w) - self.dg1(v) - self.dg2(v) + self.c_nu*self.bc1(w) - self.bc1(v) + self.bc3(w) - self.bc3(v) + self.dgc(w, v)
        #if self.smooth: return_value += self.dg3(w)
        return return_value

    def compute_bending_energy(self, state, COMM):
        return COMM.allreduce( dolfinx.fem.assemble_scalar( dolfinx.fem.form( self.energy(state)[1] )), op=MPI.SUM)

    def compute_membrane_energy(self, state, COMM):
        return COMM.allreduce( dolfinx.fem.assemble_scalar( dolfinx.fem.form( self.energy(state)[2] )), op=MPI.SUM)

    def compute_coupling_energy(self, state, COMM):
        return COMM.allreduce( dolfinx.fem.assemble_scalar( dolfinx.fem.form( self.energy(state)[3] )), op=MPI.SUM)

    def compute_penalisation(self, state, COMM):
        return COMM.allreduce( dolfinx.fem.assemble_scalar( dolfinx.fem.form( self.penalisation(state) )), op=MPI.SUM)

    def compute_penalisation_terms_w(self, state, COMM):
        w = state["w"]
        dg1_computed = COMM.allreduce( dolfinx.fem.assemble_scalar( dolfinx.fem.form(self.dg1(w))), op=MPI.SUM)
        dg2_computed = COMM.allreduce( dolfinx.fem.assemble_scalar( dolfinx.fem.form(self.dg2(w))), op=MPI.SUM)
        bc1_computed = COMM.allreduce( dolfinx.fem.assemble_scalar( dolfinx.fem.form(self.bc1(w))), op=MPI.SUM)
        bc3_computed = COMM.allreduce( dolfinx.fem.assemble_scalar( dolfinx.fem.form(self.bc3(w))), op=MPI.SUM)
        dg3_computed = COMM.allreduce( dolfinx.fem.assemble_scalar( dolfinx.fem.form(self.dg3(w))), op=MPI.SUM)
        return self.c_nu*dg1_computed, dg2_computed, self.c_nu*bc1_computed, bc3_computed, dg3_computed

    def compute_penalisation_terms_v(self, state, COMM):
        v = state["v"]
        dg1_computed = COMM.allreduce( dolfinx.fem.assemble_scalar( dolfinx.fem.form(self.dg1(v))), op=MPI.SUM)
        dg2_computed = COMM.allreduce( dolfinx.fem.assemble_scalar( dolfinx.fem.form(self.dg2(v))), op=MPI.SUM)
        bc1_computed = COMM.allreduce( dolfinx.fem.assemble_scalar( dolfinx.fem.form(self.bc1(v))), op=MPI.SUM)
        bc3_computed = COMM.allreduce( dolfinx.fem.assemble_scalar( dolfinx.fem.form(self.bc3(v))), op=MPI.SUM)
        return dg1_computed, dg2_computed, bc1_computed, bc3_computed

    def compute_total_penalisation_w(self, state, COMM):
        return_value = 0
        for element in self.compute_penalisation_terms_w(state, COMM): return_value += element
        return return_value

    def compute_total_penalisation_v(self, state, COMM):
        return_value = 0
        for element in self.compute_penalisation_terms_v(state, COMM): return_value += element
        return return_value

    def compute_penalisation_coupling(self, state, COMM):
        v = state["v"]
        w = state["w"]
        return COMM.allreduce( dolfinx.fem.assemble_scalar( dolfinx.fem.form(self.dgc(w, v))), op=MPI.SUM)


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

        membrane = 1/2 * inner( hessian(v), hessian(v) ) * dx
        bending = self.c_nu/2 * inner( hessian(w), hessian(w) ) * dx
        coupling = 1/2 * inner(self.σ(v), outer(grad(w), grad(w))) * dx
        energy = bending - membrane

        W = lambda f : self.W(f)
        h = ufl.CellDiameter(self.mesh)
        n = ufl.FacetNormal(self.mesh)

        α2 = self.alpha_penalty

        self.dg1 = lambda u: - 1/2 * ( jump(grad(u),n)*avg(dot(grad(grad(u)) * n, n)) ) * dS # < -- Original
        #self.dg1 = lambda u: - 1/2 * dot(jump(grad(u)), avg(grad(grad(u)) * n)) * dS # Slighly different <-- from above

        self.dg2 = lambda u: + 1/2 * self.alpha_penalty/avg(h) *( jump(grad(u),n) * jump(grad(u),n) ) * dS

        self.bc1 = lambda u: - 1/2 * ( inner(grad(u), n) * inner(grad(grad(u)), outer(n, n)) ) * ds
        self.bc3 = lambda u: 1/2 * self.alpha_penalty/h * ( dot(grad(u),n) * dot(grad(u),n) ) * ds



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

        return  self.c_nu*self.dg1(w) + self.dg2(w) - self.dg1(v) - self.dg2(v) + self.c_nu*self.bc1(w) - self.bc1(v) + self.bc3(w) - self.bc3(v)
    
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

