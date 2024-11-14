import ufl
from ufl import avg, ds, dS, outer, div, grad, inner, dot, jump
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

laplacian = lambda f: div(grad(f))
hessian = lambda f: grad(grad(f))


class Biharmonic:
    def __init__(self, mesh, model_parameters={}) -> None:
        self.alpha_penalty = model_parameters.get(
            "alpha_penalty", default_model_parameters["alpha_penalty"]
        )
        self.mesh = mesh

    def energy(self, state):
        u = state["u"]
        dx = ufl.Measure("dx")

        return (1 / 2 * (inner(div(grad(u)), div(grad(u))))) * dx

    def penalisation(self, state):
        u = state["u"]
        α = self.alpha_penalty
        h = ufl.CellDiameter(self.mesh)
        n = ufl.FacetNormal(self.mesh)

        dS = ufl.Measure("dS")
        ds = ufl.Measure("ds")

        dg1 = lambda u: 1 / 2 * dot(jump(grad(u)), avg(grad(grad(u)) * n)) * dS
        dg2 = lambda u: 1 / 2 * α / avg(h) * inner(jump(grad(u)), jump(grad(u))) * dS
        bc1 = lambda u: 1 / 2 * α / h * inner(grad(u), grad(u)) * ds

        return -dg1(u) + dg2(u) + bc1(u)


class ToyPlateFVK:
    def __init__(self, mesh, model_parameters={}) -> None:
        self.alpha_penalty = model_parameters.get(
            "alpha_penalty", default_model_parameters["alpha_penalty"]
        )

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

        dg1 = lambda u: 1 / 2 * dot(jump(grad(u)), avg(grad(grad(u)) * n)) * dS
        dg2 = lambda u: 1 / 2 * α / avg(h) * inner(jump(grad(u)), jump(grad(u))) * dS
        bc1 = lambda u: 1 / 2 * inner(grad(u), grad(grad(u)) * n) * ds
        bc2 = lambda u: 1 / 2 * α / h * inner(grad(u), grad(u)) * ds

        return -dg1(w) + dg2(w) - dg1(v) + dg2(v) - bc1(w) - bc2(w) - bc1(v) + bc2(v)

    def _hessian(self, u):
        return hessian(u)

    def _laplacian(self, u):
        return laplacian(u)


class NonlinearPlateFVK(ToyPlateFVK):
    def __init__(self, mesh, model_parameters={}, adimensional=False) -> None:
        self.alpha_penalty = model_parameters.get(
            "alpha_penalty", default_model_parameters["alpha_penalty"]
        )
        self.nu = model_parameters.get("nu", default_model_parameters["nu"])
        self.E = model_parameters.get("E", default_model_parameters["E"])
        self.t = model_parameters.get(
            "thickness", default_model_parameters["thickness"]
        )
        self.D = self.E * self.t**3 / (12 * (1 - self.nu**2))

        if adimensional:
            self.adimensional = True
            self.E = 1
            self.t = 1
            self.D = self.E * self.t**3 / (12 * (1 - self.nu**2))
        else:
            self.adimensional = False

        if model_parameters["higher_regularity"]:
            self.higher_regularity = True
        else:
            self.higher_regularity = False

        self.mesh = mesh

        X = basix.ufl.element("P", str(mesh.ufl_cell()), model_parameters["order"])
        self.Q = dolfinx.fem.functionspace(mesh, basix.ufl.mixed_element([X, X]))

    def energy(self, state):
        v = state["v"]
        w = state["w"]
        dx = ufl.Measure("dx")

        D = self.D
        nu = self.nu
        Eh = self.E * self.t
        k_g = -D * (1 - nu)

        membrane = (
            1 / (2 * Eh) * inner(hessian(v), hessian(v))
            - nu / (2 * Eh) * self.bracket(v, v)
        ) * dx
        bending = (
            D / 2 * (inner(laplacian(w), laplacian(w))) + k_g / 2 * self.bracket(w, w)
        ) * dx
        coupling = (
            1 / 2 * inner(self.σ(v), outer(grad(w), grad(w))) * dx
        )  # compatibility coupling term
        energy = bending - membrane + coupling

        return energy, bending, membrane, coupling

    def M(self, f):
        return grad(grad(f)) + self.nu * self.σ(f)

    def P(self, f):
        return grad(grad(f)) - self.nu * self.σ(f)

    def W(self, f):
        J = ufl.as_matrix([[0, -1], [1, 0]])
        return -0.5 * J.T * (outer(grad(f), grad(f))) * J

    def penalisation(self, state):
        v = state["v"]
        w = state["w"]
        α = self.alpha_penalty
        h = ufl.CellDiameter(self.mesh)
        n = ufl.FacetNormal(self.mesh)
        nu = self.nu

        dS = ufl.Measure("dS")
        ds = ufl.Measure("ds")
        c_nu = 12 * (1 - nu**2)

        # M = lambda f : grad(grad(f)) + nu*self.σ(f)
        M = lambda f: self.M(f)
        P = lambda f: self.P(f)
        W = lambda f: self.W(f)

        dg1 = lambda u: -1 / 2 * dot(jump(grad(u)), avg(grad(grad(u)) * n)) * dS
        dg2 = lambda u: +1 / 2 * α / avg(h) * inner(jump(grad(u)), jump(grad(u))) * dS
        dgc = lambda w, g: avg(inner(W(w), outer(n, n))) * jump(grad(g), n) * dS
        dg3 = (
            lambda w: 1
            / 2
            * α
            / avg(h)
            * inner(jump(hessian(w), n), jump(hessian(w), n))
            * dS
        )
        # bc1 = lambda u: - 1/2 * inner(grad(u), n) * inner(M(u), outer(n, n)) * ds
        bc1 = (
            lambda u: -1
            / 2
            * inner(grad(u), n)
            * inner(grad(grad(u)), outer(n, n))
            * ds
        )
        # bc2 = lambda u: - 1/2 * inner(grad(u), n) * inner(P(u), outer(n, n)) * ds
        bc2 = (
            lambda u: -1
            / 2
            * inner(grad(u), n)
            * inner(grad(grad(u)), outer(n, n))
            * ds
        )
        bc3 = lambda u: 1 / 2 * α / h * inner(grad(u), grad(u)) * ds

        c1 = self.D
        c2 = 1 / (self.E * self.t)

        self._dgw = c1 * dg1(w) + c1 * dg2(w)
        self._dgv = -c2 * dg1(v) - c2 * dg2(v)
        self._dgc = dgc(w, v)
        self._bcv = -c2 * bc2(v) - c1 * bc3(v)
        self._bcw = c2 * bc1(w) + c1 * bc3(w)

        penalisation = (
            (c1 * dg1(w) + c1 * dg2(w))
            - c2 * dg1(v)
            - c2 * dg2(v)
            + c2 * bc1(w)
            - c2 * bc2(v)
            + c1 * bc3(w)
            - c1 * bc3(v)
            + dgc(w, v)
        )

        if self.higher_regularity:
            print("Higher regularity")
            penalisation += c1 * dg3(w)
            self._dgw += c1 * dg3(w)

        return penalisation

    def gaussian_curvature(self, w, order=1):
        mesh = self.mesh
        DG_e = basix.ufl.element("DG", str(mesh.ufl_cell()), order)
        DG = dolfinx.fem.functionspace(mesh, DG_e)
        kappa = self.bracket(w, w)
        kappa_expr = dolfinx.fem.Expression(kappa, DG.element.interpolation_points())
        Kappa = dolfinx.fem.Function(DG)
        Kappa.interpolate(kappa_expr)

        return Kappa


class NonlinearPlateFVK_brenner(NonlinearPlateFVK):

    def energy(self, state):
        v = state["v"]
        w = state["w"]
        dx = ufl.Measure("dx")

        D = self.D
        nu = self.nu
        Eh = self.E * self.t
        k_g = -D * (1 - nu)

        membrane = (
            1 / (2 * Eh) * inner(hessian(v), hessian(v))
            - nu / (2 * Eh) * self.bracket(v, v)
        ) * dx
        bending = (
            D / 2 * (inner(laplacian(w), laplacian(w))) + k_g / 2 * self.bracket(w, w)
        ) * dx
        coupling = (
            1 / 2 * inner(self.σ(v), outer(grad(w), grad(w))) * dx
        )  # compatibility coupling term
        energy = bending - membrane

        return energy, bending, membrane, coupling

    def coupling_term(self, state, v_test, w_test):
        v = state["v"]
        w = state["w"]

        _function_space = self.Q
        _test = ufl.TestFunction(_function_space)

        # Split the test function into subspaces for _w_test and _v_test
        # _w_test, _v_test = ufl.split(_test)
        # w_test, v_test = ufl.split(_test)

        dx = ufl.Measure("dx")
        dS = ufl.Measure("dS")
        ds = ufl.Measure("ds")

        n = ufl.FacetNormal(self.mesh)

        coupling_in_edge = (
            lambda f1, f2, test: (
                dot(dot(avg(self.σ(f1)), grad(f2("+"))), n("+"))
                + dot(dot(avg(self.σ(f1)), grad(f2("-"))), n("-"))
            )
            * avg(test)
            * dS
        )
        coupling_bnd_edge = (
            lambda f1, f2, test: dot(dot(self.σ(f1), grad(f2)), n) * test * ds
        )

        cw_bulk = -self.bracket(w, v) * w_test * dx
        cw_in_edges = 0.5 * coupling_in_edge(v, w, w_test) + 0.5 * coupling_in_edge(
            w, v, w_test
        )
        cw_bnd_edges = 0.5 * coupling_bnd_edge(v, w, w_test) + 0.5 * coupling_bnd_edge(
            w, v, w_test
        )

        cv_bulk = -0.5 * self.bracket(w, w) * v_test * dx
        cv_in_edges = 0.5 * coupling_in_edge(w, w, v_test)
        cv_bnd_edges = 0.5 * coupling_bnd_edge(w, w, v_test)

        return (
            cw_bulk + cv_bulk + cw_bnd_edges + cv_bnd_edges + cv_in_edges + cw_in_edges
        )

    def penalisation(self, state):
        v = state["v"]
        w = state["w"]
        α = self.alpha_penalty
        h = ufl.CellDiameter(self.mesh)
        n = ufl.FacetNormal(self.mesh)
        nu = self.nu

        dS = ufl.Measure("dS")
        ds = ufl.Measure("ds")
        c_nu = 12 * (1 - nu**2)

        # M = lambda f : grad(grad(f)) + nu*self.σ(f)
        M = lambda f: self.M(f)
        P = lambda f: self.P(f)
        W = lambda f: self.W(f)

        # dg1 = lambda u: - 1/2 * dot(jump(grad(u)), avg(grad(grad(u)) * n)) * dS # CFe: 6-10-24 commented out
        # dg2 = lambda u: + 1/2 * α/avg(h) * inner(jump(grad(u)), jump(grad(u))) * dS # CFe: 6-10-24 commented out
        dg1 = (
            lambda u: -1 / 2 * jump(grad(u), n) * avg(dot(grad(grad(u)) * n, n)) * dS
        )  # CFe: 6-10-24 added
        dg2 = (
            lambda u: +1 / 2 * α / avg(h) * jump(grad(u), n) * jump(grad(u), n) * dS
        )  # CFe: 6-10-24 added
        # dgc = lambda w, g: avg(inner(W(w), outer(n, n)))*jump(grad(g), n)*dS

        # bc1 = lambda u: - 1/2 * inner(grad(u), n) * inner(M(u), outer(n, n)) * ds
        # bc2 = lambda u: - 1/2 * inner(grad(u), n) * inner(P(u), outer(n, n)) * ds
        bc1 = (
            lambda u: -1
            / 2
            * inner(grad(u), n)
            * inner(grad(grad(u)), outer(n, n))
            * ds
        )
        bc2 = (
            lambda u: -1
            / 2
            * inner(grad(u), n)
            * inner(grad(grad(u)), outer(n, n))
            * ds
        )
        # bc3 = lambda u: 1/2 * α/h * inner(grad(u), grad(u)) * ds
        bc3 = lambda u: 1 / 2 * α / h * dot(grad(u), n) * dot(grad(u), n) * ds
        c1 = self.D
        c2 = 1 / (self.E * self.t)

        self._dgw = c1 * dg1(w) + c1 * dg2(w)
        self._dgv = -c2 * dg1(v) - c2 * dg2(v)
        self._dgc = dolfinx.fem.Constant(self.mesh, 0.0) * v * ds
        self._bcv = -c2 * bc2(v) - c1 * bc3(v)
        self._bcw = c2 * bc1(w) + c1 * bc3(w)

        return (
            (c1 * dg1(w) + c1 * dg2(w))
            - c2 * dg1(v)
            - c2 * dg2(v)
            + c2 * bc1(w)
            - c2 * bc2(v)
            + c1 * bc3(w)
            - c1 * bc3(v)
        )


class NonlinearPlateFVK_carstensen(NonlinearPlateFVK_brenner):
    def coupling_term(self, state, v_test, w_test):
        v = state["v"]
        w = state["w"]

        _function_space = v.ufl_operands[0].ufl_function_space()
        _test = ufl.TestFunction(_function_space)

        # Split the test function into subspaces for _w_test and _v_test
        # _w_test, _v_test = ufl.split(_test)

        dx = ufl.Measure("dx")
        dS = ufl.Measure("dS")
        ds = ufl.Measure("ds")

        n = ufl.FacetNormal(self.mesh)

        cw_bulk = -self.bracket(w, v) * w_test * dx
        cv_bulk = -0.5 * self.bracket(w, w) * v_test * dx

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
        points_list = [point.tolist()[0] for point in points]
        params["loading"] = {"points": points_list, "signs": signs}

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


def calculate_rescaling_factors(params):
    """
    Calculate rescaling factors and store them in the params dictionary.

    Args:
    - params (dict): Dictionary containing geometry, material, and model parameters.

    Returns:
    - params (dict): Updated dictionary with rescaling factors.
    """
    # Extract necessary parameters
    _E = params["model"]["E"]
    nu = params["model"]["nu"]
    thickness = params["model"]["thickness"]

    _D = _E * thickness**3 / (12 * (1 - nu**2))

    # Calculate rescaling factors
    # w_scale = np.sqrt(2 * _D / (_E * thickness))
    # v_scale = _D
    # f_scale = np.sqrt(2 * _D**3 / (_E * thickness))

    w_scale = thickness
    v_scale = _E * thickness**3
    f_scale = 1 / _E

    adim_coeff = params["geometry"]["radius"] / thickness

    energy_scale = _E * (thickness**3) / (adim_coeff**2)

    # Store rescaling factors in the params dictionary
    params["model"]["w_scale"] = float(w_scale)
    params["model"]["v_scale"] = float(v_scale)
    params["model"]["f_scale"] = float(f_scale)
    params["model"]["energy_scale"] = float(energy_scale)

    return params


def _transverse_load_exact_solution(x, params, adimensional=False):
    f_scale = params["model"]["f_scale"]
    if adimensional:
        f_scale = 1

    _p = (40 / 3) * (1 - x[0] ** 2 - x[1] ** 2) ** 4 + (16 / 3) * (
        11 + x[0] ** 2 + x[1] ** 2
    )

    return f_scale * _p


def initialise_exact_solution_compatible_transverse(Q, params, adimensional=False):
    """
    Initialize the exact solutions for v and w using the provided parameters.

    Args:
    - Q: The function space.
    - params: A dictionary of parameters containing geometric properties (e.g., radius).

    Returns:
    - v_exact: Exact solution for v.
    - w_exact: Exact solution for w.
    """

    # Extract the necessary parameters
    # v_scale = params["model"]["v_scale"]
    # w_scale = params["model"]["w_scale"]

    _D = (
        params["model"]["E"]
        * params["model"]["thickness"] ** 3
        / (12 * (1 - params["model"]["nu"] ** 2))
    )

    w_scale = np.sqrt(2 * _D / (params["model"]["E"] * params["model"]["thickness"]))
    v_scale = _D

    if adimensional:
        v_scale /= params["model"]["v_scale"]
        w_scale /= params["model"]["w_scale"]

    q_exact = dolfinx.fem.Function(Q)
    v_exact, w_exact = q_exact.split()

    # Define the exact solution for v
    def _v_exact(x):
        rq = x[0] ** 2 + x[1] ** 2

        a1 = -1 / 12
        a2 = -1 / 18
        a3 = -1 / 24

        _v = a1 * (1 - rq) ** 2 + a2 * (1 - rq) ** 3 + a3 * (1 - rq) ** 4

        return _v * v_scale  # Apply scaling

    # Define the exact solution for w
    def _w_exact(x):
        return w_scale * (1 - x[0] ** 2 - x[1] ** 2) ** 2

    # Interpolate the exact solutions over the mesh
    v_exact.interpolate(_v_exact)
    w_exact.interpolate(_w_exact)
    exact = {"v": v_exact, "w": w_exact}

    return exact


def initialise_exact_solution_dipole(Q, params, adimensional=False):
    """
    Initialize the exact solutions for v and w using the provided parameters.

    Args:
    - Q: The function space.
    - params: A dictionary of parameters containing geometric properties (e.g., radius).

    Returns:
    - v_exact: Exact solution for v.
    - w_exact: Exact solution for w.
    """
    v_scale = params["model"]["E"] * params["model"]["thickness"]

    if adimensional:
        v_scale /= params["model"]["v_scale"]
        # w_scale /= params["model"]["w_scale"]

    q_exact = dolfinx.fem.Function(Q)
    v_exact, w_exact = q_exact.split()

    # compute distance between disclinations
    distance = np.linalg.norm(
        np.array(params["loading"]["points"][0])
        - np.array(params["loading"]["points"][1])
    )

    def _v_exact(x):
        rq = x[0] ** 2 + x[1] ** 2
        _v = (1 / (16 * np.pi)) * (
            (x[1] ** 2 + (x[0] - distance / 2) ** 2)
            * (
                np.log(4.0)
                + np.log(x[1] ** 2 + (x[0] - distance / 2) ** 2)
                - np.log(4 - 4 * x[0] * distance + rq * distance**2)
            )
            - (1 / 4)
            * (4 * (x[1] ** 2) + (2 * x[0] + distance) ** 2)
            * (
                np.log(4)
                + np.log(x[1] ** 2 + (x[0] + distance / 2) ** 2)
                - np.log(4 + 4 * x[0] * distance + rq * distance**2)
            )
        )
        return _v * v_scale

    def _w_exact(x):
        _w = np.zeros_like(x[0])
        return _w

    v_exact.interpolate(_v_exact)
    w_exact.interpolate(_w_exact)

    exact = {"v": v_exact, "w": w_exact}

    return exact


def exact_energy_dipole(parameters):
    # it should depend on the signs as well
    distance = np.linalg.norm(
        np.array(parameters["loading"]["points"][0])
        - np.array(parameters["loading"]["points"][1])
    )

    return (
        parameters["model"]["E"]
        * parameters["model"]["thickness"] ** 3
        * parameters["geometry"]["radius"] ** 2
        / (8 * np.pi)
        * distance**2
        * (np.log(4 + distance**2) - np.log(4 * distance))
    )


from disclinations.utils import _logger


def assemble_penalisation_terms(model):
    # Define and compute penalisation terms
    _penalisation_terms = {
        "dgw": model._dgw,
        "dgv": model._dgv,
        "dgc": model._dgc,
        "bcv": model._bcv,
        "bcw": model._bcw,
    }

    # Assemble each penalisation term and log its value
    assembled_penalisation_terms = {}
    for label, term in _penalisation_terms.items():
        assembled_value = dolfinx.fem.assemble_scalar(dolfinx.fem.form(term))
        assembled_penalisation_terms[label] = assembled_value
        _logger.critical(f"Penalisation term {label}: {assembled_value:.2e}")

    return assembled_penalisation_terms
