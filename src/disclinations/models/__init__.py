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

from .forms import laplacian, hessian, airy_stress, bracket, c0ip_quadratic_terms
from .material import PlateMaterial
from .forms import fvk_coupling_residual


class Biharmonic:
    """Scalar clamped Delta² model; energy() returns 1/2 integral (Delta u)^2."""

    def __init__(self, mesh, model_parameters=None) -> None:
        model_parameters = {} if model_parameters is None else model_parameters
        self.alpha_penalty = model_parameters.get(
            "alpha_penalty", default_model_parameters["alpha_penalty"]
        )
        self.mesh = mesh

    def energy(self, state):
        u = state["u"]
        dx = ufl.Measure("dx")

        return (1 / 2 * (inner(div(grad(u)), div(grad(u))))) * dx

    def penalisation(self, state):
        """Homogeneous clamped C0-IP correction to 1/2 integral (Delta u)^2."""
        u = state["u"]
        terms = c0ip_quadratic_terms(
            u, self.mesh, self.alpha_penalty, moment=laplacian(u) * ufl.Identity(2)
        )
        return sum(terms.values())


class ToyPlateFVK:
    def __init__(self, mesh, model_parameters=None) -> None:
        model_parameters = {} if model_parameters is None else model_parameters
        self.alpha_penalty = model_parameters.get(
            "alpha_penalty", default_model_parameters["alpha_penalty"]
        )

    def σ(self, v):
        return airy_stress(v)

    def bracket(self, f, g):
        return bracket(f, g)

    def penalisation(self, state):
        """Operator-only base class: no standalone mechanical model is defined."""
        raise NotImplementedError(
            "Choose a concrete Biharmonic or NonlinearPlateFVK model"
        )


class NonlinearPlateFVK(ToyPlateFVK):
    """Plane-stress FvK plate: v [N m], w [m], energy components [J]."""

    def __init__(self, mesh, model_parameters=None) -> None:
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
        self.D = self.material.bending_stiffness

        self.mesh = mesh

        if model_parameters.get("order", default_model_parameters["order"]) < 2:
            raise ValueError("Fourth-order C0-IP models require polynomial order >= 2")
        X = basix.ufl.element(
            "P",
            str(mesh.ufl_cell()),
            model_parameters.get("order", default_model_parameters["order"]),
        )
        self.Q = dolfinx.fem.functionspace(mesh, basix.ufl.mixed_element([X, X]))

    def energy(self, state):
        """Return (saddle, bending, membrane, coupling) UFL scalar forms [J]."""
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

    def _quadratic_penalisation(self, state):
        v, w = state["v"], state["w"]
        bending = c0ip_quadratic_terms(w, self.mesh, self.alpha_penalty, self.M(w))
        membrane = c0ip_quadratic_terms(v, self.mesh, self.alpha_penalty, self.P(v))
        return self.material.bending_stiffness * sum(
            bending.values()
        ) - self.material.airy_compliance * sum(membrane.values())

    def penalisation(self, state):
        """Signed quadratic and nonlinear facet corrections to the saddle [J]."""
        v, w = state["v"], state["w"]
        n = ufl.FacetNormal(self.mesh)
        coupling = (
            avg(inner(self.W(w), outer(n, n)))
            * jump(grad(v), n)
            * ufl.Measure("dS", domain=self.mesh)
        )
        return self._quadratic_penalisation(state) + coupling

    def gaussian_curvature(self, w, order=1):
        mesh = self.mesh
        DG_e = basix.ufl.element("DG", str(mesh.ufl_cell()), order)
        DG = dolfinx.fem.functionspace(mesh, DG_e)
        kappa = 0.5 * self.bracket(w, w)  # det Hess(w), small-slope Gaussian curvature
        kappa_expr = dolfinx.fem.Expression(kappa, DG.element.interpolation_points())
        Kappa = dolfinx.fem.Function(DG)
        cell_map = mesh.topology.index_map(mesh.topology.dim)
        cells = np.arange(cell_map.size_local + cell_map.num_ghosts, dtype=np.int32)
        # Supplying both cell arrays works across the DOLFINx 0.9 builds used by
        # the audit.  Some later builds no longer accept the empty cells1 array
        # produced by the one-argument Python convenience path.
        Kappa.interpolate(kappa_expr, cells, cells)

        return Kappa


class NonlinearPlateFVK_brenner(NonlinearPlateFVK):
    def energy(self, state):
        """Return uncoupled saddle and bulk diagnostics; coupling is in residual."""
        _, bending, membrane, coupling = super().energy(state)
        return bending - membrane, bending, membrane, coupling

    def coupling_term(self, state, v_test, w_test):
        """Nonlinear bracket residual with broken facet fluxes."""
        return fvk_coupling_residual(
            state, v_test, w_test, self.mesh, facet_fluxes=True
        )

    def penalisation(self, state):
        """Quadratic saddle correction [J]; explicit residual handles coupling."""
        return self._quadratic_penalisation(state)


class NonlinearPlateFVK_carstensen(NonlinearPlateFVK_brenner):
    def coupling_term(self, state, v_test, w_test):
        """Nonlinear bracket residual with volume terms only."""
        return fvk_coupling_residual(
            state, v_test, w_test, self.mesh, facet_fluxes=False
        )


def create_disclinations(mesh, params, points=None, signs=None):
    """Return (rank-local point arrays, params) for global signed point data.

    Flat (3,) input is one point; (N,3) input is N points. params['loading']
    takes precedence when it contains both points and signs. Strength units
    follow the caller's incompatibility convention; this helper sets no physical
    Frank-angle sign. Points use the mesh coordinate units. Signs are retained
    as the global ordered list in params on every rank.
    """
    loading = params.get("loading")
    if loading is not None and "points" in loading and "signs" in loading:
        points, signs = loading["points"], loading["signs"]
    else:
        points = [[0.0, 0.0, 0.0]] if points is None else points
        signs = [1.0] if signs is None else signs
    points = np.asarray(points, dtype=mesh.geometry.x.dtype)
    if points.shape == (3,):
        points = points.reshape(1, 3)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) != len(signs):
        raise ValueError("Require one 3D point per disclination strength")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(signs)):
        raise ValueError("Defect data must be finite")
    params["loading"] = {"points": points.tolist(), "signs": list(signs)}

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

    # _D = params["model"]["D"]
    _D = PlateMaterial(_E, nu, thickness).bending_stiffness

    # Calculate rescaling factors
    w_scale = np.sqrt(2 * _D / (_E * thickness))
    v_scale = _D
    f_scale = np.sqrt(2 * _D**3 / (_E * thickness))

    # Store rescaling factors in the params dictionary
    params["model"]["w_scale"] = float(w_scale)
    params["model"]["v_scale"] = float(v_scale)
    params["model"]["f_scale"] = float(f_scale)

    return params
