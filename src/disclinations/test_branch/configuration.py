"""
PURPOSE OF THE SCRIPT
Analyze configuration of interest
Using the Veriational FE formulationin its NON-dimensional formulation (see "models/adimensional.py").
Dimensionless parameters:
beta := R/h >> 1
gamma := p0/E << 1
f := gamma beta**4

ARTICLE RELATED SECTION
"""

import json
import logging
import os
import pdb
import sys
from pathlib import Path
import importlib.resources as pkg_resources

import dolfinx
import dolfinx.plot
from dolfinx import log
import dolfinx.io
from dolfinx.io import XDMFFile, gmshio
import dolfinx.mesh
from dolfinx.fem import Constant, dirichletbc
from dolfinx.fem.petsc import (assemble_matrix, create_vector, create_matrix, assemble_vector)

import ufl
from ufl import (CellDiameter, FacetNormal, dx)

from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import petsc4py

import yaml
import warnings
import basix

import matplotlib
from matplotlib.ticker import PercentFormatter, ScalarFormatter, MaxNLocator
matplotlib.use('WebAgg')
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pyvista

from disclinations.models.adimensional import A_NonlinearPlateFVK
from disclinations.meshes import mesh_bounding_box
from disclinations.meshes.primitives import mesh_circle_gmshapi
from disclinations.utils import load_parameters
from disclinations.utils.la import compute_cell_contributions, compute_disclination_loads
from disclinations.utils.viz import plot_scalar, plot_profile, plot_mesh
from disclinations.utils.sample_function import sample_function, interpolate_sample
from disclinations.solvers import SNESSolver, SNESProblem
from visuals import visuals

visuals.matplotlibdefaults(useTex=False)

PATH_TO_PARAMETERS_FILE = 'disclinations.test'

logging.basicConfig(level=logging.INFO)

REQUIRED_VERSION = "0.8.0"

if dolfinx.__version__ != REQUIRED_VERSION:
    warnings.warn(f"We need dolfinx version {REQUIRED_VERSION}, but found version {dolfinx.__version__}. Exiting.")
    sys.exit(1)


petsc4py.init(sys.argv)
log.set_log_level(log.LogLevel.WARNING)

comm = MPI.COMM_WORLD

def monitor(snes, it, norm):
    logging.info(f"Iteration {it}, residual {norm}")
    print(f"Iteration {it}, residual {norm}")
    return PETSc.SNES.ConvergedReason.ITERATING

# Set output directory
OUTDIR = os.path.join("output", "configuration")

if comm.rank == 0:
    Path(OUTDIR).mkdir(parents=True, exist_ok=True)

print("OUTDIR: ", OUTDIR)
X_COORD = 0
Y_COORD = 1
AIRY = 0
TRANSVERSE = 1
ABS_TOLLERANCE = 1e-11
REL_TOLLERANCE = 1e-11
SOL_TOLLERANCE = 1e-11

def hessian(u): return ufl.grad(ufl.grad(u))
def sigma(u):
    J = ufl.as_matrix([[0, -1], [1, 0]])
    return J.T*( hessian(u) ) * J
def mongeAmpere(u1, u2): return ufl.inner( sigma(u1), hessian(u2))

def ouward_unit_normal_x(x): return x[X_COORD] / np.sqrt(x[X_COORD]**2 + x[Y_COORD]**2)
def ouward_unit_normal_y(x): return x[Y_COORD] / np.sqrt(x[X_COORD]**2 + x[Y_COORD]**2)
def counterclock_tangent_x(x): return -x[Y_COORD] / np.sqrt(x[X_COORD]**2 + x[Y_COORD]**2)
def counterclock_tangent_y(x): return x[X_COORD] / np.sqrt(x[X_COORD]**2 + x[Y_COORD]**2)

import numpy as np

def compute_fourier_coefficients(x_values, function_values):
    """
    Compute the Fourier coefficients of a function given its values and positions.

    Parameters:
    - x_values: array-like, x-coordinates where the function is sampled (should be evenly spaced).
    - function_values: array-like, function values at corresponding x-coordinates.

    Returns:
    - frequencies: Array of frequencies.
    - coeff_real: Array of real parts of Fourier coefficients.
    - coeff_imag: Array of imaginary parts of Fourier coefficients.
    """
    x_values = np.array(x_values)
    function_values = np.array(function_values)

    N = len(function_values) # Number of sample points

    # Compute Fourier coefficients
    fourier_coeffs = np.fft.fft(function_values) / N  # Normalize coefficients

    # Compute corresponding frequencies
    dx = x_values[1] - x_values[0]  # Spacing between points (assumed uniform)
    print("dx = ", dx)

    frequencies = np.fft.fftfreq(N, d=dx)
    coeff_real = np.real(fourier_coeffs)
    coeff_imag = np.imag(fourier_coeffs)

    # Compute wavelengths
    wavelengths = np.zeros_like(frequencies)
    wavelengths[frequencies != 0] = 1 / np.abs(frequencies[frequencies != 0])
    wavelengths[frequencies == 0] = np.inf  # The zero frequency corresponds to infinite wavelength

    return frequencies, coeff_real, coeff_imag, wavelengths

# SET DISCRETE DISTRIBUTION OF DISCLINATIONS
# disclination_points_list = [[-0.7, 0.0, 0], [0.0, 0.7, 0], [0.7, 0.0, 0], [0.0, -0.7, 0],
#                             [0.5, 0.5, 0], [0.5, -0.5, 0], [-0.5, -0.5, 0], [-0.5, 0.5, 0],
#                             [0, 0, 0]] #
#disclination_power_list = [-1, -1, -1, -1, +0.5, 0.5, 0.5, 0.5, 0]

eps = 0.0
#disclination_points_list = [[-eps, 0.0, 0], [0.0, eps, 0], [eps, 0.0, 0], [0.0, -eps, 0]]
#disclination_points_list = [[-eps, 0.0, 0], [eps, 0.0, 0]]
#disclination_power_list = [0, 0, 0, 0]
#disclination_power_list = [-0.5, -0.5, -0.5, -0.5]
#disclination_power_list = [0.1, 0.1]

disclination_points_list = [[0.0, 0.0, 0]]
disclination_power_list = [1.0]

# READ PARAMETERS FILE
#with pkg_resources.path(PATH_TO_PARAMETERS_FILE, 'parameters.yml') as f:
    #parameters, _ = load_parameters(f)
with open('../test/parameters.yml') as f:
    parameters = yaml.load(f, Loader=yaml.FullLoader)

#parameters["geometry"]["mesh_size"] = 0.03

Eyoung =  parameters["model"]["E"]
nu = parameters["model"]["nu"]
thickness = parameters["model"]["thickness"]
R = parameters["geometry"]["radius"]
mesh_size = parameters["geometry"]["mesh_size"]
# mesh_size = 0.03
IP = parameters["model"]["alpha_penalty"]

# UPDATE OUTDIR
info_experiment = f"mesh_{mesh_size}_IP_{IP}_E_{Eyoung:.2e}_h_{thickness:.2e}_s_{disclination_power_list}_eps_{eps}"
OUTDIR = os.path.join(OUTDIR, info_experiment)
if not os.path.exists(OUTDIR): os.makedirs(OUTDIR)

# COMPUTE DIMENSIONLESS PARAMETERS
#a -> beta
beta = R / thickness
rho_g = 2e4 # Density of the material times g-accelleration
N = 1 # N-times plate's own weight
p0 = rho_g * thickness
gamma = N * p0 / Eyoung
f0 = (beta**4) * gamma

print(10*"*")
print("Dimensionless parameters: ")
print("beta := R/h = ", beta)
print("f := beta^4 * gamma = ", f0)
print(10*"*")

# LOAD MESH
parameters["geometry"]["geom_type"] = "circle"
model_rank = 0
tdim = 2
gmsh_model, tdim = mesh_circle_gmshapi( parameters["geometry"]["geom_type"], 1, mesh_size, tdim )
mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, model_rank, tdim)

h = CellDiameter(mesh)
n = FacetNormal(mesh)

# FUNCTION SPACES
X = basix.ufl.element("P", str(mesh.ufl_cell()), parameters["model"]["order"])
Q = dolfinx.fem.functionspace(mesh, basix.ufl.mixed_element([X, X]))

q = dolfinx.fem.Function(Q)
v, w = ufl.split(q)
state = {"v": v, "w": w}

# SET DIRICHLET BC
mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
bndry_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
bndry_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
dofs_v = dolfinx.fem.locate_dofs_topological(V=Q.sub(AIRY), entity_dim=1, entities=bndry_facets)
dofs_w = dolfinx.fem.locate_dofs_topological(V=Q.sub(TRANSVERSE), entity_dim=1, entities=bndry_facets)
bcs_v = dirichletbc( np.array(0, dtype=PETSc.ScalarType), dofs_v, Q.sub(AIRY) )
bcs_w = dirichletbc( np.array(0, dtype=PETSc.ScalarType), dofs_w, Q.sub(TRANSVERSE) )
_bcs = {AIRY: bcs_v, TRANSVERSE: bcs_w}
bcs = list(_bcs.values())


# DEFINE THE VARIATIONAL PROBLEM
model = A_NonlinearPlateFVK(mesh, parameters["model"])
energy = model.energy(state)[0]

# Volume load
#def volume_load(x): return - f0 * (x[0]**2 + x[1]**2)
def volume_load(x): return - f0 * (1 + 0*x[0]**2 + 0*x[1]**2)
f = dolfinx.fem.Function(Q.sub(TRANSVERSE).collapse()[0])
f.interpolate(volume_load)
dx = ufl.Measure("dx")
W_ext = f * w * dx # CFe: external work
penalisation = model.penalisation(state)

# Disclinations
if mesh.comm.rank == 0:
    disclinations = []
    for dp in disclination_points_list: disclinations.append(np.array([dp], dtype=mesh.geometry.x.dtype))
else:
    for dp in disclination_points_list: disclinations.append(np.zeros((0, 3), dtype=mesh.geometry.x.dtype))

# Functional
L = energy - W_ext + penalisation

# DEFINE THE FEM (WEAK) PROBLEM
F = ufl.derivative(L, q, ufl.TestFunction(Q))

Q_v, Q_v_to_Q_dofs = Q.sub(AIRY).collapse()

dp_list = [element*(beta**2) for element in disclination_power_list]

b = compute_disclination_loads(disclinations, dp_list, Q, V_sub_to_V_dofs=Q_v_to_Q_dofs, V_sub=Q_v)

solver_parameters = {
        "snes_type": "newtonls",  # Solver type: NGMRES (Nonlinear GMRES)
        "snes_max_it": 50,  # Maximum number of iterations
        "snes_rtol": REL_TOLLERANCE ,  # Relative tolerance for convergence
        "snes_atol": ABS_TOLLERANCE,  # Absolute tolerance for convergence
        "snes_stol": SOL_TOLLERANCE,  # Tolerance for the change in solution norm
        "snes_monitor": None,  # Function for monitoring convergence (optional)
        "snes_linesearch_type": "basic",  # Type of line search
    }

solver = SNESSolver(
    F_form=F,
    u=q,
    bcs=bcs,
    petsc_options=solver_parameters, #parameters.get("solvers").get("elasticity").get("snes"),
    prefix='plate_configuration',
    b0=b.vector,
    monitor=monitor,
)

solver.solve() # CFe: solve FEM problem
#convergence = solver.getConvergedReason()
#print("convergence: ", convergence)

# DISPLAY COMPUTED ENERGY VALUES
energy_scale = Eyoung * (thickness**3) / (beta**2)
energy_components = {
    "bending": energy_scale*model.energy(state)[1],
    "membrane": energy_scale*model.energy(state)[2],
    "coupling": energy_scale*model.energy(state)[3],
    "external_work": p0* thickness * (R**2) *W_ext
    }
# penalty_components = {"dg1_w": dg1_w(w), "dg2": dg2(w), "dg1_v": dg1_v(v), "dgc": dgc(v, w)}
# boundary_components = {"bc1_w": bc1_w(w), "bc2": bc2(w), "bc1_v": bc1_v(v)}

computed_energy_terms = {label: comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(energy_term)), op=MPI.SUM) for
                         label, energy_term in energy_components.items()}

print("Dimensional energy values: ", computed_energy_terms)


# DEFINE AIRY AND TRANSVERSE DISPLACEMENT FOR POST-PROCESSING
print("v_scale := Eh^3 = ", model.v_scale)
print("w_scale := h = ", model.w_scale)
v_pp, w_pp = q.split()
V_v, dofs_v = Q.sub(AIRY).collapse()
V_w, dofs_w = Q.sub(TRANSVERSE).collapse()
#v_pp.vector.array.real[dofs_v] = model.v_scale * v_pp.vector.array.real[dofs_v]
#w_pp.vector.array.real[dofs_w] = model.w_scale * w_pp.vector.array.real[dofs_w]

# COMPUTE STRESSES
sigma_xx = dolfinx.fem.Function(V_v)
sigma_xy = dolfinx.fem.Function(V_v)
sigma_yy = dolfinx.fem.Function(V_v)

sigma_xx_expr = dolfinx.fem.Expression( hessian(v_pp)[Y_COORD, Y_COORD], V_v.element.interpolation_points() )
sigma_xy_expr = dolfinx.fem.Expression( - hessian(v_pp)[X_COORD, Y_COORD], V_v.element.interpolation_points() )
sigma_yy_expr = dolfinx.fem.Expression( hessian(v_pp)[X_COORD, X_COORD], V_v.element.interpolation_points() )

sigma_xx.interpolate(sigma_xx_expr)
sigma_xy.interpolate(sigma_xy_expr)
sigma_yy.interpolate(sigma_yy_expr)

n_x = dolfinx.fem.Function(V_v)
n_y = dolfinx.fem.Function(V_v)
t_x = dolfinx.fem.Function(V_v)
t_y = dolfinx.fem.Function(V_v)
n_x.interpolate(ouward_unit_normal_x)
n_y.interpolate(ouward_unit_normal_y)
t_x.interpolate(counterclock_tangent_x)
t_y.interpolate(counterclock_tangent_y)

sigma_n_x = dolfinx.fem.Function(V_v)
sigma_n_y = dolfinx.fem.Function(V_v)
sigma_nx_expr = dolfinx.fem.Expression( sigma_xx*n_x + sigma_xy*n_y, V_v.element.interpolation_points() )
sigma_ny_expr = dolfinx.fem.Expression( sigma_xy*n_x + sigma_yy*n_y, V_v.element.interpolation_points() )
#sigma_nx_expr = dolfinx.fem.Expression( n_x, V_v.element.interpolation_points() )
#sigma_ny_expr = dolfinx.fem.Expression( n_y, V_v.element.interpolation_points() )
sigma_n_x.interpolate(sigma_nx_expr)
sigma_n_y.interpolate(sigma_ny_expr)
sigma_n = np.column_stack((sigma_n_x.x.array.real, sigma_n_y.x.array.real, np.zeros_like(sigma_n_x.x.array.real)))

sigma_nn = dolfinx.fem.Function(V_v)
sigma_nn_expr = dolfinx.fem.Expression( sigma_n_x*n_x + sigma_n_y*n_y , V_v.element.interpolation_points() )
sigma_nn.interpolate(sigma_nn_expr)

sigma_nt = dolfinx.fem.Function(V_v)
sigma_nt_expr = dolfinx.fem.Expression( sigma_n_x*t_x + sigma_n_y*t_y , V_v.element.interpolation_points() )
sigma_nt.interpolate(sigma_nt_expr)

sigma_tt = dolfinx.fem.Function(V_v)
sigma_tt_expr = dolfinx.fem.Expression( ( sigma_xx*t_x + sigma_xy*t_y )*t_x + ( sigma_xy*t_x + sigma_yy*t_y )*t_y , V_v.element.interpolation_points() )
sigma_tt.interpolate(sigma_tt_expr)

# COMPUTE MONGE-AMPERE BRACKET
ma_w = dolfinx.fem.Function(V_v)
ma_w_expr = dolfinx.fem.Expression( mongeAmpere(w_pp, w_pp), V_v.element.interpolation_points() )
ma_w.interpolate(ma_w_expr)

ma_vw = dolfinx.fem.Function(V_v)
ma_vw_expr = dolfinx.fem.Expression( mongeAmpere(w_pp, v_pp), V_v.element.interpolation_points() )
ma_vw.interpolate(ma_vw_expr)

# PLOT MESH
plt.figure()
ax = plot_mesh(mesh)
fig = ax.get_figure()
fig.savefig(f"{OUTDIR}/mesh_{info_experiment}.png")


# PLOT WITH PYVISTA
IMG_WIDTH = 1600
IMG_HEIGHT = 1200
PNG_SCALE = 2.0
LINEWIDTH = 5
FONTSIZE = 30
if pyvista.OFF_SCREEN: pyvista.start_xvfb(wait=0.1)
transparent = False
figsize = 800
#sargs = dict(height=0.8, width=0.1, vertical=True, position_x=0.05, position_y=0.05, fmt="%1.2e", title_font_size=40, color="black", label_font_size=25)

scalar_bar_args = {
    "vertical": True,
    "title_font_size": 18,  # Increase the title font size
    "label_font_size": 18,  # Increase the label font size
    "width": 0.15,          # Adjust the width of the scalar bar
    "height": 0.3,          # Adjust the height of the scalar bar
    "position_x": 0.87,      # X position (between 0 and 1, relative to the viewport)
    "position_y": 0.35       # Y position (between 0 and 1, relative to the viewport)
}

topology, cells, geometry = dolfinx.plot.vtk_mesh(Q_v)
grid = pyvista.UnstructuredGrid(topology, cells, geometry)

# PLOT FEM AND ANALYTICAL SOLUTIONS


# Airy, countour plot
subplotter = pyvista.Plotter(shape=(1, 2))
grid.point_data["v"] = v_pp.x.array.real[dofs_v]
grid.set_active_scalars("v")
subplotter.subplot(0, 0)
subplotter.add_text("v", font_size=14, color="black", position="upper_edge")
scalar_bar_args["title"] = "v"
subplotter.add_mesh(
    grid,
    show_edges=False,
    edge_color="white",
    show_scalar_bar=True,
    scalar_bar_args=scalar_bar_args,
    cmap="viridis")
#subplotter.view_xy()


grid.set_active_scalars("v")
subplotter.subplot(0, 1)
subplotter.add_text("v", position="upper_edge", font_size=14, color="black")
scalar_bar_args["title"] = "v"
subplotter.add_mesh( grid.warp_by_scalar( scale_factor = 1/max(np.abs(w_pp.x.array.real[dofs_v] )) ), show_edges=False, edge_color="white", show_scalar_bar=True, scalar_bar_args=scalar_bar_args, cmap="viridis")
subplotter.window_size = (IMG_WIDTH, IMG_HEIGHT)
subplotter.screenshot(f"{OUTDIR}/visualization_Airy_{info_experiment}.png", scale = PNG_SCALE)
# subplotter.export_html(f"{OUTDIR}/visualization_Airy_{info_experiment}.html")
pdb.set_trace()

# Transverse displacement, countour
subplotter = pyvista.Plotter(shape=(1, 2))
grid.point_data["w"] = w_pp.x.array.real[dofs_w]
grid.set_active_scalars("w")
scalar_bar_args["title"] = "w"
subplotter.subplot(0, 0)
subplotter.add_text("w", font_size=14, color="black", position="upper_edge")
subplotter.add_mesh(
    grid,
    show_edges=False,
    edge_color="white",
    show_scalar_bar=True,
    scalar_bar_args=scalar_bar_args,
    cmap="plasma")
#subplotter.view_xy()

# Transverse displacement, 3D view
grid.set_active_scalars("w")
subplotter.subplot(0, 1)
subplotter.add_text("w", position="upper_edge", font_size=14, color="black")
subplotter.add_mesh(
    grid.warp_by_scalar( scale_factor = 1/max(np.abs(w_pp.x.array.real[dofs_w] )) ),
    show_edges=False,
    edge_color="white",
    show_scalar_bar=True,
    scalar_bar_args=scalar_bar_args,
    cmap="plasma")
#subplotter.show_grid(xlabel="X-axis", ylabel="Y-axis", zlabel="Height (u)")
subplotter.window_size = (IMG_WIDTH, IMG_HEIGHT)
subplotter.screenshot(f"{OUTDIR}/visualization_w_{info_experiment}.png", scale = PNG_SCALE)
# subplotter.export_html(f"{OUTDIR}/visualization_w_{info_experiment}.html")

#pdb.set_trace()

# Cauchy stresses
subplotter = pyvista.Plotter(shape=(1, 3))
grid.point_data["sigma_xx"] = sigma_xx.x.array.real
grid.point_data["sigma_yy"] = sigma_yy.x.array.real
grid.point_data["sigma_xy"] = sigma_xy.x.array.real


# grid["vectors"] = sigma_n
# grid.set_active_vectors("vectors")
# glyphs = grid.glyph(scale="vector_magnitude", orient="vectors", factor=0, density=0)
# subplotter.add_mesh(glyphs, lighting=False, scalar_bar_args={"title": "Vector Magnitude"})
# subplotter.add_mesh(grid, color="grey", ambient=0.6, opacity=0.5, show_edges=False)
# subplotter.export_html(f"{OUTDIR}/visualization_sigman_{info_experiment}.html")

#pdb.set_trace()

grid.set_active_scalars("sigma_xx")
scalar_bar_args["title"] = "sigma_xx"
subplotter.subplot(0, 0)
subplotter.add_text("sigma_xx", position="upper_edge", font_size=14, color="black")
subplotter.add_mesh(
    grid.warp_by_scalar( scale_factor = 1 / max(np.abs(sigma_xx.x.array.real )) ),
    show_edges=False,
    edge_color="white",
    show_scalar_bar=True,
    scalar_bar_args=scalar_bar_args,
    cmap="viridis")
#subplotter.show_grid(xlabel="X-axis", ylabel="Y-axis", zlabel="Height (u)")

grid.set_active_scalars("sigma_yy")
scalar_bar_args["title"] = "sigma_yy"
subplotter.subplot(0, 1)
subplotter.add_text("sigma_yy", position="upper_edge", font_size=14, color="black")
subplotter.add_mesh(
    grid.warp_by_scalar( scale_factor = 1 / max(np.abs(sigma_yy.x.array.real )) ),
    show_edges=False,
    edge_color="white",
    show_scalar_bar=True,
    scalar_bar_args=scalar_bar_args,
    cmap="viridis")
#subplotter.show_grid(xlabel="X-axis", ylabel="Y-axis", zlabel="Height (u)")

grid.set_active_scalars("sigma_xy")
scalar_bar_args["title"] = "sigma_xy"
subplotter.subplot(0, 2)
subplotter.add_text("sigma_xy", position="upper_edge", font_size=14, color="black")
subplotter.add_mesh( grid.warp_by_scalar( scale_factor = 1 / max(np.abs(sigma_xy.x.array.real )) ), show_edges=False, edge_color="white", show_scalar_bar=True, scalar_bar_args=scalar_bar_args, cmap="viridis")
subplotter.window_size = (IMG_WIDTH, IMG_HEIGHT)
subplotter.screenshot(f"{OUTDIR}/visualization_CauchyStresses_{info_experiment}.png", scale = PNG_SCALE)
# subplotter.export_html(f"{OUTDIR}/visualization_CauchyStresses_{info_experiment}.html")
#pdb.set_trace()

# PLOT SIGMA N MAGINUTE
subplotter = pyvista.Plotter(shape=(1, 1))
scalar_bar_args["title"] = "sigma_r"
subplotter.subplot(0, 0)
grid["sigma_n"] = np.linalg.norm(sigma_n, axis=1)
grid.set_active_scalars("sigma_n")
subplotter.add_text("magnitude sigma_r", position="upper_edge", font_size=14, color="black")
subplotter.add_mesh( grid.warp_by_scalar( scale_factor = 1 / max(np.linalg.norm(sigma_n, axis=1)) ), show_edges=False, edge_color="white", show_scalar_bar=True, scalar_bar_args=scalar_bar_args, cmap="coolwarm")
subplotter.window_size = (IMG_WIDTH, IMG_HEIGHT)
subplotter.screenshot(f"{OUTDIR}/visualization_sigma_r_abs_{info_experiment}.png", scale = PNG_SCALE)
# subplotter.export_html(f"{OUTDIR}/visualization_sigma_r_abs_{info_experiment}.html")

# PLOT SIGMA N VECTOR PLOT
subplotter = pyvista.Plotter(shape=(1, 1))
subplotter.subplot(0, 0)
normalized_sigma_n = sigma_n /  (10*np.max(sigma_n)) #np.linalg.norm(sigma_n, axis=1)[:, None]
grid["normalized_sigma_n"] = normalized_sigma_n
grid["sigma_n_magnitude"] = np.linalg.norm(sigma_n, axis=1)
grid.set_active_vectors("normalized_sigma_n")
glyphs = grid.glyph(orient="normalized_sigma_n", factor=0.5, geom=pyvista.Arrow(), scale=False, tolerance=0.1)
scalar_bar_args["title"] = "Magnitude sigma_r"
subplotter.add_mesh(glyphs, scalars="sigma_n_magnitude", lighting=False, cmap="coolwarm", scalar_bar_args=scalar_bar_args )
subplotter.add_mesh(grid, color="lightgray", opacity=0.5, show_edges=True, edge_color="black")
subplotter.window_size = (IMG_WIDTH, IMG_HEIGHT)
subplotter.screenshot(f"{OUTDIR}/visualization_sigma_n_vec_{info_experiment}.png", scale = PNG_SCALE)
# subplotter.export_html(f"{OUTDIR}/visualization_sigma_n_vec_{info_experiment}.html")

plotter = pyvista.Plotter()
# Set the disk geometry and vector field if not already part of the grid
# For example, if grid is a pyvista.StructuredGrid or UnstructuredGrid
# Make sure sigma_n is set as a vector array on the grid
grid.set_active_vectors("normalized_sigma_n")  # Set 'sigma_n' as the active vector field
plotter.add_arrows(grid.points, grid["normalized_sigma_n"], mag=0.2, cmap="coolwarm", scalar_bar_args={"title": "Vector Magnitude"}) # Add the vector field arrows
plotter.add_mesh(grid, show_edges=True, edge_color="blue", color="lightgray", opacity=0.5) # Add the disk boundary
plotter.add_text("s = 1, σn", font_size=14, color="black", position="upper_edge")
subplotter.window_size = (IMG_WIDTH, IMG_HEIGHT)
# subplotter.export_html(f"{OUTDIR}/visualization_sigma_n_vec2_{info_experiment}.html")

# PLOT SIGMA NN
subplotter = pyvista.Plotter(shape=(1, 2))
grid["sigma_nn"] = sigma_nn.x.array.real
grid.set_active_scalars("sigma_nn")
subplotter.subplot(0, 0)
subplotter.add_text(r"sigma_rr", font_size=14, color="black", position="upper_edge")
scalar_bar_args["title"] = "sigma_rr"
subplotter.add_mesh(
    grid,
    show_edges=False,
    edge_color="white",
    show_scalar_bar=True,
    scalar_bar_args=scalar_bar_args,
    cmap="coolwarm")
contours = grid.contour(isosurfaces=10)
subplotter.add_mesh(contours, color="black",line_width=2, show_scalar_bar=False)
subplotter.subplot(0, 1)
scalar_bar_args["title"] = "sigma_rr"
subplotter.add_text("sigma_rr", position="upper_edge", font_size=14, color="black")
subplotter.add_mesh( grid.warp_by_scalar( scale_factor = 1 / max(np.abs(sigma_nn.x.array.real )) ), show_edges=False, edge_color="white", show_scalar_bar=True, scalar_bar_args=scalar_bar_args, cmap="coolwarm")
subplotter.window_size = (IMG_WIDTH, IMG_HEIGHT)
subplotter.screenshot(f"{OUTDIR}/visualization_sigma_rr_{info_experiment}.png", scale = PNG_SCALE)
# subplotter.export_html(f"{OUTDIR}/visualization_sigma_rr_{info_experiment}.html")

subplotter = pyvista.Plotter(shape=(1, 2))
grid["sigma_nt"] = sigma_nt.x.array.real
grid.set_active_scalars("sigma_nt")
subplotter.subplot(0, 0)
subplotter.add_text(r"sigma_nt", font_size=14, color="black", position="upper_edge")
scalar_bar_args["title"] = "sigma_nt"
subplotter.add_mesh(
    grid,
    show_edges=False,
    edge_color="white",
    show_scalar_bar=True,
    scalar_bar_args=scalar_bar_args,
    cmap="coolwarm")
contours = grid.contour(isosurfaces=10)
subplotter.add_mesh(contours, color="black",line_width=2, show_scalar_bar=False)
subplotter.subplot(0, 1)
subplotter.add_text("sigma_nt", position="upper_edge", font_size=14, color="black")
scalar_bar_args["title"] = "sigma_nt"
subplotter.add_mesh( grid.warp_by_scalar( scale_factor = 1 / max(np.abs(sigma_nt.x.array.real )) ), show_edges=False, edge_color="white", show_scalar_bar=True, scalar_bar_args=scalar_bar_args, cmap="coolwarm")
subplotter.window_size = (IMG_WIDTH, IMG_HEIGHT)
subplotter.screenshot(f"{OUTDIR}/visualization_sigma_nt_{info_experiment}.png", scale = PNG_SCALE)
# subplotter.export_html(f"{OUTDIR}/visualization_sigma_nt_{info_experiment}.html")

subplotter = pyvista.Plotter(shape=(1, 2))
grid["sigma_tt"] = sigma_tt.x.array.real
grid.set_active_scalars("sigma_tt")
subplotter.subplot(0, 0)
subplotter.add_text(r"sigma_tt", font_size=14, color="black", position="upper_edge")
scalar_bar_args["title"] = "sigma_tt"
subplotter.add_mesh(
    grid,
    show_edges=False,
    edge_color="white",
    show_scalar_bar=True,
    scalar_bar_args=scalar_bar_args,
    cmap="coolwarm")
contours = grid.contour(isosurfaces=10)
subplotter.add_mesh(contours, color="black",line_width=2, show_scalar_bar=False)
subplotter.subplot(0, 1)
subplotter.add_text("sigma_tt", position="upper_edge", font_size=14, color="black")
scalar_bar_args["title"] = "sigma_tt"
subplotter.add_mesh( grid.warp_by_scalar( scale_factor = 1 / max(np.abs(sigma_tt.x.array.real )) ), show_edges=False, edge_color="white", show_scalar_bar=True, scalar_bar_args=scalar_bar_args, cmap="coolwarm")
subplotter.window_size = (IMG_WIDTH, IMG_HEIGHT)
subplotter.screenshot(f"{OUTDIR}/visualization_sigma_tt_{info_experiment}.png", scale = PNG_SCALE)
# subplotter.export_html(f"{OUTDIR}/visualization_sigma_tt_{info_experiment}.html")

# PLOT MONGE-AMPERE W
subplotter = pyvista.Plotter(shape=(1, 2))
grid["ma_w"] = ma_w.x.array.real
grid.set_active_scalars("ma_w")
subplotter.subplot(0, 0)
subplotter.add_text(r"w", font_size=14, color="black", position="upper_edge")
scalar_bar_args["title"] = "[w, w]"
subplotter.add_mesh(
    grid,
    show_edges=False,
    edge_color="white",
    show_scalar_bar=True,
    scalar_bar_args=scalar_bar_args,
    cmap="coolwarm")
contours = grid.contour(isosurfaces=10)
subplotter.add_mesh(contours, color="black",line_width=2, show_scalar_bar=False)
subplotter.subplot(0, 1)
subplotter.add_text("ma_w", position="upper_edge", font_size=14, color="black")
scalar_bar_args["title"] = "[w, w]"
subplotter.add_mesh( grid.warp_by_scalar( scale_factor = 1 / max(np.abs(ma_w.x.array.real )) ), show_edges=False, edge_color="white", show_scalar_bar=True, scalar_bar_args=scalar_bar_args, cmap="coolwarm")
subplotter.window_size = (IMG_WIDTH, IMG_HEIGHT)
subplotter.screenshot(f"{OUTDIR}/viz_[w,w]_{info_experiment}.png", scale = PNG_SCALE)
# subplotter.export_html(f"{OUTDIR}/viz_[w,w]_{info_experiment}.html")

subplotter = pyvista.Plotter(shape=(1, 1))
grid["ma_vw"] = ma_vw.x.array.real
grid.set_active_scalars("ma_vw")
subplotter.subplot(0, 0)
subplotter.add_text("ma_vw", position="upper_edge", font_size=14, color="black")
scalar_bar_args["title"] = "[v,w]"
subplotter.add_mesh( grid.warp_by_scalar( scale_factor = 1 / max(np.abs(ma_vw.x.array.real )) ), show_edges=False, edge_color="white", show_scalar_bar=True, scalar_bar_args=scalar_bar_args, cmap="coolwarm")
subplotter.window_size = (IMG_WIDTH, IMG_HEIGHT)
#subplotter.screenshot(f"{OUTDIR}/viz_[v,w]_{info_experiment}.png", scale = PNG_SCALE)
# subplotter.export_html(f"{OUTDIR}/viz_[v,w]_{info_experiment}.html")


# PYVISTA PROFILE PLOTS
grid.set_active_scalars("w")
points = grid.points
y0 = 0
tolerance = 1e-2
x_values = points[np.abs(points[:, 1] - y0) < tolerance, 0]  # Select x-coordinates at y = 0
w_slice = grid['w'][np.abs(points[:, 1] - y0) < tolerance]
sorted_indices = np.argsort(x_values) # Sort data for plotting
x_sorted = x_values[sorted_indices]
w_sliceSorted = w_slice[sorted_indices]
scale_w_slice = f"{np.max(np.abs(w_sliceSorted)):.1e}"
plt.figure(figsize=(15, 11))
plt.plot(x_sorted, w_sliceSorted, label=f'w', color='blue', linestyle='solid', linewidth=LINEWIDTH)
plt.xticks(fontsize=FONTSIZE)
plt.yticks(fontsize=FONTSIZE)
plt.xlabel(r"$\xi_1$", fontsize=FONTSIZE)
plt.ylabel(r"$w$", fontsize=FONTSIZE)
plt.title(rf"Profile of w at $\xi_2$ = {y0}", fontsize=FONTSIZE)
ax = plt.gca() # use scientific notation for y axis
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax.yaxis.get_offset_text().set_fontsize(FONTSIZE)
ax.xaxis.set_major_locator(MaxNLocator(nbins=5))  # Adjust the number of bins to choose the number of ticks
#plt.legend(fontsize=FONTSIZE)
#plt.grid(True)
plt.savefig(f"{OUTDIR}/profile_w_y_{y0}_{info_experiment}.png", dpi=300)

grid.set_active_scalars("v")
points = grid.points
v_slice = grid['v'][np.abs(points[:, 1] - y0) < tolerance]
v_sliceSorted = v_slice[sorted_indices]
scale_v_slice = f"{np.max(np.abs(v_sliceSorted)):.1e}"
plt.figure(figsize=(15, 11))
plt.plot(x_sorted, v_sliceSorted, label=f'v', color='red', linestyle='solid', linewidth=LINEWIDTH)
plt.xticks(fontsize=FONTSIZE)
plt.yticks(fontsize=FONTSIZE)
plt.xlabel(r"$\xi_1$", fontsize=FONTSIZE)
plt.ylabel(r"$v$", fontsize=FONTSIZE)
plt.title(fr"Profile of v at $\xi_2$ = {y0}", fontsize=FONTSIZE)
ax = plt.gca() # use scientific notation for y axis
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax.yaxis.get_offset_text().set_fontsize(FONTSIZE)
ax.xaxis.set_major_locator(MaxNLocator(nbins=5))  # Adjust the number of bins to choose the number of ticks
#plt.legend(fontsize=FONTSIZE)
#plt.grid(True)
plt.savefig(f"{OUTDIR}/profile_v_y_{y0}_{info_experiment}.png", dpi=300)

grid.set_active_scalars("w")
points = grid.points
x0 = 0
y_values = points[np.abs(points[:, 0] - x0) < tolerance, 1]  # Select y-coordinates at x = 0
w_slice_2 = grid['w'][np.abs(points[:, 0] - x0) < tolerance]
y_sorted_indices = np.argsort(y_values) # Sort data for plotting
y_values = y_values[y_sorted_indices]
w_sliceSorted_2 = w_slice_2[y_sorted_indices]
scale_w_slice = f"{np.max(np.abs(w_sliceSorted_2)):.1e}"
plt.figure(figsize=(15, 11))
plt.plot(y_values, w_sliceSorted_2, label=f'w', color='blue', linestyle='solid', linewidth=LINEWIDTH)
plt.xticks(fontsize=FONTSIZE)
plt.yticks(fontsize=FONTSIZE)
plt.xlabel(r"$\xi_2$", fontsize=FONTSIZE)
plt.ylabel(r"$w$", fontsize=FONTSIZE)
plt.title(rf"Profile of w at $\xi_1$ = {x0}", fontsize=FONTSIZE)
ax = plt.gca() # use scientific notation for y axis
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax.yaxis.get_offset_text().set_fontsize(FONTSIZE)
ax.xaxis.set_major_locator(MaxNLocator(nbins=5))  # Adjust the number of bins to choose the number of ticks
plt.savefig(f"{OUTDIR}/profile_w_x_{x0}_{info_experiment}.png", dpi=300)

grid.set_active_scalars("v")
points = grid.points
v_slice_2 = grid['v'][np.abs(points[:, 0] - x0) < tolerance]
v_sliceSorted_2 = v_slice_2[y_sorted_indices]
scale_v_slice = f"{np.max(np.abs(v_sliceSorted_2)):.1e}"
plt.figure(figsize=(15, 11))
plt.plot(y_values, v_sliceSorted_2, label=f'v', color='red', linestyle='solid', linewidth=LINEWIDTH)
plt.xticks(fontsize=FONTSIZE)
plt.yticks(fontsize=FONTSIZE)
plt.xlabel(r"$\xi_2$", fontsize=FONTSIZE)
plt.ylabel(r"$v$", fontsize=FONTSIZE)
plt.title(fr"Profile of v at $\xi_1$ = {x0}", fontsize=FONTSIZE)
ax = plt.gca() # use scientific notation for y axis
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax.yaxis.get_offset_text().set_fontsize(FONTSIZE)
ax.xaxis.set_major_locator(MaxNLocator(nbins=5))  # Adjust the number of bins to choose the number of ticks
plt.savefig(f"{OUTDIR}/profile_v_x_{x0}_{info_experiment}.png", dpi=300)

# COMPUTE FOURIER COEFFICIENTS, FREQUENCIES
from scipy.interpolate import interp1d
x_values = np.array(x_sorted)
function_values = np.array(w_sliceSorted)
num_points = 501  # Choose the number of points for resampling
uniform_grid = np.linspace(-1, 1, num_points)
interp_func = interp1d(x_values, function_values, kind='cubic', fill_value="extrapolate") # Interpolate the function
resampled_values = interp_func(uniform_grid) # Evaluate the function on the uniform grid

plt.figure(figsize=(10, 6))
plt.plot(x_values, function_values, 'o', label='Original Data', markersize=5)
plt.plot(uniform_grid, resampled_values, '-', label='Interpolated Data', linewidth=2)
plt.xlabel(r'$\xi_1$', fontsize=14)
plt.ylabel(r'$w$', fontsize=14)
plt.title('Interpolation and Resampling on Uniform Grid', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.savefig(f"{OUTDIR}/interp_original_w_y_{y0}_{info_experiment}.png", dpi=300)

N_fourier_coeff = 10
frequencies, coeff_real, coeff_imag, wavelength_list = compute_fourier_coefficients(uniform_grid, resampled_values)
coeff_abs = np.sqrt(coeff_real**2 + coeff_imag**2)
#top_indices = np.argsort(coeff_abs)[-N_fourier_coeff:][::-1]  # Sort and reverse for descending order
print(f"Top {N_fourier_coeff} Fourier Coefficients:")
print("Rank | Frequency | Absolute Value")
for rank, idx in enumerate(coeff_abs):
    print(f"{frequencies[rank]:>10.4f} | {coeff_abs[rank]:>14.4e}|{wavelength_list[rank]:>10.4f}  ")

#pdb.set_trace()
# PLOT FOURIER MODES
#threshold = np.sort(coeff_abs)[-N_fourier_coeff]
# Filter coefficients based on the threshold
#mask = (coeff_abs >= threshold) & (frequencies >= 0) # True for coefficients to keep
#filtered_frequencies = frequencies[mask]
filtered_frequencies = frequencies[0:N_fourier_coeff]
filtered_coeff_abs = coeff_abs[0:N_fourier_coeff]
plt.figure(figsize=(13, 8))
stems = plt.stem(filtered_frequencies, filtered_coeff_abs, basefmt=" ")
stems[1].set_linewidth(6)
plt.tick_params(axis='both', which='major', labelsize=30)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))  # Approx. 10 major ticks
plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1))  # 2 minor ticks between major ones
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xlabel("Frequency", fontsize=30)
plt.ylabel(f"ABS Fourier Coefficients", fontsize=30)
plt.title(f"Abs Fourier Coefficients vs Frequency. eps={disclination_points_list[0][0]}, s={disclination_power_list[0]}", fontsize=30)
plt.grid(True)
plt.tight_layout()
plt.gca().yaxis.get_offset_text().set_fontsize(35)
plt.savefig(f"{OUTDIR}/frequencies_w_y_{y0}_{info_experiment}.png", dpi=300)

# xv = np.linspace(0, 2*np.pi, 6283185)  # Uniform grid
# fv = np.sin(xv)  # Example function
# frq, cr, ci, wl = compute_fourier_coefficients(xv, fv)
# print("Frequencies: ", frq)
# print("Wavelengths: ", wl)
# print("abs ", np.sqrt(cr**2 + ci**2) )
#pdb.set_trace()

# PLOTS MATPLOTLIB
# x_samples, y_samples, v_samples = sample_function(v_pp, parameters["geometry"]["radius"])
# grid_x, grid_y, v_interp = interpolate_sample(x_samples, y_samples, v_samples, parameters["geometry"]["radius"])
#
# x_samples, y_samples, w_samples = sample_function(w_pp, parameters["geometry"]["radius"])
# grid_x, grid_y, w_interp = interpolate_sample(x_samples, y_samples, w_samples, parameters["geometry"]["radius"])
#
# x0_samples = []
# v0_samples = []
# w0_samples = []
# for i in range(len(y_samples)):
#     if abs(y_samples[i]) < 5e-11:
#         x0_samples.append(x_samples[i])
#         v0_samples.append(v_samples[i])
#         w0_samples.append(w_samples[i])
#
# # Profile plots
# plt.figure(figsize=(14, 9))
# plt.plot(x0_samples, v0_samples, color='blue', linestyle='-', label='v(x,0)', linewidth=5)
# plt.xlabel(r'$\xi_1$', fontsize=30)
# plt.ylabel(r'$\tilde{v}$', fontsize=30)
# plt.tick_params(axis='both', which='major', labelsize=30)
# plt.title(r'$\tilde{v}$',  fontsize=30)
# plt.legend(fontsize=30)
# plt.grid(True)
# plt.tight_layout()
# plt.gca().yaxis.get_offset_text().set_fontsize(35)
# plt.savefig(f"{OUTDIR}/profile_Airy_{info_experiment}.png", dpi=300)
# #plt.show()
#
# plt.figure(figsize=(14, 9))
# plt.plot(x0_samples, w0_samples, color='red', linestyle='-', label='w(x,0)', linewidth=5)
# plt.xlabel(r'$\xi_1$', fontsize=30)
# plt.ylabel(r'$\tilde{w}$', fontsize=30)
# plt.tick_params(axis='both', which='major', labelsize=30)
# plt.title(r'$\tilde{w}$',  fontsize=30)
# plt.legend(fontsize=30)
# plt.grid(True)
# plt.tight_layout()
# plt.gca().yaxis.get_offset_text().set_fontsize(35)
# plt.savefig(f"{OUTDIR}/profile_Transv_displacement_{info_experiment}.png", dpi=300)
#plt.show()

