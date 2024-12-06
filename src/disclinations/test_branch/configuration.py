"""
PURPOSE OF THE SCRIPT
Analyze configuration of interest
Using the Veriational FE formulationin its NON-dimensional formulation (see "models/adimensional.py").
Dimensionless parameters:
a := R/h >> 1
b := p0/E << 1

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

def ouward_unit_normal_x(x): return x[X_COORD] / np.sqrt(x[X_COORD]**2 + x[Y_COORD]**2)
def ouward_unit_normal_y(x): return x[Y_COORD] / np.sqrt(x[X_COORD]**2 + x[Y_COORD]**2)
def counterclock_tangent_x(x): return -x[Y_COORD] / np.sqrt(x[X_COORD]**2 + x[Y_COORD]**2)
def counterclock_tangent_y(x): return x[X_COORD] / np.sqrt(x[X_COORD]**2 + x[Y_COORD]**2)

# SET DISCRETE DISTRIBUTION OF DISCLINATIONS
# disclination_points_list = [[-0.7, 0.0, 0], [0.0, 0.7, 0], [0.7, 0.0, 0], [0.0, -0.7, 0],
#                             [0.5, 0.5, 0], [0.5, -0.5, 0], [-0.5, -0.5, 0], [-0.5, 0.5, 0],
#                             [0, 0, 0]] #
#disclination_power_list = [-1, -1, -1, -1, +0.5, 0.5, 0.5, 0.5, 0]

# OK configuration
disclination_points_list = [[-0.5, 0.0, 0], [0.0, 0.5, 0], [0.5, 0.0, 0], [0.0, -0.5, 0]]
#disclination_power_list = [0, 0, 0, 0]
#disclination_power_list = [-0.5, -0.5, -0.5, -0.5]
disclination_power_list = [0.5, 0.5, 0.5, 0.5]

#disclination_points_list = [[0.0, 0.0, 0]]
#disclination_power_list = [0]

# READ PARAMETERS FILE
#with pkg_resources.path(PATH_TO_PARAMETERS_FILE, 'parameters.yml') as f:
    #parameters, _ = load_parameters(f)
with open('../test/parameters.yml') as f:
    parameters = yaml.load(f, Loader=yaml.FullLoader)

Eyoung =  parameters["model"]["E"]
nu = parameters["model"]["nu"]
thickness = parameters["model"]["thickness"]
R = parameters["geometry"]["radius"]
mesh_size = parameters["geometry"]["mesh_size"]
IP = parameters["model"]["alpha_penalty"]

# UPDATE OUTDIR
info_experiment = f"mesh_{mesh_size}_IP_{IP}_E_{Eyoung:.2e}_h_{thickness:.2e}_s_{disclination_power_list}"
OUTDIR = os.path.join(OUTDIR, info_experiment)
if not os.path.exists(OUTDIR): os.makedirs(OUTDIR)

# COMPUTE DIMENSIONLESS PARAMETERS
a = R / thickness
rho_g = 2e4 # Density of the material times g-accelleration
N = 1
p0 = rho_g *parameters["model"]["thickness"]
f0 = N * a**4 * p0 / Eyoung

print(10*"*")
print("Dimensionless parameters: ")
print("a := R/h = ", a)
print("f := a^4 * p0/E = ", f0)
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

dp_list = [element*(a**2) for element in disclination_power_list]

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
energy_scale = Eyoung * (thickness**3) / (a**2)
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
v_pp.vector.array.real[dofs_v] = model.v_scale * v_pp.vector.array.real[dofs_v]
w_pp.vector.array.real[dofs_w] = model.w_scale * w_pp.vector.array.real[dofs_w]

# COMPUTE STRESSES
sigma_xx = dolfinx.fem.Function(V_v)
sigma_xy = dolfinx.fem.Function(V_v)
sigma_yy = dolfinx.fem.Function(V_v)

# Contribute to the total membranale stress due to the transverse displacement
sigmaxx_w = ( Eyoung / (1-nu**2) ) * (1/2) * ( ufl.grad( w_pp )[0]**2 + nu * ufl.grad( w_pp )[1]**2 )
sigmayy_w = ( Eyoung / (1-nu**2) ) * (1/2) * ( ufl.grad( w_pp )[1]**2 + nu * ufl.grad( w_pp )[0]**2 )
sigmaxy_w = ( Eyoung / (1-nu**2) ) * (1/2) * ( ufl.grad( w_pp )[0]  * ufl.grad( w_pp )[1] )

sigma_xx_expr = dolfinx.fem.Expression( hessian(v_pp)[Y_COORD, Y_COORD] + sigmaxx_w, V_v.element.interpolation_points() )
sigma_xy_expr = dolfinx.fem.Expression( - hessian(v_pp)[X_COORD, Y_COORD] + sigmaxy_w, V_v.element.interpolation_points() )
sigma_yy_expr = dolfinx.fem.Expression( hessian(v_pp)[X_COORD, X_COORD] + sigmayy_w, V_v.element.interpolation_points() )

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

# PLOT MESH
plt.figure()
ax = plot_mesh(mesh)
fig = ax.get_figure()
fig.savefig(f"{OUTDIR}/mesh_{info_experiment}.png")


# PLOT WITH PYVISTA
IMG_WIDTH = 1600
IMG_HEIGHT = 1200
PNG_SCALE = 2.0
if pyvista.OFF_SCREEN: pyvista.start_xvfb(wait=0.1)
transparent = False
figsize = 800
sargs = dict(height=0.8, width=0.1, vertical=True, position_x=0.05, position_y=0.05, fmt="%1.2e", title_font_size=40, color="black", label_font_size=25)
topology, cells, geometry = dolfinx.plot.vtk_mesh(Q_v)
grid = pyvista.UnstructuredGrid(topology, cells, geometry)

# PLOT FEM AND ANALYTICAL SOLUTIONS

subplotter = pyvista.Plotter(shape=(1, 2))

# Airy, countour plot
grid.point_data["v"] = v_pp.x.array.real[dofs_v]
grid.set_active_scalars("v")
subplotter.subplot(0, 0)
subplotter.add_text("Airy's function", font_size=14, color="black", position="upper_edge")
subplotter.add_mesh(
    grid,
    show_edges=False,
    edge_color="white",
    show_scalar_bar=True,
    scalar_bar_args={"title": "Airy's function [Nm]", "vertical": False},
    cmap="viridis")
#subplotter.view_xy()


grid.set_active_scalars("v")
subplotter.subplot(0, 1)
subplotter.add_text("v", position="upper_edge", font_size=14, color="black")
subplotter.add_mesh( grid.warp_by_scalar( scale_factor = 1/max(np.abs(w_pp.x.array.real[dofs_v] )) ), show_edges=False, edge_color="white", show_scalar_bar=True, scalar_bar_args={"title": "Airy's function [Nm]", "vertical": False}, cmap="viridis")
subplotter.window_size = (IMG_WIDTH, IMG_HEIGHT)
subplotter.screenshot(f"{OUTDIR}/visualization_Airy_{info_experiment}.png", scale = PNG_SCALE)
subplotter.export_html(f"{OUTDIR}/visualization_Airy_{info_experiment}.html")

# Transverse displacement, countour
subplotter = pyvista.Plotter(shape=(1, 2))
grid.point_data["w"] = w_pp.x.array.real[dofs_w]
grid.set_active_scalars("w")
subplotter.subplot(0, 0)
subplotter.add_text("Transverse displacement [m]", font_size=14, color="black", position="upper_edge")
subplotter.add_mesh(
    grid,
    show_edges=False,
    edge_color="white",
    show_scalar_bar=True,
    scalar_bar_args={"title": "Transverse displacement  [m]", "vertical": False},
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
    scalar_bar_args={"title": "Transverse displacement [m]", "vertical": False},
    cmap="plasma")
#subplotter.show_grid(xlabel="X-axis", ylabel="Y-axis", zlabel="Height (u)")
subplotter.window_size = (IMG_WIDTH, IMG_HEIGHT)
subplotter.screenshot(f"{OUTDIR}/visualization_w_{info_experiment}.png", scale = PNG_SCALE)
subplotter.export_html(f"{OUTDIR}/visualization_w_{info_experiment}.html")

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
subplotter.subplot(0, 0)
subplotter.add_text("sigma_xx", position="upper_edge", font_size=14, color="black")
subplotter.add_mesh(
    grid.warp_by_scalar( scale_factor = 1 / max(np.abs(sigma_xx.x.array.real )) ),
    show_edges=False,
    edge_color="white",
    show_scalar_bar=True,
    scalar_bar_args={"title": "sigma_xx [Pa m]", "vertical": True},
    cmap="viridis")
#subplotter.show_grid(xlabel="X-axis", ylabel="Y-axis", zlabel="Height (u)")

grid.set_active_scalars("sigma_yy")
subplotter.subplot(0, 1)
subplotter.add_text("sigma_yy", position="upper_edge", font_size=14, color="black")
subplotter.add_mesh(
    grid.warp_by_scalar( scale_factor = 1 / max(np.abs(sigma_yy.x.array.real )) ),
    show_edges=False,
    edge_color="white",
    show_scalar_bar=True,
    scalar_bar_args={"title": "sigma_yy [Pa m]", "vertical": True},
    cmap="viridis")
#subplotter.show_grid(xlabel="X-axis", ylabel="Y-axis", zlabel="Height (u)")

grid.set_active_scalars("sigma_xy")
subplotter.subplot(0, 2)
subplotter.add_text("sigma_xy", position="upper_edge", font_size=14, color="black")
subplotter.add_mesh( grid.warp_by_scalar( scale_factor = 1 / max(np.abs(sigma_xy.x.array.real )) ), show_edges=False, edge_color="white", show_scalar_bar=True, scalar_bar_args={"title": "sigma_xy [Pa m]", "vertical": False}, cmap="viridis")
subplotter.window_size = (IMG_WIDTH, IMG_HEIGHT)
subplotter.screenshot(f"{OUTDIR}/visualization_CauchyStresses_{info_experiment}.png", scale = PNG_SCALE)
subplotter.export_html(f"{OUTDIR}/visualization_CauchyStresses_{info_experiment}.html")
#pdb.set_trace()

# PLOT SIGMA N MAGINUTE
subplotter = pyvista.Plotter(shape=(1, 1))
subplotter.subplot(0, 0)
grid["sigma_n"] = np.linalg.norm(sigma_n, axis=1)
grid.set_active_scalars("sigma_n")
subplotter.add_text("magnitude sigma_n", position="upper_edge", font_size=14, color="black")
subplotter.add_mesh( grid.warp_by_scalar( scale_factor = 1 / max(np.linalg.norm(sigma_n, axis=1)) ), show_edges=False, edge_color="white", show_scalar_bar=True, scalar_bar_args={"title": "sigma_n [Pa m]", "vertical": True}, cmap="coolwarm")
subplotter.window_size = (IMG_WIDTH, IMG_HEIGHT)
subplotter.screenshot(f"{OUTDIR}/visualization_sigma_n_abs_{info_experiment}.png", scale = PNG_SCALE)
subplotter.export_html(f"{OUTDIR}/visualization_sigma_n_abs_{info_experiment}.html")

# PLOT SIGMA N VECTOR PLOT
normalized_sigma_n = sigma_n /  (10*np.max(sigma_n)) #np.linalg.norm(sigma_n, axis=1)[:, None]
grid["normalized_sigma_n"] = normalized_sigma_n
grid["sigma_n_magnitude"] = np.linalg.norm(sigma_n, axis=1)
grid.set_active_vectors("normalized_sigma_n")
glyphs = grid.glyph(orient="normalized_sigma_n", factor=0.5, geom=pyvista.Arrow(), scale=False, tolerance=0.1)
subplotter = pyvista.Plotter(shape=(1, 1))
subplotter.subplot(0, 0)
subplotter.add_mesh(glyphs, scalars="sigma_n_magnitude", lighting=False, cmap="coolwarm", scalar_bar_args={"title": "Magnitude"} )
subplotter.add_mesh(grid, color="lightgray", opacity=0.5, show_edges=True, edge_color="black")
subplotter.window_size = (IMG_WIDTH, IMG_HEIGHT)
subplotter.screenshot(f"{OUTDIR}/visualization_sigma_n_vec_{info_experiment}.png", scale = PNG_SCALE)
subplotter.export_html(f"{OUTDIR}/visualization_sigma_n_vec_{info_experiment}.html")

# PLOT SIGMA NN
subplotter = pyvista.Plotter(shape=(1, 1))
subplotter.subplot(0, 0)
grid["sigma_nn"] = sigma_nn.x.array.real
grid.set_active_scalars("sigma_nn")
subplotter.add_text("sigma_nn", position="upper_edge", font_size=14, color="black")
subplotter.add_mesh( grid.warp_by_scalar( scale_factor = 1 / max(np.abs(sigma_nn.x.array.real )) ), show_edges=False, edge_color="white", show_scalar_bar=True, scalar_bar_args={"title": "sigma_nn [Pa m]", "vertical": True}, cmap="coolwarm")
subplotter.window_size = (IMG_WIDTH, IMG_HEIGHT)
subplotter.screenshot(f"{OUTDIR}/visualization_sigma_nn_{info_experiment}.png", scale = PNG_SCALE)
subplotter.export_html(f"{OUTDIR}/visualization_sigma_nn_{info_experiment}.html")

subplotter = pyvista.Plotter(shape=(1, 1))
subplotter.subplot(0, 0)
grid["sigma_nt"] = sigma_nt.x.array.real
grid.set_active_scalars("sigma_nt")
subplotter.add_text("sigma_nt", position="upper_edge", font_size=14, color="black")
subplotter.add_mesh( grid.warp_by_scalar( scale_factor = 1 / max(np.abs(sigma_nt.x.array.real )) ), show_edges=False, edge_color="white", show_scalar_bar=True, scalar_bar_args={"title": "sigma_nt [Pa m]", "vertical": True}, cmap="coolwarm")
subplotter.window_size = (IMG_WIDTH, IMG_HEIGHT)
subplotter.screenshot(f"{OUTDIR}/visualization_sigma_nt_{info_experiment}.png", scale = PNG_SCALE)
subplotter.export_html(f"{OUTDIR}/visualization_sigma_nt_{info_experiment}.html")

subplotter = pyvista.Plotter(shape=(1, 1))
subplotter.subplot(0, 0)
grid["sigma_tt"] = sigma_nn.x.array.real
grid.set_active_scalars("sigma_tt")
subplotter.add_text("sigma_tt", position="upper_edge", font_size=14, color="black")
subplotter.add_mesh( grid.warp_by_scalar( scale_factor = 1 / max(np.abs(sigma_tt.x.array.real )) ), show_edges=False, edge_color="white", show_scalar_bar=True, scalar_bar_args={"title": "sigma_tt [Pa m]", "vertical": True}, cmap="coolwarm")
subplotter.window_size = (IMG_WIDTH, IMG_HEIGHT)
subplotter.screenshot(f"{OUTDIR}/visualization_sigma_tt_{info_experiment}.png", scale = PNG_SCALE)
subplotter.export_html(f"{OUTDIR}/visualization_sigma_tt_{info_experiment}.html")

#pdb.set_trace()

# PLOTS MATPLOTLIB
x_samples, y_samples, v_samples = sample_function(v_pp, parameters["geometry"]["radius"])
grid_x, grid_y, v_interp = interpolate_sample(x_samples, y_samples, v_samples, parameters["geometry"]["radius"])

x_samples, y_samples, w_samples = sample_function(w_pp, parameters["geometry"]["radius"])
grid_x, grid_y, w_interp = interpolate_sample(x_samples, y_samples, w_samples, parameters["geometry"]["radius"])

x0_samples = []
v0_samples = []
w0_samples = []
for i in range(len(y_samples)):
    if abs(y_samples[i]) < 5e-11:
        x0_samples.append(x_samples[i])
        v0_samples.append(v_samples[i])
        w0_samples.append(w_samples[i])

# Surface plots
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# surface = ax.plot_surface(grid_x, grid_y, v_interp, cmap='viridis', edgecolor='none')
# cbar = fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)
# cbar.set_label('[Nm]')
# ax.set_xlabel('x axis [m]', fontsize=25)
# ax.set_ylabel('y axis [m]', fontsize=25)
# ax.set_zlabel('v [Nm]', fontsize=25)
# ax.tick_params(axis='both', which='major', labelsize=20)  # X and Y axis major ticks
# ax.tick_params(axis='z', which='major', labelsize=20)
# ax.set_title('Airy\'s stress function [dimensional]', fontsize=30)
# plt.gca().yaxis.get_offset_text().set_fontsize(35)
# plt.savefig(f"{OUTDIR}/surface_Airy_{info_experiment}.png", dpi=300)
# #plt.show()
#
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# surface = ax.plot_surface(grid_x, grid_y, w_interp, cmap='coolwarm', edgecolor='none')
# cbar = fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)
# cbar.set_label('[m]')
# ax.set_xlabel('x axis [m]', fontsize=25)
# ax.set_ylabel('y axis [m]', fontsize=25)
# ax.set_zlabel('$w$ [m]', fontsize=25)
# ax.tick_params(axis='both', which='major', labelsize=20)  # X and Y axis major ticks
# ax.tick_params(axis='z', which='major', labelsize=20)
# ax.set_title('Transverse displacement [dimensional]', fontsize=30)
# plt.gca().yaxis.get_offset_text().set_fontsize(35)
# plt.savefig(f"{OUTDIR}/surface_Transv_displacement_{info_experiment}.png", dpi=300)
# #plt.show()
#
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# surface = ax.plot_surface(grid_x, grid_y, w_interp, cmap='coolwarm', edgecolor='none')
# cbar = fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)
# cbar.set_label('[m]')
# ax.set_xlabel('x axis [m]', fontsize=25)
# ax.set_ylabel('y axis [m]', fontsize=25)
# ax.set_zlabel('$w$ [m]', fontsize=25)
# ax.tick_params(axis='both', which='major', labelsize=20)  # X and Y axis major ticks
# ax.tick_params(axis='z', which='major', labelsize=20)
# ax.set_title('Transverse displacement [dimensional]', fontsize=30)
# ax.set_zlim([-1,1])
# plt.gca().yaxis.get_offset_text().set_fontsize(35)
# plt.savefig(f"{OUTDIR}/surface_Transv_displacement_2_{info_experiment}.png", dpi=300)
# #plt.show()
#
# if pyvista.OFF_SCREEN: pyvista.start_xvfb(wait=0.1)
# transparent = False
# figsize = 800
# sargs = dict(height=0.8, width=0.1, vertical=True, position_x=0.05, position_y=0.05, fmt="%1.2e", title_font_size=40, color="black", label_font_size=25)
# topology, cells, geometry = dolfinx.plot.vtk_mesh(Q.sub(0).collapse()[0])
# grid = pyvista.UnstructuredGrid(topology, cells, geometry)
#
# grid.point_data["v"] = v_pp.x.array.real[dofs_v]
# grid.set_active_scalars("v")
# warped = grid.warp_by_scalar()
# #pyvista.global_theme.jupyter_backend = 'html'  # Or 'panel'
# subplotter = pyvista.Plotter(shape=(1, 2))
#
# subplotter.subplot(0, 0)
# subplotter.add_text("Scalar contour field", font_size=14, color="black", position="upper_edge")
# subplotter.add_mesh(grid, show_edges=False, edge_color="white", show_scalar_bar=True, cmap="viridis")
# subplotter.view_xy()
#
# subplotter.subplot(0, 1)
# subplotter.add_text("v", position="upper_edge", font_size=14, color="black")
# subplotter.set_position([-3, 2.6, 0.3])
# subplotter.set_focus([3, -1, -0.15])
# subplotter.set_viewup([0, 0, 1])
# subplotter.add_mesh(warped, show_edges=False, edge_color="white", scalar_bar_args=sargs, cmap="viridis")
# subplotter.export_html(f"{OUTDIR}/visualization_v_{info_experiment}.html")
#
# grid.point_data["w"] = w_pp.x.array.real[dofs_w]
# grid.set_active_scalars("w")
# warped = grid.warp_by_scalar()
# #pyvista.global_theme.jupyter_backend = 'html'  # Or 'panel'
# subplotter = pyvista.Plotter(shape=(1, 2))
#
# subplotter.subplot(0, 0)
# subplotter.add_text("Scalar contour field", font_size=14, color="black", position="upper_edge")
# subplotter.add_mesh(grid, show_edges=False, edge_color="white", show_scalar_bar=True, cmap="viridis")
# subplotter.view_xy()
#
# subplotter.subplot(0, 1)
# subplotter.add_text("v", position="upper_edge", font_size=14, color="black")
# subplotter.set_position([-3, 2.6, 0.3])
# subplotter.set_focus([3, -1, -0.15])
# subplotter.set_viewup([0, 0, 1])
# subplotter.add_mesh(warped, show_edges=False, edge_color="white", scalar_bar_args=sargs, cmap="viridis")
# subplotter.export_html(f"{OUTDIR}/visualization_w_{info_experiment}.html")

# Profile plots
plt.figure(figsize=(10, 6))
plt.plot(x0_samples, v0_samples, color='red', linestyle='-', label='v(x,0)', linewidth=3)
plt.xlabel('x axes [m]', fontsize=25)
plt.ylabel('Airy\' stress function [Nm]', fontsize=25)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.title('Airy\'s stress function [Nm]',  fontsize=30)
plt.legend(fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/profile_Airy_{info_experiment}.png", dpi=300)
#plt.show()

plt.figure(figsize=(10, 6))
plt.plot(x0_samples, w0_samples, color='red', linestyle='-', label='w(x,0)', linewidth=3)
plt.xlabel('x axes [m]', fontsize=25)
plt.ylabel('Transverse displacement [m]', fontsize=25)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.title('Transverse displacement [m]',  fontsize=30)
plt.legend(fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.gca().yaxis.get_offset_text().set_fontsize(35)
plt.savefig(f"{OUTDIR}/profile_Transv_displacement_{info_experiment}.png", dpi=300)
#plt.show()

# import matplotlib.pyplot as plt
#
# plt.figure()
# ax = plot_mesh(mesh)
# fig = ax.get_figure()
# fig.savefig(f"{prefix}/mesh.png")
#
#
#
#
# # ------------------------------
#
#
# import pyvista
# from pyvista.plotting.utilities import xvfb
# from dolfinx import plot
#
# xvfb.start_xvfb(wait=0.05)
# pyvista.OFF_SCREEN = True
#
# plotter = pyvista.Plotter(title="Displacement", window_size=[1600, 600], shape=(1, 3) )
#
# v, w = q.split()
# v.name = "Airy"
# w.name = "deflection"
#
# V_v, dofs_v = Q.sub(0).collapse()
# V_w, dofs_w = Q.sub(1).collapse()
#
# # _pv_points = np.array([p[0] for p in disclinations])
# # _pv_colours = np.array(-np.array(signs))
#
# scalar_plot = plot_scalar(w, plotter, subplot=(0, 0), V_sub=V_w, dofs=dofs_w, lineproperties={'clim': [min(w.vector[:]), max(w.vector[:])]})
# #plotter.add_points(
# #        disclinations,
# #        scalars = disclination_power_list,
# #        style = 'points',
# #        render_points_as_spheres=True,
# #        point_size=15.0
# #)
#
# scalar_plot = plot_scalar(v, plotter, subplot=(0, 1), V_sub=V_v, dofs=dofs_v,
#                           lineproperties={'clim': [min(v.vector[:]), max(v.vector[:])]})
# #plotter.add_points(
# #        disclinations,
# #        scalars = disclination_power_list,
# #        style = 'points',
# #        render_points_as_spheres=True,
# #        point_size=15.0
# #)
#
# plotter.subplot(0, 2)
# cells, types, x = plot.vtk_mesh(V_v)
# grid = pyvista.UnstructuredGrid(cells, types, x)
# grid.point_data["v"] = v.x.array.real[dofs_v]
# grid.set_active_scalars("v")
#
# warped = grid.warp_by_scalar("v", scale_factor=100)
# plotter.add_mesh(warped, show_edges=False)
# #plotter.add_points(
# #        disclinations,
# #        scalars = disclination_power_list,
# #        style = 'points',
# #        render_points_as_spheres=True,
# #        point_size=15.0
# #)
#
# scalar_plot.screenshot(f"{prefix}/test_fvk.png")
# print("plotted scalar")
#
# npoints = 1001
# tol = 1e-3
# xs = np.linspace(-parameters["geometry"]["radius"] + tol, parameters["geometry"]["radius"] - tol, npoints)
# points = np.zeros((3, npoints))
# points[0] = xs
#
# fig, axes = plt.subplots(1, 2, figsize=(18, 6))
#
# _plt, data = plot_profile(w, points, None, subplot=(1, 2), lineproperties={"c": "k", "label": f"$w(x)$"}, fig=fig, subplotnumber=1)
# _plt, data = plot_profile(v, points, None, subplot=(1, 2), lineproperties={"c": "k", "label": f"$v(x)$"}, fig=fig, subplotnumber=2)
# _plt.legend()
#
# _plt.savefig(f"{prefix}/test_fvk-profiles.png")


# Plot moments
# DG_e = basix.ufl.element("DG", str(mesh.ufl_cell()), parameters["model"]["order"]-2)
# DG = dolfinx.fem.functionspace(mesh, DG_e)
#
# mxx = model.M(w)[0, 0]
# mxx_expr = dolfinx.fem.Expression(mxx, DG.element.interpolation_points())
# Mxx = dolfinx.fem.Function(DG)
# Mxx.interpolate(mxx_expr)
#
# pxx = model.P(v)[0, 0]
# pxx_expr = dolfinx.fem.Expression(pxx, DG.element.interpolation_points())
# Pxx = dolfinx.fem.Function(DG)
# Pxx.interpolate(pxx_expr)
#
# wxx = model.W(w)[0, 0]
# wxx_expr = dolfinx.fem.Expression(wxx, DG.element.interpolation_points())
# Wxx = dolfinx.fem.Function(DG)
# Wxx.interpolate(wxx_expr)
#
# plotter = pyvista.Plotter(title="Moment", window_size=[1600, 600], shape=(1, 3))
# try:
#     plotter = plot_scalar(Mxx, plotter, subplot=(0, 0))
#     plotter = plot_scalar(Pxx, plotter, subplot=(0, 1))
#     plotter = plot_scalar(Wxx, plotter, subplot=(0, 2))
#     plotter.screenshot(f"{OUTDIR}/test_tensors.png")
#     print("plotted scalar")
# except Exception as e:
#     print(e)
#
# fig, axes = plt.subplots(1, 2, figsize=(18, 6))
#
# _plt, data = plot_profile(Mxx, points, None, subplot=(1, 2), lineproperties={"c": "k",  "label": f"$Mxx(x)$"}, fig=fig, subplotnumber=1)
# _plt.legend()
# _plt, data = plot_profile(Pxx, points, None, subplot=(1, 2), lineproperties={"c": "k", "label": f"$Pxx(x)$"}, fig=fig, subplotnumber=2)
# _plt.legend()
# _plt.savefig(f"{OUTDIR}/test_fvk-Mxx-profiles.png")
