import numpy as np
import os
from pathlib import Path

import ufl
from ufl import dx
import basix
import dolfinx
from dolfinx.fem import Constant, dirichletbc
from petsc4py import PETSc
import matplotlib.pyplot as plt

from disclinations.models import NonlinearPlateFVK
from disclinations.solvers import SNESSolver
from disclinations.utils.viz import plot_profile
from disclinations.utils.la import compute_disclination_loads, compute_cell_contributions

AIRY            = 0
TRANSVERSE_DISP = 1

DEFAULT_SOLVER_PARAMETERS = {
    "snes_type": "newtonls",          # Solver type: NGMRES (Nonlinear GMRES)
    "snes_max_it": 200,               # Maximum number of iterations
    "snes_rtol": 1e-15,               # Relative tolerance for convergence
    "snes_atol": 1e-15,               # Absolute tolerance for convergence
    "snes_stol": 3e-10,               # Tolerance for the change in solution norm
    "snes_monitor": None,             # Function for monitoring convergence (optional)
    "snes_linesearch_type": "basic",  # Type of line search
}

# class mesh():
#     def __init(mesh)__:

class Fvk_plate():
    bc                     = None
    load                   = None
    mesh                   = None
    tranverse_disp         = None
    airy                   = None
    bcs_tranverse_disp     = None
    bcs_airy               = None
    bcs_list               = [None, None]
    state_function         = None
    state_dic              = None
    airy_pp                = None
    tranverse_disp_pp      = None
    inplane_displacement   = [None, None]
    model                  = None
    penalisation           = None
    ext_work               = None
    cauchy_stress          = [None, None, None, None]
    output_directory       = ""
    fe                     = None # CFe: finite element
    fs                     = None # CFe: funciton space
    load_set_bool          = False
    disclinations_set_bool = False
    disclinations_list     = []

    def __init__(self, mesh, material_properties):
        self.mesh = mesh
        self.set_finite_element()
        self.set_function_space()
        self.set_state()
        #material_properties = parameters["model"]
        self.set_model(material_properties)
        self.load = Constant(self.mesh, np.array(0.0, dtype=PETSc.ScalarType))

    def get_out_dir(self): return self.output_directory
    def get_mesh(self)   : return self.mesh
    def get_load(self)   : return self.load
    def get_bc(self)     : return self.bc
    def get_airy(self)   : return self.airy_pp
    def get_tranverse_disp(self) : return self.tranverse_disp_pp

    def get_bending_energy_obj() : return self.bending_energy
    def get_membrane_energy_obj(): return self.membrane_energy
    def get_coupling_energy_obj(): return self.coupling_energy

    def get_bending_energy_value(self)     : return dolfinx.fem.assemble_scalar(dolfinx.fem.form(self.bending_energy))
    def get_membrane_energy_value(self)    : return dolfinx.fem.assemble_scalar(dolfinx.fem.form(self.membrane_energy))
    def get_coupling_energy_value(self)    : return dolfinx.fem.assemble_scalar(dolfinx.fem.form(self.coupling_energy))
    def get_elastic_energy_value(self)     : return dolfinx.fem.assemble_scalar(dolfinx.fem.form(self.elastic_energy))
    def get_external_work_value(self)      : return dolfinx.fem.assemble_scalar(dolfinx.fem.form(self.ext_work))
    def get_penalization_energy_value(self): return dolfinx.fem.assemble_scalar(dolfinx.fem.form(self.penalisation))



    def set_out_dir(self, outdir = "output", subfolder = "fvk_plate"):
        self.output_directory = os.path.join(outdir, subfolder)
        Path(self.output_directory).mkdir(parents=True, exist_ok=True)


    #polinomia_order = parameters["model"]["order"]
    def set_finite_element(self, element = "P", polinomia_order = 3):
        self.fe = basix.ufl.element(element, str(self.mesh.ufl_cell()), polinomia_order)

    def set_function_space(self):
        self.fs = dolfinx.fem.functionspace(self.mesh, basix.ufl.mixed_element([self.fe, self.fe]))

    def set_state(self):
        #assert hasattr(self, 'fs')
        self.state_function = dolfinx.fem.Function(self.fs)
        self.airy, self.tranverse_disp = ufl.split(self.state_function)
        self.state_dic = {"v": self.airy, "w": self.tranverse_disp}

    def set_model(self, material_properties):
        self.model = NonlinearPlateFVK(self.mesh, material_properties)

    def set_dirichlet_bc(self, value, index):
        self.mesh.topology.create_connectivity(self.mesh.topology.dim - 1, self.mesh.topology.dim)
        bndry_facets = dolfinx.mesh.exterior_facet_indices(self.mesh.topology)
        dofs_f = dolfinx.fem.locate_dofs_topological(V=self.fs.sub(index), entity_dim=1, entities=bndry_facets)
        bcs_f = dirichletbc(np.array(value, dtype=PETSc.ScalarType), dofs_f, self.fs.sub(index))
        return bcs_f

    def set_dirichlet_bc_airy(self, value):
        self.bcs_airy = self.set_dirichlet_bc(value, AIRY)
        self.bcs_list[AIRY] = self.bcs_airy

    def set_dirichlet_bc_trans_displ(self, value):
        self.bcs_tranverse_disp = self.set_dirichlet_bc(value, TRANSVERSE_DISP)
        self.bcs_list[TRANSVERSE_DISP] = self.bcs_tranverse_disp

    def set_load(self, load):
        self.load = load
        self.load_set_bool = True

    def set_disclination(self, coordinates_list, frank_angle_list):
        self.disclinations_set_bool = True
        if self.mesh.comm.rank == 0:
            self.frank_angle_list = frank_angle_list
            for coordinate in coordinates_list:
                self.disclinations_list.append(np.array([coordinate], dtype=self.mesh.geometry.x.dtype))
        else:
            for coordinate in coordinates_list:
                self.disclinations_list.append(np.zeros((0, 3), dtype=self.mesh.geometry.x.dtype))

    def set_bending_energy(self): self.bending_energy = self.model.energy(self.state_dic)[1]

    def set_membrane_energy(self): self.membrane_energy = self.model.energy(self.state_dic)[2]

    def set_coupling_energy(self): self.coupling_energy = self.model.energy(self.state_dic)[3]

    def set_elastic_energy(self):
        self.set_bending_energy()
        self.set_membrane_energy()
        self.set_coupling_energy()
        self.elastic_energy = self.bending_energy - self.membrane_energy + self.coupling_energy

    def set_penalization_energy(self):
        self.penalisation = self.model.penalisation(self.state_dic)

    def compute(self, solver_parameters = DEFAULT_SOLVER_PARAMETERS):

        self.set_elastic_energy()
        self.set_penalization_energy()

        self.ext_work = self.load * self.tranverse_disp * dx

        L = self.elastic_energy - self.ext_work + self.penalisation
        F = ufl.derivative(L, self.state_function, ufl.TestFunction(self.fs))

        # CFe: compute disclination contribution
        b = dolfinx.fem.Function(self.fs)
        b.x.array[:] = 0.0

        if self.disclinations_set_bool:
            fs_airy, dofs_airy = self.fs.sub(AIRY).collapse()
            _cells, _basis_values = compute_cell_contributions(fs_airy, self.disclinations_list)

            for cell, basis_value, frank_angle in zip(_cells, _basis_values, self.frank_angle_list):
                subspace_dofs = fs_airy.dofmap.cell_dofs(cell)
                dofs = np.array(dofs_airy)[subspace_dofs]
                b.x.array[dofs] += frank_angle * basis_value

        # CFe: Solver's instance
        solver = SNESSolver(
            F_form=F,
            u=self.state_function,
            bcs=self.bcs_list,
            bounds=None,
            # petsc_options=parameters.get("solvers").get("elasticity").get("snes"),
            petsc_options=solver_parameters,
            prefix='plate_fvk',
            b0 = b.vector
        )
        solver.solve()

        self.airy_pp, self.tranverse_disp_pp = self.state_function.split()
        self.airy_pp.name = "Airy"
        self.tranverse_disp_pp.name = "Transverse Deflection"


    def plot_profile(self, function, radius, title, compare_function = None):
        tol = 1e-3
        xs = np.linspace(-radius + tol, radius - tol, 202)
        points = np.zeros((3, 202))
        points[0] = xs
        #fig = plt.plot(figsize=(18, 6))
        _plt, data = plot_profile(function, points, None, None, lineproperties={ "c": "k", "label": f"${title}$"})
        _plt.savefig(self.output_directory+f"/{title}_fvk-profile.png")

    def plot_profiles(self, radius):
        v, w = self.state_function.split()
        self.plot_profile(v, radius, "Airy")
        self.plot_profile(w, radius, "Transverse displacement")




