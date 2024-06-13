from mpi4py import MPI
import ufl
import dolfinx
from petsc4py import PETSc
import sys
import petsc4py

petsc4py.init(sys.argv)

from dolfinx.cpp.log import LogLevel, log
from dolfinx.fem import form
# from damage.utils import ColorPrint

from dolfinx.fem.petsc import (
    assemble_matrix, apply_lifting, create_vector, create_matrix, set_bc, assemble_vector)


# import pdb;
# pdb.set_trace()

comm = MPI.COMM_WORLD


class SNESSolver:
    """
    Problem class for nonlinear problems, compatible with PETSC.SNES solvers.
    
    Initialize the SNESSolver.

    Parameters
    ----------
    F_form : ufl.Form
        The form of the residual.
    u : dolfinx.fem.Function
        The solution function.
    bcs : list, optional
        List of boundary conditions (default is []).
    J_form : ufl.Form, optional
        The form of the Jacobian (default is None).
    bounds : tuple, optional
        Bounds for the solution as (lower, upper) (default is None).
    petsc_options : dict, optional
        Dictionary of PETSc solver options (default is {}).
    form_compiler_parameters : dict, optional
        Parameters for form compiler (default is {}).
    jit_parameters : dict, optional
        Parameters for JIT compilation (default is {}).
    monitor : function, optional
        Monitor function for the solver (default is None).
    prefix : str, optional
        Unique prefix for PETSc solver options (default is None).
    """
    
    def __init__(
        self,
        F_form: ufl.Form,
        u: dolfinx.fem.Function,
        bcs=[],
        J_form: ufl.Form = None,
        bounds=None,
        petsc_options={},
        form_compiler_parameters={},
        jit_parameters={},
        monitor=None,
        prefix=None,
        b0=PETSc.Vec | None
    ):

        self.u = u
        self.bcs = bcs
        self.bounds = bounds

        # Give PETSc solver options a unique prefix
        if prefix is None:
            prefix = "snes_{}".format(str(id(self))[0:4])

        self.prefix = prefix

        if self.bounds is not None:
            self.lb = bounds[0]
            self.ub = bounds[1]

        V = self.u.function_space
        self.comm = V.mesh.comm
        self.F_form = dolfinx.fem.form(F_form)

        if J_form is None:
            J_form = ufl.derivative(F_form, self.u, ufl.TrialFunction(V))

        self.J_form = dolfinx.fem.form(J_form)
        self.petsc_options = petsc_options

        self.b = create_vector(self.F_form)
        # if b0 is not None:
        self.b0 = b0
        self.a = create_matrix(self.J_form)

        self.monitor = monitor
        self.solver = self.solver_setup()
        # self.solver = self.solver_setup_demo()

    def set_petsc_options(self, debug=False):
        """
        Set PETSc solver options.

        Parameters
        ----------
        debug : bool, optional
            If True, print the PETSc options (default is False).
        """
        opts = PETSc.Options()
        opts.prefixPush(self.prefix)
        if debug is True:
            print(self.petsc_options)

        for k, v in self.petsc_options.items():
            opts[k] = v

        opts.prefixPop()

    def solver_setup_demo(self):
        snes = PETSc.SNES().create(self.comm)
        
        snes.setFunction(self.F, self.b)
        snes.setJacobian(self.J, self.a)

        snes.setTolerances(rtol=1.0e-9, max_it=10)
        snes.getKSP().setType("preonly")
        snes.getKSP().setTolerances(rtol=1.0e-9)
        snes.getKSP().getPC().setType("lu")
        
        if self.monitor is not None:
            snes.setMonitor(self.monitor)

        return snes
    
    def solver_setup(self):
        """
        Set up the PETSc.SNES solver.

        Returns
        -------
        PETSc.SNES
            The configured PETSc.SNES solver object.
        """
        
        snes = PETSc.SNES().create(self.comm)

        # Set options
        snes.setOptionsPrefix(self.prefix)
        self.set_petsc_options()
        snes.setFunction(self.F, self.b)
        snes.setJacobian(self.J, self.a)

        snes.setTolerances(rtol=1.0e-9, max_it=10)
        snes.getKSP().setType("preonly")
        snes.getKSP().setTolerances(rtol=1.0e-9)
        snes.getKSP().getPC().setType("lu")

        # We set the bound (Note: they are passed as reference and not as values)
        if self.monitor is not None:
            snes.setMonitor(self.monitor)

        if self.bounds is not None:
            snes.setVariableBounds(self.lb.vector, self.ub.vector)

        snes.setFromOptions()

        return snes

    def F(self, snes: PETSc.SNES, x: PETSc.Vec, b: PETSc.Vec):
        """
        Assemble the residual F into the vector b.

        Parameters
        ----------
        snes : PETSc.SNES
            The SNES solver object.
        x : PETSc.Vec
            Vector containing the latest solution.
        b : PETSc.Vec
            Vector to assemble the residual into.
        """
        
        # We need to assign the vector to the function

        x.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                      mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.vector)
        self.u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                  mode=PETSc.ScatterMode.FORWARD)
        # Zero the residual vector
        with b.localForm() as b_local:
            b_local.set(0.0)
            
        assemble_vector(b, self.F_form)
        
        if isinstance(self.b0, PETSc.Vec):
            with self.b0.localForm() as b0_local, b.localForm() as b_local:
                b_local.axpy(1, b0_local)
        
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                                  mode=PETSc.ScatterMode.FORWARD)
        # Apply boundary conditions
        apply_lifting(b, [self.J_form], bcs=[self.bcs], x0=[x], scale=-1.0)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                      mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, self.bcs, x, -1.0)

    def J(self, snes: PETSc.SNES, x: PETSc.Vec, A: PETSc.Mat, P: PETSc.Mat):
    # def J(self, snes, x: PETSc.Vec, A: PETSc.Mat, P: PETSc.Mat):
        """
        Assemble the Jacobian matrix.

        Parameters
        ----------
        snes : PETSc.SNES
            The SNES solver object.
        x : PETSc.Vec
            Vector containing the latest solution.
        A : PETSc.Mat
            Matrix to assemble the Jacobian into.
        P : PETSc.Mat
            Preconditioner matrix (not used in this context).
        """
        
        A.zeroEntries()
        assemble_matrix(A, self.J_form, self.bcs)
        A.assemble()

    def solve(self):
        """
        Solve the nonlinear problem.

        Returns
        -------
        tuple
            A tuple containing the number of iterations and the reason for convergence.

        Raises
        ------
        RuntimeError
            If the solver fails to converge.
        """
        
        log(LogLevel.INFO, f"Solving {self.prefix}")

        try:
            self.solver.solve(None, self.u.vector)
            print(
               f"{self.prefix} SNES solver converged in",
               self.solver.getIterationNumber(),
               "iterations",
               "with converged reason",
               self.solver.getConvergedReason(),
            )
            self.u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                      mode=PETSc.ScatterMode.FORWARD)
            return (self.solver.getIterationNumber(),
                    self.solver.getConvergedReason())

        except Warning:
            log(
                LogLevel.WARNING,
                f"WARNING: {self.prefix} solver failed to converge, what's next?",
            )
            raise RuntimeError(f"{self.prefix} solvers did not converge")


from ufl import TestFunction, TrialFunction, derivative, dx, grad, inner


class SNESProblem:
    def __init__(self, F, u, bc, monitor = None):
        V = u.function_space
        du = TrialFunction(V)
        self.F_form = form(F)
        self.J_form = form(derivative(F, u, du))
        self.bc = bc
        # self._F, self._J = None, None
        self.u = u
        self.monitor = monitor
        self.snes = self.setup()
    
    def setup(self):
        b = create_vector(self.F_form)
        J = create_matrix(self.J_form)

        snes = PETSc.SNES().create()
        snes.setFunction(self.F, b)
        snes.setJacobian(self.J, J)

        snes.setTolerances(rtol=1.0e-9, max_it=10)
        snes.getKSP().setType("preonly")
        snes.getKSP().setTolerances(rtol=1.0e-9)
        snes.getKSP().getPC().setType("lu")
        
        if self.monitor is not None:
            snes.setMonitor(self.monitor)
            
        return snes

    def F(self, snes: PETSc.SNES, x: PETSc.Vec, F: PETSc.Vec):
        """Assemble residual vector."""
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.vector)
        self.u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        with F.localForm() as f_local:
            f_local.set(0.0)
        assemble_vector(F, self.F_form)
        apply_lifting(F, [self.J_form], bcs=[self.bc], x0=[x], scale=-1.0)
        F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(F, self.bc, x, -1.0)

    def J(self, snes: PETSc.SNES, x: PETSc.Vec, J: PETSc.Mat, P: PETSc.Mat):
        """Assemble Jacobian matrix."""
        J.zeroEntries()
        assemble_matrix(J, self.J_form, bcs=self.bc)
        J.assemble()