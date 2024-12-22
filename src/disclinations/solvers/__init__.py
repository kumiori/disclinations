from mpi4py import MPI
import ufl
import dolfinx
from petsc4py import PETSc
import sys
import petsc4py

petsc4py.init(sys.argv)
import logging
from dolfinx.cpp.log import LogLevel, log
from dolfinx.fem import form
# from damage.utils import ColorPrint

from dolfinx.fem.petsc import (
    assemble_matrix, apply_lifting, create_vector, create_matrix, set_bc, assemble_vector)

# CFe starting
import pdb
from petsc4py import PETSc
from slepc4py import SLEPc

import numpy as np

# def set_initial_guess(snes, initial_vector=None, random_seed=None):
#     """
#     Set the initial guess for SNES solver.
#
#     Parameters:
#     -----------
#     snes : PETSc.SNES
#         The SNES solver object
#     initial_vector : array-like, optional
#         Initial vector to set. If None, uses random values
#     random_seed : int, optional
#         Seed for random number generator if using random initial guess
#
#     Returns:
#     --------
#     x : PETSc.Vec
#         The initial guess vector
#     """
#     # Get the solution vector
#     x = snes.getSolution()
#     if initial_vector is not None:
#         # Convert initial_vector to numpy array if it isn't already
#         initial_array = np.array(initial_vector, dtype=float)
#
#         # Check if dimensions match
#         pdb.set_trace()
#         if x.getSize() != len(initial_array):
#             raise ValueError(f"Initial vector size {len(initial_array)} does not match SNES vector size {x.getSize()}")
#
#         # Set the values
#         x.setArray(initial_array)
#
#     else:
#         # Set random initial guess if no vector provided
#         if random_seed is not None:
#             np.random.seed(random_seed)
#
#         # Generate random values between -1 and 1
#         random_array = 2 * np.random.random(x.getSize()) - 1
#         x.setArray(random_array)
#
#     # Ensure the vector is assembled
#     x.assemble()
#
#     return x


# Count zero entries
def counts_zero_entries(A):
    nz_count = 0
    row_start, row_end = A.getOwnershipRange()  # Get local row range
    for row in range(row_start, row_end):
        cols, vals = A.getRow(row)  # Access columns and values for the row
        nz_count += sum(1 for v in vals if v != 0)  # Count zero entries in this row
    print(f"Number of non-zero entries: {nz_count}")
    return nz_count


def calculate_condition_number_1(A):
    """
    Purpose: computing the conditioning number of the matrix A using eigenvalues
    """

    eps = SLEPc.EPS().create() # Create eigenvalue solver

    # Set up the eigenvalue problem
    eps.setOperators(A)
    eps.setProblemType(SLEPc.EPS.ProblemType.NHEP)
    eps.setDimensions(A.size[0])  # Request two eigenvalues

    # Set some solver options for better convergence
    eps.setTolerances(tol=1e-6, max_it=1000)
    eps.setKrylovSchurRestart(0.7)  # Restart with 70% of the vectors

    # Solve for largest and smallest eigenvalues in one go
    eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_MAGNITUDE)
    eps.solve()

    # Check convergence
    nconv = eps.getConverged()
    if nconv < 2:
        print(f"Warning: Only {nconv} eigenvalues converged. Condition number estimate may be inaccurate.")
        return None

    # Get the largest and smallest eigenvalues
    largest = abs(eps.getEigenpair(0))
    smallest = abs(eps.getEigenpair(nconv-1))

    return largest / smallest # Calculate and return the condition number

def calculate_condition_number_2(A):
    """
    Purpose: computing the conditioning number of the matrix A using numpy and dense array
    """
    import numpy as np
    dense_A = A.convert("dense")
    dense_array = dense_A.getDenseArray()
    c = np.linalg.cond(dense_array) # compute the condition number
    return c


comm = MPI.COMM_WORLD
DEBUG = logging.getLogger().getEffectiveLevel() == logging.DEBUG

def monitor(snes, it, norm):
    logging.info(f"Iteration {it}, residual {norm}")
    #print(f"Iteration {it}, residual {norm}")

    # Get the step vector Δx (update to the solution)
    delta_x = snes.getSolutionUpdate()

    # Compute the norm of Δx to see the magnitude of the change
    delta_x_norm = delta_x.norm(PETSc.NormType.NORM_2)

    # Print the current iteration, function norm, and Δx norm
    print(f"SNES Iteration {it}: Residual norm = {norm:.6e}, Δx norm = {delta_x_norm:.6e}")

    return PETSc.SNES.ConvergedReason.ITERATING

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
        monitor=monitor,
        prefix=None,
        b0=PETSc.Vec | None
    ):

        self.u = u
        self.bcs = bcs
        self.bounds = bounds

        # Give PETSc solver options a unique prefix
        if prefix is None:
            prefix = "snes_{}".format(str(id(self))[0:4])

        self.prefix = prefix + "_"

        # if self.bounds is not None:
        #     self.lb = bounds[0]
        #     self.ub = bounds[1]
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

    def create_snes_solution(self):
            x = self.u.vector.copy()
            with x.localForm() as _x, self.u.vector.localForm() as _solution:
                _x[:] = _solution
            return x

    def set_petsc_options(self):
        """
        Set PETSc solver options.

        Parameters
        ----------
        debug : bool, optional
            If True, print the PETSc options (default is False).
        """
        opts = PETSc.Options()
        opts.prefixPush(self.prefix)
        
        if DEBUG:
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
        snes.setFunction(self.F, self.b)
        snes.setJacobian(self.J, self.a)

        #snes.setTolerances(rtol=1.0e-9, max_it=10)
        snes.setTolerances(rtol=1.0e-16, atol=1e-16, stol=1e-16, max_it=100)
        ksp = snes.getKSP()
        ksp.setType("preonly")
        #ksp.setType(PETSc.KSP.Type.GMRES)


        #ksp.setTolerances(rtol=1.0e-9)
        ksp.setTolerances(rtol=1.0e-16, atol=1e-16)
        #snes.setTolerances(rtol=1e-5, atol=1e-10, stol=1e-8, max_it=20)
        ksp.getPC().setType("lu")

        self.set_petsc_options()
        opts = PETSc.Options()

        if self.monitor is not None:
            snes.setMonitor(self.monitor)

        # if self.bounds is not None:
        #     snes.setVariableBounds(self.lb.vector, self.ub.vector)

        snes.setFromOptions()

        if DEBUG: snes.view()

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
        #pdb.set_trace()
        #counts_zero_entries(A)

        # CFe: added Apply diagonal scaling
        # D = P.getDiagonal()
        # D.reciprocal()
        # P.diagonalScale(L=D, R=D)
        # return PETSc.Mat.Structure.SAME_NONZERO_PATTERN
        # # CFe: end

        # # CFe: check conditioning
        #pdb.set_trace()
        # cond_number = calculate_condition_number_2(A)
        # print("Condition number of the Jacobian of the Residual :", cond_number)
        # if cond_number > 1e10: print("***********************************************************************************")
        # # CFe: end check


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

        with dolfinx.common.Timer(f"~First Order: min-max equilibrium") as timer:
            try:
                #print("Initial guess: ", self.u.vector.array)
                #self.u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                #self.solver.setSolution(self.u.vector)

                self.solver.solve(None, self.u.vector)
                print(
                f"{self.prefix} SNES solver converged in",
                self.solver.getIterationNumber(),
                "iterations",
                "with converged reason",
                self.solver.getConvergedReason(), #"Initial guess", self.solver.getInitialGuess()
                )
                self.u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

                residual_vector = self.solver.getFunction()[0]
                jacobian_matrix_1 = self.solver.getJacobian()[0]
                jacobian_matrix_2 = self.solver.getJacobian()[1]
                #pdb.set_trace()
                jacobian_norm_1 = jacobian_matrix_1.norm(PETSc.NormType.FROBENIUS)
                jacobian_norm_2 = jacobian_matrix_2.norm(PETSc.NormType.FROBENIUS)
                print(f"Final residual 2 norm: {residual_vector.norm(PETSc.NormType.NORM_2)}")
                #print(f"Jacobian matrix: {jacobian_matrix_}")
                print(f"Jacobian Frobenius_1 norm: {jacobian_norm_1}")
                print(f"Jacobian Frobenius_2 norm: {jacobian_norm_2}")
                # cond_number = calculate_condition_number_2(jacobian_matrix_1)
                #print("Condition number of the Jacobian of the Residual :", cond_number)

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
