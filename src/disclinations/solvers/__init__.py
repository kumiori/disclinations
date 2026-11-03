"""PETSc SNES assembly for scalar and mixed DOLFINx 0.9 residuals.

Point-load b0 is an already accumulated dual vector added to the residual:
F_h(u) + b0 = 0. Its sign is part of the calling model's convention.
"""
import logging
from uuid import uuid4

import dolfinx
from dolfinx.fem import form
from dolfinx.fem.petsc import (
    assemble_matrix,
    apply_lifting,
    create_vector,
    create_matrix,
    set_bc,
    assemble_vector,
)
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
import ufl


def counts_zero_entries(A):
    """Return the global count of numerically nonzero owned matrix entries."""
    lo, hi = A.getOwnershipRange()
    count = sum(np.count_nonzero(A.getRow(row)[1]) for row in range(lo, hi))
    return A.comm.tompi4py().allreduce(int(count), op=MPI.SUM)


def calculate_condition_number_1(A):
    """Return the singular-value 2-norm condition number using SLEPc.

    Unlike an eigenvalue ratio this also applies to nonnormal matrices.
    Raises when either extremal singular-value solve fails to converge.
    """
    from slepc4py import SLEPc

    values = []
    for which in (SLEPc.SVD.Which.LARGEST, SLEPc.SVD.Which.SMALLEST):
        svd = SLEPc.SVD().create(A.comm)
        try:
            svd.setOperator(A)
            svd.setDimensions(1)
            svd.setWhichSingularTriplets(which)
            svd.setTolerances(tol=1e-10, max_it=1000)
            svd.solve()
            if svd.getConverged() < 1:
                raise RuntimeError("Extremal singular value did not converge")
            values.append(svd.getValue(0))
        finally:
            svd.destroy()
    return values[0] / values[1] if values[1] > 0 else float("inf")


def calculate_condition_number_2(A):
    """Exact dense 2-norm condition number, serial small matrices only."""
    if A.comm.size != 1:
        raise ValueError("Dense condition number is only available in serial")
    dense = A.convert("dense")
    try:
        return np.linalg.cond(dense.getDenseArray())
    finally:
        dense.destroy()


def monitor(snes, iteration, norm):
    logging.info("SNES iteration %s, residual %g", iteration, norm)


class SNESSolver:
    """Solve F(u;v)+b0(v)=0 with Dirichlet lifting and checked convergence.

    F_form has rank one, J_form rank two. b0 is an already assembled PETSc
    vector, including reverse scatter; only owned entries are added. Bounds
    require a PETSc variational-inequality SNES type selected by the caller.
    Options are scoped by a collective unique prefix, then removed from the
    global database once consumed. The underlying solver remains inspectable.
    """

    def __init__(
        self,
        F_form,
        u,
        bcs=None,
        J_form=None,
        bounds=None,
        petsc_options=None,
        form_compiler_parameters=None,
        jit_parameters=None,
        monitor=monitor,
        prefix=None,
        b0=None,
    ):
        self.u, self.bcs, self.bounds = u, list(bcs or []), bounds
        self.comm = u.function_space.mesh.comm
        if len(F_form.arguments()) != 1:
            raise ValueError("F_form must have rank one")
        if J_form is None:
            J_form = ufl.derivative(F_form, u, ufl.TrialFunction(u.function_space))
        if len(J_form.arguments()) != 2:
            raise ValueError("J_form must have rank two")
        token = self.comm.bcast(uuid4().hex if self.comm.rank == 0 else None, root=0)
        self.prefix = (prefix or "snes_" + token).rstrip("_") + "_"
        kwargs = {
            "form_compiler_options": form_compiler_parameters or {},
            "jit_options": jit_parameters or {},
        }
        self.F_form, self.J_form = form(F_form, **kwargs), form(J_form, **kwargs)
        if b0 is not None and not isinstance(b0, PETSc.Vec):
            raise TypeError("b0 must be an assembled PETSc.Vec or None")
        self.b0 = b0
        self.b, self.a = create_vector(self.F_form), create_matrix(self.J_form)
        self.petsc_options = dict(petsc_options or {})
        self.monitor = monitor
        self.solver = self.solver_setup()

    def create_snes_solution(self):
        return self.u.x.petsc_vec.copy()

    def solver_setup(self):
        snes = PETSc.SNES().create(self.comm)
        snes.setOptionsPrefix(self.prefix)
        snes.setFunction(self.F, self.b)
        snes.setJacobian(self.J, self.a)
        snes.getKSP().setType("preonly")
        snes.getKSP().getPC().setType("lu")
        if self.monitor is not None:
            snes.setMonitor(self.monitor)
        opts = PETSc.Options()
        previous = opts.getAll()
        try:
            for key, value in self.petsc_options.items():
                opts[self.prefix + key] = value
            snes.setFromOptions()
        finally:
            for key in self.petsc_options:
                name = self.prefix + key
                if name in previous:
                    opts[name] = previous[name]
                else:
                    del opts[name]
        if self.bounds is not None:
            snes.setVariableBounds(*(f.x.petsc_vec for f in self.bounds))
        return snes

    def _update_state(self, x):
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.x.petsc_vec)
        self.u.x.scatter_forward()

    def F(self, snes, x, b):
        self._update_state(x)
        with b.localForm() as local:
            local.set(0)
        assemble_vector(b, self.F_form)
        apply_lifting(b, [self.J_form], bcs=[self.bcs], x0=[x], alpha=-1.0)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        if self.b0 is not None:
            b.axpy(1.0, self.b0)  # owned entries; do not reverse-scatter b0 twice
        set_bc(b, self.bcs, x, alpha=-1.0)

    def J(self, snes, x, A, P):
        self._update_state(x)  # Jacobian may be requested without a residual call
        for matrix in [A] if A.handle == P.handle else [A, P]:
            matrix.zeroEntries()
            assemble_matrix(matrix, self.J_form, bcs=self.bcs)
            matrix.assemble()

    def solve(self):
        """Return (iterations, positive reason); raise on failure/nonfinite state."""
        self.solver.solve(None, self.u.x.petsc_vec)
        self.u.x.scatter_forward()
        reason = self.solver.getConvergedReason()
        finite = self.comm.allreduce(
            bool(np.all(np.isfinite(self.u.x.array))), op=MPI.LAND
        )
        if reason <= 0 or not finite or not np.isfinite(self.solver.getFunctionNorm()):
            raise RuntimeError(
                f"{self.prefix} failed to converge: SNES reason {reason}"
            )
        return self.solver.getIterationNumber(), reason


class SNESProblem(SNESSolver):
    """Compatibility interface exposing the configured solver as .snes."""

    def __init__(self, F, u, bc, monitor=None):
        super().__init__(F, u, bcs=bc, monitor=monitor)
        self.bc, self.snes = self.bcs, self.solver
