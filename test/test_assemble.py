import dolfinx
import ufl
from mpi4py import MPI
from dolfinx.mesh import create_unit_square
from dolfinx.fem import functionspace, form, assemble_scalar
from ufl import div, grad, dx
import basix

# Step 1: Create a mesh (unit square domain)
mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)

# Step 2: Define a function space (e.g., continuous Galerkin elements of degree 1)
V = functionspace(mesh, ("CG", 1))

# Step 3: Define a test function 'v' in the function space
v = dolfinx.fem.Function(V)

# We are assembling the integral of ... * dx
form_expr = ufl.inner(ufl.grad(v), ufl.grad(v)) * dx

# form_expr = v*v * dx

a = form(form_expr)

result = assemble_scalar(a)

print(f"Result of the assembly: {result}")

from dolfinx.fem import Function, FunctionSpace, form, assemble_scalar


# Step 1: Create a mesh (unit square domain)
mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)
# Step 2: Define a scalar finite element (P1 element)
element_v = basix.ufl.element("P", str(mesh.ufl_cell()), degree=1)

# Step 3: Create a function space using the scalar element
V_v = functionspace(mesh, element_v)

# Step 4: Define a function 'v' in the function space
v = Function(V_v, name="Airy")

# Step 5: Define the form to be assembled (div(grad(v)) is the Laplacian of v for scalar fields)
# form_expr = div(grad(v)) * dx
form_expr = ufl.inner(ufl.grad(v), ufl.grad(v)) * dx
# form_expr = ufl.inner(ufl.div(ufl.grad(v)), ufl.div(ufl.grad(v))) * dx

# Step 6: Convert the expression into a form
a = form(form_expr)

# Step 7: Assemble the scalar value
result = assemble_scalar(a)

# Step 8: Print the assembled scalar value
print(f"Result of the assembly: {result}")


element_z = basix.ufl.element("P", str(mesh.ufl_cell()), degree=2)
V_z = functionspace(mesh, element_z)
z = Function(V_z)
form_expr = ufl.inner(div(grad(z)), div(grad(z))) * dx
a = form(form_expr)
result = assemble_scalar(a)
print(f"Result of the assembly: {result}")
z.interpolate(v)
