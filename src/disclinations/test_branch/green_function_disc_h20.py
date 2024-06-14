import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Define symbols
x1, y1, x0, y0 = sp.symbols('x1 y1 x0 y0')
r = sp.sqrt((x1 - x0)**2 + (y1 - y0)**2)

# Define the fundamental solution
Phi = (r**2 * sp.log(r) - r**2 / 2) / (8 * sp.pi)

# Define the transformations for the method of images
def transform(x, y):
    r_squared = x**2 + y**2
    return x / r_squared, y / r_squared

# Green's function
def green_function(x1, y1, x0, y0):
    x1_prime, y1_prime = transform(x1, y1)
    x0_prime, y0_prime = transform(x0, y0)
    
    Phi_1 = Phi.subs({x1: x1, y1: y1, x0: x0, y0: y0})
    Phi_2 = Phi.subs({x1: x1_prime, y1: y1_prime, x0: x0, y0: y0})
    Phi_3 = Phi.subs({x1: x1, y1: y1, x0: x0_prime, y0: y0_prime})
    Phi_4 = Phi.subs({x1: x1_prime, y1: y1_prime, x0: x0_prime, y0: y0_prime})
    
    return Phi_1 - Phi_2 - (x0**2 + y0**2) * Phi_3 + (x1**2 + y1**2) * (x0**2 + y0**2) * Phi_4

# Example evaluation for specific points
x1_val, y1_val = 0.5, 0.5
x0_val, y0_val = 0.1, 0.1

G = green_function(x1, y1, x0, y0)
G_val = G.evalf(subs={x1: x1_val, y1: y1_val, x0: x0_val, y0: y0_val})

print(f"Green's function value at ({x1_val}, {y1_val}) for source ({x0_val}, {y0_val}): {G_val}")


import numpy as np
import matplotlib.pyplot as plt

def green_function_numeric(x1_val, y1_val, x0_val, y0_val):
    return float(green_function(x1, y1, x0, y0).evalf(subs={x1: x1_val, y1: y1_val, x0: x0_val, y0: y0_val}))

# Grid points for evaluation
x_vals = np.linspace(-1, 1, 50)
y_vals = np.linspace(-1, 1, 50)
x_grid, y_grid = np.meshgrid(x_vals, y_vals)

# Source point
x0_val, y0_val = 0.0, 0.0

# Evaluate Green's function at each grid point
G_vals = np.zeros_like(x_grid)
for i in range(len(x_vals)):
    print(f"Progress: {i / len(x_vals) * 100:.2f}%", end='\r')
    for j in range(len(y_vals)):
        if x_vals[i]**2 + y_vals[j]**2 <= 1:
            G_vals[j, i] = green_function_numeric(x_vals[i], y_vals[j], x0_val, y0_val)

# Plot
plt.figure(figsize=(8, 6))
plt.contourf(x_grid, y_grid, G_vals, levels=100, cmap='viridis')
plt.colorbar(label="Green's Function Value")
plt.scatter([x0_val], [y0_val], color='red', label='Source Point')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Green's Function for Biharmonic Equation in Unit Disc")
plt.legend()
plt.show()
plt.savefig("green_function_disc_h20.png")

# ----------

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Define symbols
x1, y1, x0, y0 = sp.symbols('x1 y1 x0 y0')
r = sp.sqrt((x1 - x0)**2 + (y1 - y0)**2)

# Define the fundamental solution
Phi = (r**2 * sp.log(r) - r**2 / 2) / (8 * sp.pi)

# Define the transformations for the method of images
def transform(x, y):
    r_squared = x**2 + y**2
    return x / r_squared, y / r_squared

# Green's function
def green_function(x1, y1, x0, y0):
    x1_prime, y1_prime = transform(x1, y1)
    x0_prime, y0_prime = transform(x0, y0)
    
    Phi_1 = Phi.subs({x1: x1, y1: y1, x0: x0, y0: y0})
    Phi_2 = Phi.subs({x1: x1_prime, y1: y1_prime, x0: x0, y0: y0})
    Phi_3 = Phi.subs({x1: x1, y1: y1, x0: x0_prime, y0: y0_prime})
    Phi_4 = Phi.subs({x1: x1_prime, y1: y1_prime, x0: x0_prime, y0: y0_prime})
    
    return Phi_1 - Phi_2 - (x0**2 + y0**2) * Phi_3 + (x1**2 + y1**2) * (x0**2 + y0**2) * Phi_4

# Example evaluation for specific points
x1_val, y1_val = 0.0, 0.5
x0_val, y0_val = 0.0, 0.0

G = green_function(x1, y1, x0, y0)
G_val = G.evalf(subs={x1: x1_val, y1: y1_val, x0: x0_val, y0: y0_val})

print(f"Green's function value at ({x1_val}, {y1_val}) for source ({x0_val}, {y0_val}): {G_val}")

# Evaluate Green's function along y-axis
def green_function_numeric(x1_val, y1_val, x0_val, y0_val):
    return float(green_function(x1, y1, x0, y0).evalf(subs={x1: x1_val, y1: y1_val, x0: x0_val, y0: y0_val}))


y_vals = np.linspace(-1, 1, 100)
G_vals = np.array([green_function_numeric(0, y, x0_val, y0_val) for y in y_vals])

# Plot the Green's function along the y-axis
plt.figure(figsize=(8, 6))
plt.plot(y_vals, G_vals, label="Green's Function along y-axis", color='blue')
plt.scatter([y0_val], [green_function_numeric(0, y0_val, x0_val, y0_val)], color='red', label='Source Point')
plt.xlabel('y')
plt.ylabel("Green's Function Value")
plt.title("Green's Function for Biharmonic Equation in Unit Disc (along y-axis)")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("green_function_along_y_axis.png")