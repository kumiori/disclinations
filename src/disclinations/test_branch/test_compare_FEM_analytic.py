"""
PURPOSE OF THE SCRIPT
Compare the three FE formulations (Variational, Brenner, Carstensen) with a known solution.
The script can use both the dimensional or the NON-dimensional FE formulations (see respectivelly "models/__init__.py" and "models/adimensional.py")

ARTICLE RELATED SECTION
"Tests 1: kinematically compatible plate under volume forces"
"Tests 2: kinematically incompatible plate"
"""

from disclinations.utils.viz import plot_scalar, plot_profile, plot_mesh
import yaml
import numpy as np
import matplotlib.pyplot as plt
import pdb
import os
import pandas as pd
import argparse

# parser = argparse.ArgumentParser(description="Which comparison would you like to run?")
# parser.add_argument("test type", help="Either disclination or transverse")
# parser.add_argument("dimensional / adimensional model", help="Either disclination or transverse")

SCRIPT_VAR = "test_fvk_vs_analytics_transverse_Adim"
SCRIPT_BRN = "test_fvk_vs_analytics_transverse-brenner_Adim"
SCRIPT_CAR = "test_fvk_vs_analytics_transverse-carstensen_Adim"

#SCRIPT_VAR = "test_fvk_disclinations_dipole_Adim"
#SCRIPT_BRN = "test_fvk_disclinations_dipole-brenner_Adim"
#SCRIPT_CAR = "test_fvk_disclinations_dipole-carstensen_Adim"

#SCRIPT_VAR = "test_fvk_vs_analytics_transverse"
#SCRIPT_BRN = "test_fvk_vs_analytics_transverse-brenner"
#SCRIPT_CAR = "test_fvk_vs_analytics_transverse-carstensen"

#SCRIPT_VAR = "test_fvk_disclinations_dipole"
#SCRIPT_BRN = "test_fvk_disclinations_dipole-brenner"
#SCRIPT_CAR = "test_fvk_disclinations_dipole-carstensen"

OUTDIR = os.path.join("output", "compare", SCRIPT_VAR)
if not os.path.exists(OUTDIR): os.makedirs(OUTDIR)

# Define a function to run a script and return a variable
def run_script(filename):
    globals_dict = {}  # Dictionary to capture script's variables
    with open(filename) as file:
        exec(file.read(), globals_dict)
    return globals_dict

#pdb.set_trace()

script_var = run_script(SCRIPT_VAR+".py")
script_brn = run_script(SCRIPT_BRN+".py")
script_car = run_script(SCRIPT_CAR+".py")

V_v = script_var["V_v"]
V_w = script_var["V_w"]
dofs_v = script_var["dofs_v"]
dofs_w = script_var["dofs_w"]

v_exact = script_var["v_exact"]
v_var = script_var["vpp"]
v_brn = script_brn["vpp"]
v_car = script_car["vpp"]

w_exact = script_var["w_exact"]
w_var = script_var["wpp"]
w_brn = script_brn["wpp"]
w_car = script_car["wpp"]

# Compare energies
columns = ["Error % Bending Energy (Var)", "Error % Bending Energy (Brn)", "Error % Bending Energy (Car)",
           "Error % Membrane Energy (Var)", "Error % Membrane Energy (Brn)", "Error % Membrane Energy (Car)",
           "Error % Coupling Energy (Var)", "Error % Coupling Energy (Brn)", "Error % Coupling Energy (Car)",
           "Bending Energy (Exact)", "Bending Energy (Var)", "Bending Energy (Brn)", "Bending Energy (Car)",
           "Membrane Energy (Exact)", "Membrane Energy (Var)", "Membrane Energy (Brn)", "Membrane Energy (Car)",
           "Coupling Energy (Exact)", "Coupling Energy (Var)", "Coupling Energy (Brn)", "Coupling Energy (Car)",
           "Mesh size", "Interior Penalty (IP)", "Thickness", "Young modulus", "Poisson ratio"]

#experimental_data = pd.DataFrame(columns=columns)
e_data = {}
#pdb.set_trace()
e_data["Bending Energy (Var)"] = script_var["energy_terms"]["bending"]
e_data["Bending Energy (Brn)"] = script_brn["energy_terms"]["bending"]
e_data["Bending Energy (Car)"] = script_car["energy_terms"]["bending"]

e_data["Membrane Energy (Var)"] = script_var["energy_terms"]["membrane"]
e_data["Membrane Energy (Brn)"] = script_brn["energy_terms"]["membrane"]
e_data["Membrane Energy (Car)"] = script_car["energy_terms"]["membrane"]

e_data["Coupling Energy (Var)"] = script_var["energy_terms"]["coupling"]
e_data["Coupling Energy (Brn)"] = script_brn["energy_terms"]["coupling"]
e_data["Coupling Energy (Car)"] = script_car["energy_terms"]["coupling"]

e_data["Bending Energy (Exact)"] = script_var["ex_bending_energy"]
e_data["Membrane Energy (Exact)"] = script_var["ex_membrane_energy"]
e_data["Coupling Energy (Exact)"] = script_var["ex_coupl_energy"]

errorBending_var = "N.A."
errorBending_brn = "N.A."
errorBending_car = "N.A."
errorMembrane_var = "N.A."
errorMembrane_brn = "N.A."
errorMembrane_car = "N.A."
errorCoupling_var = "N.A."
errorCoupling_brn = "N.A."
errorCoupling_car = "N.A."

if e_data["Bending Energy (Exact)"] != 0.0:
    errorBending_var = 100*( e_data["Bending Energy (Var)"] - e_data["Bending Energy (Exact)"] ) / e_data["Bending Energy (Exact)"]
    errorBending_brn = 100*( e_data["Bending Energy (Brn)"] - e_data["Bending Energy (Exact)"] ) / e_data["Bending Energy (Exact)"]
    errorBending_car = 100*( e_data["Bending Energy (Car)"] - e_data["Bending Energy (Exact)"] ) / e_data["Bending Energy (Exact)"]

if e_data["Membrane Energy (Exact)"] != 0.0:
    errorMembrane_var = 100*( e_data["Membrane Energy (Var)"] - e_data["Membrane Energy (Exact)"] ) / e_data["Membrane Energy (Exact)"]
    errorMembrane_brn = 100*( e_data["Membrane Energy (Brn)"] - e_data["Membrane Energy (Exact)"] ) / e_data["Membrane Energy (Exact)"]
    errorMembrane_car = 100*( e_data["Membrane Energy (Car)"] - e_data["Membrane Energy (Exact)"] ) / e_data["Membrane Energy (Exact)"]

if e_data["Coupling Energy (Exact)"] != 0.0:
    errorCoupling_var = 100*( e_data["Coupling Energy (Var)"] - e_data["Coupling Energy (Exact)"] ) / e_data["Coupling Energy (Exact)"]
    errorCoupling_brn = 100*( e_data["Coupling Energy (Brn)"] - e_data["Coupling Energy (Exact)"] ) / e_data["Coupling Energy (Exact)"]
    errorCoupling_car = 100*( e_data["Coupling Energy (Car)"] - e_data["Coupling Energy (Exact)"] ) / e_data["Coupling Energy (Exact)"]

print("RESULTS")
print("----------------------------------------------")
print("Percent errorBending_var: ", errorBending_var)
print("Percent errorBending_brn: ", errorBending_brn)
print("Percent errorBending_car: ", errorBending_car)
print("----------------------------------------------")
print("Percent errorMembrane_var: ", errorMembrane_var)
print("Percent errorMembrane_brn: ", errorMembrane_brn)
print("Percent errorMembrane_car: ", errorMembrane_car)
print("----------------------------------------------")
print("Percent errorCoupling_var: ", errorCoupling_var)
print("Percent errorCoupling_brn: ", errorCoupling_brn)
print("Percent errorCoupling_car: ", errorCoupling_car)
print("----------------------------------------------")

#pdb.set_trace()
exp_dict = {
    "Bending Energy (Var)": [e_data["Bending Energy (Var)"]],
    "Bending Energy (Brn)":  [e_data["Bending Energy (Brn)"]],
    "Bending Energy (Car)": [e_data["Bending Energy (Car)"]],
    "Membrane Energy (Var)": [e_data["Membrane Energy (Var)"]],
    "Membrane Energy (Brn)": [e_data["Membrane Energy (Brn)"]],
    "Membrane Energy (Car)": [e_data["Membrane Energy (Car)"]],
    "Coupling Energy (Var)": [e_data["Coupling Energy (Var)"]],
    "Coupling Energy (Brn)": [e_data["Coupling Energy (Brn)"]],
    "Coupling Energy (Car)": [e_data["Coupling Energy (Car)"]],
    "Bending Energy (Exact)": [e_data["Bending Energy (Exact)"]],
    "Membrane Energy (Exact)": [e_data["Membrane Energy (Exact)"]],
    "Coupling Energy (Exact)": [e_data["Coupling Energy (Exact)"]],
    "Mesh size": [script_var["mesh_size"]],
    "Interior Penalty (IP)": [script_var["parameters"]["model"]["alpha_penalty"]],
    "Thickness": [script_var["thickness"]], "Young modulus": [script_var["_E"]],
    "Poisson ratio": [script_var["nu"]],
    "Error % Bending Energy (Var)": [errorBending_var],
    "Error % Bending Energy (Brn)": [errorBending_brn],
    "Error % Bending Energy (Car)": [errorBending_car],
    "Error % Membrane Energy (Var)": [errorMembrane_var],
    "Error % Membrane Energy (Brn)": [errorMembrane_brn],
    "Error % Membrane Energy (Car)": [errorMembrane_car],
    "Error % Coupling Energy (Var)": [errorCoupling_var],
    "Error % Coupling Energy (Brn)": [errorCoupling_brn],
    "Error % Coupling Energy (Car)": [errorCoupling_car]
    }

experimental_data = pd.DataFrame(exp_dict)
experimental_data.to_excel(f'{OUTDIR}/Models_comparison.xlsx', index=False)

# Plot profiles:
import pyvista

pyvista.OFF_SCREEN = True

plotter = pyvista.Plotter(title="Displacement", window_size=[1200, 600], shape=(2, 2))


with open("parameters.yml") as f: parameters = yaml.load(f, Loader=yaml.FullLoader)
tol = 1e-3
xs = np.linspace(-parameters["geometry"]["radius"] + tol, parameters["geometry"]["radius"] - tol, 101)
points = np.zeros((3, 101))
points[0] = xs

fig, axes = plt.subplots(1, 1, figsize=(24, 18))

_plt, data = plot_profile(w_var, points, None, subplot=(1, 1), lineproperties={"c": "b", "lw":5, "label": f"w_var"}, fig=fig, subplotnumber=1)
#pdb.set_trace()
_plt, data = plot_profile(w_brn, points, None, subplot=(1, 1), lineproperties={"c": "r", "lw":5, "label": f"w_brn", "ls": ":"}, fig=fig, subplotnumber=1)
_plt, data = plot_profile(w_car, points, None, subplot=(1, 1), lineproperties={"c": "g", "lw":5, "label": f"w_car", "ls": "--"}, fig=fig, subplotnumber=1)
_plt, data = plot_profile(w_exact, points, None, subplot=(1, 1), lineproperties={"c": "k", "lw":5, "label": f"w_e", "ls": "--"}, fig=fig, subplotnumber=1)

_plt.xlabel("x [m]", fontsize=30)
_plt.ylabel("Transverse displacement [m]", fontsize=30)
_plt.xticks(fontsize=30)
_plt.yticks(fontsize=30)
_plt.title(f"Comparison between FE models. Transverse displacement. Mesh = {parameters["geometry"]["mesh_size"]}. IP = {parameters["model"]["alpha_penalty"]}", size = 30)
_plt.grid(True)
_plt.legend(fontsize=30)
_plt.savefig(f"{OUTDIR}/{SCRIPT_VAR}-w-profiles.png")

fig, axes = plt.subplots(1, 1, figsize=(24, 18))

_plt, data = plot_profile(v_var, points, None, subplot=(1, 1), lineproperties={"c": "b", "lw":5, "label": f"v_var"}, fig=fig, subplotnumber=1)
_plt, data = plot_profile(v_brn, points, None, subplot=(1, 1), lineproperties={"c": "r", "lw":5, "label": f"v_brn", "ls": ":"}, fig=fig, subplotnumber=1)
_plt, data = plot_profile(v_car, points, None, subplot=(1, 1), lineproperties={"c": "g", "lw":5, "label": f"v_car", "ls": "--"}, fig=fig, subplotnumber=1)
_plt, data = plot_profile(v_exact, points, None, subplot=(1, 1), lineproperties={"c": "k", "lw":5, "label": f"v_e", "ls": "--"}, fig=fig, subplotnumber=1)

_plt.xlabel("x [m]", fontsize=30)
_plt.ylabel("Airy's function [Nm]", fontsize=30)
_plt.xticks(fontsize=30)
_plt.yticks(fontsize=30)
_plt.title(f"Comparison between FE models. Airy's function. Mesh = {parameters["geometry"]["mesh_size"]}. IP = {parameters["model"]["alpha_penalty"]}", size = 30)
_plt.grid(True)
_plt.legend(fontsize=30)
_plt.savefig(f"{OUTDIR}/{SCRIPT_VAR}-v-profiles.png")
