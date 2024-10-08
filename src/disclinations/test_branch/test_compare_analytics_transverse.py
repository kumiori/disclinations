from disclinations.utils.viz import plot_scalar, plot_profile, plot_mesh
import yaml
import numpy as np
import matplotlib.pyplot as plt
import pdb
import os
import pandas as pd

#SCRIPT_VAR = "test_fvk_vs_analytics_transverse.py"
#SCRIPT_BRN = "test_fvk_vs_analytics_transverse-brenner.py"
#SCRIPT_CAR = "test_fvk_vs_analytics_transverse-carstensen.py"
SCRIPT_VAR = "test_fvk_vs_analytics_transverse.py"
SCRIPT_BRN = "test_fvk_vs_analytics_transverse-brenner.py"
SCRIPT_CAR = "test_fvk_vs_analytics_transverse-carstensen.py"

OUTDIR = os.path.join("output", "test_compare_analytic_transverse")

# Define a function to run a script and return a variable
def run_script(filename):
    globals_dict = {}  # Dictionary to capture script's variables
    with open(filename) as file:
        exec(file.read(), globals_dict)
    return globals_dict

#pdb.set_trace()

script_var = run_script(SCRIPT_VAR)
script_brn = run_script(SCRIPT_BRN)
script_car = run_script(SCRIPT_CAR)

V_v = script_var["V_v"]
V_w = script_var["V_w"]
dofs_v = script_var["dofs_v"]
dofs_w = script_var["dofs_w"]

v_exact = script_var["v_exact"]
v_var = script_var["v"]
v_brn = script_brn["v"]
v_car = script_car["v"]

w_exact = script_var["w_exact"]
w_var = script_var["w"]
w_brn = script_brn["w"]
w_car = script_car["w"]

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
e_data["Bending Energy (Var)"] = script_var["computed_energy_terms"]["bending"]
e_data["Bending Energy (Brn)"] = script_brn["energy_terms"]["bending"]
e_data["Bending Energy (Car)"] = script_car["energy_terms"]["bending"]

e_data["Membrane Energy (Var)"] = script_var["computed_energy_terms"]["membrane"]
e_data["Membrane Energy (Brn)"] = script_brn["energy_terms"]["membrane"]
e_data["Membrane Energy (Car)"] = script_car["energy_terms"]["membrane"]

e_data["Coupling Energy (Var)"] = script_var["computed_energy_terms"]["coupling"]
e_data["Coupling Energy (Brn)"] = script_brn["energy_terms"]["coupling"]
e_data["Coupling Energy (Car)"] = script_car["energy_terms"]["coupling"]

e_data["Bending Energy (Exact)"] = script_var["ex_bending_energy"]
e_data["Membrane Energy (Exact)"] = script_var["ex_membrane_energy"]
e_data["Coupling Energy (Exact)"] = script_var["ex_coupl_energy"]

#pdb.set_trace()
exp_dict = {"Bending Energy (Var)": [script_var["computed_energy_terms"]["bending"]], "Bending Energy (Brn)":  [script_brn["energy_terms"]["bending"]], "Bending Energy (Car)": [script_car["energy_terms"]["bending"]],
         "Membrane Energy (Var)": [script_var["computed_energy_terms"]["membrane"]], "Membrane Energy (Brn)": [script_brn["energy_terms"]["membrane"]], "Membrane Energy (Car)": [script_car["energy_terms"]["membrane"]],
         "Coupling Energy (Var)": [script_var["computed_energy_terms"]["coupling"]], "Coupling Energy (Brn)": [script_brn["energy_terms"]["coupling"]], "Coupling Energy (Car)": [script_car["energy_terms"]["coupling"]],
         "Bending Energy (Exact)": [script_var["computed_energy_terms"]["bending"]], "Membrane Energy (Exact)": [script_var["computed_energy_terms"]["membrane"]], "Coupling Energy (Exact)": [script_var["computed_energy_terms"]["coupling"]],
         "Mesh size": [script_var["mesh_size"]], "Interior Penalty (IP)": [script_var["parameters"]["model"]["alpha_penalty"]], "Thickness": [script_var["thickness"]], "Young modulus": [script_var["_E"]], "Poisson ratio": [script_var["nu"]],
         "Error % Bending Energy (Var)": [100*( e_data["Bending Energy (Var)"]- e_data["Bending Energy (Exact)"] ) / e_data["Bending Energy (Exact)"]],
         "Error % Bending Energy (Brn)": [100*( e_data["Bending Energy (Brn)"]- e_data["Bending Energy (Exact)"] ) / e_data["Bending Energy (Exact)"]],
         "Error % Bending Energy (Car)": [100*( e_data["Bending Energy (Car)"]- e_data["Bending Energy (Exact)"] ) / e_data["Bending Energy (Exact)"]],
         "Error % Membrane Energy (Var)": [100*( e_data["Membrane Energy (Var)"] - e_data["Membrane Energy (Exact)"] ) / e_data["Membrane Energy (Exact)"]],
         "Error % Membrane Energy (Brn)": [100*( e_data["Membrane Energy (Brn)"] - e_data["Membrane Energy (Exact)"] )/ e_data["Membrane Energy (Exact)"]],
         "Error % Membrane Energy (Car)": [100*( e_data["Membrane Energy (Car)"] - e_data["Membrane Energy (Exact)"] )/ e_data["Membrane Energy (Exact)"]],
         "Error % Coupling Energy (Var)": [100*( e_data["Coupling Energy (Var)"] - e_data["Coupling Energy (Exact)"] ) / e_data["Coupling Energy (Exact)"]],
         "Error % Coupling Energy (Brn)": [100*( e_data["Coupling Energy (Brn)"] - e_data["Coupling Energy (Exact)"] ) / e_data["Coupling Energy (Exact)"]],
         "Error % Coupling Energy (Car)": [100*( e_data["Coupling Energy (Car)"] - e_data["Coupling Energy (Exact)"] ) / e_data["Coupling Energy (Exact)"]]
         }

#experimental_data = experimental_data.append(exp_dict, ignore_index=True)
experimental_data = pd.DataFrame(exp_dict)
experimental_data.to_excel(f'{OUTDIR}/Models_comparison.xlsx', index=False)
# Plot profiles:

import pyvista

pyvista.OFF_SCREEN = True

plotter = pyvista.Plotter(title="Displacement", window_size=[1200, 600], shape=(2, 2))


#scalar_plot = plot_scalar(v, plotter, subplot=(0, 0), V_sub=V_v, dofs=dofs_v)
#scalar_plot = plot_scalar(w, plotter, subplot=(0, 1), V_sub=V_w, dofs=dofs_w)
#
#scalar_plot = plot_scalar(v_exact, plotter, subplot=(1, 0), V_sub=V_v, dofs=dofs_v)
#scalar_plot = plot_scalar(w_exact, plotter, subplot=(1, 1), V_sub=V_w, dofs=dofs_w)
#
#scalar_plot.screenshot(f"{prefix}/test_fvk.png")
#print("plotted scalar")
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
_plt.savefig(f"{OUTDIR}/test_fvk-w-profiles.png")

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
_plt.savefig(f"{OUTDIR}/test_fvk-v-profiles.png")
