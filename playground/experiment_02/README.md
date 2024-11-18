# Numerical Experiments with FVK Plates and Disclinations

## Series 001 Overview

This script contains material for conducting numerical experiments on FVK (Foppl-von Kármán) plates subjected to an array of disclination loadings (flower configuration) and its own weight. The experiments aim to investigate the nonlinear behavior of FVK plates, the distinction between traction and compression, the formation of wrinkling patterns (under compression), and the apparent stiffness reduction under tension.

## Model Description:

The model used in these experiments is based on the Foppl-von Kármán plate theory, which describes the deformation of thin elastic plates. The equations governing the FVK plate model are solved numerically using finite element methods.

$$
\mathcal {E}(y) = -E_m(v) + E_b(w) + E_c(v, w) - \frac{R^4}{E h^4} p_0 \mathcal L_{\text{tr}}(w) +\frac{R^2}{h^2} \sum_i^4 \langle \delta(x-x_i), v\rangle
$$

where the state is $y=(v, w)$, $v\in H^2_0(B_R)$ is Airy stress potential, $w\in H^2_0(B_R)$ is the transverse displacement, and $p_0$ is set so that the coefficient of $\mathcal L_{\text{tr}}(w)$ is one.

## Loading Conditions

A flower of disclinations, topological (so-called) defects in the crystal lattice.

Experiments of the 002 Series run parametrically with respect to the imposed pressure on the plate to study the relative contribution of transverse pressure and disclinations in the stress distribution in the plates.

## Rationale

The numerical experiments are conducted to:

-   Investigate the deformation behavior of FVK plates under different loading conditions.
-   Study the influence of the thickness on the stress distribution in the plates.
-   Validate the numerical results against analytical solutions where possible.

## Repository Structure

-   **output/**: Directory for storing visualisations and analysis results.

## Usage

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/fvk-plate-experiments.git
    ```

2. Navigate to the repository directory:

    ```bash
    cd fvk-plate-experiments
    ```

3. Modify the input parameters in the configuration files located in the `data/` directory.

4. Run the experiments using the provided scripts:

    ```bash
    python3 experiment_01.py
    ```

5. Analyse the results and visualise the data using your preferred tools.

## Output Data

The output data from the numerical experiments include (TBI):

-   Deformation profiles of the FVK plates.
-   Stress distributions in the plates.
-   Comparison of numerical results with analytical solutions (where applicable).
-   Plots and visualisations of key parameters and quantities of interest.

## References

-   [Reference paper or resource for FVK plate theory]
-   [Reference paper or resource for disclinations]

---

Feel free to customise and expand upon this template to suit your specific experiments and requirements.
