# Numerical Experiments with FVK Plates and Disclinations

## Series 000 Overview
### Test 1: Kinematically compatible plate under volume forces

This script contains material for conducting numerical experiments on FVK (Foppl-von Kármán) plates subjected to transverse pressure loading. The experiments aim to investigate the behavior of FVK plates under various loading conditions and parameter configurations.

## Model Description: compatible load

The model used in these experiments is based on the Foppl-von Kármán plate theory, which describes the deformation of thin elastic plates. The equations governing the FVK plate model are solved numerically using finite element methods.

$$
\mathcal {E}(y) = -E_m(v) + E_b(w) + E_c(v, w) - \frac{R^4}{E h^4} p_0 \mathcal L_{\text{tr}}(w)
$$
where the state is $y=(v, w)$, $v\in H^2_0(B_R)$ is Airy stress potential, $w\in H^2_0(B_R)$ is the transverse displacement, and $p_0$ is set so that the coefficient of $\mathcal L_{\text{tr}}(w)$ is one.

$\frac{R^4}{E h^4} p_0:= c*a^4$, where $c:=p_0/E$, $a:=R/h$

## Loading Conditions

The transverse load is polynomial, 
$
f(x) = \frac{40}{3} (1 - x^2 - y^2)^4 + \frac{16}{3} (11 + x^2 + y^2)
$

Experiments of the 000 Series run parametrically with respect to the .... of the plate to study the ...

## Rationale

The numerical experiments are conducted to:

- Investigate the deformation behavior of FVK plates under different loading conditions.
- Study the influence of the thickness on the stress distribution in the plates.
- Validate the numerical results against analytical solutions where possible.

## Repository Structure

- **output/**: Directory for storing visualisations and analysis results.

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
    python3 experiment_00.py
    ```

5. Analyse the results and visualise the data using your preferred tools.

## Output Data

The output data from the numerical experiments include (TBI):

- Deformation profiles of the FVK plates.
- Stress distributions in the plates.
- Comparison of numerical results with analytical solutions (where applicable).
- Plots and visualisations of key parameters and quantities of interest.

## References

- [Reference paper or resource for FVK plate theory]
- [Reference paper or resource for disclinations]

---

Feel free to customise and expand upon this template to suit your specific experiments and requirements.