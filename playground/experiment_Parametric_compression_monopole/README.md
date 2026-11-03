# Numerical Experiments with FVK Plates and Disclinations

## Series 000 Overview
### Test X: Plate under pressure and disclination, parametric 

This script contains material for conducting numerical experiments on FVK (Foppl-von Kármán) plates subjected to transverse pressure loading and a disclination. The experiments aim to investigate the behavior of FVK plates parametrically with respect to the nondimensional parameter a, keeping c fixed to 1.

## Model Description: monopole + transverse pressure


$$
\mathcal {E}(y) = -E_m(v) + E_b(w) + E_c(v, w) - \frac{R^4}{E h^4} p_0 \mathcal L_{\text{tr}}(w) + a^2 s \langle \delta(x-x_0), v\rangle
$$

where $a = \frac{R}{h}$, $c:=\frac{R^4}{E h^4} p_0$, $s = \pm 1$