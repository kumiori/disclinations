# Numerical Experiments

## 1. Numerical Setup

### 1.1 Variational Problem

-   Brief description of the problem being solved.
-   Reference to the finite element discretization and solver configuration.

### 1.2 Implementation Details

-   Software environment: dolfinx/FEniCSx, PETSc.
-   Solver configurations (e.g., SNES settings, tolerance).
-   Meshes

---

## 2. Verification Tests

### 2.1 Plate with Kinematically Compatible Loads, Parametric

-   **Setup**: Plate loaded with surface pressure.
-   **Objectives**: Compare numerical results against known analytic solutions.
-   **Results**: Error analysis, visualizations.

### 2.2 Plate with Kinematically Incompatible Loads, Parametric

-   **Setup**: Plate loaded with a dipole of disclinations.
-   **Objectives**: Validate the implementation for nontrivial incompatible loading.
-   **Results**: Error analysis, visualizations.

---

## 3. Parametric Experiments

### 3.1 Mixed Loading: Varying Thickness

-   **Setup**: Surface pressure combined with disclination monopole; thickness is varied.
-   **Objectives**: Study the role of aspect ratio on the nonlinear mechanics.
-   **Results**: Parametric plots (e.g., energy terms, deformation profiles).

### 3.2 Mixed Loading: Varying Pressure

-   **Setup**: Surface pressure combined with disclination monopole; pressure is varied.
-   **Objectives**: Investigate the effect of increasing loading intensity.
-   **Results**: Parametric plots (e.g., energy terms, deformation profiles).

---

## 4. Complex Experiment: Flower Configuration

### 4.1 Setup

-   **Configuration**: A flower-like arrangement of disclinations mimicking engineered graphene sheets.
-   **Objectives**: Demonstrate the capability of the model for complex and nonstandard configurations.

### 4.2 Results

-   Energy analysis, deformation profiles.
-   Discussion of insights from this experiment.

---

## 5. Summary of Numerical Results

-   Recap key findings from the verification and parametric experiments.
-   Discuss implications for the mechanics of thin plates under complex loading scenarios.
