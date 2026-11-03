"""Isotropic Kirchhoff–Love plate constants (SI inputs, plane stress)."""
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class PlateMaterial:
    """Young's modulus E [Pa], Poisson ratio nu [-], thickness [m]."""

    E: float
    nu: float
    thickness: float

    def __post_init__(self):
        if not all(isfinite(v) for v in (self.E, self.nu, self.thickness)):
            raise ValueError("Material parameters must be finite")
        if self.E <= 0 or self.thickness <= 0 or not -1 < self.nu < 0.5:
            raise ValueError("Require E > 0, thickness > 0 and -1 < nu < 0.5")

    @property
    def bending_stiffness(self):
        """D = E t³/[12(1-nu²)] [N m]."""
        return self.E * self.thickness**3 / (12 * (1 - self.nu**2))

    @property
    def airy_compliance(self):
        """1/(E t) [m/N], for an Airy potential of stress resultants."""
        return 1 / (self.E * self.thickness)

    @property
    def dimensionless_bending(self):
        return 1 / (12 * (1 - self.nu**2))

    def energy_scale(self, radius):
        """E t^5/R² [J], for x=R xi, w=t w~, v=E t³ v~."""
        if not isfinite(radius) or radius <= 0:
            raise ValueError("radius must be positive and finite")
        return self.E * self.thickness**5 / radius**2
