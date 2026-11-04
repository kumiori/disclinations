"""Pytest collection policy for the maintained validation suite.

The ignored modules are retained as historical experiment scripts.  They use
pre-audit APIs and produce research output, so collecting them as unattended
unit tests would fail before the maintained tests run.
"""

collect_ignore = [
    "test/test_dipole.py",
    "test/test_transverse.py",
    "src/disclinations/test/test_monopole_adimensional.py",
    "src/disclinations/test/test_parametric-dipole.py",
    "src/disclinations/test/test_parametric-dipole-relative_work.py",
]
