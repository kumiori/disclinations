"""Compare checkpoint/repaired linear Airy blocks using the SAME repaired solver.

This isolates variational changes; it does not claim the old executable worked
with DOLFINx 0.9. The old solver itself used removed APIs.
"""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from mpi4py import MPI
from disclinations.models.material import PlateMaterial
from examples import verify_clamped as benchmark

REFERENCE = "71e11e4d43174d26149130e97bd98b2f9635e09e"


def main():
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("Checkpoint comparison is a serial experiment")
    text = subprocess.check_output(
        ["git", "show", REFERENCE + ":src/disclinations/models/__init__.py"], text=True
    )
    with tempfile.TemporaryDirectory() as folder:
        # The archived module loads a sibling parameter file.
        path = Path(folder) / "reference.py"
        path.write_text(text)
        params = subprocess.check_output(
            [
                "git",
                "show",
                REFERENCE + ":src/disclinations/models/default_parameters.yml",
            ],
            text=True,
        )
        path.with_name("default_parameters.yml").write_text(params)
        spec = importlib.util.spec_from_file_location("reference_models", path)
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)

        class Reference(original.NonlinearPlateFVK):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.material = PlateMaterial(self.E, self.nu, self.t)

        repaired = benchmark.NonlinearPlateFVK
        output = []
        for n in (4, 8, 16):
            for model, label in (
                (Reference, "checkpoint_forms"),
                (repaired, "repaired_forms"),
            ):
                benchmark.NonlinearPlateFVK = model
                result, _ = benchmark.solve_case(n, 3)
                result["implementation"] = label
                output.append(result)
        benchmark.NonlinearPlateFVK = repaired
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
