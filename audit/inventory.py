"""Inventory tracked and non-ignored textual source, including notebook code/ZIP TeX."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import zipfile

PATTERN = re.compile(
    r"\b[Cc]_?[12]\b|c_nu|hess_l2_norm|LinearProblem|Burgers|Frank|crack|core.radius|energy_(scale|dimension)|1\s*/\s*\(.*E.*t|self\.D|\b(?:D|_D|Eh|k_g)\s*=|b_scale|v_scale|w_scale"
)
SUFFIXES = {".py", ".yml", ".yaml", ".toml", ".md", ".tex", ".ipynb", ".sh", ".txt"}


def main():
    paths = (
        subprocess.check_output(["git", "ls-files", "-co", "--exclude-standard", "-z"])
        .decode()
        .split("\0")
    )
    sources = []
    hits = []

    def inspect(name, contents):
        lines = contents.splitlines()
        sources.append(
            {
                "path": name,
                "lines": len(lines),
                "sha256": hashlib.sha256(contents.encode()).hexdigest(),
            }
        )
        for i, line in enumerate(lines, 1):
            if PATTERN.search(line):
                hits.append({"path": name, "line": i, "text": line.strip()})

    for name in sorted(set(paths)):
        if (
            not name
            or name.startswith(("audit/", "tests/", "examples/", "docs/"))
            or name in ("SCIENTIFIC_AUDIT_AND_REPAIRS.md", "README.md")
        ):
            continue
        p = Path(name)
        if p.suffix == ".zip":
            with zipfile.ZipFile(p) as z:
                for item in z.namelist():
                    if Path(item).suffix in SUFFIXES:
                        inspect(name + "!" + item, z.read(item).decode())
        elif p.suffix == ".ipynb":
            notebook = json.loads(p.read_text())
            for i, cell in enumerate(notebook["cells"]):
                inspect(name + f"!cell{i}", "".join(cell.get("source", [])))
        elif p.suffix in SUFFIXES:
            inspect(name, p.read_text())
    Path("audit/source_inventory.json").write_text(json.dumps(sources, indent=2) + "\n")
    Path("audit/constants_and_scope_inventory.json").write_text(
        json.dumps(hits, indent=2) + "\n"
    )
    print(f"{len(sources)} text sources/cells scanned; {len(hits)} relevant lines")


if __name__ == "__main__":
    main()
