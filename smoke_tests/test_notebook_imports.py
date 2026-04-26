"""
Smoke test for the marimo notebook itself.

We don't try to *run* the reactive graph — that's marimo's job. We just confirm:
  - the notebook file parses as valid Python
  - every cell body is syntactically valid
  - every name imported in the setup cell actually exists in src.visualizations

Run from the repo root:
    python -m smoke_tests.test_notebook_imports
"""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NOTEBOOKS = [
    ROOT / "notebooks" / "walkthrough.py",
    ROOT / "snapkv_notebook.py",
]


def check_parses(path: Path) -> list[str]:
    if not path.exists():
        return [f"file does not exist: {path}"]
    try:
        ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as e:
        return [f"SyntaxError in {path.name}: {e}"]
    return []


def check_visualization_imports(path: Path) -> list[str]:
    """Find names imported from src.visualizations and verify they exist."""
    if not path.exists():
        return []
    tree = ast.parse(path.read_text(encoding="utf-8"))
    expected: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "src.visualizations":
            for alias in node.names:
                expected.add(alias.name)
    if not expected:
        return []
    mod = importlib.import_module("src.visualizations")
    missing = sorted(n for n in expected if not hasattr(mod, n))
    return [f"{path.name}: missing from src.visualizations: {n}" for n in missing]


def main() -> int:
    fails: list[str] = []
    for nb in NOTEBOOKS:
        rel = nb.relative_to(ROOT)
        parse_fails = check_parses(nb)
        if parse_fails:
            fails.extend(parse_fails)
            print(f"  FAIL  {rel}  (parse)")
            continue
        import_fails = check_visualization_imports(nb)
        if import_fails:
            fails.extend(import_fails)
            print(f"  FAIL  {rel}  (missing imports)")
            continue
        print(f"  ok    {rel}")

    print()
    if fails:
        print(f"{len(fails)} failure(s):")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("all notebooks ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
