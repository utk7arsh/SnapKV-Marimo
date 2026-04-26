"""
Run every smoke test in this folder. Exits non-zero if any fail.

    python -m smoke_tests.run_all
    venv/Scripts/python.exe smoke_tests/run_all.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smoke_tests import test_notebook_imports, test_visualizations  # noqa: E402

SUITES = [
    ("notebook imports", test_notebook_imports.main),
    ("visualizations",   test_visualizations.main),
]


def main() -> int:
    overall = 0
    for name, fn in SUITES:
        print(f"\n=== {name} ===")
        rc = fn()
        if rc != 0:
            overall = rc
    print()
    print("ALL SMOKE TESTS PASSED" if overall == 0 else "SMOKE TESTS FAILED")
    return overall


if __name__ == "__main__":
    sys.exit(main())
