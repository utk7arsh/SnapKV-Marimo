# Smoke tests

Quick sanity checks for the SnapKV notebook project. Not a test suite — just
"does the thing import and render without exploding."

## What gets checked

- **`test_visualizations.py`** — calls every `plot_*` / `run_demo` /
  `render_algo_step` function with reasonable defaults and verifies:
  - it returns without raising
  - chart-producing functions emit the safe Chart.js loader pattern
    (unique canvas id, single CDN load, destroy-existing-chart guard).
    These are the things that previously caused charts to silently
    fail to render after a cell re-run.
- **`test_notebook_imports.py`** — parses each marimo notebook and
  verifies that every name it imports from `src.visualizations` actually
  exists. Catches the "I renamed the function but forgot to update the
  notebook" class of bug.

## Run

From the repo root, using the project venv:

```bash
venv/Scripts/python.exe smoke_tests/run_all.py
```

Or pick one:

```bash
venv/Scripts/python.exe smoke_tests/test_visualizations.py
venv/Scripts/python.exe smoke_tests/test_notebook_imports.py
```

Exit code is non-zero on any failure, so it's CI-friendly.
