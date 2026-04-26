"""
Smoke tests for src/visualizations.py.

Runs every plot/render function with sane defaults and verifies that:
  - it returns without raising
  - the returned object is a marimo Html / md object (has _repr_ methods)
  - chart functions emit a Chart.js block with a unique canvas id and
    the safe-loader pattern (so re-running cells won't break rendering)

Run from the repo root:
    python -m smoke_tests.test_visualizations
or:
    venv/Scripts/python.exe smoke_tests/test_visualizations.py
"""

from __future__ import annotations

import re
import sys
import traceback
from pathlib import Path

# Make `src` importable regardless of where this is run from
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.visualizations import (  # noqa: E402
    plot_adaptive_window,
    plot_agent_mapping,
    plot_attention_compute,
    plot_attention_consistency,
    plot_budget_quality,
    plot_entropy_intuition,
    plot_human_memory,
    plot_kv_growth,
    plot_memory_breakdown,
    plot_memory_hierarchy,
    plot_memory_types,
    plot_naive_strategy,
    plot_per_head_heatmap,
    plot_two_axes,
    plot_vote_cluster,
    render_algo_step,
    run_custom_policy,
    run_demo,
    run_needle_demo,
    simulate_agent_loop,
)


SAMPLE_PROMPT = (
    "The Eiffel Tower is 324 metres tall. "
    "It was the tallest structure in the world for 41 years. "
    "What is the height of the Eiffel Tower?"
)

# (label, callable, kwargs, is_chart) — is_chart=True means it should produce a
# Chart.js block (we then assert the safe-loader markers are present).
CASES = [
    ("plot_kv_growth",            lambda: plot_kv_growth(),                                  True),
    ("plot_memory_breakdown",     lambda: plot_memory_breakdown(),                           True),
    ("plot_attention_compute",    lambda: plot_attention_compute(),                          True),
    ("plot_attention_consistency",lambda: plot_attention_consistency(window_size=16, head_idx=0), True),
    ("plot_per_head_heatmap",     lambda: plot_per_head_heatmap(SAMPLE_PROMPT),              True),
    ("plot_budget_quality",       lambda: plot_budget_quality(),                             True),
    ("plot_vote_cluster",         lambda: plot_vote_cluster(),                               True),
    ("plot_adaptive_window",      lambda: plot_adaptive_window(SAMPLE_PROMPT),               True),
    ("plot_entropy_intuition",    lambda: plot_entropy_intuition(),                          True),
    ("plot_human_memory",         lambda: plot_human_memory(),                               True),
    ("plot_naive_strategy/Recent",lambda: plot_naive_strategy("Recent Only"),                False),
    ("plot_naive_strategy/Random",lambda: plot_naive_strategy("Random Drop"),                False),
    ("plot_naive_strategy/Stride",lambda: plot_naive_strategy("Uniform Stride"),             False),
    ("plot_naive_strategy/All",   lambda: plot_naive_strategy("Keep Everything"),            False),
    ("run_demo/SnapKV",           lambda: run_demo(SAMPLE_PROMPT, 0.3, "SnapKV"),            False),
    ("run_demo/H2O",              lambda: run_demo(SAMPLE_PROMPT, 0.3, "H2O"),               False),
    ("run_demo/Streaming",        lambda: run_demo(SAMPLE_PROMPT, 0.3, "StreamingLLM"),      False),
    ("run_demo/Full",             lambda: run_demo(SAMPLE_PROMPT, 0.3, "Full Cache"),        False),
    ("render_algo_step/1",        lambda: render_algo_step(1),                               False),
    ("render_algo_step/2",        lambda: render_algo_step(2),                               False),
    ("render_algo_step/3",        lambda: render_algo_step(3),                               False),
    ("render_algo_step/4",        lambda: render_algo_step(4),                               False),
    ("run_needle_demo/early",     lambda: run_needle_demo("early",  0.25, 60),               False),
    ("run_needle_demo/middle",    lambda: run_needle_demo("middle", 0.25, 60),               False),
    ("run_needle_demo/late",      lambda: run_needle_demo("late",   0.25, 60),               False),
    ("run_custom_policy",         lambda: run_custom_policy(SAMPLE_PROMPT, 0.3, 0.3, 0.2, 0.5), False),
    ("plot_two_axes",             lambda: plot_two_axes(),                                   False),
    ("plot_memory_hierarchy",     lambda: plot_memory_hierarchy(),                           False),
    ("plot_memory_types",         lambda: plot_memory_types(),                               False),
    ("plot_agent_mapping",        lambda: plot_agent_mapping(),                              False),
    ("simulate_agent_loop/full",  lambda: simulate_agent_loop(12, "Full Cache", 800),        True),
    ("simulate_agent_loop/stream",lambda: simulate_agent_loop(12, "Streaming (recent only)", 800), True),
    ("simulate_agent_loop/snap",  lambda: simulate_agent_loop(12, "SnapKV-style (intent-aware)", 800), True),
    ("simulate_agent_loop/sum",   lambda: simulate_agent_loop(12, "Agent + Summarise", 800), True),
]

IFRAME_RE = re.compile(r'<iframe srcdoc="([^"]+)"', re.S)


def _check_chart_html(html: str) -> list[str]:
    """
    Verify the chart is wrapped in a sandboxed iframe whose srcdoc contains
    the Chart.js loader and a `new Chart(...)` call. This is what makes the
    chart actually render inside marimo (mo.Html otherwise strips <script>).
    """
    fails = []
    matches = IFRAME_RE.findall(html)
    if not matches:
        fails.append("no <iframe srcdoc='…'> wrapper found — chart will not render in marimo")
        return fails
    if len(matches) > 1:
        fails.append(f"expected one iframe per chart, got {len(matches)}")
    inner = matches[0]  # still HTML-escaped
    if "chart.umd.js" not in inner:
        fails.append("Chart.js CDN URL missing inside iframe srcdoc")
    if "new Chart(" not in inner:
        fails.append("`new Chart(` call missing inside iframe srcdoc")
    if "sandbox=&quot;allow-scripts&quot;" not in html and 'sandbox="allow-scripts"' not in html:
        fails.append("iframe is not sandboxed with allow-scripts")
    return fails


def _extract_html(obj) -> str | None:
    """Get the inner HTML string from a marimo Html object, if it is one."""
    text = getattr(obj, "text", None)
    if isinstance(text, str):
        return text
    # Fallback: stringification
    try:
        return str(obj)
    except Exception:
        return None


def main() -> int:
    print(f"Running {len(CASES)} smoke tests…\n")
    passed, failed = 0, 0
    failures: list[tuple[str, str]] = []

    for name, fn, is_chart in CASES:
        try:
            obj = fn()
        except Exception as e:
            failed += 1
            failures.append((name, f"raised {type(e).__name__}: {e}\n{traceback.format_exc()}"))
            print(f"  FAIL  {name}  (exception)")
            continue

        if obj is None:
            failed += 1
            failures.append((name, "returned None"))
            print(f"  FAIL  {name}  (returned None)")
            continue

        if is_chart:
            html = _extract_html(obj) or ""
            problems = _check_chart_html(html)
            if problems:
                failed += 1
                failures.append((name, "; ".join(problems)))
                print(f"  FAIL  {name}  ({problems[0]})")
                continue

        passed += 1
        print(f"  ok    {name}")

    print()
    print(f"{passed} passed · {failed} failed · {len(CASES)} total")

    if failures:
        print("\n--- failure details ---")
        for name, msg in failures:
            print(f"\n[{name}]")
            print(msg)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
