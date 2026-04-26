"""
Targeted regression checks for two notebook interactions:
  - the method picker should prefer SnapKV for the notebook's stated
    long-context, single-shot, drop-in scenario
  - the SnapKV-style agent loop should no longer collapse to the same
    behaviour as plain streaming

This test stubs `marimo` so it can run in lightweight environments where the
full notebook stack is not installed.
"""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _FakeHtml:
    def __init__(self, text: str):
        self.text = text

    def __str__(self) -> str:
        return self.text


sys.modules.setdefault("marimo", types.SimpleNamespace(Html=_FakeHtml))

from src.visualizations import (  # noqa: E402
    plot_memory_breakdown,
    run_method_picker,
    simulate_agent_loop,
)


LONG_CONTEXT = "Long (\u0033\u0032\u00e2\u20ac\u201c128K)"


def _best_fit_name(html: str) -> str:
    match = re.search(
        r"BEST FIT</span>\s*<span[^>]*color:var\(--color-text-primary\)\">([^<]+)</span>",
        html,
        re.S,
    )
    if not match:
        raise AssertionError("Could not extract BEST FIT name from method-picker output")
    return match.group(1).strip()


def _still_accessible(html: str) -> int:
    match = re.search(
        r"Still accessible</div>\s*<div[^>]*color:#1D9E75\">([0-9,]+)</div>",
        html,
        re.S,
    )
    if not match:
        raise AssertionError("Could not extract 'Still accessible' value from agent-loop output")
    return int(match.group(1).replace(",", ""))


def main() -> int:
    memory_html = plot_memory_breakdown().text
    assert "data:[2e-06" in memory_html or "data:[0.000002" in memory_html, (
        "Memory breakdown chart should keep strictly positive values on its log axis"
    )
    print("ok    memory-breakdown chart keeps non-zero log-scale data")

    picker_html = run_method_picker(
        context_length=LONG_CONTEXT,
        workload="Single-shot Q&A",
        drop_in_required=True,
        recovery_needed=False,
    ).text
    best_fit = _best_fit_name(picker_html)
    assert best_fit == "SnapKV", f"Expected SnapKV to be BEST FIT, got {best_fit}"

    stream_html = simulate_agent_loop(12, "Streaming (recent only)", 800).text
    snap_html = simulate_agent_loop(12, "SnapKV-style (intent-aware)", 800).text
    stream_accessible = _still_accessible(stream_html)
    snap_accessible = _still_accessible(snap_html)

    assert snap_html != stream_html, "SnapKV-style output should differ from Streaming output"
    assert snap_accessible > stream_accessible, (
        "SnapKV-style should retain more accessible context than plain Streaming"
    )

    print("ok    method-picker long/drop-in scenario prefers SnapKV")
    print("ok    SnapKV-style agent loop retains more context than Streaming")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
