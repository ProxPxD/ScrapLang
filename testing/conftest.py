# conftest.py
from typing import Dict, Set

import pydash as _
import pytest

FAILED_MARKERS: Dict[str, Set[str]] = {}
PASSED_MARKERS: Dict[str, Set[str]] = {}


def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    """Collect markers for failed tests, but do NOT print yet."""
    if call.when == "call":
        marks = {'/'.join((m.name, *_.map_(m.args, str))) for m in item.iter_markers() if not m.name.startswith('parametrize')}
        if call.excinfo is not None:
            FAILED_MARKERS[item.nodeid] = marks
        else:
            PASSED_MARKERS[item.nodeid] = marks

@pytest.hookimpl(trylast=True)
def pytest_terminal_summary(terminalreporter, exitstatus):
    """Print marker info AFTER all tracebacks and summary."""
    if not FAILED_MARKERS:
        return

    tr = terminalreporter
    tr.write_line("")   # spacing
    tr.write_line("=== Marker summary for FAILED tests ===")

    passed_markers = set.union(set(), *PASSED_MARKERS.values())
    tr.write_line(f'\nPassed markers: {passed_markers}')
    failing_markers = set.union(set(), *FAILED_MARKERS.values())
    tr.write_line(f'\nFailed markers: {failing_markers}')


    # Common markers across failures
    common_failing = (set.intersection(*FAILED_MARKERS.values()) if FAILED_MARKERS else set()) - passed_markers
    tr.write_line(f'\nCommon markers: {common_failing or "{}"}')
