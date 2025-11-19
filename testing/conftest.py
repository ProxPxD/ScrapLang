# conftest.py
from typing import Dict, Set

import pydash as _
import pytest

FAILED_MARKERS: Dict[str, Set[str]] = {}


def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    """Collect markers for failed tests, but do NOT print yet."""
    if call.when == "call" and call.excinfo is not None:
        FAILED_MARKERS[item.nodeid] = {'/'.join((m.name, *_.map_(m.args, str))) for m in item.iter_markers() if not m.name.startswith('parametrize')}

@pytest.hookimpl(trylast=True)
def pytest_terminal_summary(terminalreporter, exitstatus):
    """Print marker info AFTER all tracebacks and summary."""
    if not FAILED_MARKERS:
        return

    tr = terminalreporter
    tr.write_line("")   # spacing
    tr.write_line("=== Marker summary for FAILED tests ===")

    appearing_markers = set.union(*FAILED_MARKERS.values())
    tr.write_line(f'\nAppearing markers: {appearing_markers}')

    # Common markers across failures
    common = set.intersection(*FAILED_MARKERS.values())
    tr.write_line(f'\nCommon markers: {common or "{}"}')
