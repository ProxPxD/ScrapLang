import sys
import os
from pathlib import Path

import pytest

if __name__ == '__main__':
    # os.chdir(Path(__file__).parent.parent)
    pytest.main(sys.argv)  # --capture=sys -v --full-trace testing/tests/system/
