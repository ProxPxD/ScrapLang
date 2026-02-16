from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    try:
        from src.constants import Paths
        from src.app_managing import AppMgr

        AppMgr(
            conf_path=Paths.CONF_FILE,
            valid_data_file=Paths.VALID_DATA_FILE,
            short_mem_file=Paths.SHORT_MEM_FILE,
            lang_script_file=Paths.LANG_SCRIPT_FILE
        ).run()
    except KeyboardInterrupt:
        pass
    except:
        raise


if __name__ == '__main__':
    main()
