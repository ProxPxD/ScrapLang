import logging
import sys
import warnings
from argparse import Namespace
from dataclasses import asdict

from box import Box

from src.glosbe.constants import Paths
from src.glosbe.context import Context


def adjust_dict_like_obj(obj: Context | Namespace) -> Box:
    match obj:
        case object() if obj.__class__.__name__ == 'Context': context = asdict(obj)
        case Namespace(): context = vars(obj)
        case None: context = {}
        case _: raise ValueError(f'Unexpected valued for context of type {type(obj)}: {obj}')
    return Box(context, default_box=True)


def setup_logging(context: Context | Namespace = None) -> None:
    root_logger = logging.getLogger()
    context = adjust_dict_like_obj(context)

    # 1. Clean up existing configuration
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)
    # 2. Set fresh handlers
    handlers = [logging.StreamHandler(sys.stdout)]
    if context.debug:
        handlers.append(logging.FileHandler(Paths.LOG_DIR, encoding='utf-8'))
    # 3. Configure with current debug state
    logging.basicConfig(
        level=logging.DEBUG if context.debug else logging.INFO,
        format='%(levelname)s: %(message)s',
        handlers=handlers,
        force=True  # Critical: Overrides any existing config
    )
    # 4. Handle warnings
    warnings.filterwarnings('default' if context.debug else 'ignore')
