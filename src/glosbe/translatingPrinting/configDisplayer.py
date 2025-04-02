from typing import Any

# from ..configurations import Configurations
from ..constants import FLAGS as F
from ..layoutAdjusting.layoutAdjuster import LayoutAdjustmentsMethods
from ..translating.scrapping import TranslationTypes
from ..translatingPrinting.translationPrinter import TranslationPrinter


# TODO: move it in translator printer
# class ConfigDisplayer:
#
#     @classmethod
#     def display_configs(cls, **config_names: str):
#         for conf, value in config_names.items():
#             cls.display_config(conf, value)
#
#     @classmethod
#     def display_config(cls, config_name: str, value: Any = None, show_possible=False):
#         conf_to_display = value if value is not None else Configurations.get_conf(config_name)
#         conf_to_display = cls.format_config(conf_to_display)
#         TranslationPrinter.out(f'{config_name[2:]}: {conf_to_display}')
#
#         if show_possible:
#             cls.display_possible_values(config_name)
#         TranslationPrinter.out('\n')
#
#     @classmethod
#     def format_config(cls, config: list | str):
#         if isinstance(config, list):
#             return cls.list_to_comma_string(config)
#         return str(config)
#
#     @classmethod
#     def list_to_comma_string(cls, list_to_transform: list | str):
#         return str(list_to_transform)[1:-1].replace("'", '')
#
#     @classmethod
#     def display_possible_values(cls, config_name: str):  # TODO: test
#         match config_name:
#             case F.C.DOUBLE_MODE_STYLE_LONG_FLAG:
#                 TranslationPrinter.out('Possible values: ', end='')
#                 TranslationPrinter.out(str([TranslationTypes.LANG, TranslationTypes.WORD, TranslationTypes.DOUBLE])[1:-1])
#             case F.C.LAYOUT_ADJUSTMENT_METHOD_LONG_FLAG:
#                 TranslationPrinter.out('Possible values: ', end='')
#                 TranslationPrinter.out(str([LayoutAdjustmentsMethods.NATIVE, LayoutAdjustmentsMethods.KEYBOARD, LayoutAdjustmentsMethods.NONE])[1:-1])
#             case F.C.LAYOUT_ADJUSTMENT_LANG_LONG_FLAG:
#                 TranslationPrinter.out('Possible values: ', end='')
#                 TranslationPrinter.out(str(['uk', 'de', 'zh'])[1:-1])