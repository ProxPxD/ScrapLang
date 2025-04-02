
from dataclasses import dataclass
from itertools import islice, chain
from pathlib import Path
from typing import Any, Iterable

import yaml
from box import Box
from more_itertools import partition

from returns.functions import tap

from .constants import FLAGS
from .layoutAdjusting.layoutAdjuster import LayoutAdjustmentsMethods
from .translating.scrapping import TranslationTypes
from .translatingPrinting.translationPrinter import TranslationPrinter
from more_itertools import nth

C = FLAGS.CONFIGURATIONAL
M = FLAGS.MODES


@dataclass(frozen=True)
class ConfigMessages:
    LANGUAGE_IN_SAVED: str = 'Language {} is already saved'
    LANGUAGE_NOT_IN_SAVED: str = 'Language {} has not been in saved'

#
# class Configurations:
#
#     _configs: dict = None
#     _current_config_file: Path = ''
#
#     defaults = {
#         C.DEFAULT_MODE_LONG_FLAG: M.SINGLE_LONG_FLAG,
#         C.LANGS_SHOW_LONG_FLAG: [],
#         C.LANG_LIMIT_LONG_FLAG: 3,
#         C.LAYOUT_ADJUSTMENT_METHOD_LONG_FLAG: LayoutAdjustmentsMethods.NONE,
#         C.LAYOUT_ADJUSTMENT_LANG_LONG_FLAG: '',
#         C.DOUBLE_MODE_STYLE_LONG_FLAG: TranslationTypes.LANG,
#         C.SHOW_INFO_MODE_FLAG_LONG: False
#     }
#
#     @classmethod
#     def init(cls, file_name=Paths.DEFAULT_CONF_FILE_NAME, default=None) -> None:
#         config_file_to_init = Paths.RESOURCES_DIR / file_name
#         if not cls._configs or cls._current_config_file != config_file_to_init:
#             cls._current_config_file = config_file_to_init
#             Paths.RESOURCES_DIR.mkdir(exist_ok=True)
#             if not cls._current_config_file.exists():
#                 cls.init_file(default)
#             cls._configs = cls._get_configurations()
#
#     @classmethod
#     def _get_configurations(cls) -> dict[str, Any]:
#         with open(cls._current_config_file, 'r') as f:
#             configs = yaml.safe_load(f) or {}
#         return configs
#
#     @classmethod
#     def remove_current_configuration(cls):
#         cls._current_config_file.unlink(missing_ok=True)
#
#     @classmethod
#     def save(cls) -> None:
#         Configurations._save(Configurations._configs)
#
#     @classmethod
#     def save_and_close(cls) -> None:
#         Configurations.save()
#         Configurations._configs = None
#
#     @classmethod
#     def _save(cls, configs: dict) -> None:
#         with open(cls._current_config_file, 'w') as config_file:
#             yaml.safe_dump(configs, config_file, default_flow_style=False)
#
#     @classmethod
#     def init_file(cls, values=None) -> None:
#         cls._current_config_file.touch()
#         values = values or cls.defaults
#         Configurations._save(values)
#
#     @classmethod
#     def change_conf(cls, conf: str, value) -> None:
#         if Configurations._configs is None:
#             Configurations.init()
#         Configurations._configs[conf] = cls.purify_value(value)
#
#     @classmethod
#     def purify_value(cls, value: str) -> str | bool | int | list:
#         if isinstance(value, (list, int, bool)):
#             return value
#         if value.isnumeric():
#             return int(value)
#
#         true_values = ('true', '1')
#         false_values = ('false', '0')
#         if value.lower() in chain(true_values, false_values):
#             return value.lower() in true_values
#         return value
#
#     @classmethod
#     def set_conf(cls, conf: str, value) -> None:
#         '''
#         Alias for change_conf
#         '''
#         cls.change_conf(conf, value)
#
#     @classmethod
#     def get_all_configs(cls) -> dict[str, Any]:
#         return dict(cls._configs)
#
#     @classmethod
#     def get_conf(cls, name: str) -> list | str | bool | int:  # TODO: think of default configs and setting them in case of new features
#         if name not in Configurations._configs:
#             cls._set_default_value_for(name)
#         return Configurations._configs[name]
#
#     @classmethod
#     def _set_default_value_for(cls, conf: str) -> None:
#         default = cls._get_default_value_for(conf)
#         if default is not None:
#             cls._configs[conf] = default
#
#     @classmethod
#     def _get_default_value_for(cls, conf: str) -> Any:
#         return cls.defaults[conf] if conf in cls.defaults else None
#
#     @classmethod
#     def add_langs(cls, *arguments: str) -> None:
#         langs = Configurations.get_saved_languages()
#         not_saved, saved = partition(lambda lang: lang in langs, arguments)
#         for lang in saved:
#             TranslationPrinter.out(ConfigMessages.LANGUAGE_IN_SAVED.format(lang))
#         for lang in not_saved:
#             langs.append(lang)
#
#     @classmethod
#     def remove_langs(cls, *arguments: str):
#         langs = cls.get_saved_languages()
#         not_saved, saved = partition(lambda lang: lang in langs, arguments)
#         for lang in saved:
#             langs.remove(lang)
#         for lang in not_saved:
#             TranslationPrinter.out(ConfigMessages.LANGUAGE_NOT_IN_SAVED.format(lang))
#
#     @classmethod
#     def get_default_translation_mode(cls) -> str:
#         return cls.get_conf(C.DEFAULT_MODE_LONG_FLAG)
#
#     @classmethod
#     def get_saved_languages(cls) -> list[str]:
#         return cls.get_conf(C.LANGS_SHOW_LONG_FLAG)
#
#     @classmethod
#     def get_from_language(cls) -> str:
#         return cls.get_nth_saved_language(1)
#
#     @classmethod
#     def get_nth_saved_language(cls, index: int, *to_skips: str) -> str:
#         langs = cls.load_config_languages(*to_skips)
#         nth_lang = nth(langs, index)
#         return nth_lang
#
#     @classmethod
#     def get_adjustment_method(cls) -> str:
#         return cls.get_conf(C.LAYOUT_ADJUSTMENT_METHOD_LONG_FLAG)
#
#     @classmethod
#     def get_adjustment_lang(cls) -> str:
#         return cls.get_conf(C.LAYOUT_ADJUSTMENT_LANG_LONG_FLAG)
#
#     @classmethod
#     def load_config_languages_by_limit(cls, *to_skips: str, limit=None) -> Iterable[str]:
#         if limit is None:
#             limit = int(Configurations.get_conf(C.LANG_LIMIT_LONG_FLAG))
#         langs = cls.load_config_languages(*to_skips)
#         return islice(langs, limit)
#
#     @classmethod
#     def load_config_languages(cls, *to_skips: str) -> Iterable[str]:
#         langs: list = Configurations.get_conf(C.LANGS_SHOW_LONG_FLAG)[:]
#         return filter(lambda lang: lang not in to_skips, langs)
#
#     @classmethod
#     def change_last_used_languages(cls, *langs: str) -> None:
#         languages: list[str] = Configurations.get_saved_languages()
#         for lang in reversed(langs):
#             if lang in languages:
#                 languages.remove(lang)
#                 languages.insert(0, lang)
#         Configurations.change_conf(C.LANGS_SHOW_LONG_FLAG, languages)
#


class ConfLoader:
    @classmethod
    def load(cls, path: str | Path) -> Box:
        match Path(path).suffix:
            case '.yaml' | '.yml': return cls.load_yaml(path)
            case '.toml': return cls.load_toml(path)
            case _: raise ValueError(f'Unsupported file format: {path}')

    @classmethod
    def load_yaml(cls, path: str | Path) -> Box:
        import yaml
        with open(path, 'r') as f:
            return Box(yaml.safe_load(f))

    @classmethod
    def load_toml(cls, path: str | Path) -> Box:
        import toml
        with open(path, 'r') as f:
            return Box(toml.load(f))
