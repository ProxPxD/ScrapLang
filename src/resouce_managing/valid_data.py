import logging
from dataclasses import replace, dataclass, asdict
from functools import cache
from pathlib import Path
from typing import Iterable, Sequence, Callable, Collection

import pandas as pd
import pydash as _
from box import Box
from pandas import DataFrame
from pandas.core.groupby import DataFrameGroupBy
from pydantic import BaseModel, field_validator, ConfigDict
from pydash import chain as c

from .file import FileMgr
from ..constants import supported_languages, preinitialized
from ..context import Context
# from ..lang_detecting.preprocessing.data import LSC # TODO: fix imports
from ..scrapping import Outcome, MainOutcomeKinds as Kinds
from ..scrapping.wiktio.parsing import WiktioResult


@preinitialized
@dataclass
class ValidDataColumns:
    LANG: str = 'lang'
    WORD: str = 'word'
    DIALECT: str = 'dialect'
    PRONUNCIATIONS: str = 'pronunciations'
    FEATURES: str = 'features'

VDC = ValidDataColumns

class ValidArgs(BaseModel):
    model_config = ConfigDict(extra='forbid')

    lang: list[str]
    word: list[str]
    dialect: str
    pronunciations: list[str]
    features: list[str]

    @field_validator(VDC.LANG, mode='before')
    @classmethod
    def val_langs(cls, langs: list[str]):  # Truly not-needed, they have to be supported at this point
        supported = set(supported_languages.keys())
        actual = set(langs)
        if invalid := actual - supported:
            raise ValueError(f'"{list(invalid)}" languages are not supported')
        return langs


class ValidDataMgr:
    def __init__(self, conf_file: Path | str, context: Context, n_parsed: int = 32):
        self.context = context
        self._valid_data_file_mgr = FileMgr(conf_file)
        self._n_parsed: int = n_parsed

    def gather(self, scrap_results: Iterable[Outcome]) -> bool:
        logging.debug('Searching something not-yet-gathered')
        success_results = [replace(sr, args=Box(sr.args, default_box=True)) for sr in scrap_results if sr.is_success()]
        cols = list(asdict(ValidDataColumns).values())
        success_data = DataFrame(c().concat(
                self._gather_for_from_main_translations(success_results),
                self._gather_for_lang_data(success_results),
            list(self._gather_for_wiktio(success_results)),
            ).map(c().concat([None]*len(cols)).take(len(cols)))([]),
            columns=cols,
        )
        # success_data = success_data[success_data[VDC.LANG] != self.]  # TODO: Fix for gathering only supported languages
        if not success_data.empty:
            logging.debug('Found potential new data for gathering')
            old = self._valid_data_file_mgr.load()
            valid_data = pd.concat([old, success_data], ignore_index=True)
            valid_data = self._merge_matching(valid_data)
            valid_data.sort_values(by=cols[:2], inplace=True)
            self._valid_data_file_mgr.save(valid_data.drop_duplicates())
            return True
        return False

    def is_arg_set_valid(self, kinds: Collection[str], lang_arg: str) -> Callable[[Outcome], bool]:
        return _.over_every([
            c().get('kind').apply(kinds.__contains__),
            c().get(f'args.{lang_arg}').apply(self.context.all_langs.__contains__),
        ])

    def _gather_for_lang_data(self, scrap_results: Iterable[Outcome]) -> list[Sequence[str]]:
        kinds = (Kinds.DEFINITION, Kinds.INFLECTION,)
        args = (lang_arg:='lang', 'word')
        return c(scrap_results).filter(self.is_arg_set_valid(kinds, lang_arg)).map(c().get('args').at(*args)).value()

    def _gather_for_from_main_translations(self, scrap_results: Iterable[Outcome]) -> list[Sequence[str]]:
        kinds = (Kinds.MAIN_TRANSLATION, )
        args = (lang_arg:='from_lang', 'word')
        return c(scrap_results).filter(self.is_arg_set_valid(kinds, lang_arg)).map(c().get('args').at(*args)).value()

    def _gather_for_wiktio(self, scrap_results: Iterable[Outcome]) -> Iterable[Sequence[str]]:
        kinds = (Kinds.WIKTIO, )
        wiki_outcomes: list[Outcome] = c(scrap_results).filter(c().get('kind').apply(kinds.__contains__)).value()
        for outcome in wiki_outcomes:
            wiki: WiktioResult = outcome.results
            lang, word = _.at(outcome.args, 'lang', 'word')
            if lang not in self.context.all_langs:
                continue
            for meaning in wiki.meanings:
                for pronunciation in meaning.pronunciations or wiki.pronunciations:
                    dialect = pronunciation.name
                    ipas = ':'.join(pronunciation.ipas)
                    yield lang, word, dialect, ipas, str(meaning.rel_data)


    @classmethod
    def _merge_matching(cls, df: DataFrame) -> DataFrame:
        def process_group(group: DataFrameGroupBy):
            other_columns = group.columns[2:]
            notna = group[other_columns].notna()
            if notna.any(axis=1).any():
                return group[notna.all(axis=1).values]
            else:
                return group

        return df.groupby([VDC.LANG, VDC.WORD]).apply(process_group).reset_index(drop=True)

    def remove_entries_of_lang(self, lang: str) -> None:
        valid_data: DataFrame = self._valid_data_file_mgr.load()
        cleaned_valid_data = valid_data[valid_data['lang'] != lang].drop_index()
        self._valid_data_file_mgr.save(cleaned_valid_data)
