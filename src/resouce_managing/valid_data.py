import logging
from dataclasses import replace, dataclass, asdict
from pathlib import Path
from typing import Iterable, Sequence, Callable, Collection, Sized

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
    IS_MAPPED: str = 'is_mapped'
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
        self.valid_data_file_mgr = FileMgr(conf_file)
        self._n_parsed: int = n_parsed

    @property
    def data(self) -> DataFrame:
        return self.valid_data_file_mgr.content

    def gather(self, scrap_results: Iterable[Outcome]) -> bool:
        logging.debug('Searching something not-yet-gathered')
        success_results = [replace(sr, args=Box(sr.args, default_box=True)) for sr in scrap_results if sr.is_success()]
        cols = list(asdict(ValidDataColumns).values())
        success_data = DataFrame(c(success_results).apply(_.over([
                self._gather_for_from_main_translations,
                self._gather_for_lang_data,
                self._gather_for_wiktio,
            ])).map(list).flatten().map(c().concat([None]*len(cols)).take(len(cols))).value(),
            columns=cols,
        )
        if not success_data.empty:
            logging.debug('Found potential new data for gathering')
            old = self.valid_data_file_mgr.load().sort_values(by=cols[:2]).convert_dtypes()
            valid_data = pd.concat([old, success_data], ignore_index=True)
            valid_data = self._merge_matching(valid_data)
            valid_data = valid_data.sort_values(by=cols[:2]).drop_duplicates().reset_index(drop=True).convert_dtypes()
            if not valid_data.equals(old):
                self.valid_data_file_mgr.save(valid_data)
                return True
        return False

    def is_arg_set_valid(self, kinds: Collection[str], lang_arg: str) -> Callable[[Outcome], bool]:
        return lambda o: _.over_every([
            c().get('kind').apply(kinds.__contains__),
            c().get(f'args.{lang_arg}').apply(self.context.langs.__contains__),
            lambda o: isinstance(o.results, Sized) and len(o.results) > 1 or isinstance(o.results, DataFrame)
        ])(o)

    def _gather_for_lang_data(self, scrap_results: Iterable[Outcome]) -> Iterable[Sequence[str]]:
        kinds = (Kinds.DEFINITION, Kinds.INFLECTION,)
        args = (lang_arg:='lang', 'word')
        for lang, word in c(scrap_results).filter(self.is_arg_set_valid(kinds, lang_arg)).map(c().get('args').at(*args)).value():
            yield lang, word, False
            if word in self.context.words and self.context.is_mapped(word):
                yield lang, self.context.get_unmmapped(word), True

    def _gather_for_from_main_translations(self, scrap_results: Iterable[Outcome]) -> Iterable[Sequence[str]]:
        kinds = (Kinds.MAIN_TRANSLATION, )
        args = (lang_arg:='from_lang', 'word')
        for lang, word in c(scrap_results).filter(self.is_arg_set_valid(kinds, lang_arg)).map(c().get('args').at(*args)).value():
            yield lang, word, False
            if self.context.is_mapped(word):
                yield lang, self.context.get_unmmapped(word), True

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
                    yield lang, word, False, dialect, ipas, str(meaning.rel_data)
                    if self.context.is_mapped(word):
                        yield lang, self.context.get_unmmapped(word), True


    @classmethod
    def _merge_matching(cls, df: DataFrame) -> DataFrame:
        def process_group(group: DataFrameGroupBy):
            if group.shape[0] <= 1:
                return group
            other_columns_i = group.columns.to_list().index(VDC.DIALECT)
            other_columns = group.columns[other_columns_i:]
            notna = group[other_columns].notna()
            if notna.any(axis=1).any():
                return group[notna.all(axis=1).values]
            else:
                return group

        return df.groupby([VDC.LANG, VDC.WORD, VDC.IS_MAPPED]).apply(process_group).reset_index(drop=True)

    def remove_entries_of_lang(self, lang: str) -> None:
        valid_data: DataFrame = self.valid_data_file_mgr.load()
        cleaned_valid_data = valid_data[valid_data[VDC.LANG] != lang].drop_index()
        self.valid_data_file_mgr.save(cleaned_valid_data)
