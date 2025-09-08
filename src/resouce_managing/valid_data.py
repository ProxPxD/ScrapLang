from dataclasses import replace
from itertools import repeat, chain
from pathlib import Path
from typing import Iterable

import pandas as pd
import pydash as _
from box import Box
from pandas import DataFrame
from pydantic import BaseModel, field_validator
from pydash import chain as c

from .file import FileMgr
from ..constants import supported_languages
from ..context import Context
from ..scrapping import ScrapResult, MainScrapKinds
from ..scrapping.parsing import TranslationKind


class ValidArgs(BaseModel):

    lang: list[str]
    word: list[str]

    @field_validator('lang', mode='before')
    @classmethod
    def val_langs(cls, langs: list[str]):  # Truly not-needed, they have to be supported at this point
        supported = set(supported_languages.keys())
        actual = set(langs)
        if invalid := actual - supported:
            raise ValueError(f'"{list(invalid)}" languages are not supported')
        return langs



class ValidArgsMgr:
    def __init__(self, conf_file: Path | str, context: Context, n_parsed: int = 32):
        self._valid_data_mgr = FileMgr(conf_file)
        self.context: Context = context
        self._n_parsed: int = n_parsed

    def gather(self, scrap_results: Iterable[ScrapResult]) -> None:
        is_gatherable = lambda sr: sr.is_success() and sr.kind in MainScrapKinds.all() and sr.kind != MainScrapKinds.INDIRECT_TRANSLATION
        success_results = [replace(sr, args=Box(sr.args, default_box=True)) for sr in scrap_results if is_gatherable(sr)]
        success_data = DataFrame(_.concat(
            self._gather_for_from_langs(success_results),
            self._gather_for_langs(success_results),
            self._gather_for_to_langs(success_results)
        ), columns=['lang', 'word'])
        if not success_data.empty:
            valid_data = pd.concat([self._valid_data_mgr.load(as_box=False), success_data], ignore_index=True)
            valid_data.sort_values(by=['lang', 'word'], inplace=True)
            self._valid_data_mgr.save(valid_data.drop_duplicates())

    def _gather_for_langs(self, scrap_results: Iterable[ScrapResult]) -> list[list[str]]:
        return [_.at(sr.args, 'lang', 'word') for sr in scrap_results if sr.args.lang]

    def _gather_for_from_langs(self, scrap_results: Iterable[ScrapResult]) -> list[list[str]]:
        return [_.at(sr.args, 'from_lang', 'word') for sr in scrap_results if sr.args.from_lang]

    def _gather_for_to_langs(self, scrap_results: Iterable[ScrapResult]) -> list[list[str]]:
        return [[lang, trans.word] for sr in scrap_results for lang, trans in zip(repeat(sr.args.to_lang), sr.content) if sr.args.to_lang and trans.kind == TranslationKind.MAIN]