from __future__ import annotations

import operator as op
from typing import TYPE_CHECKING, Any, Optional

import pydash as _
from GlotScript import sp
from pandas import DataFrame, Series
from pydash import chain as c
from pydash import flow

from src.lang_detecting.advanced_detecting.data.preprocessing.core.consts import Cols
from src.lang_detecting.advanced_detecting.data.preprocessing.core.filtering import ColFilter
from src.lang_detecting.advanced_detecting.data.preprocessing.core.step import Resources, SeqStep, SimpleStep, Step
from src.lang_detecting.advanced_detecting.data.preprocessing.core.transforming import RowTransform
from src.lang_detecting.advanced_detecting.data.preprocessing.specific.augmenting import Augmenter
from src.lang_detecting.advanced_detecting.data.preprocessing.specific.grouper import Grouper
from src.lang_detecting.advanced_detecting.tokenizer import MultiKindTokenizer, Tokens
from src.resouce_managing.valid_data import VDC

if TYPE_CHECKING:
    from src.lang_detecting.advanced_detecting.conf import Conf
    from src.lang_detecting.advanced_detecting.model_io_mging import KindToTargets


class PreprocessorFactory:
    def __init__(self,
            tokenizer: MultiKindTokenizer,
            conf: Conf,
            kinds_to_targets: KindToTargets,
        ):
        """
        put - adds a column
        """
        self.resources = Resources(tokenizer=tokenizer, conf=conf)
        self.kind_to_targets = kinds_to_targets
        self.init_preprocessor: Optional[Step] = None
        self.train_preprocessor: Optional[Step] = None
        self.val_preprocessor: Optional[Step] = None
        self._create_preprocessors()

    @property
    def tokenizer(self) -> MultiKindTokenizer:
        return self.resources.tokenizer

    def _create_preprocessors(self) -> None:
        """
        Naming Convention
        put - adds a column
        """
        resources = self.resources
        tokenizer, conf = self.resources.tokenizer, self.resources.conf

        is_not_mapped = ColFilter(col=VDC.IS_MAPPED, mask_func=lambda col: ~col, precond=lambda df: VDC.IS_MAPPED in df.columns)
        targets = _.flatten(self.kind_to_targets.values())
        is_valid_target = ColFilter(col=VDC.LANG, mask_func=lambda col: col.isin(targets))
        uniq_lang_word = ColFilter(output_cols=[VDC.LANG, VDC.WORD])
        uniq_lang_word_kind = ColFilter(output_cols=[VDC.LANG, VDC.WORD, Cols.KIND])
        ensure_word_seq = RowTransform(col=VDC.WORD, func=tuple)
        ensure_bos = RowTransform(col=VDC.WORD, func=lambda w: (Tokens.BOS, *tuple(w), Tokens.BOS) if Tokens.BOS not in w else tuple(w))
        pad_word = SeqStep(ensure_word_seq, ensure_bos)

        def get_kind_save(word: list[str]) -> Optional[str]:
            chars = [ch for ch in word if len(ch) == 1]
            details: dict = sp(''.join(chars))[-1]['details']
            return next(iter(details.keys())) if details else None
        is_with_kind = ColFilter(col=Cols.KIND, mask_func=flow(DataFrame.isna, op.inv))
        is_multilang_kind = ColFilter(lambda df: df.groupby(Cols.KIND)[VDC.LANG].transform('nunique') > 1)
        put_kind_safe = RowTransform(to_col=Cols.KIND, from_col=VDC.WORD, func=get_kind_save, post_func=is_with_kind, precond=lambda df: Cols.KIND not in df.columns)
        put_tokens = RowTransform(to_col=Cols.TOKENS, func=lambda row: tokenizer.tokenize_word(row[VDC.WORD], row[Cols.KIND]))
        put_specs = RowTransform(to_col=Cols.SPECS, func=lambda row: tokenizer.tokenize_spec_groups(row[VDC.WORD], row[Cols.KIND]))
        put_tokens_specs = SeqStep(put_tokens, put_specs)
        put_kind_tokens_specs = SeqStep(put_kind_safe, put_tokens_specs)
        put_decode = RowTransform(to_col=Cols.DECODE, func=lambda row: tokenizer.detokenize_word(row[Cols.TOKENS], row[Cols.KIND]))
        put_len = RowTransform(to_col=Cols.LEN, from_col=Cols.TOKENS, func=len)
        put_n_uniq = RowTransform(to_col=Cols.N_UNIQ, from_col=Cols.DECODE, func=c().filter(Tokens.UNK.__contains__).apply(len))
        expand_rows = SeqStep(put_decode, put_len, put_n_uniq)

        #self.chunker = Chunker(size=5, to_col=VDC.WORD, from_col=VDC.WORD)

        def constrain(data: DataFrame) -> Series[Any]:
            w = conf.data.word
            offset = 2
            m_uniq = data[Cols.N_UNIQ] > 0
            m_enough_non_uniq = data[Cols.LEN] - data[Cols.N_UNIQ] >= w.n_non_uniq + offset
            m_right_ratio = (data[Cols.LEN] - 5 - offset) / data[Cols.N_UNIQ] >= w.len_to_uniq_ratio
            m_long = data[Cols.LEN] >= w.len_thresh + offset
            return m_long & (~m_uniq | m_enough_non_uniq & m_right_ratio)

        enough_uniq = ColFilter(mask_func=constrain)

        def enough_count_func(data: DataFrame) -> Series[Any]:
            class_counts = data[VDC.LANG].value_counts()
            numerous_enough = class_counts[class_counts >= conf.data.min_record_n_thresh].index
            return data[VDC.LANG].isin(numerous_enough)

        enough_count = ColFilter(mask_func=enough_count_func)
        filter_out_bad_data = SeqStep(enough_uniq, enough_count)
        self.group = Grouper()

        self.init_preprocessor = SeqStep(is_not_mapped, is_valid_target, uniq_lang_word, pad_word, put_kind_safe, is_multilang_kind, put_tokens_specs, expand_rows, filter_out_bad_data)
        ensure_dropped = SimpleStep(lambda df: df.drop(columns=[Cols.DECODE, Cols.N_UNIQ], errors='ignore'))
        augment_filter = SeqStep(Augmenter(resources=resources), uniq_lang_word_kind, put_kind_tokens_specs, expand_rows, filter_out_bad_data,
                                 precond=_.constant(conf.data.augment.is_augmenting))
        self.train_preprocessor = SeqStep(augment_filter)
        self.val_preprocessor = SeqStep(ensure_dropped)

