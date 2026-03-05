from __future__ import annotations

import operator as op

import pydash as _
from GlotScript import sp
from pandas import DataFrame
from pydash import chain as c, flow

from src.lang_detecting.advanced_detecting.conf import Conf
from src.lang_detecting.advanced_detecting.data.preprocessing.core.consts import Cols
from src.lang_detecting.advanced_detecting.data.preprocessing.core.filtering import ColFilter
from src.lang_detecting.advanced_detecting.data.preprocessing.core.step import Resources, SeqStep, SimpleStep, Step
from src.lang_detecting.advanced_detecting.data.preprocessing.core.transforming import RowTransform
from src.lang_detecting.advanced_detecting.data.preprocessing.specific.augmenting import Augmenter
from src.lang_detecting.advanced_detecting.data.preprocessing.specific.grouper import Grouper
from src.lang_detecting.advanced_detecting.tokenizer import MultiKindTokenizer, Tokens
from src.resouce_managing.valid_data import VDC


class PreprocessorFactory:
    def __init__(self,
            tokenizer: MultiKindTokenizer,
            conf: Conf,
        ):
        """
        put - adds a column
        """
        self.resources = Resources(tokenizer=tokenizer, conf=conf)
        self.init_preprocessor: Step | None = None
        self.train_preprocessor: Step | None = None
        self.val_preprocessor: Step | None = None
        self._create_preprocessors()

    def _create_preprocessors(self) -> None:
        """
        Naming Convention
        put - adds a column
        """
        resources = self.resources
        tokenizer, conf = self.resources.tokenizer, self.resources.conf

        is_not_mapped = ColFilter(col=VDC.IS_MAPPED, mask_func=lambda col: ~col, precond=lambda df: VDC.IS_MAPPED in df.columns)
        uniq_lang_word = ColFilter(output_cols=[VDC.LANG, VDC.WORD])
        uniq_lang_word_kind = ColFilter(output_cols=[VDC.LANG, VDC.WORD, Cols.KIND])

        def get_kind(word: str):
            details: dict = sp(''.join(word))[-1]['details']
            return next(iter(details.keys())) if details else None

        ensure_word_seq = RowTransform(col=VDC.WORD, func=tuple)
        strip_bos = RowTransform(col=VDC.WORD, func=c().apply(tuple).reject(Tokens.BOS.__eq__))
        enclose_with_bos = RowTransform(col=VDC.WORD, func=lambda w: tuple([Tokens.BOS, *tuple(w), Tokens.BOS]))
        ensure_bos = SeqStep(strip_bos, enclose_with_bos)
        is_with_kind = ColFilter(col=Cols.KIND, mask_func=flow(DataFrame.isna, op.inv))
        put_kind = RowTransform(to_col=Cols.KIND, from_col=VDC.WORD, func=get_kind, post_func=is_with_kind)
        ensure_kind_safe_for_bos = SeqStep(strip_bos, put_kind, enclose_with_bos, precond=lambda df: Cols.KIND not in df.columns)
        put_tokens = RowTransform(to_col=Cols.TOKENS, func=lambda row: tokenizer.tokenize_input(row[VDC.WORD], row[Cols.KIND]))
        put_specs = RowTransform(to_col=Cols.SPECS,
                                 func=lambda row: tokenizer.tokenize_spec_groups(row[VDC.WORD], row[Cols.KIND]))
        form_model_form = SeqStep(ensure_word_seq, ensure_kind_safe_for_bos, put_tokens, put_specs)
        put_decode = RowTransform(to_col=Cols.DECODE, func=lambda row: tokenizer.detokenize_input(row[Cols.TOKENS], row[Cols.KIND]))
        put_len = RowTransform(to_col=Cols.LEN, from_col=Cols.TOKENS, func=len)
        put_n_uniq = RowTransform(to_col=Cols.N_UNIQ, from_col=Cols.DECODE, func=c().filter(tokenizer.unknown.__contains__).apply(len))
        expand_rows = SeqStep(ensure_bos, put_decode, put_len, put_n_uniq)

        def constrain(data) -> DataFrame:
            w = conf.data.word
            m_uniq = data[Cols.N_UNIQ] > 0
            m_enough_non_uniq = data[Cols.LEN] - data[Cols.N_UNIQ] >= w.n_non_uniq
            m_right_ratio = (data[Cols.LEN] - 5) / data[Cols.N_UNIQ] >= w.len_to_uniq_ratio
            m_long = data[Cols.LEN] >= w.len_thresh
            return m_long & (~m_uniq | m_enough_non_uniq & m_right_ratio)

        enough_uniq = ColFilter(mask_func=constrain)

        def enough_count_func(data: DataFrame) -> DataFrame:
            class_counts = data[VDC.LANG].value_counts()
            numerous_enough = class_counts[class_counts >= conf.data.min_record_n_thresh].index
            return data[VDC.LANG].isin(numerous_enough)

        enough_count = ColFilter(mask_func=enough_count_func)
        filter_out_bad_data = SeqStep(enough_uniq, enough_count)
        self.group = Grouper()

        self.init_preprocessor = SeqStep(is_not_mapped, uniq_lang_word, form_model_form, expand_rows, filter_out_bad_data)
        ensure_dropped = SimpleStep(lambda df: df.drop(columns=[Cols.DECODE, Cols.N_UNIQ], errors='ignore'))
        augment_filter = SeqStep(Augmenter(resources=resources), uniq_lang_word_kind, form_model_form, expand_rows, filter_out_bad_data,
                                 precond=_.constant(conf.data.augment.is_augmenting))
        self.train_preprocessor = SeqStep(augment_filter)
        self.val_preprocessor = SeqStep(ensure_dropped)

