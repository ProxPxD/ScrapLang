import logging
import operator as op
from abc import ABC
from collections import OrderedDict
from itertools import combinations
from typing import Collection, Callable

from pydash import chain as c
import pydash as _
from pydash import flow, spread

from src.constants import Paths
from src.lang_detecting.advanced_detecting import colutils
from src.lang_detecting.preprocessing.data import LSC
from src.resouce_managing.file import FileMgr

Kind = Vocab = Class = str
KindToVocab = OrderedDict[Kind, Vocab]
KindToTargets = OrderedDict[Kind, list[Class]]
KindToTokensTargets = OrderedDict[Kind, KindToVocab | KindToTargets]
SpecialGroup = OrderedDict[str, Vocab]  # Default
KindToSpecialGroup = OrderedDict[Kind, SpecialGroup]
TokenizedKindToGroupedVocab = OrderedDict[Kind, OrderedDict[str, list[int]]]

ALL = 'all'


class ModelIOMgr:
    def __init__(self):
        self.model_io = FileMgr(Paths.MODEL_IO_FILE, create_if_not=True)

    @classmethod
    def filter_any_shared_chars(cls, chars_group: Collection[str]):
        combineds = c(chars_group).apply(lambda cs: combinations(map(set, cs), 2))
        any_shareds = combineds.map(spread(op.and_))
        all_any_shareds = any_shareds.reduce(lambda a, b: a | b, set())
        as_strs = all_any_shareds.to_list().sort().join()
        return as_strs.value()

    def extract_kinds_to_vocab_classes(self, lang_script) -> KindToTokensTargets:
        script_lang = lang_script.explode(LSC.SCRIPTS).rename(columns={LSC.SCRIPTS: LSC.SCRIPT})
        sclc = script_common_langs_chars = (script_lang.groupby(LSC.SCRIPT, as_index=False).agg({
            LSC.LANG: flow(sorted, list),
            LSC.CHARS: self.filter_any_shared_chars
        }).rename(columns={LSC.LANG: LSC.LANGS}))
        sclc = sclc[(sclc[LSC.LANGS].map(len) > 1) & (sclc[LSC.CHARS].map(len) > 0)].reset_index(drop=True)
        shary_script = OrderedDict([
            (row[LSC.SCRIPT], OrderedDict([
                (LSC.LANGS, row[LSC.LANGS]),
                (LSC.CHARS, ''.join(row[LSC.CHARS]))
            ])) for idx, row in sclc.iterrows()
        ])
        return shary_script

    def update_model_io_if_needed(self, kinds_to_vocab_classes: KindToTokensTargets) -> None:
        kinds_to_vocab_classes = colutils.order_dict_to_dict(kinds_to_vocab_classes)
        old_script_langs = self.model_io.load()
        if old_script_langs != kinds_to_vocab_classes:
            logging.debug(f'Updating model IO')
            self.model_io.save(kinds_to_vocab_classes) # TODO: Uncomment after retraining functionality
            # raise ValueError("Model requires retraining and that's unsupported and unhandled for now")


class KindToMgr:
    @classmethod
    def separate_kinds_tos(cls, kinds_to_tokens_targets: KindToTokensTargets) -> tuple[KindToVocab, KindToTargets]:
        kinds_to_vocab, kinds_to_targets = OrderedDict(), OrderedDict()
        for kind, vc in kinds_to_tokens_targets.items():
            kinds_to_vocab[kind] = vc[LSC.CHARS]
            kinds_to_targets[kind] = vc[LSC.LANGS]
        return kinds_to_vocab, kinds_to_targets

    @classmethod
    def map_kind_to_to_target_to_shared(cls, kinds_to_tokens_targets: KindToTokensTargets) -> dict[str, str]:
        return {
            target: tokens_targets[LSC.CHARS]
            for kind, tokens_targets in kinds_to_tokens_targets.items()
            for target in tokens_targets[LSC.LANGS]
        }
