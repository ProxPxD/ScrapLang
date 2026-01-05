import logging
import operator as op
from itertools import combinations
from typing import Collection

import pydash as _
from pydash import chain as c
from pydash import flow, spread

from src.constants import Paths
from src.lang_detecting.preprocessing.data import LSC
from src.resouce_managing.file import FileMgr

Kind = Vocab = Class = str
KindToVocab = dict[Kind, Vocab]
KindToClasses = dict[Kind, list[Class]]
KindToVocabClasses = dict[Kind, KindToVocab | KindToClasses]
GroupedVocab = dict[str, Vocab]  # Default
KindToGroupedVocab = dict[Kind, GroupedVocab]

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

    def extract_kinds_to_vocab_classes(self, lang_script) -> KindToVocabClasses:
        script_lang = lang_script.explode(LSC.SCRIPTS).rename(columns={LSC.SCRIPTS: LSC.SCRIPT})
        sclc = script_common_langs_chars = (script_lang.groupby(LSC.SCRIPT, as_index=False).agg({
            LSC.LANG: flow(sorted, list),
            LSC.CHARS: self.filter_any_shared_chars
        }).rename(columns={LSC.LANG: LSC.LANGS}))
        sclc = sclc[(sclc[LSC.LANGS].map(len) > 1) & (sclc[LSC.CHARS].map(len) > 0)].reset_index(drop=True)
        shary_script = {row[LSC.SCRIPT]: {LSC.LANGS: row[LSC.LANGS], LSC.CHARS: ''.join(row[LSC.CHARS])} for idx, row in
                        sclc.iterrows()}
        return shary_script

    def update_model_io_if_needed(self, kinds_to_vocab_classes: KindToVocabClasses) -> None:
        old_script_langs = self.model_io.load()
        if old_script_langs != kinds_to_vocab_classes:
            logging.debug(f'Updating model IO')
            self.model_io.save(kinds_to_vocab_classes) # TODO: Uncomment after retraining functionality
            # raise ValueError("Model requires retraining and that's unsupported and unhandled for now")

    @classmethod
    def separate_kinds_tos(cls, kinds_to_vocab_classes: KindToVocabClasses) -> tuple[KindToVocab, KindToClasses]:
        kinds_to_vocab = _.map_values(kinds_to_vocab_classes, c().get(LSC.CHARS))
        kinds_to_classes = _.map_values(kinds_to_vocab_classes, c().get(LSC.LANGS))
        return kinds_to_vocab, kinds_to_classes

    @classmethod
    def group_up_kinds_to_vocab(cls, kinds_to_vocab: KindToVocab) -> KindToGroupedVocab:
        kind_grouped = {}
        for kind, vocab in kinds_to_vocab.items():
            group_up_kind_vocab = getattr(cls, f'group_up_{kind.lower()}_vocab', lambda ts: {})
            kind_grouped[kind] = grouped = group_up_kind_vocab(vocab)
            grouped[ALL] = vocab
        return kind_grouped

    @classmethod
    def group_up_latn_vocab(cls, vocab: str) -> GroupedVocab:
        enhance_with_upper = lambda l: [l, l.upper()]
        grouped = {
            'letter': c(vocab).filter(str.isalpha).join().value(),
            'upper': c(vocab).filter(str.isupper).join().value(),
            'vowels': c(vocab).intersection(list('aeiouy')).flat_map(enhance_with_upper).join().value(),
            'consonants': c(vocab).intersection(list('bcdfghjklmnpqrstvwxyz')).flat_map(enhance_with_upper).join().value(),
        }
        return grouped

    @classmethod
    def group_up_cyrl_vocab(cls, vocab: str) -> GroupedVocab:
        enhance_with_upper = lambda l: [l, l.upper()]
        grouped = {
            'letter': c(vocab).filter(str.isalpha).join().value(),
            'upper': c(vocab).filter(str.isupper).join().value(),
            'vowels': c(vocab).intersection(list('аоуэыиеёєяюї')).flat_map(enhance_with_upper).join().value(),
            'consonants': c(vocab).intersection(list('бвгґджзйклмнпрстфхцчшщ')).flat_map(enhance_with_upper).join().value(),
            'function': c(vocab).intersection(list('ьъ')).flat_map(enhance_with_upper).join().value(),
        }
        return grouped
