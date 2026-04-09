import logging
import operator as op
from collections import OrderedDict
from collections.abc import Collection
from itertools import combinations

from pandas import DataFrame
from pydash import chain as c
from pydash import flow, spread

from src.constants import Paths
from src.lang_detecting.advanced_detecting import colutils
from src.lang_detecting.advanced_detecting.conf import Data
from src.lang_detecting.preprocessing.data import LSC
from src.resouce_managing.file import FileMgr
from src.resouce_managing.valid_data import VDC

Kind = Vocab = Target = str
KindToVocab = OrderedDict[Kind, list[Vocab]]
KindToTargets = OrderedDict[Kind, list[Target]]
KindToTokensTargets = OrderedDict[Kind, KindToVocab | KindToTargets]
SpecialGroup = OrderedDict[str, Vocab]  # Default
KindToSpecialGroup = OrderedDict[Kind, SpecialGroup]
TokenizedKindToGroupedVocab = OrderedDict[Kind, OrderedDict[str, list[int]]]


class ModelIOMgr:
    def __init__(self) -> None:
        self.model_io = FileMgr(Paths.MODEL_IO_FILE, create_if_not=True)
        self.data_conf = Data()

    @classmethod
    def filter_any_shared_chars(cls, chars_group: Collection[str]) -> str:
        combineds = c(chars_group).apply(lambda cs: combinations(map(set, cs), 2)) # type: ignore[arg-type]
        any_shareds = combineds.map(spread(op.and_))
        all_any_shareds = any_shareds.reduce(lambda a, b: a | b, set())
        as_strs = all_any_shareds.to_list().sort().join()
        return as_strs.value()

    def generate_model_io(self, lang_script: DataFrame, data: DataFrame) -> None:
        kinds_to_vocab_classes = self.extract_kinds_to_vocab_classes(lang_script, data)
        self.update_model_io_if_needed(kinds_to_vocab_classes)

    def extract_kinds_to_vocab_classes(self, lang_script: DataFrame, data: DataFrame) -> KindToTokensTargets:
        # data
        m_unmapped = ~data[VDC.IS_MAPPED]
        m_long_enough = data[VDC.WORD].str.len() >= self.data_conf.word.len_thresh
        lang_to_words = data[m_unmapped & m_long_enough].reset_index()
        m_enough_words = lang_to_words.groupby(VDC.LANG)[VDC.WORD].transform('count') >= self.data_conf.min_n_samples
        qualified_langs = set(lang_to_words[m_enough_words][VDC.LANG])

        #script_lang
        lang_script = lang_script[lang_script[VDC.LANG].isin(qualified_langs)]
        script_lang = lang_script.explode(LSC.SCRIPTS).rename(columns={LSC.SCRIPTS: LSC.SCRIPT})
        sclc = script_common_langs_chars = (script_lang.groupby(LSC.SCRIPT, as_index=False).agg({  # noqa: F841
            LSC.LANG: flow(sorted, list),  # type: ignore[arg-type]
            LSC.CHARS: self.filter_any_shared_chars,
        }).rename(columns={LSC.LANG: LSC.LANGS}))
        sclc = sclc[(sclc[LSC.LANGS].map(len) > 1) & (sclc[LSC.CHARS].map(len) > 0)].reset_index(drop=True)
        shary_script = OrderedDict([
            (row[LSC.SCRIPT], OrderedDict([
                (LSC.LANGS, row[LSC.LANGS]),
                (LSC.CHARS, ''.join(row[LSC.CHARS])),
            ])) for idx, row in sclc.iterrows()
        ])
        return shary_script

    def update_model_io_if_needed(self, kinds_to_vocab_classes: KindToTokensTargets) -> None:
        kinds_to_vocab_classes = colutils.order_dict_to_dict(kinds_to_vocab_classes)
        old_script_langs = self.model_io.load()
        if old_script_langs != kinds_to_vocab_classes:
            logging.debug('Updating model IO')
            self.model_io.save(kinds_to_vocab_classes) # TODO: Uncomment after retraining functionality
            # raise ValueError("Model requires retraining and that's unsupported and unhandled for now")

    @classmethod
    def enhance_tokens(cls, kind_to_vocab: KindToVocab, tokens: list[Vocab]) -> KindToVocab:
        return OrderedDict([(kind, tokens + list(vocab)) for kind, vocab in kind_to_vocab.items()])

class KindToMgr:
    @classmethod
    def separate_kinds_tos(cls, kinds_to_tokens_targets: KindToTokensTargets) -> tuple[KindToVocab, KindToTargets]:
        kinds_to_vocab, kinds_to_targets = OrderedDict(), OrderedDict()
        for kind, vc in kinds_to_tokens_targets.items():
            kinds_to_vocab[kind] = list(vc[LSC.CHARS])
            kinds_to_targets[kind] = vc[LSC.LANGS]
        return kinds_to_vocab, kinds_to_targets

