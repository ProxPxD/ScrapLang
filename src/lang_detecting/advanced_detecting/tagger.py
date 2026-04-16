import math
import re
from math import floor, log2
from os import rename
from typing import TYPE_CHECKING, Callable, Collection

import pydash as _
from pydash import chain as c
from pydash import flow
from toolz import curry

from src.lang_detecting.advanced_detecting.conf import Conf
from src.lang_detecting.advanced_detecting.model import Moe

if TYPE_CHECKING:
    from src.lang_detecting.advanced_detecting.model import Expert


def str_flat(array: Collection | str) -> list[str]:
    match array:
        case None: return []
        case str(): return [array]
        case _: return c(array).map_(str_flat).flatten().value()

TagS = None | str | Collection[str]

@curry
def decom(base: int, n: float) -> tuple[int, int]:
    exp = floor(math.log(n, base))
    coef = n / base**exp
    return int(coef), exp

def depercent(n: float) -> int:
    return int(n*100)

def get_decom_tag(*, base: int = 10, add_coef: bool = False, add_base: bool = True, skip_10: bool = True) -> Callable[[int], str]:
    def decom_tag(n: float) -> str:
        coef, exp = decom(base, n)
        is_skipping_10 = base == 10 and skip_10
        match add_base, is_skipping_10:
            case (True, False): base_str = f'{base}^'
            case (True, True): base_str = 'e'
            case _: base_str = ''
        match add_coef, is_skipping_10:
            case (True, False): coef_str = f'{coef}*'
            case (True, True): coef_str = f'{coef}'
            case _: coef_str = ''
        return f'{coef_str}{base_str}{exp}'
    return decom_tag

class Tagger:
    TAG_FUNC_PAT = re.compile(r'^_\w+_tags?$')

    def __init__(self, conf: Conf, moe: Moe):
        self.conf: Conf = conf
        self.moe: Moe = moe

    @classmethod
    def deltags(cls) -> TagS:
        return ['autodel']

    @property
    def tags(self) -> list[str]:
        tag_funcs: list[Callable[[], TagS]] = [getattr(self, name) for name in dir(self) if self.TAG_FUNC_PAT.fullmatch(name)]
        tags = str_flat([func() for func in tag_funcs])
        return tags

    def _augm_tags(self) -> TagS:
        return 'augmented' if self.conf.data.augment.is_augmenting else 'non-augmented'

    def _size_tags(self) -> TagS:
        match exp := flow(log2, floor)(self.conf.train.epochs):
            case _ if exp < 8: size = 'tiny'
            case 8: size = 'lil'
            case 9: size = 'mid'
            case 10: size = 'mid-big'
            case _ if exp > 10: size = 'big'
            case _: return None
        return f'size/{size}'

    def _dropout_tags(self) -> TagS:
        tags, expert = [], self.conf.expert
        place_prob = {
            'attn': expert.p_attn_dropout,
            'emb': expert.p_emb_dropout,
            'conv': expert.p_conv_dropout,
        }
        tags = [f'dropout/{place}/{p}' for place, p in place_prob.items()]
        return tags

    def _act_tags(self) -> TagS:
        expert: Expert = self.moe.experts[0]
        pref_kind = [
            ('act/conv', expert.hid_act.__class__.__name__.replace('ReLU', 'Relu')),
            ('pool/attn', expert.post_attn_pool_name),
        ]
        tags = [f'{pref}/{_.snake_case(kind)}' for pref, kind in pref_kind]
        return tags

    def _min_n_samples(self) -> TagS:
        return f'min_n_samples/{self.conf.data.min_n_samples}'

    def _smoothing_tag(self) -> TagS:
        smoothing = self.conf.train.smoothing
        if not smoothing.is_on:
            return None
        alpha = str(int(smoothing.alpha*100))
        return f'smoothing/{alpha}'

    def _train_tags(self) -> TagS:
        params = dict(
            lr=[get_decom_tag(add_coef=True)],
            weight_decay=[get_decom_tag()],
            gamma=float,
            epochs=[get_decom_tag(base=2), int],
        )
        return [
            f'{param}/{trans(getattr(self.conf.train, param))}'
            for param, trans_s in params.items()
            for trans in _.to_list(trans_s)
        ]

    def _weight_tags(self) -> TagS:
        params = dict(
            c_pos=float,
            freq_bias=float,
        )
        return [
            f'{param}/{trans(getattr(self.conf.weights, param))}'
            for param, trans_s in params.items()
            for trans in _.to_list(trans_s)
        ]

    def _conv_tags(self) -> TagS:
        e = self.conf.expert
        coef_tags = [
            f'c_{name}/{c(getattr(e, name)).join("_").value()}'
            for name in ('kernels', 'hidden_channels', 'paddings')
        ]
        return [
            f'n_conv/{len(e.kernels)}',
            *coef_tags,
        ]
