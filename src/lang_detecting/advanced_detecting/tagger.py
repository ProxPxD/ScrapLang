import math
import re
from math import floor, log2
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
def decom(base: int, n: float) -> tuple[int, int, int]:
    exp = floor(math.log(n, base))
    coef = n / base**exp
    return int(coef), base, exp

def depercent(n: float) -> int:
    return int(n*100)

@curry
def deexp(base: int, n: float) -> str:
    coef, base, exp = decom(base, n)
    return f'{coef}_e{base}_{exp}'

deexp10 = deexp(10)
deexp2 = deexp(2)


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
            'attn': expert.p_dropout,
            'emb': expert.p_dropout,
            'conv': expert.p_conv_dropout,
        }
        tags = [f'dropout/{place}/{int(100*p)}' for place, p in place_prob.items() if p]
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
            lr=[deexp10, float],
            weight_decay=[deexp10, float],
            gamma=[depercent, float],
            epochs=[int],
        )
        return [
            f'{param}/{trans(getattr(self.conf.train, param))}'
            for param, trans_s in params.items()
            for trans in _.to_list(trans_s)
        ]
