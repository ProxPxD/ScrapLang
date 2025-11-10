import shlex
from dataclasses import dataclass, field, replace
from itertools import product, permutations
from pathlib import Path
from typing import Iterable, Any, Sequence

import pydash as _
from pydash import chain as c
import pytest
import yaml
from box import Box
from more_itertools import windowed, unique

from src.app_managing import AppMgr
from src.context_domain import assume, indirect, gather_data, infervia, groupby
from testing.core import TCG
from testing.core.mocking import mocked_scrap, CallCollector

SYSTEM_PATH = Path(__file__).parent
TMP_DIR = SYSTEM_PATH / 'tmp'
TEST_CONF = TMP_DIR / 'test_conf.yaml'

TMP_DIR.mkdir(exist_ok=True)



@dataclass
class InputCase:
    input: Iterable[str] | str
    tags: Iterable[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    conf: Box = field(default_factory=Box)
    replacement: dict = field(default_factory=dict)

IC = InputCase

@dataclass
class TC:  # TODO: think of disabling queries
    descr: str
    input: Iterable[InputCase | str] | InputCase | str
    context: dict[str, Any] = field(default_factory=dict)
    tags: Iterable[str] = field(default_factory=list)
    conf: dict[str, Any] = field(default_factory=Box)

    replacement: dict = field(default_factory=dict)



@dataclass
class Tc:
    descr: str
    tags: Sequence[str]

    input: str
    conf: dict[str, Any]
    context: dict[str, Any]


class SystemTCG(TCG):

    color_confs = ('no', 'blue', dict(main='blue'), dict(main='yellow', pronunciation=[0, 255, 0]))
    mappings = dict(
        simple_single=dict(eo={'Cx': 'Ĉ', 'Gx': 'Ĝ', 'Hx': 'Ĥ', 'Jx': 'Ĵ', 'Sx': 'Ŝ', 'Ux': 'Ŭ', 'cx': 'ĉ', 'gx': 'ĝ', 'hx': 'ĥ', 'jx': 'ĵ', 'sx': 'ŝ', 'ux': 'ŭ'}),
        ordered=dict(ru=[{'lu': 'лю'}, {'u': 'у'}]),
        regex=dict(ru=[{'l([aeuo])': 'łi\1', 'li': 'łi'}, {'[ji]a': 'я', '[ji]e': 'е', '[ji]o': 'ё', '[ji]u': 'ю', 'j[''q]': 'й'}]),
    )
    single_confs = [
        *[dict(indirect=indirect_val) for indirect_val in indirect],
        *[dict(assume=assume_val) for assume_val in assume],
        *[dict(gather_data=gather_data_val) for gather_data_val in gather_data],
        *[dict(infervia=infervia_val) for infervia_val in infervia],
        *[dict(groupby=groupby_val) for groupby_val in groupby],
        *[dict(color=color_conf) for color_conf in color_confs],
        *[dict(mappings=mapping) for mapping in mappings.values()]
    ]

    perm_base = Box({
        'from_lang': 'en',
        'to_langs': ('pl', 'de'),
        'words': ('water', 'bass'),
    })
    multi_perm = Box({
        'langs': (perm_base.from_lang, *perm_base.to_langs),
        'args': (perm_base.from_lang, *perm_base.to_langs, *perm_base.words),
        **perm_base,
    })

    @classmethod
    def generate_tcs(cls) -> list:
        return [
        TC(
            descr='Exact allflag configless call',
            tags=['position', 'permutation'],
            input=_.map_(permutations({'-w Herr', '-f de', '-t pl'}, 3), ' '.join),
            context={
                'from_lang': 'de',
                'words': frozenset('Herr'),
                'to_langs': frozenset('pl'),
            }
        ),
        TC(
            descr=f'Conf Loading',
            tags=['conf', 'load'],
            input=[
                IC(
                    tags=[f'{next(iter(conf.keys()))}/{next(iter(conf.values()))}'],  # TODO: implement in TCG marker/tagger
                    input=f'-h',  # TODO think
                    context={f'_conf.{next(iter(conf.keys()))}': next(iter(conf.values()))},
                    conf=conf,
                ) for conf in cls.single_confs
            ],
        ),
        TC(
            descr=f'Conf Setting',
            tags=['conf', 'set'],
            input=[
                IC(
                    tags=[f'{next(conf.keys())}/{next(conf.values())}'],
                    input=f'set {next(conf.keys())} {next(conf.values())}',
                    context={f'_conf.{next(conf.keys())}': next(conf.values())},
                ) for conf in cls.single_confs if not isinstance(conf, (dict, list))
            ],
        ),
        TC(
            descr='Assume sunny resolution',
            input={
                'es de -w en de orden --assume lang',  # lang
                'en de --assume word',  # word
                'en de -w en de',  # Default
                '',
            },
            context={
                'from_lang': ('es'),
                'words': frozenset({'en', 'de'}),
                'to_langs': frozenset({'de'}),
            },
            conf=(just_langs_es_de_pl_en_conf := Box({
                'langs': ['es', 'de', 'pl', 'en'],
            })),
        ),
        TC(
            descr='Assume rainy resolution',
            input={
                'es pl de --assume lang'
            },
            context={
                'from_lang': ('es'),
                'to_langs': frozenset({'pl', 'de'})  # TODO: Error no word
            },
            conf=just_langs_es_de_pl_en_conf,
        ),
        TC(
            descr='Single translation of any arg placement resolution',
            tags={'permutation', 'position', 'single-from-lang', 'single-to-lang', 'single-word'},
            input={
                'Herr de pl',
                'de Herr pl',
                'de pl Herr',
            },
            context={
                'from_lang': 'de',
                'words': frozenset({'Herr'}),
                'to_langs': frozenset({'pl'}),
            },
            conf=(base_langs_es_de_pl_en_conf := Box(dict(**just_langs_es_de_pl_en_conf, assume='lang'))),
        ),
        TC(
            descr='Multiword from single lang to multilang translation of any arg placement resolution',
            tags={'permutation', 'position', 'multi-word', 'multi-to-lang', 'single-from-lang'},
            input={
                ' '.join(perm)
                for perm
                in permutations(cls.multi_perm.args, len(cls.multi_perm.args))
                if all(i < j for group in (cls.multi_perm.langs, cls.multi_perm.words) for i, j in windowed(map(perm.index, group), 2))
            },
            context={
                'from_lang': ('en'),
                'words': frozenset({'water', 'bass'}),
                'to_langs': frozenset({'pl', 'de'}),
            },
            conf=base_langs_es_de_pl_en_conf,
        ),
        TC(
            descr='Main Replacements',
            tags={'replacement'},
            input={
                'es pl orden conocer <WORD_FLAG> en de',
                'pl orden conocer -w en de <FROM_LANG> es',
                'es orden conocer -w en de <TO_LANG> pl',
            },
            replacement={
                'WORD_FLAG': ['--word', '--words', '-word', '-words'],
                'FROM_LANG': ['--from-lang', '--from', '-from'],
                'TO_LANG':   ['--to-lang', '--to-langs', '--to', '-to'],
            },
            context={
                'words': frozenset({'orden', 'conocer', 'en', 'de'}),
                'from_lang': ('es'),
                'to_langs': frozenset({'pl'}),
            },
            conf=base_langs_es_de_pl_en_conf,
        ),
        TC(
            descr='Inflection',
            tags=set(),
            input=[
                IC(
                    tags={'permutation', 'position'},
                    input=_.map_(permutations({'Herr', 'de', '-i'}, 3), ' '.join),
                    context=(herr_context := {
                        'from_lang': ('de'),
                        'inflection': True,
                        'words': ['Herr']
                    }),
                ),
                IC(
                    input='Herr de {INFL_FLAG}',
                    replacement={'INFL_FLAG': ['--inflection', '--infl', '-infl']},
                    context=herr_context,
                ),
                # TODO: extend with concrete inflection types and tables to test
            ],
            conf=base_langs_es_de_pl_en_conf,
        ),
        TC(
            descr='Definition',
            tags=set(),
            input=[
                IC(
                    tags={'permutation', 'position'},
                    input=_.map_(permutations({'bass', 'en', '-d'}, 3), ' '.join),
                    context=(bass_context := {
                        'from_lang': ('en'),
                        'definition': True,
                        'words': ['bass']
                    }),
                ),
                IC(
                    input='bass <DEF_FLAG> en',
                    replacement={
                        'DEF_FLAG': ['--definitions', '--definition', '-definitions', '-definition', '--def', '-def']},
                    context=bass_context,
                ),
                # TODO: extend with definition formatting
            ],
            conf=base_langs_es_de_pl_en_conf,
        ),
        TC(
            descr='Reverse',
            tags=set(),
            input=[
                IC(
                    tags={'permutation', 'position'},
                    input={
                        'pl es conocer -r',
                        'pl es -r conocer',
                        'pl -r es conocer',
                    },
                    context=(conocer_context := {
                        'from_lang': ('es'),
                        'to_langs': frozenset({'pl'}),
                        'words': ['conocer']
                    }),
                ),
                IC(
                    tags={'from-conf'},
                    input='-r conocer',
                    conf=Box({
                        'langs': ['pl', 'es'],
                        'assume': 'lang',
                    }),
                    context=conocer_context,
                )
            ],
            conf=base_langs_es_de_pl_en_conf,
        ),
        TC(
            descr='GroupBy',
            tags=set(),
            input=[
                IC(
                    input='es pl de -w conocer orden --groupby lang'
                    # TODO expected
                ),

                IC(
                    tags={'replacement',},
                    input='es pl de -w conocer orden -by word'
                    # TODO expected
                ),
            ],
            conf=base_langs_es_de_pl_en_conf,
        ),
    ]

    @classmethod
    def map_to_many(cls, tc: TC) -> Iterable[Tc]:
        tcs = cls.map_inputs(tc)  # input > InputCase (single)
        tcs = _.flat_map(tcs, cls.replace_placeholders)
        tcs = _.map_(tcs, cls.singularize)
        return tcs

    @classmethod
    def map_inputs(cls, tc: TC) -> list:
        match tc.input:
            case InputCase(): return [tc]
            case str(): return [replace(tc, input=InputCase(input=tc.input))]
            case _ if isinstance(tc.input, Iterable):
                return _.flat_map((replace(tc, input=input) for input in tc.input), cls.map_inputs)
            case _: raise ValueError(f'Unexpected input type: {type(tc.input)}')

    @classmethod
    def replace_placeholders(cls, tc: TC) -> Iterable[TC]:
        def replace_at(replacements: dict[str, list], tc: TC):
            for placetaker_batch in product(*replacements.values()):
                curr_tc = replace(tc)
                curr_input: str = tc.input.input
                for placeholder, placetaker in zip(replacements.keys(), placetaker_batch):
                    curr_input = curr_input.replace(f'<{placeholder}>', placetaker)
                curr_tc = replace(curr_tc, input=replace(tc.input, input=curr_input))
                yield curr_tc
        tcs = replace_at(tc.input.replacement, tc)
        tcs = _.uniq_by(tcs, lambda tc: tc.input.input)
        top_repl_tcs = (subtc for tc in tcs for subtc in replace_at(tc.replacement, tc))
        tcs = c(top_repl_tcs).uniq_by(lambda tc: tc.input.input).value()
        return tcs

    @classmethod
    def singularize(cls, tc: TC) -> Tc:
        return Tc(
            descr=tc.descr,
            tags=list(tc.tags) + list(tc.input.tags),
            input=tc.input.input,
            context={**tc.context, **tc.input.context},
            conf={**tc.conf, **tc.input.conf},
        )

    @classmethod
    def gather_tags(cls, tc) -> Iterable[str]:
        return tc.tags



@pytest.fixture(autouse=True)
def patch(mocker):
    mocker.patch(
        'src.core.scrap_adapting.ScrapAdapter.scrap',
        side_effect=mocked_scrap,
    )


@SystemTCG.parametrize('tc')
def test(tc: TC):
    with open(TEST_CONF, 'w') as f:
        yaml.dump(tc.conf, f, default_flow_style=False, allow_unicode=True)
    collector = CallCollector()
    app_mgr = AppMgr(conf_path=TEST_CONF, printer=collector)

    app_mgr.run_single(shlex.split(tc.input))

    output = collector.output