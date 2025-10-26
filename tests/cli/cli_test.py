import shlex
from dataclasses import dataclass, field, replace
from typing import Iterable

from box import Box

from src.app_managing import AppMgr
from src.cli import CLI
from src.context import Context
from tests.TCG import TCG


@dataclass
class TC:
    descr: str
    input: str
    e_parsed: dict
    conf: Box

    replacement: dict = field(default_factory=dict)



class CliTCG(TCG):
    tcs = [
        TC(
            descr='single trans with conf word first',
            input='obituary en pl',
            e_parsed=dict(
                words=['obituary'],
                from_lang='en',
                to_langs=['pl']
            ),
            conf=(base_conf := Box({
                'assume': 'lang',
                'langs': ['en', 'pl', 'de'],
                'mappings': {},
            }))
        ),
        TC(
            descr='single trans with conf word last',
            input='en pl obituary',
            e_parsed=dict(
                words=['obituary'],
                from_lang='en',
                to_langs=['pl']
            ),
            conf=base_conf,
        ),
        TC(
            descr='multiple word trans with conf',
            input='en pl <WORD_FLAG> obituary cat',
            replacement={'WORD_FLAG': ['-w', '--words', '-words']},
            e_parsed=dict(
                words=['obituary', 'cat'],
                from_lang='en',
                to_langs=['pl']
            ),
            conf=base_conf,
        ),
        TC(
            descr='multiple lang trans',
            input='obituary en <LANG_FLAG> pl de',
            replacement={'LANG_FLAG': ['-t', '-l', '--to', '--to-lang']},
            e_parsed=dict(
                words=['obituary'],
                from_lang='en',
                to_langs=['pl', 'de']
            ),
            conf=base_conf,
        ),
        TC(
            descr='multiple lang no flag trans',
            input='obituary en pl de',
            e_parsed=dict(
                words=['obituary'],
                from_lang='en',
                to_langs=['pl', 'de']
            ),
            conf=base_conf,
        ),
        TC(
            descr='single word trans with conf with from flags',
            input='obituary pl <FROM_FLAG> en',
            replacement={'FROM_FLAG': ['--from', '-f', '--from-lang']},
            e_parsed=dict(
                words=['obituary'],
                from_lang='en',
                to_langs=['pl']
            ),
            conf=base_conf,
        ),
        TC(
            descr='inflection word from lang',
            input='obituary <INFLECTION_FLAG> en',
            replacement={'INFLECTION_FLAG': ['-i', '-infl', '--infl', '--inflection', '--conjugation', '--conj', '-conj', '-c', '--declension', '--decl', '-decl', '--table', '-tab']},
            e_parsed=dict(
                words=['obituary'],
                from_lang='en',
                inflection=True,
            ),
            conf=base_conf,
        ),
        TC(
            descr='definition word from lang',
            input='obituary <DEFINITION_FLAG> en',
            replacement={'DEFINITION_FLAG': ['-d', '-def', '--def', '--definition', '--definitions']},
            e_parsed=dict(
                words=['obituary'],
                from_lang='en',
                definition=True,
            ),
            conf=base_conf,
        ),
        TC(
            descr='reverse word from lang to lang',
            input='obituary pl en <REVERSE_FLAG>',
            replacement={'REVERSE_FLAG': ['-r', '--reverse', '--reversed']},
            e_parsed=dict(
                words=['obituary'],
                from_lang='en',
                to_langs=['pl']
            ),
            conf=base_conf,
        ),
        TC(
            descr='groupby lang from lang to langs',
            input='obituary en pl de cat <GROUP_BY_FLAG> lang',
            replacement={'GROUP_BY_FLAG': ['-by', '--groupby']},
            e_parsed=dict(
                words=['obituary', 'cat'],
                from_lang='en',
                to_langs=['pl', 'de'],
                groupby='lang'
            ),
            conf=base_conf,
        )
    ]

    @classmethod
    def map_to_many(cls, tc) -> Iterable | list:
        tcs = []
        if not tc.replacement:
            tcs.append(tc)
        else:
            for placeholder, placetakers in tc.replacement.items():
                for placetaker in placetakers:
                    new_tc = replace(tc, input=tc.input.replace(f'<{placeholder}>', placetaker))
                    tcs.append(new_tc)
        return tcs


@CliTCG.parametrize('tc')
def test(tc: TC):
    context = Context(tc.conf)
    cli = CLI(tc.conf, context)
    a_parsed = cli.parse(shlex.split(tc.input))
    for key, e_val in tc.e_parsed.items():
        a_val = getattr(a_parsed, key)
        assert a_val == e_val