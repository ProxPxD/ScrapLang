from dataclasses import dataclass

from box import Box

from src.glosbe.cli import CLI
from tests.TCG import TCG


@dataclass
class TC:
    descr: str
    input: str
    e_parsed: dict
    conf: dict


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
            conf=(base_conf := {
                'assume': 'lang',
                'langs': ['en', 'pl'],
                'mappings': {},
            })
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
            input='en pl -w obituary cat',
            e_parsed=dict(
                words=['obituary', 'cat'],
                from_lang='en',
                to_langs=['pl']
            ),
            conf=base_conf,
        ),
        TC(
            descr='single word trans with conf with from/to flags',
            input='obituary -f en -t pl',
            e_parsed=dict(
                words=['obituary'],
                from_lang='en',
                to_langs=['pl']
            ),
            conf=base_conf,
        )
    ]


@CliTCG.parametrize('tc')
def test(tc: TC):
    cli = CLI(Box(tc.conf))
    a_parsed = cli.parse(tc.input)
    for key, e_val in tc.e_parsed.items():
        a_val = getattr(a_parsed, key)
        assert a_val == e_val