from dataclasses import dataclass

import pytest
from box import Box

from src.glosbe.cli import CLI
from tests.TCG import TCG


@dataclass
class ConfParsed:
    conf: dict
    parsed: dict


@dataclass
class TC:
    descr: str
    input: str
    e_parsed: dict
    conf: dict


class CliTCG(TCG):
    tcs = [
        TC(
            descr='single trans with conf',
            input='obituary en pl',
            e_parsed=dict(
                words=['obituary'],
                from_lang='en',
                to_langs=['pl']
            ),
            conf={
                'assume': 'lang',
                'langs': ['de', 'zh'],
                'mappings': {},
            }
        )
    ]


@CliTCG.parametrize('tc')
def test(tc: TC):
    cli = CLI(Box(tc.conf))
    a_parsed = cli.parse(tc.input)
    for key, e_val in tc.e_parsed.items():
        a_val = getattr(a_parsed, key)
        assert a_val == e_val