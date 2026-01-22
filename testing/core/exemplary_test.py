from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Iterable, Optional

from TCG import TCG


def exemplary_logic(string: str) -> str:
    return ' '.join(string.upper())


@dataclass
class TC:
    descr: str

    input: str | list[str]
    expected: str | list[str]

    tags: set[str] = field(default_factory=set)


@dataclass
class Tc:
    descr: str

    input: str
    expected: str

    tags: set[str] = field(default_factory=set)


class ExemplaryTCG(TCG):
    tcs = [
        TC('Trivial - already upper case', input='X', expected='X', tags={'upper'}),
        TC('Simple', input='h', expected='H'),
        TC('Double test for tags passing', input=['lol', 'lmao'], expected=['L O L', 'L M A O']),
    ]

    @classmethod
    def generate_tcs(cls) -> list:
        tcs = cls.tcs
        tcs.extend([
            TC('Test that needs functional operations', input='kidding', expected='K I D D I N G'),
        ])
        return tcs

    @classmethod
    def map_to_many(cls, tc: TC) -> Iterable[Tc]:
        match tc.input, tc.expected:
            case (str(), str()): return [Tc(**asdict(tc))]
            case (list(), list()):
                if len(tc.input) != len(tc.expected):
                    raise ValueError('TC should have the same number of inputs and expected')
                return [Tc(**{**asdict(tc), 'input': i, 'expected': e}) for i, e in zip(tc.input, tc.expected)]
            case _: raise ValueError('Unexpected input or expected fields of TC')

    @classmethod
    def gather_tags(cls, tc: Tc) -> Iterable[str]:
        if 'functional' in tc.descr:
            tc.tags.add('functional')
        return tc.tags

    @classmethod
    def create_name(cls, tc) -> Optional[str]:
        return f'{tc.descr}: "{tc.input}", ({", ".join(tc.tags)})'


@ExemplaryTCG.parametrize('tc')
def test(tc: Tc):
    actual = exemplary_logic(tc.input)
    assert actual == tc.expected
