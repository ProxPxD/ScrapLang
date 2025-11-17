from pathlib import Path
from typing import Callable

import pydash as _
from bs4 import Tag
from pydash import chain as c
from requests import HTTPError, Response

from src.scrapping.core.parsing import Result, ParsingException

PAGES = Path(__file__).parent.parent / 'pages'


def get_filename_from_url(url: str) -> str:
    site = _.filter_({'glosbe', 'wiktio'}, url.__contains__)[0]
    params = []
    more_info = ''
    match site:
        case 'wiktio':
            params = [url.split('/wiki/')[1]]
        case 'glosbe':
            params = url.split('.com/')[1].split('/')[:3]
            more_info = c({'details', 'indirect'}).filter_(url.__contains__).get(0, '').value()
        case _: raise ValueError('Unexpected site!')

    name = f'{site}-{"-".join(params)}'
    if more_info:
        name = f'{name}-{more_info}'
    fullname = f'{name}.html'
    return fullname


def mocked_scrap(url: str, parse: Callable[[Response | Tag | str], list[Result] | ParsingException | HTTPError]) -> list[Result] | HTTPError | ParsingException:
    filename = get_filename_from_url(url)
    with open(PAGES / filename, 'r') as f:
        content = f.read()
    return parse(content)


class CallCollector:
    def __init__(self, *,
                 line_mapper: Callable[[str], str] = None,
                 msg_mapper: Callable[[str], str] = None,
                 sep: str = '\n',
        ):
        self._buffor = []
        self._sep = sep
        self._msg_mapper = msg_mapper or _.identity
        self._line_mapper = line_mapper or _.identity

    def __call__(self, msg):
        self._buffor.append(self._msg_mapper(str(msg)))

    @property
    def output(self):
        return c(self._buffor).map(c().split(self._sep)).flatten().map(self._line_mapper).join(self._sep).value()

    def clear(self):
        self._buffor.clear()
