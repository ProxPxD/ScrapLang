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
            # if more_info:
            #     params = _.at(params, 1, -1)

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
    def __init__(self):
        self._buffor = []

    def __call__(self, msg):
        self._buffor.append(str(msg))

    @property
    def output(self):
        return '\n'.join(self._buffor)

    def clear(self):
        self._buffor.clear()