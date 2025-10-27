from typing import Any, Callable

from bs4 import Tag
from requests import HTTPError, Response

from src.scrapping.core.parsing import Result, ParsingException
import pydash as _
from pydash import chain as c

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
    with open(filename, 'r') as f:
        content = f.read()
    return parse(content)