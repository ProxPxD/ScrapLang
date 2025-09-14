from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
from typing import Iterator

from bs4 import BeautifulSoup, PageElement
from bs4.element import Tag
from requests import Response


def ensure_tag(to_parse: Response | Tag | str) -> Tag:
    match to_parse:
        case Tag(): return to_parse
        case str(): return BeautifulSoup(to_parse, features="html5lib")
        case Response(): return ensure_tag(to_parse.text)
        case _: raise ValueError(f'Cannot handle type {type(to_parse)} of {to_parse}!')

def with_ensured_tag(func):
    @wraps(func)
    def wrapper(self, tag, *args, **kwargs):
        return func(self, ensure_tag(tag), *args, **kwargs)
    return wrapper


@dataclass(frozen=True)
class Result:
    pass


class ParsingException(ValueError):
    pass


class CaptchaException(ParsingException):
    pass



class Parser(ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    @abstractmethod
    def parse(cls, to_parse: Response | Tag | str) -> list:
        raise NotImplementedError

    @classmethod
    @with_ensured_tag
    def is_captcha(cls, tag: Tag) -> bool:
        return bool(tag.find('div', {'class': 'g-recaptcha'}))

    @classmethod
    def filter_to_tags(cls, elems: Iterator[PageElement]) -> Iterator[Tag]:
        return (t for t in elems if isinstance(t, Tag))