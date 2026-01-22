from functools import lru_cache
from typing import Any, Callable, Iterable, Optional

from bs4 import Tag
from requests import Response, Session
from requests.exceptions import HTTPError

from .parsing import CaptchaException, Parser, ParsingException, Result


class ScrapAdapter:
    def __init__(self, session: Session = None):
        self.session: Optional[Session] = session

    def scrap(self,
              url: str,
              parse: Callable[[Response | Tag | str], list[Result] | ParsingException],
              params: dict = None,
              headers: dict = None
        ) -> list[Result] | HTTPError | ParsingException:
        try:
            # response = self.session.get(url, allow_redirects=True, params=params, headers=headers)
            response = self.get_response(
                url,
                params=params and tuple(params.items()),
                headers=headers and tuple(headers.items()),
            )
            response.raise_for_status()
        except HTTPError as e:
            return e
        else:
            if Parser.is_captcha(response):
                return CaptchaException('Captcha appeared, robot identified!')
            return parse(response)

    @lru_cache(maxsize=2)
    def get_response(self, url: str, params: Iterable[tuple[Any, Any]], headers: Iterable[tuple[Any, Any]]) -> Response:
        return self.session.get(url, allow_redirects=True, params=params and dict(params), headers=headers and dict(headers))
