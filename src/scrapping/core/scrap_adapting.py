from typing import Optional, Callable

from bs4 import Tag
from requests import Session, Response
from requests.exceptions import HTTPError

from .parsing import Result, ParsingException, CaptchaException, Parser


class ScrapAdapter:
    def __init__(self, session: Session = None):
        self.session: Optional[Session] = session

    def scrap(self,
              url: str,
              parse: Callable[[Response | Tag | str], list[Result] | ParsingException],
              params=None,
              headers=None
        ) -> list[Result] | HTTPError | ParsingException:
        try:
            response = self.session.get(url, allow_redirects=True, params=params, headers=headers)
            response.raise_for_status()
        except HTTPError as e:
            return e
        else:
            if Parser.is_captcha(response):
                return CaptchaException('Captcha appeared, robot identified!')
            return parse(response)
