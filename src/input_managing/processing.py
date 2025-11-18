import json
import logging
import re
from argparse import Namespace
from functools import reduce

from box import Box

from src.context import Context
from src.input_managing.outstemming import Outstemmer


import pydash as _
from pydash import chain as c

from src.lang_detecting.detecting import Detector
from src.lang_detecting.preprocessing.data import DataProcessor


class InputProcessor:
    def __init__(self, context: Context,
                 data_processor: DataProcessor = None,
        ):
        self.context = context
        self.outstemmer = Outstemmer()
        self.data_processor = data_processor
        self.detector = Detector(self.data_processor.lang_script_mgr.load()) if self.data_processor.lang_script_mgr else None

    def process(self, parsed: Namespace) -> Namespace:
        parsed = self._word_outstemming(parsed)
        parsed = self._fill_args(parsed)
        parsed = self._reverse_if_needed(parsed)
        origs = list(parsed.words)
        parsed = self._apply_mapping(parsed)
        parsed.mapped = [o != m for o, m in zip(origs, parsed.words)]
        logging.debug(f'Processed: {parsed}')
        return parsed

    def _word_outstemming(self, parsed: Namespace) -> Namespace:
        parsed.words = self.outstemmer.join_outstem_syntax(parsed.words)
        parsed.words = [outstemmed for word in parsed.words for outstemmed in self.outstemmer.outstem(word)]
        return parsed

    def _fill_args(self, parsed: Namespace) -> Namespace:
        parsed = self._detect_lang(parsed)
        parsed = self._fill_last_used(parsed)
        return parsed

    def _detect_lang(self, parsed: Namespace) -> Namespace:
        if parsed.from_lang:
            logging.debug('There exist "from_lang", not inferring')
            return parsed
        if self.context.infervia in {'all', 'ai'} and self.detector:
            logging.debug('Inferring thru a simple detector')
            if from_lang := self.detector.detect_simple(parsed.words):
                logging.debug(f'Inferred {from_lang}')
                parsed.from_lang = from_lang
                return parsed
            # log
        return parsed

    def _fill_last_used(self, parsed: Namespace) -> Namespace:
        used = _.filter_([parsed.from_lang] + parsed.to_langs)
        pot_defaults = [lang for lang in self.context.langs if lang not in used]; logging.debug(f'Potential defaults: {pot_defaults}')
        if len(self.context.langs) < (n_needed := int(not parsed.from_lang) + int(not parsed.to_langs)):
            raise ValueError(f'Config has not enough defaults! Needed {n_needed}, but possible to choose only: {pot_defaults}')
        # Do not require to translate on definition or inflection
        if not parsed.to_langs and (parsed.definition or parsed.inflection or parsed.wiktio):
            if parsed.at.startswith('n'):
                n_needed -= 1
        to_fill = pot_defaults[:n_needed]; logging.debug(f'Chosen defaults: {to_fill}')
        if not parsed.from_lang and to_fill:
            from_lang = to_fill.pop(0); logging.debug(f'Filling from_lang with "{from_lang}"')
            parsed.from_lang = from_lang
        if not parsed.to_langs and to_fill:
            to_lang = to_fill.pop(0); logging.debug(f'Filling to_lang with "{to_lang}"')
            parsed.to_langs.append(to_lang)
        return parsed

    def _reverse_if_needed(self, parsed: Namespace) -> Namespace:
        if parsed.reverse:
            old_from, old_first_to = parsed.from_lang, parsed.to_langs[0]
            logging.debug(f'Reversing: {old_from, old_first_to} => {old_first_to, old_from}')
            parsed.from_lang = old_first_to
            parsed.to_langs[0] = old_from
        return parsed

    def _uniq_langs(self, parsed: Namespace) -> Namespace:
        # TODO: test
        parsed.to_langs = _.uniq(parsed.to_langs)
        return parsed

    def _apply_mapping(self, parsed: Namespace) -> Namespace:
        # todo: test żurawel (regex)

        whole_lang_mapping: Box
        if not (whole_lang_mapping := self.context.mappings.get(parsed.from_lang)) or whole_lang_mapping and not whole_lang_mapping[0]:
            return parsed
        logging.debug(f'Applying mapping for "{parsed.from_lang}" with map:\n{json.dumps(whole_lang_mapping, indent=4, ensure_ascii=False)}')
        for single_mapping in whole_lang_mapping:
            patts, repls = zip(*sorted(single_mapping.items(), key=c().get(0).size(), reverse=True))
            whole_lang_mapping: list = list(map(lambda patt, repl: (re.compile(patt), repl), patts, repls))
            logging.debug(f'from_lang_map: {whole_lang_mapping}')
            parsed.words = [
                reduce(lambda w, patt_repl: patt_repl[0].sub(patt_repl[1], w), whole_lang_mapping, word)
                for word in parsed.words
            ]
        return parsed
