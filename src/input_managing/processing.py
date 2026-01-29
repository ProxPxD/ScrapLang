import json
import logging
import re
from argparse import Namespace
from collections.abc import Callable
from functools import reduce
from itertools import cycle
from typing import Optional

import pydash as _
from box import Box
from pydash import chain as c
from toolz import itemmap

from src.context import Context
from src.input_managing.outstemming import Outstemmer
from src.lang_detecting.detecting import Detector
from src.lang_detecting.preprocessing.data import DataProcessor


class InputProcessor:
    def __init__(self, context: Context,
                 data_processor: DataProcessor = None,
        ):
        self.context = context
        self.outstemmer = Outstemmer()
        self.data_processor = data_processor
        self.detector = Detector(self.context, self.data_processor.lang_script, valid_data_mgr=self.data_processor.valid_data_mgr) if self.data_processor.lang_script_mgr else None

    def process(self, parsed: Namespace) -> Namespace:
        parsed = self._word_outstemming(parsed)
        parsed = self._fill_args(parsed)
        parsed = self._reverse_if_needed(parsed)
        parsed = self._uniq(parsed)
        origs = list(parsed.words)
        parsed = self._apply_mapping(parsed)
        parsed.unmapped = origs
        logging.debug(f'Processed: {parsed}')
        return parsed

    def _word_outstemming(self, parsed: Namespace) -> Namespace:
        parsed.words = self.outstemmer.join_outstem_syntax(parsed.words)
        # TODO: Add test for veryfying that the same word in outstemming do not appear
        # TODO: Add test for allowing the same word for different from_langs
        parsed.words = _.flat_map(parsed.words, self.outstemmer.outstem)
        return parsed

    def _fill_args(self, parsed: Namespace) -> Namespace:
        # TODO: Refactor to have it cleaner which filling is needed - when from, when to-lang? (see all occurencese of: orig_from_langs)
        msg, func = self._is_filling_needless_and_get_map_parsed(parsed)
        if msg:
            logging.debug(msg)
            return (func or _.identity)(parsed)
        parsed = self._infer_lang(parsed)
        parsed = self._fill_last_used(parsed)
        return parsed

    def _is_filling_needless_and_get_map_parsed(self, parsed: Namespace) -> tuple[Optional[str | bool], Optional[Callable[[Namespace], Namespace]]]:
        # TODO: add test for verifying argument fullfilling if only from_lang is specified
        if parsed.orig_from_langs or parsed.orig_to_langs:
            return 'There exist from- and to- langs, not filling', None
        if parsed.set or parsed.add or parsed.delete:
            return 'Conf editing is run, not filling', None
        if parsed.retrain is True:
            return 'Just retraining, not filling', None
        if isinstance(parsed.loop, bool) or self.context.loop is True:
            # TODO: verify if it's enough and that replacement is not needed later, test "-r" in loop
            # TODO: test loop
            retrieve_context_langs = lambda parsed: c(dict(
                from_langs=parsed.from_langs or self.context.from_langs,
                to_langs=parsed.to_langs or self.context.to_langs,
            ).items()).map(lambda name_langs: setattr(parsed, name_langs[0], name_langs[1])).value() and parsed
            return 'Just managing the loop, not filling', retrieve_context_langs
        return False, None

    def _infer_lang(self, parsed: Namespace) -> Namespace:
        if msg := self._is_infer_needless(parsed):
            logging.debug(msg)
            return parsed
        logging.debug('Inferring thru a simple detector')
        inferred_lang = self.detector.detect_simple(parsed.words)
        if inferred_lang in parsed.from_langs:  # TODO: test not replacing with the same: t przekaz pl en nośnik
            return parsed
        if not inferred_lang:
            #inferred_lang = self.detector.detect_advanced()
            return parsed

        logging.debug(f'Inferred {inferred_lang}')
        parsed.to_langs.extend(parsed.from_langs)
        parsed.from_langs = [inferred_lang]
        return parsed

    def _is_infer_needless(self, parsed: Namespace) -> Optional[str]:
        if parsed.orig_from_langs:
            return 'From lang explicitly specified, not inferring'
        if parsed.reverse:  # TODO: add test for reversing!
            return 'Last used langs should be reversed, not inferring'
        if not self.detector or self.context.infervia not in {'all', 'ai'}:
            return 'AI Inferring disabled, not inferring'
        return None

    def _fill_last_used(self, parsed: Namespace) -> Namespace:
        used = _.filter_(parsed.from_langs + parsed.to_langs)
        pot_defaults = [lang for lang in self.context.langs if lang not in used]; logging.debug(f'Potential defaults: {pot_defaults}')
        if len(self.context.langs) < (n_needed := int(not parsed.from_langs) + int(not parsed.to_langs)):
            raise ValueError(f'Config has not enough defaults! Needed {n_needed}, but possible to choose only: {pot_defaults}')
        # Do not require to translate on definition or inflection
        if not parsed.to_langs and (parsed.definition or parsed.inflection or parsed.wiktio or parsed.grammar):
            if parsed.at.startswith('n'):
                n_needed -= 1
        to_fill = pot_defaults[:n_needed]; logging.debug(f'Chosen defaults: {to_fill}')
        if not parsed.from_langs and to_fill:
            from_lang = to_fill.pop(0); logging.debug(f'Filling from_lang with "{from_lang}"')
            parsed.from_langs = [from_lang]
        if not parsed.to_langs and to_fill:
            to_lang = to_fill.pop(0); logging.debug(f'Filling to_lang with "{to_lang}"')
            parsed.to_langs.append(to_lang)
        return parsed

    def _reverse_if_needed(self, parsed: Namespace) -> Namespace:
        if parsed.reverse is True:
            old_from, old_first_to = parsed.from_langs[0], parsed.to_langs[0]
            logging.debug(f'Reversing: {old_from, old_first_to} => {old_first_to, old_from}')
            parsed.from_langs[0] = old_first_to
            parsed.to_langs[0] = old_from
        return parsed

    def _uniq(self, parsed: Namespace) -> Namespace:
        parsed.to_langs = _.uniq(parsed.to_langs)
        if len(parsed.from_langs) == 1:
            parsed.words = _.uniq(parsed.words)
        return parsed

    def _apply_mapping(self, parsed: Namespace) -> Namespace:
        whole_lang_mapping: Box
        mapped_words = []
        for from_lang, word in zip(cycle(self.context.from_langs or parsed.from_langs), parsed.words):
            if not (whole_lang_mapping := self.context.mappings.get(from_lang)) or whole_lang_mapping and not whole_lang_mapping[0]:
                mapped_words.append(word)
                continue
            logging.debug(f'Applying mapping for "{from_lang}" with map:\n{json.dumps(whole_lang_mapping, indent=4, ensure_ascii=False)}')
            for single_mapping in whole_lang_mapping:
                patts, repls = zip(*sorted(single_mapping.items(), key=c().get(0).size(), reverse=True))
                whole_lang_mapping: list = list(map(lambda patt, repl: (re.compile(patt), repl), patts, repls))
                logging.debug(f'from_lang_map: {whole_lang_mapping}')
                word = reduce(lambda w, patt_repl: patt_repl[0].sub(patt_repl[1], w), whole_lang_mapping, word)
            mapped_words.append(word)
        parsed.words = mapped_words
        return parsed

    def retrain_detector(self) -> None:
        self.data_processor.generate_script_summary()
        if not self.detector.advanced_detector:
            raise ValueError('Advanced detector requires torch lib to work')
        logging.debug('Retraining model')
        self.detector.advanced_detector.retrain_model()
