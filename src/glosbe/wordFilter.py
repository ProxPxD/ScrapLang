from .constants import supported_languages


class WordFilter:

    alphabet = 'abcdefghijklmnopqrstuvwxyz'

    def __init__(self):
        self._to_swap = []
        self._is_any_word_moved_from_from_langs = False
        self._is_any_word_moved_from_to_langs = False

    def reset(self):
        self._is_any_word_moved_from_from_langs = False
        self._is_any_word_moved_from_to_langs = False

    def is_any_word_moved_from_from_langs(self) -> bool:
        return self._is_any_word_moved_from_from_langs

    def is_any_word_moved_from_to_langs(self) -> bool:
        return self._is_any_word_moved_from_to_langs

    def get_to_swap(self) -> list:
        return self._to_swap

    def is_lang(self, to_test: str) -> bool:
        return to_test in supported_languages

    def is_any_lang_misplaced(self, *to_check: str):
        return any(not self.is_lang(elem) for elem in to_check)

    def _membership(self, to_test):
        return 0 if self.is_lang(to_test) else 1

    def split_into_langs_and_words(self, to_split: SmartList[str]) -> tupleLists_2:
        langs_words: tupleLists_2 = (SmartList(), SmartList())
        for to_test in to_split:
            langs_words[self._membership(to_test)].append(to_test)
        return langs_words

    def split_from_from_and_to_langs(self, from_lang: SmartList[str], to_langs: SmartList[str]) -> tupleLists_3:
        new_from_lang, from_langs_words = self.split_into_langs_and_words(from_lang)
        new_to_lang, to_langs_words = self.split_into_langs_and_words(to_langs)
        if from_langs_words:
            self._is_any_word_moved_from_from_langs = True
        if to_langs_words:
            self._is_any_word_moved_from_to_langs = True
        return new_from_lang, new_to_lang, from_langs_words + to_langs_words


