from parameterized import parameterized

from src.cli import CURRENT_MODES_COL
from tests.abstractCliTest import AbstractCliTest


class MultiWordCliTest(AbstractCliTest):
    @classmethod
    def _get_test_name(cls) -> str:
        return 'Word Mode CLI'

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.cli.root.get_collection(CURRENT_MODES_COL).set_default('-w')

    @parameterized.expand([
        ('all_arguments', ['schlafen', 'Herr', 'nicht'], 'de', 'pl', 't de pl schlafen Herr nicht'),
        ('additional_with_flag', ['schlafen', 'Herr', 'nicht'], 'de', 'pl', 't de pl schlafen Herr -w nicht'),
        ('no_from_lang_with_flag', ['schlafen', 'Herr', 'nicht'], 'de', 'pl', 't pl -w schlafen Herr nicht'),
        ('only_words_with_flag', ['schlafen', 'Herr', 'nicht'], 'de', 'pl', 't -w schlafen Herr nicht'),
    ])
    def test_parsing(self, name: str, e_words: str, e_from_lang: str, e_to_lang: str, input_line: str):
        words, from_lang, to_langs = self.cli.root.get_collections('words', 'from_langs', 'to_langs')
        from_lang.set_default(e_from_lang)
        to_langs.set_default(e_to_lang)

        self.cli.parse_without_actions(input_line)

        self.assertCountEqual(e_words, words.get_plain())
        self.assertEqual(e_from_lang, from_lang.get())
        self.assertEqual(e_to_lang, to_langs.get())
