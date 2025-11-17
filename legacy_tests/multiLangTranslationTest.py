from parameterized import parameterized

from src.input_managing.cli import CURRENT_MODES_COL
from tests.abstractCliTest import AbstractCliTest


class MultiLangCliTest(AbstractCliTest):

    @classmethod
    def _get_test_name(cls) -> str:
        return 'Lang Mode CLI'

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.cli.root.get_collection(CURRENT_MODES_COL).set_default('-m')

    @parameterized.expand([
        ('all_arguments', 'to be', 'en', ['zh', 'es', 'ru'], 't to\ be en zh es ru'),
        ('additional_with_flag', 'pensar', 'es', ['zh', 'en', 'ru'], 't pensar es zh en -m ru'),
        ('no_from_lang_with_flag', 'pensar', 'es', ['zh', 'en', 'ru'], 't pensar -m zh en ru'),
        ('only_word_with_flag', 'pensar', 'es', ['zh', 'en', 'ru'], 't pensar -m'),
    ])
    def test_parsing(self, name: str, e_word: str, e_from_lang: str, e_to_langs: str, input_line: str):
        words, from_lang, to_langs = self.cli.root.get_collections('words', 'from_langs', 'to_langs')
        from_lang.set_default(e_from_lang)
        to_langs.set_default(e_to_langs)

        self.cli.parse_without_actions(input_line)

        self.assertEqual(e_word, words.get())
        self.assertEqual(e_from_lang, from_lang.get())
        self.assertCountEqual(e_to_langs, to_langs.get_plain())

