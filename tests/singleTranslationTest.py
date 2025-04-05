from parameterized import parameterized

from src.glosbe.cli import CURRENT_MODES_COL
from tests.abstractCliTest import AbstractCliTest


class SingleModeCliTest(AbstractCliTest):

    @classmethod
    def _get_test_name(cls) -> str:
        return 'Single Mode CLI'

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.cli.root.get_collection(CURRENT_MODES_COL).set_default('-s')

    @parameterized.expand([
        ('all_arguments', 'tak', 'pl', 'zh', 't tak pl zh'),
        ('no_from_lang', 'tak', 'pl', 'zh', 't tak zh'),
        ('only_word', 'tak', 'pl', 'zh', 't tak'),
    ])
    def test_order_parsing(self, name: str, e_word: str, e_from_lang: str, e_to_lang: str, input_line: str):
        words, from_langs, to_langs = self.cli.root.get_collections('words', 'from_langs', 'to_langs')
        from_langs.set_default(e_from_lang)
        to_langs.set_default(e_to_lang)

        self.cli.parse_without_actions(input_line)

        self.assertEqual(e_word, words.get())
        self.assertEqual(e_from_lang, from_langs.get())
        self.assertEqual(e_to_lang, to_langs.get())
