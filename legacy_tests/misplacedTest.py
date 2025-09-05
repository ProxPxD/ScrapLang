from parameterized import parameterized

from src.resouce_managing.configurating import Configurations
from src.cli import CURRENT_MODES_COL
from tests.abstractCliTest import AbstractCliTest


class MisplacedTest(AbstractCliTest):

    @classmethod
    def _get_test_name(cls) -> str:
        return 'Misplaced'

    def setUp(self) -> None:
        super().setUp()
        Configurations.init(self.get_file_name(), default=self.get_default_configs())

    def tearDown(self) -> None:
        super().tearDown()
        Configurations.remove_current_configuration()

    @parameterized.expand([
        ('single_once', ['suchen'], ['de'], ['pl'], 't de suchen pl', '-s', []),
        ('single_twice', ['suchen'], ['de'], ['pl'], 't de pl suchen', '-s', []),
        ('single_no_from_lang', ['suchen'], ['de'], ['pl'], 't pl suchen', '-s', ['pl', 'de']),
        ('lang_misplaced_before', ['suchen'], ['de'], ['pl', 'fr'], 't de suchen -m pl fr', None, []),
        ('lang_misplaced_after', ['suchen'], ['de'], ['pl', 'fr'], 't -m pl suchen fr', None, []),
        ('word_misplaced_one_word', ['suchen', 'nehmen', 'krank', 'Weib'], ['de'], ['pl'], 't pl suchen -w nehmen krank Weib', None, ['pl', 'de']),
        ('word_misplaced_two_words', ['suchen', 'nehmen', 'krank', 'Weib'], ['de'], ['pl'], 't suchen nehmen -w krank Weib', None, ['pl', 'de']),
    ])
    def test_misplaced(self, name: str, e_words: list[str], e_from_langs: list[str], e_to_langs: list[str], input_line: str, mode: str, last_langs: list[str, str]):
        words, from_langs, to_langs = self.cli.root.get_collections('words', 'from_langs', 'to_langs')
        if last_langs:
            Configurations.change_last_used_languages(*last_langs)
        if mode:
            self.cli.root.get_collection(CURRENT_MODES_COL).set_default(mode)

        self.cli.parse(input_line)

        if not from_langs:
            from_langs += Configurations.get_from_language()
        if not to_langs:
            to_langs += Configurations.get_nth_saved_language(0, from_langs.get())
        self.assertCountEqual(e_words, words.get_as_list())
        self.assertCountEqual(e_from_langs, from_langs.get_as_list())
        self.assertCountEqual(e_to_langs, to_langs.get_as_list())

    # def test_misplaced(self):
    #     self.run_current_test_with_params()
