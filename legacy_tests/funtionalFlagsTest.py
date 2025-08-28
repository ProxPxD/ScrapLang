import unittest

from parameterized import parameterized

from src.configurating import Configurations
from tests.abstractCliTest import AbstractCliTest


class FunctionalFlagsTest(AbstractCliTest):
	@classmethod
	def _get_test_name(cls) -> str:
		return 'Functional Flags'

	def setUp(self) -> None:
		super().setUp()
		Configurations.init(self.get_file_name(), default=self.get_default_configs())

	def tearDown(self) -> None:
		super().tearDown()
		Configurations.remove_current_configuration()

	@parameterized.expand([
		('short_reverse_flag', 't żal de pl -r', ['żal'], ['pl'], ['de']),
	])
	def test_parsed_arguments(self, name: str, input_line: str, e_words: list[str], e_from_langs: list[str], e_to_langs: list[str]):
		words, from_lang, to_langs = self.cli.root.get_collections('words', 'from_langs', 'to_langs')
		self.cli.parse(input_line)

		self.assertCountEqual(e_words, words.get_as_list())
		self.assertCountEqual(e_from_langs, from_lang.get_as_list())
		self.assertCountEqual(e_to_langs, to_langs.get_as_list())

	@parameterized.expand([
		('', '', [], [], [], '')
	])
	@unittest.skip
	def test_from_lang_flag(self, name: str, input_line: str, e_words: list[str], e_from_langs: list[str], e_to_langs: list[str], default_mode: str):
		self.fail(NotImplementedError.__name__)