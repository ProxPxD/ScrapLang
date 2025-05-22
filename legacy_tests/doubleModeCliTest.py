import unittest

from parameterized import parameterized

from tests.abstractCliTest import AbstractCliTest


class DoubleModeCliTest(AbstractCliTest):
	@classmethod
	def _get_test_name(cls) -> str:
		return 'Double Mode'

	@parameterized.expand([
		('', [], '', [], '')
	])
	@unittest.skip
	def test_parsing(self, name: str, e_word: list[str], e_from_lang: str, e_to_langs: list[str], input_line: str):
		self.fail(NotImplementedError.__name__)
