from typing import Any, Type

from parameterized import parameterized

from src.glosbe.translatingPrinting.formatting import GenderFormatter, PartOfSpeechFormatter, AbstractFormatter
from tests.abstractCliTest import AbstractCliTest


class FormattingTest(AbstractCliTest):

	@classmethod
	def _get_test_name(cls) -> str:
		return 'Formatting'

	@parameterized.expand([
		('feminine', 'fem', GenderFormatter),
		('masculine', 'masc', GenderFormatter),
		('neuter', 'neut', GenderFormatter),
		('adjective', 'adj.', PartOfSpeechFormatter),
		('adverb', 'adv.', PartOfSpeechFormatter),
	],
		name_func=lambda method, param_num, param: f'{method.__name__}_{param_num}_format_{param.args[0]}_with_{param.args[2].__name__}'
	)
	def test_formatter(self, arg: Any, expected: Any, formatter: Type[AbstractFormatter]):
		self.assertEqual(expected, formatter.format(arg))