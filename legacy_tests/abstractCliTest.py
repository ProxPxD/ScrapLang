import abc

from src.cli import TranslatorCli
from tests.abstractTest import AbstractTest


class AbstractCliTest(AbstractTest, abc.ABC):
	cli: TranslatorCli

	@classmethod
	def setUpClass(cls) -> None:
		super().setUpClass()
		cls.cli = TranslatorCli()
		cls.cli.turn_off_translating()

	@classmethod
	def tearDownClass(cls) -> None:
		super().tearDownClass()
		cls.cli.turn_on_translating()

	def get_file_name(self) -> str:
		return self._get_test_name()

	@classmethod
	def get_default_configs(cls):
		return {
			'--default-mode': '--single',
			'--langs': ['pl', 'de', 'fr', 'zh'],
			'--limit': 3,
			'--layout_adjustment_mode': 'none',
			'--adjustment_lang': '',
			'--double-mode-style': '--lang',
    	}
