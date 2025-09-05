from parameterized import parameterized

from src.resouce_managing.configurating import Configurations
from src.constants import FLAGS as F
from tests.abstractCliTest import AbstractCliTest


class ConfigurationTest(AbstractCliTest):

	@classmethod
	def _get_test_name(cls) -> str:
		return 'Configuration'

	def setUp(self) -> None:
		super().setUp()
		Configurations.init(self.get_file_name())

	def tearDown(self) -> None:
		super().tearDown()
		Configurations.remove_current_configuration()

	def test_take_nth_lang_from_no_langs(self):
		Configurations.change_conf('--langs', [])
		try:
			Configurations.get_nth_saved_language(0)
			Configurations.get_nth_saved_language(1)
		except Exception:
			self.fail()

	@parameterized.expand([
		('first', 'pl', 0),
		('second', 'zh', 1),
		('third', 'es', 2),
		('forth', 'de', 3),
		('out_of_range', None, 4),
	])
	def test_take_nth_lang(self, name: str, expected: str, n: int):
		Configurations.change_conf('--langs', ['pl', 'zh', 'es', 'de'])
		actual = Configurations.get_nth_saved_language(n)
		self.assertEqual(expected, actual)

	@parameterized.expand([
		('equal_limit_to_conf_langs_count', 3, ['de', 'fr', 'zh'], ['de', 'fr', 'zh']),
		('smaller_limit_than_conf_langs_count', 2, ['de', 'fr'], ['de', 'fr', 'zh', 'en', 'ru']),
		('greater_limit_than_conf_langs_count', 10, ['de', 'fr'], ['de', 'fr']),
	])
	def test_limit(self, name: str, limit: int, e_to_langs: list[str], conf_langs: list[str]):
		Configurations.change_conf(F.C.LANGS_SHOW_LONG_FLAG, conf_langs)
		Configurations.change_conf(F.C.LANG_LIMIT_LONG_FLAG, limit)
		self.cli.parse('t mieć pl -m')
		a_to_langs = self.cli.to_langs
		self.assertCountEqual(e_to_langs, a_to_langs)

	# TODO: test other flags printing and possible values