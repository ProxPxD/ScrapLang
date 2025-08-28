from parameterized import parameterized
from smartcli.nodes.smartList import SmartList

from src.configurating import Configurations
from src.constants import FLAGS as F
from src.scrapping import TranslationTypes
from src.printer import Printer
from tests.abstractCliTest import AbstractCliTest


class FlagSettingTest(AbstractCliTest):

	@classmethod
	def _get_test_name(cls) -> str:
		return 'Flag Setting'

	def get_file_name(self) -> str:
		return 'flag_test'

	def setUp(self) -> None:
		super().setUp()
		Configurations.init(self.get_file_name())
		Printer.turn(True)

	def tearDown(self) -> None:
		super().tearDown()
		Configurations.remove_current_configuration()

	@parameterized.expand([
		('short_limit', F.C.LANG_LIMIT_SHORT_FLAG, F.C.LANG_LIMIT_LONG_FLAG, 3, [3]),
		('long_limit', F.C.LANG_LIMIT_LONG_FLAG, F.C.LANG_LIMIT_LONG_FLAG, 1, [1]),
		('short_default_mode_original', F.C.DEFAULT_MODE_SHORT_FLAG, F.C.DEFAULT_MODE_LONG_FLAG, F.M.SINGLE_LONG_FLAG, ['single']),
		('long_default_mode_original', F.C.DEFAULT_MODE_LONG_FLAG, F.C.DEFAULT_MODE_LONG_FLAG, F.M.LANG_LONG_FLAG, ['lang']),
		('short_default_mode_with_change', F.C.DEFAULT_MODE_SHORT_FLAG, F.C.DEFAULT_MODE_LONG_FLAG, F.M.WORD_LONG_FLAG, ['w']),
		('long_default_mode_with_change', F.C.DEFAULT_MODE_LONG_FLAG, F.C.DEFAULT_MODE_LONG_FLAG, F.M.LANG_LONG_FLAG, ['m']),
	])
	def test_set_configuration(self, name: str, flag: str, conf_to_get: str, expected, flag_args: list[str]):
		self.cli.parse(f't {flag} {" ".join(map(str, flag_args))}')
		conf_value = Configurations.get_conf(conf_to_get)
		self.assertEqual(expected, conf_value)

	@parameterized.expand([
		('short_limit', F.C.LANG_LIMIT_SHORT_FLAG, F.C.LANG_LIMIT_LONG_FLAG, [3]),
		('long_limit', F.C.LANG_LIMIT_LONG_FLAG, F.C.LANG_LIMIT_LONG_FLAG, [1]),
		('short_default_mode_original', F.C.DEFAULT_MODE_SHORT_FLAG, F.C.DEFAULT_MODE_LONG_FLAG, [F.M.SINGLE_LONG_FLAG]),
		('long_default_mode_original', F.C.DEFAULT_MODE_LONG_FLAG, F.C.DEFAULT_MODE_LONG_FLAG, [F.M.LANG_LONG_FLAG]),
		('short_default_mode_with_change', F.C.DEFAULT_MODE_SHORT_FLAG, F.C.DEFAULT_MODE_LONG_FLAG, [F.M.WORD_LONG_FLAG]),
		('long_default_mode_with_change', F.C.DEFAULT_MODE_LONG_FLAG, F.C.DEFAULT_MODE_LONG_FLAG, [F.M.LANG_LONG_FLAG]),
		('short_langs', F.C.LANGS_SHOW_SHORT_FLAG, F.C.LANGS_SHOW_LONG_FLAG, ['pl', 'es', 'de']),
		('long_langs', F.C.LANGS_SHOW_LONG_FLAG, F.C.LANGS_SHOW_LONG_FLAG, ['pl', 'es', 'de']),
		('short_show_info', F.C.SHOW_INFO_MODE_FLAG_SHORT, F.C.SHOW_INFO_MODE_FLAG_LONG, [False]),
		('long_show_info', F.C.SHOW_INFO_MODE_FLAG_LONG, F.C.SHOW_INFO_MODE_FLAG_LONG, [True]),
	])
	def test_display_configuration(self, name: str, flag: str, conf_to_set: str, to_saves: list[str]):
		to_save = to_saves[0] if len(to_saves) == 1 else to_saves
		Configurations.set_conf(conf_to_set, to_save)
		text = SmartList()

		self.cli.set_out_stream(text.__iadd__)
		self.cli.parse(f't {flag}')
		text = ''.join(text).replace('\n', '')
		expected_ending = ', '.join(to_save) if 'langs' in name else str(to_save)
		self.assertTrue(text.endswith(expected_ending), f'"{text}" does not end with "{expected_ending}"')

	@parameterized.expand([
		('last_1st', F.C.LAST_LANG_LONG_FLAG, [1], 0),
		('last_3rd', F.C.LAST_LANG_LONG_FLAG, [3], 2),
		('long_last_first', F.C.LAST_1_LONG_FLAG, [], 0),
		('short_last_first', F.C.LAST_1_SHORT_FLAG, [], 0),
		('long_last_second', F.C.LAST_2_LONG_FLAG, [], 1),
		('short_last_second', F.C.LAST_2_SHORT_FLAG, [], 1),

	])
	def test_display_last_language(self, name: str, flag: str, flag_args: list[str], lang_to_take: int):
		Configurations.set_conf(F.C.LANGS_SHOW_LONG_FLAG, ['pl', 'es', 'zh', 'de'])
		text = SmartList()
		self.cli.set_out_stream(text.__iadd__)

		self.cli.parse(f't {flag} {" ".join(map(str, flag_args))}')

		actual_lang = '\n'.join(text).split(' ')[-1].replace('\n', '')
		expected_lang = Configurations.get_nth_saved_language(lang_to_take)
		self.assertEqual(expected_lang, actual_lang)

	@parameterized.expand([
		('long_flag', F.C.ADD_LANG_LONG_FLAG, 'pl'),
		('short_flag', F.C.ADD_LANG_SHORT_FLAG, 'de'),
	])
	def test_add_language(self, name, command: str, lang: str):
		self.cli.parse(f't {command} {lang} {F.F.SILENT_LONG_FLAG}')
		langs = Configurations.get_saved_languages()
		self.assertIn(f'{lang}', langs)

	@parameterized.expand([
		('long_flag', F.C.REMOVE_LANG_LONG_FLAG, 'pl'),
		('short_flag', F.C.REMOVE_LANG_SHORT_FLAG, 'de'),
	])
	def test_remove_language(self, name, command: str, lang: str):
		Configurations.add_langs(lang)
		self.cli.parse(f't {command} {lang}')
		langs = Configurations.get_saved_languages()
		self.assertNotIn(f'{lang}', langs)

	@parameterized.expand([
		('long_settings', F.C.SETTINGS_LONG_FLAG),
		('short_settings', F.C.SETTINGS_SHORT_FLAG),
	])
	def test_display_settings(self, name, flag):
		text = SmartList()
		self.cli.set_out_stream(text.__iadd__)

		self.cli.parse(f't {flag}')

		printed_keys = [f'--{t.split(":")[0]}' for t in text if '\n' not in t]
		configs = list(Configurations.get_all_configs().keys())

		self.assertEqual(configs, printed_keys)

	def test_add_multiple_langs(self):
		Configurations.add_langs('pl')
		self.cli.parse(f't -al pl pl pl pl pl {F.F.SILENT_LONG_FLAG}')
		self.assertEqual(1, len(list(filter(lambda l: l == 'pl', Configurations.get_saved_languages()))))

	@parameterized.expand([
		('by_lang', TranslationTypes.LANG, F.C.DOUBLE_MODE_STYLE_LONG_FLAG),
		('by_word', TranslationTypes.WORD, F.C.DOUBLE_MODE_STYLE_LONG_FLAG),
		('by_nothing_with_single', TranslationTypes.SINGLE, F.C.DOUBLE_MODE_STYLE_LONG_FLAG),
		('by_nothing_with_double', TranslationTypes.DOUBLE, F.C.DOUBLE_MODE_STYLE_LONG_FLAG),
		('by_lang_short', TranslationTypes.LANG, F.C.DOUBLE_MODE_STYLE_SHORT_FLAG),
		('by_word_short', TranslationTypes.WORD, F.C.DOUBLE_MODE_STYLE_SHORT_FLAG),
		('by_nothing_with_single_short', TranslationTypes.SINGLE, F.C.DOUBLE_MODE_STYLE_SHORT_FLAG),
		('by_nothing_with_double_short', TranslationTypes.DOUBLE, F.C.DOUBLE_MODE_STYLE_SHORT_FLAG),
	])
	def test_double_mode_formatting_in_one_flag(self, name: str, style: TranslationTypes, flag: str):
		self.cli.parse(f't {flag} {style}')
		self.assertEqual(style, Configurations.get_conf(F.C.DOUBLE_MODE_STYLE_LONG_FLAG))