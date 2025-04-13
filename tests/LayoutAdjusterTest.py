from parameterized import parameterized

from src.glosbe.configurating import Configurations
from src.glosbe.constants import FLAGS
from src.glosbe.layoutAdjusting.layoutAdjuster import LayoutAdjustmentsMethods
from tests.abstractCliTest import AbstractCliTest


class LayoutAdjusterTest(AbstractCliTest):

	@classmethod
	def _get_test_name(cls) -> str:
		return 'Layout Adjuster'

	def setUp(self) -> None:
		super().setUp()
		Configurations.init(self.get_file_name())

	def tearDown(self) -> None:
		super().tearDown()
		Configurations.remove_current_configuration()

	@parameterized.expand([
		('keyboard_all_langs_and_flags', ['--single', 'pl', 'en'], ['trzymać'], 't trzymać зд ут -і', LayoutAdjustmentsMethods.KEYBOARD, 'uk'),
		('keyboard_some_langs_and_flags', ['--lang', '--word', 'uk', 'zh', 'es', 'de'], ['світло', 'привіт'], 't гл -m яр es ву -ц світло привіт', LayoutAdjustmentsMethods.KEYBOARD, 'uk'),
		('native_all_langs_and_flags', ['--single', 'pl', 'en'], ['trzymać'], 't trzymać пл ен -с', LayoutAdjustmentsMethods.NATIVE, 'uk'),
		('native_some_langs_and_flags', ['--lang', '--word', 'uk', 'zh', 'es', 'de'], ['світло', 'привіт'], 't ук -m дж es де -в світло привіт', LayoutAdjustmentsMethods.NATIVE, 'uk'),
		('native_chinese', ['--single', 'pl', 'en'], ['trzymać'], 't trzymać 波 英 -s', LayoutAdjustmentsMethods.NATIVE, 'zh'),
	]) # TODO: test setting adjusting method
	def test_adjuster(self, name: str, expected_adjusted: list[str], expected_not_adjusted: list[str], input_string: str, method: str, lang: str):
		Configurations.set_conf(FLAGS.CONFIGURATIONAL.LAYOUT_ADJUSTMENT_METHOD_LONG_FLAG, method)
		Configurations.set_conf(FLAGS.CONFIGURATIONAL.LAYOUT_ADJUSTMENT_LANG_LONG_FLAG, lang)

		self.cli.parse_without_actions(input_string)
		flags = list(map(Flag.get_name, self.cli.root.get_active_flags()))
		langs = self.cli.langs
		actual_not_adjusted = self.cli.words
		actual_adjusted = flags + langs

		self.assertCountEqual(expected_adjusted, actual_adjusted, 'Something has not been correctly adjusted')
		self.assertCountEqual(expected_not_adjusted, actual_not_adjusted, "Adjuster adjusted what shouldn't be")