from unittest import skipIf

from parameterized import parameterized

from tests.abstractCliTest import AbstractCliTest


class HelpTest(AbstractCliTest):

	@classmethod
	def _get_test_name(cls) -> str:
		return 'Help'

	@parameterized.expand([
		('no_args_in_single', 't -s'),
		('no_args_in_multi', 't -m'),
		('no_args_in_word', 't -w'),
		('no_args_in_double', 't -w -m'),
		('implicit_help', 't -h'),
	])
	@skipIf(lambda: HelpTest.cli.root.help == '', 'no help implemented')
	def test_help_printing(self, name: str, input_line: str):
		self.cli.parse(input_line)
		self.fail(NotImplementedError.__name__)


'''
	root.help.name = 'trans'
            root.help.short_description = 'Translates any word from and to any language'
            single_node.help.short_description = 'Translates a single word'
            single_node.help.synopsis = 'trans <WORD> [FROM_LANG] [TO_LANG]'
            word_node.help.short_description = 'Translates many words to a single node'
            word_node.help.synopsis = 'trans [FROM_LANG] [TO_LANG] [-w] <WORD>...'
            lang_node.help.short_description = 'Translates a word to many languages'
            lang_node.help.synopsis = 'trans <WORD> [FROM_LANG] [-m] <TO_LANG>...'
            double_multi_node.help.short_description = 'Translates many words into many languages'
            double_multi_node.help.synopsis = 'trans [FROM_LANG] -w <WORD>... -m <TO_LANG>... '
            '''