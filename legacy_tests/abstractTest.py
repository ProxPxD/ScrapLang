import abc
import unittest
from typing import Iterable

from src.configurating import Configurations
from src.constants import FLAGS as F


class AbstractTest(unittest.TestCase, abc.ABC):
    half_sep_length = 40
    currentResult = None

    total = 0
    failure = 0
    errors = 0
    skipped = 0

    @classmethod
    def print_sep_with_text(cls, text: str, sep: str = '*') -> None:
        with_sep_lines = sep * cls.half_sep_length + f' {text} ' + sep * cls.half_sep_length
        over_length = len(with_sep_lines) - cls.half_sep_length*2
        to_print = with_sep_lines[over_length//2 : -over_length//2]
        print(to_print)

    @classmethod
    def setUpClass(cls) -> None:
        cls.print_sep_with_text(f'Starting {cls._get_test_name()} tests!')
        Configurations.init()
        Configurations.change_conf(F.C.LANGS_SHOW_LONG_FLAG, ['pl', 'en', 'de', 'es', 'uk', 'zh'])

    def setUp(self) -> None:
        if not self.get_method_name().startswith('test_'):
            return
        super().setUp()
        print('- ', self.get_method_name(), end=' ... ')

    def tearDown(self) -> None:
        if not self.get_method_name().startswith('test_'):
            return
        super().tearDown()
        result = self.defaultTestResult()
        self._feedErrorsToResult(result, self._outcome.errors)

        is_error = any(test == self for test, text in result.errors)
        is_failure = any(test == self for test, text in result.failures)
        is_skipped = any(test == self for test, text in result.skipped)
        passed = not (is_error or is_failure or is_skipped)

        self.__class__.total += 1
        if is_error:
            self.__class__.errors += 1
        if is_failure:
            self.__class__.failure += 1
        if is_skipped:
            self.__class__.skipped += 1

        print('PASS' if passed else 'ERROR' if is_error else 'FAIL' if is_failure else 'SKIP' if is_skipped else
        'WRONG UNIT TEST OUTCOME CHECKING! Investigate (possible incompatible with a python newer than 3.10)')

    @classmethod
    def tearDownClass(cls) -> None:
        cls.print_statistics(percentage=False)

    @classmethod
    def print_statistics(cls, failure=None, errors=None, skipped=None, total=None, *, short=False, percentage=True):
        if failure is None:
            failure = cls.failure
        if errors is None:
            errors = cls.errors
        if skipped is None:
            skipped = cls.skipped
        if total is None:
            total = cls.total
        failed = failure + errors
        passed = total - failed - skipped
        if short:
            print(f'({failure}F, {errors}E, {passed}P)/{total}')
        else:
            print(f'Failed: {failed} (Failures: {failure}, Errors: {errors}), Passed: {passed}, Total: {total}')
        if percentage:
            print(
                f'Failed: {100 * failed / total:.1f}% (Failures: {100 * failure / total:.1f}%, Errors: {100 * errors / total:.1f})%, Passed: {100 * passed / total:.1f}%')

    def run(self, result: unittest.result.TestResult | None = ...) -> unittest.result.TestResult | None:
        self.currentResult = result
        unittest.TestCase.run(self, result)

    @classmethod
    @abc.abstractmethod
    def _get_test_name(cls) -> str:
        return 'unnamed'

    def get_method_name(self) -> str:
        return self.id().split('.')[-1]

    @classmethod
    def _get_test_name(cls) -> str:
        return cls.__name__.removesuffix('Test')

    def get_parameterized_methods_of_current_test(self, *method_nums) -> Iterable:
        if not method_nums:
            return iter([])

        method_prefix = self.get_method_name() + '_'
        child_methods = [name for name in dir(self) if name.startswith(method_prefix)]

        if method_nums[0] is None:
            method_nums = range(len(child_methods))
        number_marks = (name.removeprefix(method_prefix).split('_')[0] for name in child_methods)
        zero_count = len(next((num for num in number_marks if all((ch == '0' for ch in num))), ''))

        method_prefixes = (f'{method_prefix}{str(i).zfill(zero_count)}' for i in method_nums)
        methods = (getattr(self, actual_name) for expected_prefix in method_prefixes for actual_name in child_methods if
                   actual_name.startswith(expected_prefix))
        return methods

    def run_current_test_with_params(self, *method_nums):  # TODO: support custom names
        '''
        Run in a method declared after the desired parametrized method and named the same as it
        :param method_nums:
        :return:
        '''
        for method in self.get_parameterized_methods_of_current_test(*method_nums):
            method()

