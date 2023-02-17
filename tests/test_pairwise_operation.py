from unittest import TestCase, main

import numpy as np
from autodiff import ComputationalGraph, atan2, parameter


class PairwiseOperationTestCase(TestCase):

    def setUp(self, /) -> None:
        self._operand_1 = parameter(5, 4, 3, 2, 1)
        self._operand_2 = parameter(4, 1, 2, 1)

        result = atan2(self._operand_1, self._operand_2)

        self._cg = ComputationalGraph([self._operand_1, self._operand_2],
                                      [result])
        self._cg.compile()

    def test_pairwise_operation(self, /) -> None:
        operand_1_value = np.random.randn(*self._operand_1.shape)
        operand_2_value = np.random.randn(*self._operand_2.shape)

        actual_result_value, = self._cg(operand_1_value, operand_2_value)
        expected_result_value = np.arctan2(operand_1_value, operand_2_value)

        np.testing.assert_allclose(actual_result_value, expected_result_value)


if __name__ == '__main__':
    main()
