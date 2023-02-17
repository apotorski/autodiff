from unittest import TestCase, main

import numpy as np
from autodiff import ComputationalGraph, parameter


class MatrixMultiplicationTestCase(TestCase):

    def setUp(self, /) -> None:
        self._operand_1 = parameter(5, 4, 3, 2, 1)
        self._operand_2 = parameter(4, 1, 1, 2)

        result = self._operand_1 @ self._operand_2

        self._cg = ComputationalGraph([self._operand_1, self._operand_2],
                                      [result])
        self._cg.compile()

    def test_matrix_multiplication(self, /) -> None:
        operand_1_value = np.random.randn(*self._operand_1.shape)
        operand_2_value = np.random.randn(*self._operand_2.shape)

        actual_result_value, = self._cg(operand_1_value, operand_2_value)
        desired_result_value = operand_1_value @ operand_2_value

        np.testing.assert_allclose(actual_result_value, desired_result_value)


if __name__ == '__main__':
    main()
