from unittest import TestCase, main

import numpy as np
from autodiff import ComputationalGraph, parameter, tanh


class ElementwiseOperationTestCase(TestCase):

    def setUp(self, /) -> None:
        self._operand = parameter(5, 4, 3, 2, 1)

        result = tanh(self._operand)

        self._cg = ComputationalGraph([self._operand], [result])
        self._cg.compile()

    def test_elementwise_operation(self, /) -> None:
        operand_value = np.random.randn(*self._operand.shape)

        actual_result_value, = self._cg(operand_value)
        expected_result_value = np.tanh(operand_value)

        np.testing.assert_allclose(actual_result_value, expected_result_value)


if __name__ == '__main__':
    main()
