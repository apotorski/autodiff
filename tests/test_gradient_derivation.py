from unittest import TestCase, main

import numpy as np
from autodiff import ComputationalGraph, derive_gradient, parameter


class GradientDerivationTestCase(TestCase):

    def setUp(self, /) -> None:
        x = parameter()
        y = parameter()

        # Rosenbrock function
        f = (1.0 - x)**2.0 + 100.0*(y - x**2.0)**2.0

        self._cg = ComputationalGraph([x, y], [f])
        self._cg.compile()

    def test_gradient_derivation(self, /) -> None:
        actual_gradient = derive_gradient(self._cg)

        desired_gradient = lambda x, y: np.array([
            [400.0*x**3.0 + 2.0*x - 400.0*x*y - 2.0],
            [-200*x**2.0 + 200.0*y]
        ])

        x = np.array(np.random.randn())
        y = np.array(np.random.randn())

        actual_gradient_value, = actual_gradient(x, y)
        desired_gradient_value = desired_gradient(x, y)

        np.testing.assert_allclose(actual_gradient_value,
                                   desired_gradient_value)


if __name__ == '__main__':
    main()
