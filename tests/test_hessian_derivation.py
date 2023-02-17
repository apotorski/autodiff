from unittest import TestCase, main

import numpy as np
from autodiff import ComputationalGraph, derive_hessian, parameter


class HessianDerivationTestCase(TestCase):

    def setUp(self, /) -> None:
        x = parameter()
        y = parameter()

        # Rosenbrock function
        f = (1.0 - x)**2.0 + 100.0*(y - x**2.0)**2.0

        self._cg = ComputationalGraph([x, y], [f])
        self._cg.compile()

    def test_hessian_derivation(self, /) -> None:
        actual_hessian = derive_hessian(self._cg)

        desired_hessian = lambda x, y: np.array([
            [1200.0*x**2.0 - 400.0*y + 2.0, -400.0*x],
            [                       -400*x,    200.0]
        ])

        x = np.array(np.random.randn())
        y = np.array(np.random.randn())

        actual_hessian_value, = actual_hessian(x, y)
        desired_hessian_value = desired_hessian(x, y)

        np.testing.assert_allclose(actual_hessian_value, desired_hessian_value)



if __name__ == '__main__':
    main()
