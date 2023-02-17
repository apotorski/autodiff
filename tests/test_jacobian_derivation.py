from unittest import TestCase, main

import numpy as np
from autodiff import ComputationalGraph, cos, derive_jacobian, parameter, sin


class JacobianDerivationTestCase(TestCase):

    def setUp(self, /) -> None:
        theta = parameter()
        phi = parameter()

        # cartesian coordinates of point on unit sphere
        x = sin(phi)*cos(theta)
        y = sin(phi)*sin(theta)
        z = cos(phi)

        self._cg = ComputationalGraph([theta, phi], [x, y, z])
        self._cg.compile()

    def test_jacobian_derivation(self, /) -> None:
        actual_jacobian = derive_jacobian(self._cg)

        desired_jacobian = lambda theta, phi: np.array([
            [-np.sin(phi)*np.sin(theta), np.cos(phi)*np.cos(theta)],
            [ np.sin(phi)*np.cos(theta), np.cos(phi)*np.sin(theta)],
            [                       0.0,              -np.sin(phi)]
        ])

        theta = np.array(np.random.randn())
        phi = np.array(np.random.randn())

        actual_jacobian_value, = actual_jacobian(theta, phi)
        desired_jacobian_value = desired_jacobian(theta, phi)

        np.testing.assert_allclose(actual_jacobian_value,
                                   desired_jacobian_value)


if __name__ == '__main__':
    main()
