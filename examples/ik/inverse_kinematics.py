import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, patheffects
from matplotlib._enums import CapStyle, JoinStyle

from autodiff import (ComputationalGraph, Tensor, atan2, constant, cos,
                      differentiate, empty, parameter, sin)

FPS = 50

SECONDS_PER_TRANSITION = 2.0


class ArticulatedRobot:

    def __init__(self, configuration, epsilon=1e-6):
        self._epsilon = epsilon

        L = constant(configuration)
        theta = parameter(len(L), 1)
        p = empty(len(L) + 1, 2)

        current_transform = self._T(theta=theta[0])
        p[0, 0] = current_transform[0, 2]
        p[0, 1] = current_transform[1, 2]
        for i in range(len(theta) - 1):
            current_transform @= self._T(theta=theta[i + 1], p_x=L[i])
            p[i + 1, 0] = current_transform[0, 2]
            p[i + 1, 1] = current_transform[1, 2]
        current_transform @= self._T(p_x=L[5])
        p[6, 0] = current_transform[0, 2]
        p[6, 1] = current_transform[1, 2]

        y = Tensor.from_tensors([
            atan2(
                current_transform[1, 0],
                current_transform[0, 0]),
            current_transform[0, 2],
            current_transform[1, 2]
        ], shape=(3, 1))

        self._model = ComputationalGraph([theta], [p, y])
        self._model.compile()

        jacobian_tensor = differentiate(self._model, theta, y)
        self._model.output_tensors.append(jacobian_tensor)
        self._model.compile()


        self._theta_value = np.zeros((len(L), 1))

    def __call__(self, r):
        while True:
            p, y, J = self._model(self._theta_value)
            e = r - y

            if np.linalg.norm(e) <= self._epsilon:
                break

            self._theta_value += np.linalg.pinv(J) @ e
            self._theta_value %= 2.0*np.pi

        return p

    @staticmethod
    def _T(theta=constant(0.0), p_x=constant(0.0), p_y=constant(0.0)):
        return Tensor.from_tensors([
            cos(theta),      -sin(theta),           p_x,
            sin(theta),       cos(theta),           p_y,
            constant(0.0), constant(0.0), constant(1.0)
        ], (3, 3))


if __name__ == '__main__':
    articulated_robot = ArticulatedRobot(configuration=np.ones(6))

    steps_per_movement = int(SECONDS_PER_TRANSITION * FPS)

    end_effector_configurations = np.array([
        [[-np.pi / 2.0], [2.0], [2.0]],
        [[-np.pi / 2.0], [2.0], [1.0]],
        [[-np.pi / 2.0], [1.0], [1.0]],
        [[-np.pi / 2.0], [2.0], [2.0]],
        [[         0.0], [2.0], [2.0]],
        [[-np.pi / 2.0], [2.0], [2.0]]
    ])

    interpolated_end_effector_configurations = np.concatenate([
        np.linspace(end_effector_configuration_1, end_effector_configuration_2,
                    steps_per_movement)
        for end_effector_configuration_1, end_effector_configuration_2
        in zip(end_effector_configurations[:-1],
               end_effector_configurations[1:])
    ])

    joint_positions = [
        articulated_robot(interpolated_end_effector_configuration)
        for interpolated_end_effector_configuration
        in interpolated_end_effector_configurations
    ]


    # create animation
    px = 1.0 / plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(figsize=(640*px, 640*px))
    ax.grid()
    ax.set_aspect('equal')
    ax.set_xlim([-1.0, 4.0])
    ax.set_ylim([-1.0, 4.0])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.plot(end_effector_configurations[:4, 1],
            end_effector_configurations[:4, 2],
            'k--')

    line, = ax.plot(
        [], [], lw=10,
        path_effects=[patheffects.SimpleLineShadow(), patheffects.Normal()],
        solid_capstyle=CapStyle.round, solid_joinstyle=JoinStyle.round)

    fig.tight_layout(pad=2.0)

    def animate(joint_positions):
        line.set_data(joint_positions[:, 0], joint_positions[:, 1])

        return line,

    animation.FuncAnimation(
        fig, animate, joint_positions, interval=int(1e3 / FPS), blit=True) \
            .save('inverse_kinematics_demo.gif', fps=FPS)
