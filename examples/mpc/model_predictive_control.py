import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib._enums import CapStyle
from matplotlib.patches import Circle, Rectangle

from autodiff import (ComputationalGraph, constant, cos, differentiate, empty,
                      parameter, sin)

CART_WIDTH = 0.5
CART_HEIGHT = 0.25

BALL_RADIUS = 0.1

WARM_UP_STEPS = 50

SIMULATION_LENGTH = 10.0


class Environment:

    _g = 9.807

    def __init__(self, m_p=0.1, m_c=1.0, l=1.0, u_max=16.0, dt=2e-2):
        self._m_p = m_p
        self._m_c = m_c
        self._l = l

        self._u_max = u_max

        self._dt = dt

        self._x = np.zeros(4)

    def __call__(self, u):
        u = np.clip(u, -self._u_max, self._u_max)

        self._x[0] += self._x[2]*self._dt
        self._x[1] += self._x[3]*self._dt

        self._x[2] += \
            (u + self._m_p*np.sin(self._x[1])*(self._l*self._x[3]**2.0
            + self._g*np.cos(self._x[1]))) \
            / (self._m_c + self._m_p*np.sin(self._x[1])**2.0) \
                * self._dt
        self._x[3] += \
            (-u*np.cos(self._x[1]) - self._m_p*self._l*self._x[3]**2.0 \
            * np.cos(self._x[1])*np.sin(self._x[1]) - (self._m_c + self._m_p)
            * self._g*np.sin(self._x[1])) \
            / (self._l*(self._m_c + self._m_p*np.sin(self._x[1])**2.0)) \
                * self._dt

        return self._x

    def reset(self):
        self._x = np.zeros(4)

        return self._x.copy()

    @property
    def dt(self):
        return self._dt

    @property
    def g(self):
        return self._g

    @property
    def m_p(self):
        return self._m_p

    @property
    def m_c(self):
        return self._m_c

    @property
    def l(self):
        return self._l

    @property
    def u_max(self):
        return self._u_max


class ModelPredictiveController:

    def __init__(self, environment, alpha=1e-3, beta=0.9, gamma=1e-2,
                 epsilon=1e-8, T=100, optimization_steps=128):
        self._environment = environment

        dt = constant(environment.dt)
        g = constant(environment.g)

        m_p = constant(environment.m_p)
        m_c = constant(environment.m_c)
        l = constant(environment.l)

        x_0 = parameter(4)
        r = parameter(4)
        u = parameter(T, 1)

        x = empty(T, 4)


        x[0, 0] = x_0[0] + x_0[2]*dt
        x[0, 1] = x_0[1] + x_0[3]*dt

        x[0, 2] = x_0[2] \
            + (u[0, 0] + m_p*sin(x_0[1])*(l*x_0[3]**2.0
            + g*cos(x_0[1])))/(m_c + m_p*sin(x_0[1])**2.0) * dt
        x[0, 3] = x_0[3] \
            + (-u[0, 0]*cos(x_0[1]) - m_p*l*x_0[3]**2.0
            * cos(x_0[1])*sin(x_0[1]) - (m_c + m_p)*g
            * sin(x_0[1]))/(l*(m_c + m_p*sin(x_0[1])**2.0)) * dt

        for t in range(1, T):
            x[t, 0] = x[t - 1, 0] + x[t - 1, 2]*dt
            x[t, 1] = x[t - 1, 1] + x[t - 1, 3]*dt

            x[t, 2] = x[t - 1, 2] \
                + (u[t, 0] + m_p*sin(x[t - 1, 1])*(l*x[t - 1, 3]**2.0
                + g*cos(x[t - 1, 1])))/(m_c + m_p*sin(x[t - 1, 1])**2.0) * dt
            x[t, 3] = x[t - 1, 3] \
                + (-u[t, 0]*cos(x[t - 1, 1]) - m_p*l*x[t - 1, 3]**2.0
                * cos(x[t - 1, 1])*sin(x[t - 1, 1]) - (m_c + m_p)*g
                * sin(x[t - 1, 1]))/(l*(m_c + m_p*sin(x[t - 1, 1])**2.0)) * dt

        e = r - x

        e_mask = np.zeros(e.shape)
        e_mask[:-1, :2] = e_mask[-1, :] = 1.0
        e *= e_mask

        loss = 0.5*((e**2.0).mean() + gamma*(u**2.0).mean())

        self._model = ComputationalGraph([r, x_0, u], [loss])
        self._model.compile()

        loss_gradient = differentiate(self._model, u, loss) \
            .transpose().reshape(u.shape)
        self._model.output_tensors = [loss_gradient]
        self._model.compile()

        self._u = np.random.uniform(-1.0, 1.0, size=u.shape)

        self._alpha = alpha
        self._beta = beta
        self._epsilon = epsilon

        self._s = np.zeros_like(self._u)

        self._optimization_steps = optimization_steps

    def __call__(self, r, x_0):
        for _ in range(self._optimization_steps):
            g, = self._model(r, x_0, self._u*environment.u_max)

            self._s = self._beta*self._s + (1.0 - self._beta)*g**2.0
            self._u -= self._alpha*g/np.sqrt(self._s + self._epsilon)

            self._u = np.clip(self._u, -1.0, 1.0)

        u = self._u[0]*environment.u_max
        self._u[:-1] = self._u[1:]

        return u


if __name__ == '__main__':
    environment = Environment()
    controller = ModelPredictiveController(environment)

    simulation_steps = int(SIMULATION_LENGTH / environment.dt)

    states = []
    control_signals = []

    commanded_state = np.array([
        [0.0],
        [np.pi],
        [0.0],
        [0.0]
    ])

    state = environment.reset()

    for _ in range(WARM_UP_STEPS):
        controller(commanded_state, state)

    for step in range(simulation_steps):
        states.append(state.copy())

        control_signal = controller(commanded_state, state)
        control_signals.append(control_signal.copy())

        state = environment(control_signal)

        print(f'{step + 1} / {simulation_steps}')

    states = np.stack(states)


    # create animation
    px = 1.0 / plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(figsize=(640*px, 640*px))
    ax.grid()
    ax.set_aspect('equal')
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.plot((-2.0, 2.0), (0.0, 0.0), color='tab:gray')

    x, theta, _, _ = states[0]

    cart = Rectangle((
        x - CART_WIDTH / 2.0,
        0.0 - CART_HEIGHT / 2.0
    ), width=CART_WIDTH, height=CART_HEIGHT, color='tab:blue', zorder=4,
    capstyle=CapStyle.round)
    ax.add_patch(cart)

    ball = Circle((
        x + environment.l*np.cos(theta - np.pi / 2.0),
        environment.l*np.sin(theta - np.pi / 2.0)
    ), radius=BALL_RADIUS, color='tab:blue', zorder=4)
    ax.add_patch(ball)

    pendulum, = ax.plot(
        (x, x + environment.l*np.cos(theta - np.pi / 2.0)),
        (0.0, environment.l*np.sin(theta - np.pi / 2.0)),
        color='tab:blue', lw=2, solid_capstyle=CapStyle.round,
    )

    fig.tight_layout()

    def animate(state):
        x, theta, _, _ = state

        cart.set(xy=(
            x - CART_WIDTH / 2.0,
            0.0 - CART_HEIGHT / 2.0
        ))

        ball.set(center=(
            x + environment.l*np.cos(theta - np.pi / 2.0),
            environment.l*np.sin(theta - np.pi / 2.0)
        ))

        pendulum.set_data(
            (x, x + environment.l*np.cos(theta - np.pi / 2.0)),
            (0.0, environment.l*np.sin(theta - np.pi / 2.0))
        )

        return cart, ball, pendulum

    animation.FuncAnimation(
        fig, animate, states, interval=int(environment.dt*1e3), blit=True) \
            .save('model_predictive_control_demo.gif',
                  fps=int(1 / environment.dt))