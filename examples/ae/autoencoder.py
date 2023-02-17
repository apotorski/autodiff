import abc

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from autodiff import ComputationalGraph, Tensor, differentiate, parameter, tanh

DATASET_SIZE = 16

EPOCHS = 5000

LOG_PRINT_PERIOD = 200
LOG_PLOT_PERIOD = 10

ANIMATION_FPS = 50


class Linear:

    def __init__(self, input_features, output_features):
        self._W = parameter(output_features, input_features)
        self._B = parameter(output_features, 1)

        # normalized Xavier weight initialization
        bound = np.sqrt(6.0) / np.sqrt(self._W.shape[0] + self._W.shape[1])
        self._W_value = np.random.uniform(-bound, bound, self._W.shape)
        self._B_value = np.zeros(self._B.shape)

        self._parameters = [
            self._W,
            self._B
        ]

    def __call__(self, X):
        return self._W @ X + self._B

    @property
    def parameters(self):
        return Tensor.from_tensors(
            self._parameters, shape=(sum(map(len, self._parameters)), 1))

    def generate_initial_parameter_values(self):
        return np.expand_dims(
            np.concatenate([
                self._W_value.flatten(),
                self._B_value.flatten()
            ]), axis=-1)


class AutoencoderModule(abc.ABC):

    def __init__(self, *linears):
        self._parameters = Tensor.from_tensors(
            parameters := [linear.parameters for linear in linears],
            shape=(sum(map(len, parameters)), 1))

        self._parameter_values = np.concatenate(
            [linear.generate_initial_parameter_values() for linear in linears])

    @abc.abstractmethod
    def __call__(self, X):
        ...

    @property
    def parameters(self):
        return self._parameters

    @property
    def parameter_values(self):
        return self._parameter_values

    @parameter_values.setter
    def parameter_values(self, value):
        self._parameter_values = value


class Encoder(AutoencoderModule):

    def __init__(self):
        self._input_linear = Linear(2, 10)
        self._hidden_linear = Linear(10, 1)

        super().__init__(
            self._input_linear,
            self._hidden_linear
        )

    def __call__(self, X):
        return self._hidden_linear(tanh(self._input_linear(X)))


class Decoder(AutoencoderModule):

    def __init__(self):
        self._hidden_linear = Linear(1, 10)
        self._output_linear = Linear(10, 2)

        super().__init__(
            self._hidden_linear,
            self._output_linear
        )

    def __call__(self, H):
        return self._output_linear(tanh(self._hidden_linear(H)))


class Optimizer:

    """Gradient descent with momentum."""

    def __init__(self, parameters, epsilon=1e-2, mu=0.99):
        self._epsilon = epsilon
        self._mu = mu

        self._v = np.zeros(parameters.shape)

    def __call__(self, g):
        self._v = self._mu*self._v - self._epsilon*g

        return self._v





if __name__ == '__main__':
    # generate dataset - points on unit circle
    theta = np.linspace(0.0, 2.0*np.pi, num=DATASET_SIZE, endpoint=False)
    input_data = \
        np.expand_dims(
            np.stack([
                np.cos(theta),
                np.sin(theta)
            ], axis=-1),
            axis=-1)


    # create model
    alpha = 1e-4

    encoder = Encoder()
    decoder = Decoder()

    X = parameter(*input_data.shape)

    H = encoder(X)
    R = decoder(H)

    reconstruction_loss = ((X - R)**2.0).mean()
    regularization_loss = 0.5*((encoder.parameters**2.0).sum() \
                             + (decoder.parameters**2.0).sum())

    loss = reconstruction_loss + alpha*regularization_loss

    cg = ComputationalGraph([X, encoder.parameters, decoder.parameters],
                            [loss])
    cg.compile()

    encoder_gradient = differentiate(cg, encoder.parameters, loss).transpose()
    decoder_gradient = differentiate(cg, decoder.parameters, loss).transpose()
    cg.output_tensors.append(encoder_gradient)
    cg.output_tensors.append(decoder_gradient)
    cg.compile()


    # initialize optimizers
    encoder_optimizer = Optimizer(encoder.parameters)
    decoder_optimizer = Optimizer(decoder.parameters)


    # train model
    reconstructed_data = []

    codes = []

    epochs = []
    losses = []

    for epoch in range(EPOCHS):
        loss_value, encoder_gradient_value, decoder_gradient_value = \
            cg(input_data, encoder.parameter_values, decoder.parameter_values)

        encoder.parameter_values += encoder_optimizer(encoder_gradient_value)
        decoder.parameter_values += decoder_optimizer(decoder_gradient_value)

        if epoch % LOG_PRINT_PERIOD == 0:
            print(f'epoch {epoch} / {EPOCHS}, loss = {loss_value:.6f}')

        if epoch % LOG_PLOT_PERIOD == 0:
            reconstructed_data.append(R.access())

            codes.append(H.access().flatten())

            epochs.append(epoch)
            losses.append(loss_value)


    # create animation
    px = 1.0 / plt.rcParams['figure.dpi']
    fig = plt.figure(figsize=(640*px, 640*px))

    data_ax = fig.add_subplot(221)
    data_ax.grid()
    data_ax.set_aspect('equal')
    data_ax.set_xlabel('x')
    data_ax.set_ylabel('y')
    data_ax.set_xlim([-2.0, 2.0])
    data_ax.set_ylim([-2.0, 2.0])
    data_ax.set_xticklabels([])
    data_ax.set_yticklabels([])

    code_ax = fig.add_subplot(222)
    code_ax.grid()
    code_ax.set_xlabel('original code')
    code_ax.set_ylabel('generated code')
    code_ax.set_ylim([-2.0*np.pi, 2.0*np.pi])

    loss_ax = fig.add_subplot(212)
    loss_ax.grid()
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    loss_ax.set_yscale('log')

    input_data_scatter = data_ax.scatter(input_data[:, 0], input_data[:, 1],
                                         s=80, facecolors='none',
                                         edgecolors='b', label='input data')
    reconstructed_data_scatter = data_ax.scatter([], [], marker='o',
                                                 label='reconstructed data')
    data_ax.legend(loc='upper right')

    code_line, = code_ax.plot(theta, codes[0], 'o')

    loss_line, = loss_ax.plot(epochs, losses)

    fig.tight_layout(pad=2.0)

    def animate(t):
        reconstructed_data_scatter.set_offsets(reconstructed_data[t])
        code_line.set_ydata(codes[t])
        loss_line.set_data(epochs[:t], losses[:t])

        return reconstructed_data_scatter, code_line, loss_line

    interval = int(1e3 / ANIMATION_FPS)
    animation.FuncAnimation(
        fig, animate, len(epochs), interval=interval, blit=True) \
            .save('autoencoder_demo.gif', fps=ANIMATION_FPS)
