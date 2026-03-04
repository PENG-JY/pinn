import numpy as np
import tensorflow as tf


class Network_FNN:
    """Fourier-feature feed-forward network.

    Architecture: RFF(x) -> softplus -> softplus -> softplus -> linear
    Input is mapped to a 512-dim Fourier feature space before the MLP.
    """

    def __init__(self, input_dim=4, fourier_dim=256, hidden=None, output_dim=1):
        if hidden is None:
            hidden = [64, 64, 64]
        self.B = tf.constant(
            np.random.normal(scale=1.0, size=(input_dim, fourier_dim)), dtype=tf.float32
        )
        layer_sizes = [fourier_dim * 2] + hidden + [output_dim]
        self.layers = layer_sizes
        self.weight, self.biases = self._init_weights(layer_sizes)
        self.variables = self.weight + self.biases

    def _init_weights(self, sizes):
        weights, biases = [], []
        for i in range(len(sizes) - 1):
            std = np.sqrt(2.0 / (sizes[i] + sizes[i + 1]))
            weights.append(tf.Variable(
                tf.random.truncated_normal([sizes[i], sizes[i + 1]], stddev=std), dtype=tf.float32
            ))
            biases.append(tf.Variable(tf.zeros([1, sizes[i + 1]], dtype=tf.float32)))
        return weights, biases

    def _fourier(self, x):
        proj = tf.matmul(x, self.B)
        return tf.concat([tf.sin(proj), tf.cos(proj)], axis=-1)

    def __call__(self, x, lb=None, ub=None):
        h = self._fourier(x)
        for W, b in zip(self.weight[:-1], self.biases[:-1]):
            h = tf.nn.softplus(tf.matmul(h, W) + b)
        return tf.matmul(h, self.weight[-1]) + self.biases[-1]
