import chainer
import chainer.functions as cf
import numpy as np
from chainer.backends import cuda


class Actnorm(chainer.Chain):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        with self.init_scope():
            shape = (1, channels, 1, 1)
            self.scale = chainer.Parameter(
                initializer=np.zeros(shape, dtype="float32"))
            self.bias = chainer.Parameter(
                initializer=np.zeros(shape, dtype="float32"))

    def forward_step(self, x):
        bias = cf.broadcast_to(self.bias, x.shape)
        scale = cf.broadcast_to(self.scale, x.shape)
        y = (x + bias) * scale

        log_det = self.compute_log_determinant(x, self.scale)

        return y, log_det

    def reverse_step(self, y):
        bias = cf.broadcast_to(self.bias, y.shape)
        scale = cf.broadcast_to(self.scale, y.shape)
        x = y / scale - bias

        log_det = -self.compute_log_determinant(x, self.scale)

        return x, log_det

    def compute_log_determinant(self, x, scale):
        h, w = x.shape[2:]
        return h * w * cf.sum(cf.log(abs(scale)))