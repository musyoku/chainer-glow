from chainer.backends import cuda
import chainer.functions as cf

from ... import base
from .parameters import Parameters


class Actnorm(base.Actnorm):
    def __init__(self, params: Parameters):
        self.params = params
        self.bias = self.params.bias
        self.scale = self.params.scale

    def __call__(self, x):
        scale = cf.tanh(self.scale + 2) + 1e-12

        y = x + cf.broadcast_to(self.bias, x.shape)
        y = y * cf.broadcast_to(scale, x.shape)

        log_det = self.compute_log_determinant(x, scale)

        return y, log_det

    def compute_log_determinant(self, x, scale):
        h, w = x.shape[2:]
        return h * w * cf.sum(cf.log(abs(scale)))


class ReverseActnorm(base.ReverseActnorm):
    def __init__(self, params: Parameters):
        self.params = params
        self.bias = self.params.bias
        self.scale = self.params.scale

    def __call__(self, y):
        scale = cf.tanh(self.scale + 2) + 1e-12

        x = y / cf.broadcast_to(scale, y.shape)
        x = x - cf.broadcast_to(self.bias, y.shape)

        log_det = self.compute_log_determinant(x, scale)

        return x, log_det

    def compute_log_determinant(self, x, scale):
        h, w = x.shape[2:]
        return -h * w * cf.sum(cf.log(abs(scale)))
