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
        y = x + cf.broadcast_to(self.bias, x.shape)
        y = y * cf.broadcast_to(self.scale, x.shape)
        log_det = self.compute_log_determinant(x)
        return y, log_det

    def compute_log_determinant(self, x):
        h, w = x.shape[2:]
        return h * w * cf.sum(cf.log(abs(self.scale)))


class ReverseActnorm(base.ReverseActnorm):
    def __init__(self, params: Parameters):
        self.params = params
        self.bias = self.params.bias
        self.scale = self.params.scale

    def __call__(self, y):
        x = y / cf.broadcast_to(self.scale, y.shape)
        x = x - cf.broadcast_to(self.bias, y.shape)
        return x
