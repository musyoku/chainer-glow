from chainer.backends import cuda
import chainer.functions as cf

from ... import base
from .parameters import Parameters


class Actnorm(base.Actnorm):
    def __init__(self, params: Parameters):
        self.params = params
        self.bias = self.params.bias.b
        self.scale = self.params.scale.W

    def __call__(self, x):
        inter = cf.bias(x, self.bias)
        y = cf.scale(inter, self.scale)
        log_det = self.compute_log_determinant(x)
        return y, log_det

    def compute_log_determinant(self, x):
        h, w = x.shape[2:]
        return h * w * cf.sum(cf.log(abs(self.scale)))


class ReverseActnorm(base.ReverseActnorm):
    def __init__(self, params: Parameters):
        self.params = params
        self.bias = self.params.bias.b
        self.scale = self.params.scale.W

    def __call__(self, y):
        inter = cf.scale(y, 1.0 / self.scale)
        x = cf.bias(inter, -self.bias)
        return x
