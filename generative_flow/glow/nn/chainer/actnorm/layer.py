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
        xp = cuda.get_array_module(self.bias.data)
        b = xp.broadcast_to(self.bias.data.reshape((1, -1, 1, 1)), x.shape).astype("float64")
        s = xp.broadcast_to(self.scale.data.reshape((1, -1, 1, 1)), x.shape).astype("float64")

        inter = cf.bias(x, -self.bias)
        y = cf.scale(inter, 1.0 / self.scale)
        log_det = self.compute_log_determinant(x)

        return (x - b) / s, log_det

        return y, log_det

    def compute_log_determinant(self, x):
        h, w = x.shape[2:]
        s = self.params.scale.W
        return h * w * cf.sum(cf.log(abs(s)))


class ReverseActnorm(base.ReverseActnorm):
    def __init__(self, params: Parameters):
        self.params = params
        self.bias = self.params.bias.b
        self.scale = self.params.scale.W

    def __call__(self, y):
        xp = cuda.get_array_module(self.bias.data)
        b = xp.broadcast_to(self.bias.data.reshape((1, -1, 1, 1)), y.shape).astype("float64")
        s = xp.broadcast_to(self.scale.data.reshape((1, -1, 1, 1)), y.shape).astype("float64")
        return y / s + b


        inter = cf.scale(y, self.scale)
        x = cf.bias(inter, self.bias)
        return x
