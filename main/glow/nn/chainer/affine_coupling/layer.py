from chainer.backends import cuda
import chainer.functions as cf

from ... import base
from .parameters import Parameters


class AffineCoupling(base.AffineCoupling):
    def __init__(self, nn):
        self.nn = nn

    def __call__(self, x):
        split = x.shape[1] // 2
        xa = x[:, :split]
        xb = x[:, split:]
        log_scale, translation = self.nn(xb)
        self.scale = cf.exp(log_scale)
        ya = self.scale * xa + translation
        yb = xb
        y = cf.concat((ya, yb), axis=1)
        return y

    def compute_log_determinant(self, h, w):
        return cf.log(abs(self.scale))


class NonlinearMapping(base.NonlinearMapping):
    def __init__(self, params: Parameters, reverse=False):
        self.params = params
        self.reverse = reverse

    def __call__(self, x):
        channels = x.shape[1]
        out = cf.relu(self.params.conv_1(x))
        out = cf.relu(self.params.conv_2(x))
        out = self.params.conv_3(x)
        log_scale = out[:, :channels]
        translation = out[:, channels:]
        if self.reverse:
            return -log_scale, -translation
        return log_scale, translation
