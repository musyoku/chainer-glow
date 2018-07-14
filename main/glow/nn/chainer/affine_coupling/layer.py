from chainer.backends import cuda
import chainer.functions as cf

from ... import base
from .parameters import Parameters


class NonlinearMapping(base.NonlinearMapping):
    def __init__(self, params: Parameters):
        self.params = params

    def __call__(self, x):
        out = x
        out = cf.relu(self.params.conv_1(out))
        out = cf.relu(self.params.conv_2(out))
        out = self.params.conv_3(out)
        log_scale = out[:, 0::2]
        translation = out[:, 1::2]
        return log_scale, translation


class AffineCoupling(base.AffineCoupling):
    def __init__(self, nn: NonlinearMapping):
        self.nn = nn

    def __call__(self, x):
        xa = x[:, 0::2]
        xb = x[:, 1::2]
        log_scale, translation = self.nn(xb)
        scale = cf.sigmoid(log_scale + 2)
        ya = scale * (xa + translation)
        yb = xb
        y = cf.concat((ya, yb), axis=1)
        log_det = self.compute_log_determinant(scale)
        return y, log_det

    def compute_log_determinant(self, scale):
        return cf.sum(cf.log(abs(scale)))


class ReverseAffineCoupling(base.ReverseAffineCoupling):
    def __init__(self, nn: NonlinearMapping):
        self.nn = nn

    def __call__(self, y):
        ya = y[:, 0::2]
        yb = y[:, 1::2]
        log_scale, translation = self.nn(yb)
        scale = cf.sigmoid(log_scale + 2)
        xa = ya / scale - translation
        xb = yb
        x = cf.concat((xa, xb), axis=1)
        return x
