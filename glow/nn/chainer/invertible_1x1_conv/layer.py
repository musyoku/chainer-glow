from chainer.functions.connection import convolution_2d
from chainer.backends import cuda
import chainer.functions as cf

from ... import base
from .parameters import Parameters, LUParameters


class Invertible1x1Conv(base.Invertible1x1Conv):
    def __init__(self, params: Parameters):
        self.params = params

    def __call__(self, x):
        log_det = self.compute_log_determinant(x)
        y = self.params.conv(x)
        return y, log_det

    def compute_log_determinant(self, x):
        h, w = x.shape[2:]
        W = self.params.conv.W
        det = cf.det(W)
        if det.data == 0:
            det += 1e-16
        return h * w * cf.log(abs(det))


class ReverseInvertible1x1Conv(base.ReverseInvertible1x1Conv):
    def __init__(self, params: Parameters):
        self.params = params

    def __call__(self, y):
        log_det = self.compute_log_determinant(y)
        x = self.params.conv(y)
        return x, log_det

    def compute_log_determinant(self, x):
        h, w = x.shape[2:]
        W = self.params.conv.W
        return h * w * cf.log(abs(cf.det(W)))


class LUInvertible1x1Conv(base.Invertible1x1Conv):
    def __init__(self, params: LUParameters):
        self.params = params

    def __call__(self, x):
        y = convolution_2d.convolution_2d(x, self.params.W, None, 1, 0)
        log_det = self.compute_log_determinant(x)
        return y, log_det

    def compute_log_determinant(self, x):
        h, w = x.shape[2:]
        return h * w * cf.sum(cf.log(abs(self.params.s)))


class LUReverseInvertible1x1Conv(base.ReverseInvertible1x1Conv):
    def __init__(self, params: Parameters):
        self.params = params

    def __call__(self, y):
        x = self.params.conv(y)
        log_det = self.compute_log_determinant(y)
        return x, log_det

    def compute_log_determinant(self, x):
        h, w = x.shape[2:]
        return h * w * cf.sum(cf.log(abs(self.params.s)))