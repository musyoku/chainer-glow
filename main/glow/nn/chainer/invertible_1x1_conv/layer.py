from chainer.backends import cuda
import chainer.functions as cf

from ... import base
from .parameters import Parameters


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
        return h * w * cf.log(cf.det(W))