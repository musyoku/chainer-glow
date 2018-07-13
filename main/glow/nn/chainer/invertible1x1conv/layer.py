from chainer.backends import cuda
import chainer.functions as cf

from ... import base
from .parameters import Parameters


class Invertible1x1Conv(base.Invertible1x1Conv):
    def __init__(self, conv):
        self.conv = conv

    def __call__(self, x):
        return self.conv(x)

    def compute_log_determinant(self, h, w):
        W = self.conv.W
        return h * w * cf.log(cf.det(W))