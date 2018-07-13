from chainer.backends import cuda
import chainer.functions as cf

from ... import base
from .parameters import Parameters


class Actnorm(base.Actnorm):
    def __init__(self, params: Parameters):
        self.params = params

    def __call__(self, x):
        print(x.shape)
        inter = self.params.scale(x)
        print(inter.shape)
        y = self.params.bias(inter)
        print(y.shape)
        log_det = self.compute_log_determinant(x)
        print(log_det)
        return y, log_det

    def compute_log_determinant(self, x):
        h, w = x.shape[2:]
        s = self.params.scale.W
        return h * w * cf.sum(cf.log(s))