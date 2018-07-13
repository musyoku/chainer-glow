from chainer.backends import cuda
import chainer.functions as cf

from ... import base
from .parameters import Parameters


class Actnorm(base.Actnorm):
    def __init__(self, params: Parameters):
        self.params = params

    def __call__(self, x):
        inter = self.params.scale(x)
        y = self.params.bias(inter)
        log_det = self.compute_log_determinant(x)
        return y, log_det

    def compute_log_determinant(self, x):
        h, w = x.shape[2:]
        s = self.params.scale.W
        return h * w * cf.sum(cf.log(s))  # keep minibatch
