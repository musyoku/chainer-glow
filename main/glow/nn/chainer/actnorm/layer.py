from chainer.backends import cuda
import chainer.functions as cf

from ... import base
from .parameters import Parameters


class Actnorm(base.Actnorm):
    def __init__(self, h, w, scale, bias):
        self.scale = scale
        self.bias = bias
        self.h = h
        self.w = w

    def __call__(self, x):
        inter = self.scale(x)
        y = self.bias(inter)
        return y

    def compute_log_determinant(self):
        s = self.scale.W
        return self.h * self.w * cf.sum(cf.log(s))