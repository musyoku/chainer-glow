from chainer.backends import cuda
import chainer.functions as cf

from .parameters import Parameters


class Conv2dZeros(object):
    def __init__(self, params: Parameters):
        self.params = params

    def __call__(self, x):
        out = x
        out = self.params.conv(out)
        bias = out * cf.broadcast_to(
            (cf.exp(self.params.scale * 3.0)), out.shape)
        # bias = out
        return bias