from chainer.backends import cuda
import chainer.functions as cf

from .parameters import Parameters


class Actnorm(object):
    def __init__(self, params: Parameters):
        self.params = params
        self.bias = self.params.bias
        self.scale = self.params.scale

    def __call__(self, x):
        bias = cf.broadcast_to(self.bias, x.shape)
        scale = cf.broadcast_to(self.scale, x.shape)
        y = (x + bias) * scale

        log_det = self.compute_log_determinant(x, self.scale)

        return y, log_det

    def compute_log_determinant(self, x, scale):
        h, w = x.shape[2:]
        return h * w * cf.sum(cf.log(abs(scale)))

    def reverse_copy(self):
        params = self.params.reverse_copy()
        return ReverseActnorm(params)


class ReverseActnorm(object):
    def __init__(self, params: Parameters):
        self.params = params
        self.bias = self.params.bias
        self.scale = self.params.scale

    def __call__(self, y):
        bias = cf.broadcast_to(self.bias, y.shape)
        scale = cf.broadcast_to(self.scale, y.shape)
        x = y / scale - bias

        log_det = self.compute_log_determinant(x, self.scale)

        return x, log_det

    def compute_log_determinant(self, x, scale):
        h, w = x.shape[2:]
        return -h * w * cf.sum(cf.log(abs(scale)))
