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
        split = out.shape[1] // 2
        log_scale = out[:, :split]
        bias = out[:, split:]
        return log_scale, bias


class AffineCoupling(base.AffineCoupling):
    def __init__(self, nn: NonlinearMapping):
        self.nn = nn

    def __call__(self, x):
        split = x.shape[1] // 2
        xa = x[:, :split]
        xb = x[:, split:]

        log_scale, bias = self.nn(xb)
        scale = cf.sigmoid(log_scale + 2)

        ya = scale * (xa + bias)
        yb = xb
        y = cf.concat((ya, yb), axis=1)

        log_det = self.compute_log_determinant(scale)

        return y, log_det

    def compute_log_determinant(self, scale):
        return cf.sum(cf.log(abs(scale)))

    def reverse_copy(self):
        params = self.nn.params.reverse_copy()
        nn = NonlinearMapping(params)
        return ReverseAffineCoupling(nn)


class ReverseAffineCoupling(base.ReverseAffineCoupling):
    def __init__(self, nn: NonlinearMapping):
        self.nn = nn

    def __call__(self, y):
        split = y.shape[1] // 2
        ya = y[:, :split]
        yb = y[:, split:]

        log_scale, bias = self.nn(yb)
        scale = cf.sigmoid(log_scale + 2)

        xa = ya / scale - bias
        xb = yb
        x = cf.concat((xa, xb), axis=1)

        log_det = self.compute_log_determinant(scale)

        return x, log_det

    def compute_log_determinant(self, scale):
        return -cf.sum(cf.log(abs(scale)))
