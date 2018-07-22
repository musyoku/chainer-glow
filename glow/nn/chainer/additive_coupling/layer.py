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
        bias = self.params.conv_3(out)
        return bias


class AdditiveCoupling(base.AdditiveCoupling):
    def __init__(self, nn: NonlinearMapping):
        self.nn = nn

    def __call__(self, x):
        split = x.shape[1] // 2
        xa = x[:, :split]
        xb = x[:, split:]

        bias = self.nn(xb)

        ya = xa + bias
        yb = xb
        y = cf.concat((ya, yb), axis=1)

        return y, 0

    def reverse_copy(self):
        params = self.nn.params.reverse_copy()
        nn = NonlinearMapping(params)
        return ReverseAdditiveCoupling(nn)


class ReverseAdditiveCoupling(base.ReverseAdditiveCoupling):
    def __init__(self, nn: NonlinearMapping):
        self.nn = nn

    def __call__(self, y):
        split = y.shape[1] // 2
        ya = y[:, :split]
        yb = y[:, split:]

        bias = self.nn(yb)

        xa = ya - bias
        xb = yb
        x = cf.concat((xa, xb), axis=1)

        return x, 0