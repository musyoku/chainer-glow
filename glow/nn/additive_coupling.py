import chainer
import chainer.functions as cf
import chainer.links as L
import numpy as np
from chainer.backends import cuda
from chainer.initializers import HeNormal, Zero


class AdditiveCouplingNonlinearMapping(chainer.Chain):
    def __init__(self, channels_x, channels_h):
        super().__init__()
        self.channels_x = channels_x
        self.channels_h = channels_h

        with self.init_scope():
            self.conv_1 = L.Convolution2D(
                channels_x,
                channels_h,
                ksize=3,
                stride=1,
                pad=1,
                initialW=HeNormal(0.05))
            self.conv_2 = L.Convolution2D(
                channels_h,
                channels_h,
                ksize=1,
                stride=1,
                pad=0,
                initialW=HeNormal(0.05))
            self.conv_3 = L.Convolution2D(
                channels_h,
                channels_x,
                ksize=3,
                stride=1,
                pad=1,
                initialW=Zero())
            self.scale = chainer.Parameter(
                initializer=np.zeros((1, channels_x, 1, 1), dtype="float32"))

    def __call__(self, x):
        out = x
        out = cf.relu(self.conv_1(out))
        out = cf.relu(self.conv_2(out))
        out = self.conv_3(out)
        bias = out * cf.broadcast_to((cf.exp(self.scale * 3.0)), out.shape)
        # bias = out
        return bias


class AdditiveCoupling(chainer.Chain):
    def __init__(self, nn: AdditiveCouplingNonlinearMapping):
        super().__init__()
        with self.init_scope():
            self.nn = nn

    def forward_step(self, x):
        split = x.shape[1] // 2
        xa = x[:, :split]
        xb = x[:, split:]

        bias = self.nn(xa)

        ya = xa
        yb = xb + bias
        y = cf.concat((ya, yb), axis=1)

        xp = cuda.get_array_module(x)
        return y, chainer.Variable(xp.array(0.0, dtype="float32"))

    def reverse_step(self, y):
        split = y.shape[1] // 2
        ya = y[:, :split]
        yb = y[:, split:]

        bias = self.nn(ya)

        xa = ya
        xb = yb - bias
        x = cf.concat((xa, xb), axis=1)

        xp = cuda.get_array_module(x)
        return x, chainer.Variable(xp.array(0.0, dtype="float32"))
