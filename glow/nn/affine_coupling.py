import chainer
import chainer.functions as cf
import chainer.links as L
import numpy as np
from chainer.backends import cuda
from chainer.initializers import HeNormal, Zero


class AffineCouplingNonlinearMapping(chainer.Chain):
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
                initialW=(HeNormal(0.1)))
            self.conv_2 = L.Convolution2D(
                channels_h,
                channels_h,
                ksize=1,
                stride=1,
                pad=0,
                initialW=(HeNormal(0.1)))
            self.conv_3 = L.Convolution2D(
                channels_h,
                channels_x * 2,
                ksize=3,
                stride=1,
                pad=1,
                initialW=Zero())

    def __call__(self, x):
        out = x
        out = cf.relu(self.conv_1(out))
        out = cf.relu(self.conv_2(out))
        out = self.conv_3(out)
        split = out.shape[1] // 2
        log_scale = out[:, :split]
        bias = out[:, split:]
        return log_scale, bias


class AffineCoupling(chainer.Chain):
    def __init__(self, nn: AffineCouplingNonlinearMapping):
        super().__init__()
        with self.init_scope():
            self.nn = nn

    def forward_step(self, x):
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

    def reverse_step(self, y):
        split = y.shape[1] // 2
        ya = y[:, :split]
        yb = y[:, split:]

        log_scale, bias = self.nn(yb)
        scale = cf.sigmoid(log_scale + 2)

        xa = ya / scale - bias
        xb = yb
        x = cf.concat((xa, xb), axis=1)

        log_det = -self.compute_log_determinant(scale)

        return x, log_det

    def compute_log_determinant(self, scale):
        return cf.sum(cf.log(abs(scale)))