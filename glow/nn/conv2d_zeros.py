import chainer
import chainer.functions as cf
import chainer.links as L
import numpy as np
from chainer.backends import cuda
from chainer.initializers import HeNormal, Zero


class Conv2dZeros(chainer.Chain):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out

        with self.init_scope():
            self.conv = L.Convolution2D(
                channels_in,
                channels_out,
                ksize=3,
                stride=1,
                pad=1,
                initialW=Zero())
            self.scale = chainer.Parameter(
                initializer=np.zeros((1, channels_out, 1, 1), dtype="float32"))

    def __call__(self, x):
        out = x
        out = self.conv(out)
        bias = out * cf.broadcast_to((cf.exp(self.scale * 3.0)), out.shape)
        # bias = out
        return bias
