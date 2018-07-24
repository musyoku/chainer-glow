import chainer
import chainer.links as L
import numpy as np
from chainer.initializers import Zero, HeNormal


class Parameters(chainer.Chain):
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

    def reverse_copy(self):
        copy = Parameters(self.channels_x, self.channels_h)
        if self.xp is not np:
            copy.to_gpu()
        copy.conv_1.W.data[...] = self.conv_1.W.data
        copy.conv_1.b.data[...] = self.conv_1.b.data
        copy.conv_2.W.data[...] = self.conv_2.W.data
        copy.conv_2.b.data[...] = self.conv_2.b.data
        copy.conv_3.W.data[...] = self.conv_3.W.data
        copy.conv_3.b.data[...] = self.conv_3.b.data
        copy.scale.data[...] = self.scale.data
        return copy