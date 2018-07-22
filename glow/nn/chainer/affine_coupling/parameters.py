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
                initialW=(HeNormal(0.1)))
            # self.conv_scale = L.Convolution2D(
            #     channels_h,
            #     channels_x // 2,
            #     ksize=3,
            #     stride=1,
            #     pad=1,
            #     initialW=Zero())
            # self.conv_bias = L.Convolution2D(
            #     channels_h,
            #     channels_x // 2,
            #     ksize=3,
            #     stride=1,
            #     pad=1,
            #     initialW=Zero())

    def reverse_copy(self):
        copy = Parameters(self.channels_x, self.channels_h)
        if self.xp is not np:
            copy.to_gpu()
        copy.conv_1.W.data[...] = self.conv_1.W.data
        copy.conv_2.W.data[...] = self.conv_2.W.data
        copy.conv_3.W.data[...] = self.conv_3.W.data
        return copy