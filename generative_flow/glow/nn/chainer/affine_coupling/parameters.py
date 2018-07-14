import chainer
import chainer.links as L
from chainer.initializers import Zero, HeNormal


class Parameters(chainer.Chain):
    def __init__(self, channels_x, channels_h):
        super().__init__()
        self.channels_x = channels_x
        self.channels_h = channels_h

        with self.init_scope():
            self.conv_1 = L.Convolution2D(
                channels_x // 2,
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
                channels_x,
                ksize=3,
                stride=1,
                pad=1,
                initialW=Zero())
