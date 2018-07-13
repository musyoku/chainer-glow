import chainer
import chainer.links as L


class Parameters(chainer.Chain):
    def __init__(self, channels):
        super().__init__()
        with self.init_scope():
            self.conv_1 = L.Convolution2D(
                channels, channels, ksize=3, stride=1, pad=1)
            self.conv_2 = L.Convolution2D(
                channels, channels, ksize=1, stride=1, pad=0)
            self.conv_3 = L.Convolution2D(
                channels, channels * 2, ksize=3, stride=1, pad=1)
