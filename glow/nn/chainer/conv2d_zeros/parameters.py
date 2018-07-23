import chainer
import chainer.links as L
import numpy as np
from chainer.initializers import Zero, HeNormal


class Parameters(chainer.Chain):
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

    def reverse_copy(self):
        copy = Parameters(self.channels_out, self.channels_in)
        if self.xp is not np:
            copy.to_gpu()
        copy.conv.W.data[...] = self.conv.W.data
        copy.scale.data[...] = self.scale.data
        return copy