import chainer
import chainer.links as L
import numpy as np


class Parameters(chainer.Chain):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        with self.init_scope():
            shape = (channels, channels)
            rotation_mat = np.linalg.qr(np.random.normal(
                size=shape))[0].astype("float32").reshape(shape + (1, 1))
            self.conv = L.Convolution2D(
                channels,
                channels,
                ksize=1,
                stride=1,
                pad=0,
                nobias=True,
                initialW=rotation_mat)


class LUDecompositionParameters(chainer.Chain):
    def __init__(self, channels):
        raise NotImplementedError
