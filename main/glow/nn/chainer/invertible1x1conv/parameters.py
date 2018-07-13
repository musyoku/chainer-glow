import chainer
import chainer.links as L


class Parameters(chainer.Chain):
    def __init__(self, channels):
        super().__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(
                channels, channels, ksize=1, stride=1, pad=0, nobias=True)


class LUDecompositionParameters(chainer.Chain):
    def __init__(self, channels):
        raise NotImplementedError
