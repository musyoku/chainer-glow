import chainer
import numpy as np


class Parameters(chainer.Chain):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        with self.init_scope():
            shape = (1, channels, 1, 1)
            self.scale = chainer.Parameter(
                initializer=np.zeros(shape, dtype="float32"))
            self.bias = chainer.Parameter(
                initializer=np.zeros(shape, dtype="float32"))
