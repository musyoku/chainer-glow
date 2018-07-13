import math
import chainer.functions as cf


def squeeze(y, factor=2):
    batchsize = y.shape[0]
    channels = y.shape[1]
    height = y.shape[2]
    width = y.shape[3]
    assert height % factor == 0
    assert width % factor == 0
    out = cf.reshape(y, (batchsize, channels, height // factor, factor,
                         width // factor, factor))
    out = cf.transpose(out, (0, 1, 4, 3, 2, 5))
    out = cf.reshape(out, (batchsize, int(channels * factor**2),
                           height // factor, width // factor))
    return out