import math
import chainer.functions as cf
from chainer.backends import cuda


def squeeze(x, factor=2, module=cf):
    batchsize = x.shape[0]
    channels = x.shape[1]
    height = x.shape[2]
    width = x.shape[3]
    assert height % factor == 0
    assert width % factor == 0
    out = module.reshape(x, (batchsize, channels, height // factor, factor,
                             width // factor, factor))
    out = module.transpose(out, (0, 1, 3, 5, 2, 4))
    out = module.reshape(out, (batchsize, int(channels * factor**2),
                               height // factor, width // factor))
    return out


def unsqueeze(x, factor=2, module=cf):
    batchsize = x.shape[0]
    channels = x.shape[1]
    height = x.shape[2]
    width = x.shape[3]
    out = module.reshape(
        x, (batchsize, channels // (factor**2), factor, factor, height, width))
    out = module.transpose(out, (0, 1, 4, 2, 5, 3))
    out = module.reshape(
        out,
        (batchsize, channels // (factor**2), height * factor, width * factor))
    return out


def standard_normal_nll(x):
    nll = 0.5 * (math.log(2 * math.pi) + x * x)
    return cf.sum(nll)


def split_channel(x):
    n = x.shape[1] // 2
    return x[:, :n], x[:, n:]


def factor_z(z, levels, squeeze_factor=2):
    xp = cuda.get_array_module(z)
    factorized_z = []
    for level in range(levels):
        z = squeeze(z, module=xp, factor=squeeze_factor)
        if level == levels - 1:
            factorized_z.append(z)
        else:
            zi, z = split_channel(z)
            factorized_z.append(zi)
    return factorized_z