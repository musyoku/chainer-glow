import math
import chainer.functions as cf

def squeeze(x, factor=2):
    batchsize = x.shape[0]
    channels = x.shape[1]
    height = x.shape[2]
    width = x.shape[3]
    assert height % factor == 0
    assert width % factor == 0
    out = cf.reshape(x, (batchsize, channels, height // factor, factor,
                         width // factor, factor))
    out = cf.transpose(out, (0, 1, 4, 3, 2, 5))
    out = cf.reshape(out, (batchsize, int(channels * factor**2),
                           height // factor, width // factor))
    return out

def gaussian_negative_log_likelihood(x, mu, var, ln_var):
    k = mu.shape[1] * mu.shape[2] * mu.shape[3]
    diff = x - mu
    return 0.5 * (k * math.log(2 * math.pi) + cf.sum(ln_var + diff * diff / var, axis=(1, 2, 3)))
