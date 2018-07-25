import os
import sys

import chainer
import chainer.functions as cf
import cupy as xp
import numpy as np
from chainer.backends import cuda

sys.path.append(os.path.join("..", ".."))
import glow

sys.path.append("..")
from model import Glow
from hyperparams import Hyperparameters


def check_layer():
    channels_x = 2
    batchsize = 1

    xp.random.seed(0)

    x = xp.random.normal(size=(batchsize, channels_x, 1, 1)).astype("float32")

    # actnorm
    layers = []
    size = 4 * 32
    for _ in range(size):
        params = glow.nn.actnorm.Parameters(channels=channels_x)
        params.to_gpu()

        params.scale.data = xp.random.uniform(
            -1.0, 1.0, size=params.scale.data.shape).astype("float32")
        params.bias.data = xp.random.uniform(
            -1.0, 1.0, size=params.bias.data.shape).astype("float32")
        actnorm = glow.nn.actnorm.Actnorm(params)
        rev_actnorm = actnorm.reverse_copy()
        layers.append((actnorm, rev_actnorm))

    y = x
    for k in range(size):
        actnorm = layers[k][0]
        y, _ = actnorm(y)
    rev_x = y
    for k in range(size):
        rev_actnorm = layers[size - k - 1][1]
        rev_x, _ = rev_actnorm(rev_x)
    error = cf.mean(abs(x - rev_x))
    print("actnorm:", error)

    # invertible 1x1 convolution
    layers = []
    size = 4 * 32
    for _ in range(size):
        params = glow.nn.invertible_1x1_conv.Parameters(channels=channels_x)
        params.to_gpu()

        shape = params.conv.W.data.shape[:2]
        # noise = xp.random.uniform(
        #     -1.0, 1.0, size=shape).astype("float32").reshape(shape + (1, 1))
        # params.conv.W.data += noise

        conv_1x1 = glow.nn.invertible_1x1_conv.Invertible1x1Conv(params)
        rev_conv_1x1 = conv_1x1.reverse_copy()
        layers.append((conv_1x1, rev_conv_1x1))

    y = x
    for k in range(size):
        conv_1x1 = layers[k][0]
        y, _ = conv_1x1(y)
    rev_x = y
    for k in range(size):
        rev_conv_1x1 = layers[size - k - 1][1]
        rev_x, _ = rev_conv_1x1(rev_x)
    error = cf.mean(abs(x - rev_x))
    print("inv_1x1:", error)

    # invertible 1x1 convolution (LU)
    # layers = []
    # size = 4 * 32
    # for _ in range(size):
    #     params = glow.nn.invertible_1x1_conv.LUParameters(
    #         channels=channels_x)
    #     params.to_gpu()
    #     conv_1x1 = glow.nn.invertible_1x1_conv.LUInvertible1x1Conv(
    #         params)
    #     rev_conv_1x1 = conv_1x1.reverse_copy()
    #     layers.append((conv_1x1, rev_conv_1x1))

    # y = x
    # for k in range(size):
    #     conv_1x1 = layers[k][0]
    #     y, _ = conv_1x1(y)
    # rev_x = y
    # for k in range(size):
    #     rev_conv_1x1 = layers[size - k - 1][1]
    #     rev_x, _ = rev_conv_1x1(rev_x)
    # error = cf.mean(abs(x - rev_x))
    # print("lu_1x1:", error)

    # affine coupling layer
    params = glow.nn.additive_coupling.Parameters(
        channels_x=channels_x // 2, channels_h=128)
    params.to_gpu()
    params.conv_1(x[:, 0::2])
    params.conv_1.W.data = xp.random.uniform(
        -1.0, 1.0, size=params.conv_1.W.data.shape).astype("float32")
    params.conv_2.W.data = xp.random.uniform(
        -1.0, 1.0, size=params.conv_2.W.data.shape).astype("float32")
    params.conv_3.W.data = xp.random.uniform(
        -1.0, 1.0, size=params.conv_3.W.data.shape).astype("float32")
    params.scale.data = xp.random.uniform(
        -1.0, 1.0, size=params.scale.data.shape).astype("float32")
    nonlinear_mapping = glow.nn.additive_coupling.NonlinearMapping(params)
    coupling_layer = glow.nn.additive_coupling.AdditiveCoupling(
        nn=nonlinear_mapping)
    rev_coupling_layer = coupling_layer.reverse_copy()

    y = x
    for _ in range(size):
        y, _ = coupling_layer(y)
    rev_x = y
    for _ in range(size):
        rev_x, _ = rev_coupling_layer(rev_x)
    error = cf.mean(abs(x - rev_x))
    print("coupling:", error)


def check_model():
    depth_per_level = 32
    levels = 4
    batchsize = 3

    x = xp.random.normal(0, 1, size=(batchsize, 3, 64, 64)).astype("float32")

    hyperparams = Hyperparameters()
    hyperparams.depth_per_level = depth_per_level
    hyperparams.levels = levels
    hyperparams.nn_hidden_channels = 32

    encoder = Glow(hyperparams)
    encoder.to_gpu()

    for level in range(levels):
        for depth in range(depth_per_level):
            actnorm, conv_1x1, coupling_layer = encoder[level][depth]

            params = actnorm.params
            params.scale.data = xp.random.uniform(
                -1.0, 1.0, size=params.scale.data.shape).astype("float32")
            params.bias.data = xp.random.uniform(
                -1.0, 1.0, size=params.bias.data.shape).astype("float32")

            params = conv_1x1.params
            # shape = params.conv.W.data.shape
            # noise = xp.random.uniform(-1.0, 1.0, size=shape).astype("float32")
            # params.conv.W.data += noise

            params = coupling_layer.nn.params
            params.conv_1.W.data = xp.random.uniform(
                -1.0, 1.0, size=params.conv_1.W.data.shape).astype("float32")
            params.conv_2.W.data = xp.random.uniform(
                -1.0, 1.0, size=params.conv_2.W.data.shape).astype("float32")
            params.conv_3.W.data = xp.zeros(
                params.conv_3.W.data.shape, dtype="float32")
            params.scale.data = xp.zeros(
                params.scale.data.shape, dtype="float32")

    with encoder.reverse() as decoder:
        factorized_z_distribution, logdet = encoder.forward_step(x)
        factorized_z = []
        for (zi, mean, ln_var) in factorized_z_distribution:
            factorized_z.append(zi)
        rev_x, rev_logdet = decoder.reverse_step(factorized_z)
        error = cf.mean(abs(x - rev_x))
        print(logdet, rev_logdet)
        print(error)


def check_squeeze():
    factor = 2
    shape = (1, 3, 256, 256)
    x = xp.arange(0, np.prod(shape)).reshape(shape)
    y = glow.nn.functions.squeeze(x, factor=factor, module=xp)
    _x = glow.nn.functions.unsqueeze(y, factor=factor, module=xp)
    print(xp.mean(xp.abs(x - _x)))


def main():
    with chainer.no_backprop_mode():
        check_model()
        check_squeeze()
        check_layer()


if __name__ == "__main__":
    main()
