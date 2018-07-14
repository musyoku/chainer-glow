import os
import sys

import chainer
import chainer.functions as cf
import cupy
import numpy as np
from chainer.backends import cuda

sys.path.append(os.path.join("..", "..", ".."))
import glow

sys.path.append("..")
from model import InferenceModel, GenerativeModel, reverse_actnorm, reverse_conv_1x1, reverse_coupling_layer
from hyperparams import Hyperparameters


def check_layer():
    channels_x = 4
    batchsize = 3

    x = np.random.normal(size=(batchsize, channels_x, 3, 3)).astype("float32")

    # actnorm
    params = glow.nn.chainer.actnorm.Parameters(channels=channels_x)
    params.scale.W.data = np.random.normal(
        size=params.scale.W.data.shape).astype("float32")
    params.bias.b.data = np.random.normal(
        size=params.bias.b.data.shape).astype("float32")
    actnorm = glow.nn.chainer.actnorm.Actnorm(params)
    rev_actnorm = reverse_actnorm(actnorm)

    y, _ = actnorm(x)
    rev_x = rev_actnorm(y)
    error = cf.mean(abs(x - rev_x))
    print(error)

    # invertible 1x1 convolution
    params = glow.nn.chainer.invertible_1x1_conv.Parameters(
        channels=channels_x)
    params.conv.W.data = np.random.normal(
        size=params.conv.W.data.shape).astype("float32")
    conv_1x1 = glow.nn.chainer.invertible_1x1_conv.Invertible1x1Conv(params)
    rev_conv_1x1 = reverse_conv_1x1(conv_1x1)

    y, _ = conv_1x1(x)
    rev_x = rev_conv_1x1(y)
    error = cf.mean(abs(x - rev_x))
    print(error)

    # affine coupling layer
    params = glow.nn.chainer.affine_coupling.Parameters(
        channels_x=channels_x, channels_h=512)
    params.conv_1(x[:, 0::2])
    params.conv_1.W.data = np.random.normal(
        size=params.conv_1.W.data.shape).astype("float32")
    params.conv_2.W.data = np.random.normal(
        size=params.conv_2.W.data.shape).astype("float32")
    params.conv_3.W.data = np.random.normal(
        0, 0.001, size=params.conv_3.W.data.shape).astype("float32")
    nonlinear_mapping = glow.nn.chainer.affine_coupling.NonlinearMapping(
        params)
    coupling_layer = glow.nn.chainer.affine_coupling.AffineCoupling(
        nn=nonlinear_mapping)
    rev_coupling_layer = reverse_coupling_layer(coupling_layer)

    y, _ = coupling_layer(x)
    rev_x = rev_coupling_layer(y)
    error = cf.mean(abs(x - rev_x))
    print(error)


def check_model():
    depth_per_level = 4
    levels = 4
    batchsize = 3

    x = np.random.normal(0, 1, size=(batchsize, 3, 64, 64)).astype("float32")

    hyperparams = Hyperparameters()
    hyperparams.depth_per_level = depth_per_level
    hyperparams.levels = levels
    hyperparams.nn_hidden_channels = 32

    inference_model = InferenceModel(hyperparams)
    inference_model(x)

    for level in range(levels):
        for depth in range(depth_per_level):
            actnorm, conv_1x1, coupling_layer = inference_model[level][depth]

            params = actnorm.params
            params.scale.W.data = np.random.normal(
                size=params.scale.W.data.shape).astype("float32")
            params.bias.b.data = np.random.normal(
                size=params.bias.b.data.shape).astype("float32")

            params = conv_1x1.params
            params.conv.W.data = np.random.normal(
                size=params.conv.W.data.shape).astype("float32")

            params = coupling_layer.nn.params
            params.conv_1.W.data = np.random.normal(
                size=params.conv_1.W.data.shape).astype("float32")
            params.conv_2.W.data = np.random.normal(
                size=params.conv_2.W.data.shape).astype("float32")
            params.conv_3.W.data = np.zeros(
                params.conv_3.W.data.shape, dtype="float32")

    generative_model = GenerativeModel(inference_model)

    factorized_z, logdet = inference_model(x)
    _x = generative_model(factorized_z)
    print(x)
    print(_x)


def main():
    check_layer()
    check_model()


if __name__ == "__main__":
    main()
