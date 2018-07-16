import os
import sys

import chainer
import chainer.functions as cf
import cupy as xp
import numpy as np
from chainer.backends import cuda

sys.path.append(os.path.join("..", "..", ".."))
import glow

sys.path.append("..")
from model import InferenceModel, GenerativeModel, reverse_actnorm, reverse_conv_1x1, reverse_coupling_layer
from hyperparams import Hyperparameters


def check_layer():
    channels_x = 1
    batchsize = 1

    x = xp.random.normal(size=(batchsize, channels_x, 5,
                               5)).astype("float32")

    # actnorm
    params = glow.nn.chainer.actnorm.Parameters(channels=channels_x)
    params.to_gpu()

    params.scale.W.data = xp.random.normal(
        1.0, 1, size=params.scale.W.data.shape).astype("float32")
    params.bias.b.data = xp.random.normal(
        0.0, 1, size=params.bias.b.data.shape).astype("float32")
    actnorm = glow.nn.chainer.actnorm.Actnorm(params)
    rev_actnorm = reverse_actnorm(actnorm)
    rev_actnorm.params.to_gpu()

    scales = []
    biases = []
    for _ in range(4 * 32):
        scale_64 = xp.random.normal(
            1.0, 1, size=params.scale.W.data.shape).astype("float32")
        bias_64 = xp.random.normal(
            0.0, 1, size=params.bias.b.data.shape).astype("float32")
        
        scales.append(scale_64)
        biases.append(bias_64)

    print(x)
    y = x
    for _ in range(4 * 32):
        scale_64 = scales[_]
        bias_64 = biases[_]
        y = y - bias_64
        y = y / scale_64
    print(y)
    for _ in range(4 * 32):
        scale_64 = scales[4 * 32 - _ - 1]
        bias_64 = biases[4 * 32 - _ - 1]
        y = y * scale_64
        y = y + bias_64
    print(y)
    y = x
    for _ in range(4 * 32):
        y = y - params.bias.b.data
        y = y / params.scale.W.data
    print(y)
    for _ in range(4 * 32):
        y = y * params.scale.W.data
        y = y + params.bias.b.data
    print(y)
    exit()


    y = x
    for _ in range(4 * 32):
        y, _ = actnorm(y)
    rev_x = y
    for _ in range(4 * 32):
        rev_x = rev_actnorm(rev_x)
    error = cf.mean(abs(x - rev_x))
    print(error)

    # invertible 1x1 convolution
    params = glow.nn.chainer.invertible_1x1_conv.Parameters(
        channels=channels_x)
    params.to_gpu()

    shape = params.conv.W.data.shape[:2]
    noise = xp.random.normal(
        0.0, 1, size=shape).astype("float32").reshape(shape + (1, 1))
    params.conv.W.data += noise

    conv_1x1 = glow.nn.chainer.invertible_1x1_conv.Invertible1x1Conv(params)
    rev_conv_1x1 = reverse_conv_1x1(conv_1x1)
    rev_conv_1x1.params.to_gpu()

    y = x
    for _ in range(4 * 32):
        y, _ = conv_1x1(y)
    rev_x = y
    for _ in range(4 * 32):
        rev_x = rev_conv_1x1(rev_x)
    error = cf.mean(abs(x - rev_x))
    print(error)

    # affine coupling layer
    params = glow.nn.chainer.affine_coupling.Parameters(
        channels_x=channels_x, channels_h=128)
    params.to_gpu()
    params.conv_1(x[:, 0::2])
    params.conv_1.W.data = xp.random.normal(
        0.0, 1, size=params.conv_1.W.data.shape).astype("float32")
    params.conv_2.W.data = xp.random.normal(
        0.0, 1, size=params.conv_2.W.data.shape).astype("float32")
    params.conv_3.W.data = xp.zeros(
        params.conv_3.W.data.shape, dtype="float32")
    nonlinear_mapping = glow.nn.chainer.affine_coupling.NonlinearMapping(
        params)
    coupling_layer = glow.nn.chainer.affine_coupling.AffineCoupling(
        nn=nonlinear_mapping)
    rev_coupling_layer = reverse_coupling_layer(coupling_layer)
    rev_coupling_layer.nn.params.to_gpu()

    y = x
    for _ in range(4 * 32):
        y, _ = coupling_layer(y)
    rev_x = y
    for _ in range(4 * 32):
        rev_x = rev_coupling_layer(rev_x)
    error = cf.mean(abs(x - rev_x))
    print(error)

    forward_variables = []
    reverse_variables = []
    y = x
    for _ in range(4 * 32):
        forward_variables.append(y)
        y, _ = actnorm(y)
        y, _ = conv_1x1(y)
        y, _ = coupling_layer(y)
    rev_x = y
    for _ in range(4 * 32):
        rev_x = rev_coupling_layer(rev_x)
        rev_x = rev_conv_1x1(rev_x)
        reverse_variables.append(rev_x)
        rev_x = rev_actnorm(rev_x)
    error = cf.mean(abs(x - rev_x))
    reverse_variables.reverse()

    for f, r in zip(forward_variables, reverse_variables):
        y, _ = actnorm(f)
        print("error", cf.mean_absolute_error(y, r))
    print(error)


def check_model():
    depth_per_level = 2
    levels = 5
    batchsize = 3

    x = xp.random.normal(0, 1, size=(batchsize, 3, 64, 64)).astype("float32")

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
            params.scale.W.data = xp.random.normal(
                1.0, 0.1, size=params.scale.W.data.shape).astype("float32")
            params.bias.b.data = xp.random.normal(
                0.0, 0.1, size=params.bias.b.data.shape).astype("float32")

            params = conv_1x1.params
            shape = params.conv.W.data.shape[:2]
            noise = xp.random.normal(0.0, 0.01, size=shape).astype("float32")
            params.conv.W.data += noise

            params = coupling_layer.nn.params
            params.conv_1.W.data = xp.random.normal(
                0.0, 0.1, size=params.conv_1.W.data.shape).astype("float32")
            params.conv_2.W.data = xp.random.normal(
                0.0, 0.1, size=params.conv_2.W.data.shape).astype("float32")
            params.conv_3.W.data = xp.zeros(
                params.conv_3.W.data.shape, dtype="float32")

    generative_model = GenerativeModel(inference_model)

    factorized_z, logdet = inference_model(x)
    rev_x = generative_model(factorized_z)
    error = cf.mean(abs(x - rev_x))
    print(error)


def main():
    with chainer.using_config("train", False), chainer.using_config(
            "enable_backprop", False):
        check_layer()
        check_model()


if __name__ == "__main__":
    main()
