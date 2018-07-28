import os
import sys
import argparse

import chainer
import chainer.functions as cf
import cupy
import numpy as np
import matplotlib.pyplot as plt

from chainer.backends import cuda
from tabulate import tabulate

sys.path.append(os.path.join("..", ".."))
import glow

sys.path.append("..")
from model import Glow, to_cpu
from hyperparams import Hyperparameters


def make_uint8(array, bins):
    if array.ndim == 4:
        array = array[0]
    if (array.shape[2] == 3):
        return np.uint8(
            np.clip(
                np.floor((to_cpu(array) + 0.5) * bins) * (255 / bins), 0, 255))
    return np.uint8(
        np.clip(
            np.floor((to_cpu(array.transpose(1, 2, 0)) + 0.5) * bins) *
            (255 / bins), 0, 255))


def get_model(path, using_gpu):
    print(path)
    hyperparams = Hyperparameters(path)
    hyperparams.print()

    num_bins_x = 2.0**hyperparams.num_bits_x

    encoder = Glow(hyperparams, hdf5_path=path)
    if using_gpu:
        encoder.to_gpu()

    return encoder, num_bins_x, hyperparams


def main():
    xp = np
    using_gpu = args.gpu_device >= 0
    if using_gpu:
        cuda.get_device(args.gpu_device).use()
        xp = cupy

    model1 = get_model(args.snapshot_path_1, using_gpu)
    model2 = get_model(args.snapshot_path_2, using_gpu)
    model3 = get_model(args.snapshot_path_3, using_gpu)

    num_bins_x, hyperparams = model1[1:]

    fig = plt.figure(figsize=(12, 4))
    left = fig.add_subplot(1, 3, 1)
    center = fig.add_subplot(1, 3, 2)
    right = fig.add_subplot(1, 3, 3)

    while True:
        z = xp.random.normal(
            0, args.temperature, size=(
                1,
                3,
            ) + hyperparams.image_size).astype("float32")

        with chainer.no_backprop_mode():
            encoder = model1[0]
            with encoder.reverse() as decoder:
                hyperparams = model1[2]
                x, _ = decoder.reverse_step(z)
                x_img = make_uint8(x.data[0], num_bins_x)
                left.imshow(x_img, interpolation="none")
                left.set_title("#channels = {}".format(
                    hyperparams.nn_hidden_channels))

            encoder = model2[0]
            with encoder.reverse() as decoder:
                hyperparams = model2[2]
                x, _ = decoder.reverse_step(z)
                x_img = make_uint8(x.data[0], num_bins_x)
                center.imshow(x_img, interpolation="none")
                center.set_title("#channels = {}".format(
                    hyperparams.nn_hidden_channels))

            encoder = model3[0]
            with encoder.reverse() as decoder:
                hyperparams = model3[2]
                x, _ = decoder.reverse_step(z)
                x_img = make_uint8(x.data[0], num_bins_x)
                right.imshow(x_img, interpolation="none")
                right.set_title("#channels = {}".format(
                    hyperparams.nn_hidden_channels))

            plt.pause(.01)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot-path-1", "-snapshot-1", type=str, required=True)
    parser.add_argument(
        "--snapshot-path-2", "-snapshot-2", type=str, required=True)
    parser.add_argument(
        "--snapshot-path-3", "-snapshot-3", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--gpu-device", "-gpu", type=int, default=0)
    args = parser.parse_args()
    main()
