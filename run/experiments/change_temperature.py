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
from model import InferenceModel, GenerativeModel, to_cpu
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


def main():
    xp = np
    using_gpu = args.gpu_device >= 0
    if using_gpu:
        cuda.get_device(args.gpu_device).use()
        xp = cupy

    hyperparams = Hyperparameters(args.snapshot_path)
    hyperparams.print()

    num_bins_x = 2.0**hyperparams.num_bits_x

    encoder = InferenceModel(hyperparams, hdf5_path=args.snapshot_path)
    decoder = encoder.reverse()

    if using_gpu:
        encoder.to_gpu()
        decoder.to_gpu()

    temperatures = [0.0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    total = len(temperatures)
    fig = plt.figure(figsize=(total * 4, 4))
    subplots = []
    for n in range(total):
        subplot = fig.add_subplot(1, total, n + 1)
        subplots.append(subplot)

    with chainer.no_backprop_mode():
        while True:
            z_batch = []
            for temperature in temperatures:
                z = np.random.normal(
                    0, temperature, size=(
                        3,
                    ) + hyperparams.image_size).astype("float32")
                z_batch.append(z)
            z_batch = np.asanyarray(z_batch)
            if using_gpu:
                z_batch = cuda.to_gpu(z_batch)
            x, _ = decoder(z_batch)
            for n, (temperature, subplot) in enumerate(zip(temperatures, subplots)):
                x_img = make_uint8(x.data[n], num_bins_x)
                subplot.imshow(x_img, interpolation="none")
                subplot.set_title("temperature={}".format(temperature))
            plt.pause(.01)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot-path", "-snapshot", type=str, required=True)
    parser.add_argument("--gpu-device", "-gpu", type=int, default=0)
    args = parser.parse_args()
    main()
