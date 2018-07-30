import os
import sys
import argparse
import time

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


def main():
    xp = np
    using_gpu = args.gpu_device >= 0
    if using_gpu:
        cuda.get_device(args.gpu_device).use()
        xp = cupy

    hyperparams = Hyperparameters(args.snapshot_path)
    hyperparams.print()

    num_bins_x = 2.0**hyperparams.num_bits_x

    encoder = Glow(hyperparams, hdf5_path=args.snapshot_path)
    if using_gpu:
        encoder.to_gpu()

    total = hyperparams.levels
    fig = plt.figure(figsize=(4 * total, 4))
    subplots = []
    for n in range(total):
        subplot = fig.add_subplot(1, total, n + 1)
        subplots.append(subplot)

    with chainer.no_backprop_mode() and encoder.reverse() as decoder:
        while True:
            seed = int(time.time())

            for level in range(1, hyperparams.levels):
                xp.random.seed(seed)
                z = xp.random.normal(
                    0,
                    args.temperature,
                    size=(
                        1,
                        3,
                    ) + hyperparams.image_size,
                    dtype="float32")
                factorized_z = glow.nn.functions.factor_z(z, level + 1)

                out = glow.nn.functions.unsqueeze(
                    factorized_z.pop(-1),
                    factor=hyperparams.squeeze_factor,
                    module=xp)
                for n, zi in enumerate(factorized_z[::-1]):
                    block = encoder.blocks[level - n - 1]
                    out, _ = block.reverse_step(
                        out,
                        gaussian_eps=zi,
                        squeeze_factor=hyperparams.squeeze_factor)
                rev_x = out
                rev_x_img = make_uint8(rev_x.data[0], num_bins_x)
                subplot = subplots[level - 1]
                subplot.imshow(rev_x_img, interpolation="none")
                subplot.set_title("level = {}".format(level))

            # original #levels
            xp.random.seed(seed)
            z = xp.random.normal(
                0,
                args.temperature,
                size=(
                    1,
                    3,
                ) + hyperparams.image_size,
                dtype="float32")
            factorized_z = encoder.factor_z(z)
            rev_x, _ = decoder.reverse_step(factorized_z)
            rev_x_img = make_uint8(rev_x.data[0], num_bins_x)
            subplot = subplots[-1]
            subplot.imshow(rev_x_img, interpolation="none")
            subplot.set_title("level = {}".format(hyperparams.levels))

            plt.pause(.01)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot-path", "-snapshot", type=str, required=True)
    parser.add_argument("--temperature", "-temp", type=float, default=0.7)
    parser.add_argument("--gpu-device", "-gpu", type=int, default=0)
    args = parser.parse_args()
    main()
