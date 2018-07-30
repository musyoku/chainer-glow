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

    total = hyperparams.levels + 1
    fig = plt.figure(figsize=(4 * total, 4))
    subplots = []
    for n in range(total):
        subplot = fig.add_subplot(1, total, n + 1)
        subplots.append(subplot)

    def reverse_step(z, sampling=True):
        if isinstance(z, list):
            factorized_z = z
        else:
            factorized_z = encoder.factor_z(z)

        assert len(factorized_z) == len(encoder.blocks)

        out = None
        sum_logdet = 0

        for block, zi in zip(encoder.blocks[::-1], factorized_z[::-1]):
            out, logdet = block.reverse_step(
                out,
                gaussian_eps=zi,
                squeeze_factor=encoder.hyperparams.squeeze_factor,
                sampling=sampling)
            sum_logdet += logdet

        return out, sum_logdet

    with chainer.no_backprop_mode() and encoder.reverse() as decoder:
        while True:
            base_z = xp.random.normal(
                0,
                args.temperature,
                size=(
                    1,
                    3,
                ) + hyperparams.image_size,
                dtype="float32")
            factorized_z = encoder.factor_z(base_z)

            rev_x, _ = decoder.reverse_step(factorized_z)
            rev_x_img = make_uint8(rev_x.data[0], num_bins_x)
            subplots[0].imshow(rev_x_img, interpolation="none")

            z = xp.copy(base_z)
            factorized_z = encoder.factor_z(z)
            for n in range(hyperparams.levels - 1):
                factorized_z[n] = xp.random.normal(
                    0,
                    args.temperature,
                    size=factorized_z[n].shape,
                    dtype="float32")
            rev_x, _ = decoder.reverse_step(factorized_z)
            rev_x_img = make_uint8(rev_x.data[0], num_bins_x)
            subplots[1].imshow(rev_x_img, interpolation="none")

            # for n in range(hyperparams.levels):
            #     z = xp.copy(base_z)
            #     factorized_z = encoder.factor_z(z)
            #     for m in range(n + 1):
            #         factorized_z[m] = xp.random.normal(
            #             0,
            #             args.temperature,
            #             size=factorized_z[m].shape,
            #             dtype="float32")
            #         # factorized_z[m] = xp.zeros_like(factorized_z[m])
            #     out = None
            #     for k, (block, zi) in enumerate(
            #             zip(encoder.blocks[::-1], factorized_z[::-1])):
            #         sampling = False
            #         out, _ = block.reverse_step(
            #             out,
            #             gaussian_eps=zi,
            #             squeeze_factor=encoder.hyperparams.squeeze_factor,
            #             sampling=sampling)
            #     rev_x = out

            #     rev_x_img = make_uint8(rev_x.data[0], num_bins_x)
            #     subplots[n + 1].imshow(rev_x_img, interpolation="none")

            for n in range(hyperparams.levels):
                z = xp.copy(base_z)
                factorized_z = encoder.factor_z(z)
                factorized_z[n] = xp.random.normal(
                    0,
                    args.temperature,
                    size=factorized_z[n].shape,
                    dtype="float32")
                factorized_z[n] = xp.zeros_like(factorized_z[n])
                out = None
                for k, (block, zi) in enumerate(
                        zip(encoder.blocks[::-1], factorized_z[::-1])):
                    sampling = False if k == hyperparams.levels - n - 1 else True
                    out, _ = block.reverse_step(
                        out,
                        gaussian_eps=zi,
                        squeeze_factor=encoder.hyperparams.squeeze_factor,
                        sampling=sampling)
                rev_x = out

                rev_x_img = make_uint8(rev_x.data[0], num_bins_x)
                subplots[n + 1].imshow(rev_x_img, interpolation="none")
            plt.pause(.01)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot-path", "-snapshot", type=str, required=True)
    parser.add_argument("--temperature", "-temp", type=float, default=0.7)
    parser.add_argument("--gpu-device", "-gpu", type=int, default=0)
    args = parser.parse_args()
    main()
