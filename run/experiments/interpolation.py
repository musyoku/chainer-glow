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
from PIL import Image
from pathlib import Path

sys.path.append(os.path.join("..", ".."))
import glow

sys.path.append("..")
from model import Glow, to_cpu, to_gpu
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


def preprocess(image, num_bits_x):
    num_bins_x = 2**num_bits_x
    if num_bits_x < 8:
        image = np.floor(image / (2**(8 - num_bits_x)))
    image = image / num_bins_x - 0.5
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))
    elif image.ndim == 4:
        image = image.transpose((0, 3, 1, 2))
    else:
        raise NotImplementedError
    return image


def main():
    xp = np
    using_gpu = args.gpu_device >= 0
    if using_gpu:
        cuda.get_device(args.gpu_device).use()
        xp = cupy

    hyperparams = Hyperparameters(args.snapshot_path)
    hyperparams.print()

    num_bins_x = 2.0**hyperparams.num_bits_x

    assert args.dataset_format in ["png", "npy"]

    files = Path(args.dataset_path).glob("*.{}".format(args.dataset_format))
    if args.dataset_format == "png":
        images = []
        for filepath in files:
            image = np.array(Image.open(filepath)).astype("float32")
            image = preprocess(image, hyperparams.num_bits_x)
            images.append(image)
        assert len(images) > 0
        images = np.asanyarray(images)
    elif args.dataset_format == "npy":
        images = []
        for filepath in files:
            array = np.load(filepath).astype("float32")
            array = preprocess(array, hyperparams.num_bits_x)
            images.append(array)
        assert len(images) > 0
        num_files = len(images)
        images = np.asanyarray(images)
        images = images.reshape((num_files * images.shape[1], ) +
                                images.shape[2:])
    else:
        raise NotImplementedError

    dataset = glow.dataset.Dataset(images)
    iterator = glow.dataset.Iterator(dataset, batch_size=2)

    print(tabulate([["#image", len(dataset)]]))

    encoder = Glow(hyperparams, hdf5_path=args.snapshot_path)
    if using_gpu:
        encoder.to_gpu()

    total = args.num_steps + 2
    fig = plt.figure(figsize=(4 * total, 4))
    subplots = []
    for n in range(total):
        subplot = fig.add_subplot(1, total, n + 1)
        subplots.append(subplot)

    with chainer.no_backprop_mode() and encoder.reverse() as decoder:
        while True:
            for data_indices in iterator:
                x = to_gpu(dataset[data_indices])
                x += xp.random.uniform(0, 1.0 / num_bins_x, size=x.shape)
                factorized_z_distribution, _ = encoder.forward_step(x)

                factorized_z = []
                for (zi, mean, ln_var) in factorized_z_distribution:
                    factorized_z.append(zi)

                z = encoder.merge_factorized_z(factorized_z)
                z_start = z[0]
                z_end = z[1]

                z_batch = []
                for n in range(args.num_steps):
                    ratio = n / (args.num_steps - 1)
                    z_interp = ratio * z_end + (1.0 - ratio) * z_start
                    z_batch.append(args.temperature * z_interp)

                for n in range(args.num_steps):
                    z = z_batch[n]
                    z = z[None, ...]
                    rev_x, _ = decoder.reverse_step(z)
                    rev_x_img = make_uint8(rev_x.data[0], num_bins_x)
                    subplots[n + 1].imshow(rev_x_img, interpolation="none")

                x_start_img = make_uint8(x[0], num_bins_x)
                subplots[0].imshow(x_start_img, interpolation="none")

                x_end_img = make_uint8(x[-1], num_bins_x)
                subplots[-1].imshow(x_end_img, interpolation="none")

                plt.pause(.01)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot-path", "-snapshot", type=str, required=True)
    parser.add_argument("--num-steps", "-steps", type=int, default=5)
    parser.add_argument("--dataset-path", "-dataset", type=str, required=True)
    parser.add_argument("--gpu-device", "-gpu", type=int, default=0)
    parser.add_argument("--dataset-format", "-ext", type=str, required=True)
    parser.add_argument("--temperature", "-temp", type=float, default=0.7)
    args = parser.parse_args()
    main()
