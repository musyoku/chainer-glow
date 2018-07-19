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

sys.path.append(os.path.join("..", "..", ".."))
import glow

sys.path.append("..")
from model import InferenceModel, GenerativeModel, to_cpu, to_gpu
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

    files = Path(args.dataset_path).glob("*.png")
    images = []
    for filepath in files:
        image = np.array(Image.open(filepath)).astype("float32")
        image = image / 255.0 - 0.5
        image = image.transpose((2, 0, 1))
        images.append(image)
    images = np.asanyarray(images)

    dataset = glow.dataset.png.Dataset(images)
    iterator = glow.dataset.png.Iterator(dataset, batch_size=1)

    print(tabulate([["#image", len(dataset)]]))

    hyperparams = Hyperparameters(args.snapshot_path)
    print(
        tabulate([
            ["levels", hyperparams.levels],
            ["depth_per_level", hyperparams.depth_per_level],
            ["nn_hidden_channels", hyperparams.nn_hidden_channels],
            ["image_size", hyperparams.image_size],
            ["lu_decomposition", hyperparams.lu_decomposition],
            ["num_bits_x", hyperparams.num_bits_x],
        ]))
    num_bins_x = 2.0**hyperparams.num_bits_x

    encoder = InferenceModel(hyperparams, hdf5_path=args.snapshot_path)
    decoder = encoder.reverse()

    if using_gpu:
        encoder.to_gpu()
        decoder.to_gpu()

    fig = plt.figure(figsize=(8, 4))
    left = fig.add_subplot(1, 2, 1)
    right = fig.add_subplot(1, 2, 2)

    with chainer.no_backprop_mode():
        while True:
            for data_indices in iterator:
                x = to_gpu(dataset[data_indices])
                x += xp.random.uniform(0.0, 1.0 / 256.0, size=x.shape)
                factorized_z, _ = encoder(x)
                rev_x = decoder(factorized_z)

                x_img = make_uint8(x[0], num_bins_x)
                rev_x_img = make_uint8(rev_x.data[0], num_bins_x)

                left.imshow(x_img, interpolation="none")
                right.imshow(rev_x_img, interpolation="none")

                plt.pause(.01)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot-path", "-snapshot", type=str, required=True)
    parser.add_argument("--dataset-path", "-dataset", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--gpu-device", "-gpu", type=int, default=0)
    args = parser.parse_args()
    main()
