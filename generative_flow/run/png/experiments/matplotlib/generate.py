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

sys.path.append(os.path.join("..", "..", "..", ".."))
import glow

sys.path.append(os.path.join("..", ".."))
from model import InferenceModel, GenerativeModel, to_cpu
from hyperparams import Hyperparameters


def make_uint8(array):
    if array.ndim == 4:
        array = array[0]
    if (array.shape[2] == 3):
        return np.uint8(np.clip((to_cpu(array) + 1) * 0.5 * 255, 0, 255))
    return np.uint8(
        np.clip((to_cpu(array.transpose(1, 2, 0)) + 1) * 0.5 * 255, 0, 255))


def main():
    xp = np
    using_gpu = args.gpu_device >= 0
    if using_gpu:
        cuda.get_device(args.gpu_device).use()
        xp = cupy

    hyperparams = Hyperparameters(args.snapshot_path)
    print(
        tabulate([
            ["levels", hyperparams.levels],
            ["depth_per_level", hyperparams.depth_per_level],
            ["nn_hidden_channels", hyperparams.nn_hidden_channels],
            ["image_size", hyperparams.image_size],
        ]))

    inference_model = InferenceModel(hyperparams, hdf5_path=args.snapshot_path)
    generative_model = GenerativeModel(inference_model)

    if using_gpu:
        inference_model.to_gpu()
        generative_model.to_gpu()

    while True:
        z = xp.random.normal(
            0, args.temperature, size=(
                1,
                3,
            ) + hyperparams.image_size).astype("float32")

        with chainer.no_backprop_mode():
            x = generative_model(z)
            x_img = make_uint8(x.data[0])
            plt.imshow(x_img, interpolation="none")
            plt.pause(.01)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot-path", "-snapshot", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--gpu-device", "-gpu", type=int, default=0)
    args = parser.parse_args()
    main()
