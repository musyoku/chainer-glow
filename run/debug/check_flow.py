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

from glow.nn.functions import squeeze, unsqueeze

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


def forward_flows(x, block_enc, block_gen):
    sum_logdet = 0

    for flow_enc, flow_gen in zip(block_enc.flows, reversed(block_gen.flows)):
        xp = cuda.get_array_module(x.data)

        y, logdet = flow_enc(x)
        rev_x, _ = flow_gen(y)

        # print(
        #     xp.mean(x.data), xp.var(x.data), xp.mean(y.data), xp.var(y.data),
        #     xp.mean(rev_x.data), xp.var(rev_x.data))
        # print(xp.mean(xp.abs(x.data - rev_x.data)))

        sum_logdet += logdet
        x = y

    return y, sum_logdet


def forward_blocks(x, encoder: InferenceModel, decoder: GenerativeModel):
    z = []
    sum_logdet = 0
    out = x
    num_levels = len(encoder.blocks)

    for level, (block_enc, block_gen) in enumerate(
            zip(encoder.blocks, reversed(decoder.blocks))):
        # squeeze
        out = squeeze(out, factor=2)

        # step of flow
        out, logdet = forward_flows(out, block_enc, block_gen)
        sum_logdet += logdet

        # split
        if level == num_levels - 1:
            z.append(out)
        else:
            n = out.shape[1]
            zi = out[:, :n // 2]
            out = out[:, n // 2:]
            z.append(zi)

    return z, sum_logdet


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
            ["lu_decomposition", hyperparams.lu_decomposition],
            ["num_bits_x", hyperparams.num_bits_x],
        ]))
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
            break
        assert len(images) > 0
        num_files = len(images)
        images = np.asanyarray(images)
        images = images.reshape((num_files * images.shape[1], ) +
                                images.shape[2:])
    else:
        raise NotImplementedError

    dataset = glow.dataset.png.Dataset(images)
    iterator = glow.dataset.png.Iterator(dataset, batch_size=1)

    print(tabulate([["#image", len(dataset)]]))

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
                x += xp.random.uniform(0, 1.0 / num_bins_x, size=x.shape)
                factorized_z, _ = forward_blocks(x, encoder, decoder)

                rev_x, _ = decoder(factorized_z)
                print(
                    xp.mean(x), xp.var(x), " ->", xp.mean(rev_x.data),
                    xp.var(rev_x.data))

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
    parser.add_argument("--dataset-format", "-ext", type=str, required=True)
    args = parser.parse_args()
    main()
