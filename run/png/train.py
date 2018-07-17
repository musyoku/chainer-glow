import argparse
import math
import os
import random
import sys

from tabulate import tabulate
from PIL import Image
from pathlib import Path

import chainer
import chainer.functions as cf
import cupy
import numpy as np
from chainer.backends import cuda

sys.path.append(".")
sys.path.append(os.path.join("..", ".."))
import glow

from hyperparams import Hyperparameters
from model import InferenceModel, GenerativeModel
from optimizer import Optimizer


def to_gpu(array):
    if isinstance(array, np.ndarray):
        return cuda.to_gpu(array)
    return array


def to_cpu(array):
    if isinstance(array, cupy.ndarray):
        return cuda.to_cpu(array)
    return array


def printr(string):
    sys.stdout.write(string)
    sys.stdout.write("\r")


def main():
    try:
        os.mkdir(args.snapshot_path)
    except:
        pass

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
    iterator = glow.dataset.png.Iterator(dataset, batch_size=args.batch_size)

    print(tabulate([["#image", len(dataset)]]))

    hyperparams = Hyperparameters(args.snapshot_path)
    hyperparams.levels = args.levels
    hyperparams.depth_per_level = args.depth_per_level
    hyperparams.nn_hidden_channels = args.nn_hidden_channels
    hyperparams.image_size = image.shape[1:]
    hyperparams.serialize(args.snapshot_path)

    print(
        tabulate([
            ["levels", hyperparams.levels],
            ["depth_per_level", hyperparams.depth_per_level],
            ["nn_hidden_channels", hyperparams.nn_hidden_channels],
            ["image_size", hyperparams.image_size],
        ]))

    encoder = InferenceModel(hyperparams, hdf5_path=args.snapshot_path)
    if using_gpu:
        encoder.to_gpu()

    optimizer = Optimizer(encoder.parameters)

    # Data dependent initialization
    if encoder.need_initialize:
        for batch_index, data_indices in enumerate(iterator):
            x = to_gpu(dataset[data_indices])
            encoder.initialize_actnorm_weights(x)
            break

    current_training_step = 0

    # Training loop
    num_pixels = hyperparams.image_size[0] * hyperparams.image_size[1]
    for iteration in range(args.training_steps):
        sum_loss = 0
        for batch_index, data_indices in enumerate(iterator):
            x = to_gpu(dataset[data_indices])
            x += xp.random.uniform(0, 1.0 / 256.0, size=x.shape)
            factorized_z, logdet = encoder(x, reduce_memory=args.reduce_memory)
            logdet -= math.log(256.0) * num_pixels
            negative_log_likelihood = 0
            for zi in factorized_z:
                prior_mean = xp.zeros(zi.shape, dtype="float32")
                prior_ln_var = prior_mean
                negative_log_likelihood += cf.gaussian_nll(
                    zi, prior_mean, prior_ln_var)
            denom = args.batch_size * num_pixels
            loss = (negative_log_likelihood - logdet) / denom
            encoder.cleargrads()
            loss.backward()
            optimizer.update(current_training_step)

            current_training_step += 1

            sum_loss += float(loss.data)
            printr(
                "Iteration {}: Batch {} / {} - loss: {:.8f} - nll: {:.8f} - log_det: {:.8f}".
                format(iteration + 1, batch_index + 1, len(iterator),
                       float(loss.data),
                       float(negative_log_likelihood.data) / denom,
                       float(logdet.data) / denom))

            if batch_index % 100 == 0:
                encoder.serialize(args.snapshot_path)

        # Check model reversibility
        reconstruction_error = None
        if True:
            with chainer.no_backprop_mode():
                decoder = encoder.reverse()
                if using_gpu:
                    decoder.to_gpu()
                factorized_z, logdet = encoder(x)
                rev_x = decoder(factorized_z)
                reconstruction_error = float(cf.mean(abs(x - rev_x)).data)

        print(
            "\033[2KIteration {} - loss: {:.5f} - reconstruction error: {:.5f}  - step: {}".
            format(iteration + 1, sum_loss / len(iterator),
                   reconstruction_error, current_training_step))
        encoder.serialize(args.snapshot_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", "-dataset", type=str, required=True)
    parser.add_argument(
        "--snapshot-path", "-snapshot", type=str, default="snapshot")
    parser.add_argument("--batch-size", "-b", type=int, default=32)
    parser.add_argument("--gpu-device", "-gpu", type=int, default=0)
    parser.add_argument("--reduce-memory", action="store_true")

    parser.add_argument("--training-steps", "-i", type=int, default=100000)
    parser.add_argument("--depth-per-level", "-depth", type=int, default=32)
    parser.add_argument("--levels", "-levels", type=int, default=5)
    parser.add_argument("--nn-hidden-channels", "-nn", type=int, default=512)
    args = parser.parse_args()
    main()
