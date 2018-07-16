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
        image = (image / 255.0 * 2.0) - 1.0
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

    model = InferenceModel(hyperparams, hdf5_path=args.snapshot_path)
    if using_gpu:
        model.to_gpu()

    optimizer = Optimizer(model.parameters)

    # Data dependent initialization
    if model.need_initialize:
        for batch_index, data_indices in enumerate(iterator):
            x = to_gpu(dataset[data_indices])
            model.initialize_actnorm_weights(x)
            break

    current_training_step = 0

    # Training loop
    for iteration in range(args.training_steps):
        sum_loss = 0
        for batch_index, data_indices in enumerate(iterator):
            x = to_gpu(dataset[data_indices])
            factorized_z, logdet = model(x, reduce_memory=args.reduce_memory)
            negative_log_likelihood = 0
            for zi in factorized_z:
                prior_mean = xp.zeros(zi.shape, dtype="float32")
                prior_ln_var = prior_mean
                negative_log_likelihood += cf.gaussian_nll(
                    zi, prior_mean, prior_ln_var)
            loss = (negative_log_likelihood - logdet) / args.batch_size
            model.cleargrads()
            loss.backward()
            optimizer.update(current_training_step)

            current_training_step += 1

            sum_loss += float(loss.data)
            printr("Iteration {}: Batch {} / {} - loss: {:.3f}".format(
                iteration + 1, batch_index + 1, len(iterator),
                float(loss.data)))

            if batch_index % 100 == 0:
                model.serialize(args.snapshot_path)

        print("\033[2KIteration {} - loss: {:.3f} - step: {}".format(
            iteration + 1, sum_loss / len(iterator), current_training_step))
        model.serialize(args.snapshot_path)

        # Check model stability
        if True:
            with chainer.no_backprop_mode():
                generative_model = GenerativeModel(model)
                if using_gpu:
                    generative_model.to_gpu()
                factorized_z, logdet = model(x)
                rev_x = generative_model(factorized_z)
                error = cf.mean(abs(x - rev_x))
                print(error)


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
