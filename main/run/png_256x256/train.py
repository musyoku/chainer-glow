import argparse
import math
import os
import random
import sys
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
from model import InferenceModel
from optimizer import Optimizer


def printr(string):
    sys.stdout.write(string)
    sys.stdout.write("\r")


def to_gpu(array):
    if args.gpu_device >= 0:
        return cuda.to_gpu(array)
    return array


def to_cpu(array):
    if args.gpu_device >= 0:
        return cuda.to_cpu(array)
    return array


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
        images.append(image)
    images = np.asanyarray(images)

    dataset = glow.dataset.png.Dataset(images)
    print(len(dataset))

    hyperparams = Hyperparameters()
    model = InferenceModel(hyperparams, hdf5_path=args.snapshot_path)
    if using_gpu:
        model.to_gpu()

    optimizer = Optimizer(model.parameters)

    prior_mean = xp.zeros(
        (args.batch_size, 3) + hyperparams.image_size, dtype="float32")
    prior_ln_var = xp.zeros(
        (args.batch_size, 3) + hyperparams.image_size, dtype="float32")

    current_training_step = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", "-dataset", type=str, required=True)
    parser.add_argument("--snapshot-path", type=str, default="snapshot")
    parser.add_argument("--batch-size", "-b", type=int, default=32)
    parser.add_argument("--gpu-device", "-gpu", type=int, default=0)
    parser.add_argument("--training-steps", "-i", type=int, default=100000)
    args = parser.parse_args()
    main()
