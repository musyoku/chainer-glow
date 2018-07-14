import os
import sys
import argparse

import chainer
import chainer.functions as cf
import cupy
import numpy as np
from chainer.backends import cuda

sys.path.append(os.path.join("..", "..", ".."))
import glow

sys.path.append("..")
from model import InferenceModel, GenerativeModel
from hyperparams import Hyperparameters


def main():
    xp = np
    using_gpu = args.gpu_device >= 0
    if using_gpu:
        cuda.get_device(args.gpu_device).use()
        xp = cupy

    hyperparams = Hyperparameters(args.snapshot_path)
    inference_model = InferenceModel(hyperparams, hdf5_path=args.snapshot_path)
    generative_model = GenerativeModel(inference_model)

    if using_gpu:
        inference_model.to_gpu()
        generative_model.to_gpu()

    z = xp.random.normal(
        0, 1, size=(
            1,
            3,
        ) + hyperparams.image_size)
    x = generative_model(z)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-path", type=str, default="snapshot")
    parser.add_argument("--gpu-device", "-gpu", type=int, default=0)
    args = parser.parse_args()
    main()
