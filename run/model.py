import os
import sys
import chainer
import uuid
import cupy
import contextlib
import numpy as np
from pathlib import Path
from chainer import functions as cf
from chainer.serializers import load_hdf5, save_hdf5
from chainer.backends import cuda

sys.path.append("..")
import glow

from glow.nn.functions import squeeze, unsqueeze, split_channel, factor_z
from hyperparams import Hyperparameters


def to_gpu(array):
    if isinstance(array, np.ndarray):
        return cuda.to_gpu(array)
    return array


def to_cpu(array):
    if isinstance(array, cupy.ndarray):
        return cuda.to_cpu(array)
    return array


def forward_closure(layer):
    def func(x):
        return layer.forward_step(x)

    return func


def zeros_like(x):
    xp = cuda.get_array_module(x)
    if isinstance(x, chainer.Variable):
        x = x.data
    return xp.zeros_like(x)


class Flow(chainer.Chain):
    def __init__(self, actnorm, conv_1x1, coupling_layer):
        super().__init__()
        with self.init_scope():
            self.actnorm = actnorm
            self.conv_1x1 = conv_1x1
            self.coupling_layer = coupling_layer

    def forward_step(self, x):
        sum_logdet = 0
        out = x

        out, logdet = self.actnorm.forward_step(out)
        sum_logdet += logdet

        out, logdet = self.conv_1x1.forward_step(out)
        sum_logdet += logdet

        out, logdet = self.coupling_layer.forward_step(out)
        sum_logdet += logdet

        return out, sum_logdet

    def reverse_step(self, x):
        sum_logdet = 0
        out = x

        out, logdet = self.coupling_layer.reverse_step(out)
        sum_logdet += logdet

        out, logdet = self.conv_1x1.reverse_step(out)
        sum_logdet += logdet

        out, logdet = self.actnorm.reverse_step(out)
        sum_logdet += logdet

        return out, sum_logdet

    def __iter__(self):
        yield self.actnorm
        yield self.conv_1x1
        yield self.coupling_layer


class Block(chainer.ChainList):
    def __init__(self, flows, channels_x, split_output=False):
        super().__init__()
        assert isinstance(flows, list)

        self.flows = flows
        self.channels_x = channels_x
        self.split_output = split_output

        channels_in = channels_x // 2 if split_output else channels_x
        channels_out = channels_x if split_output else channels_x * 2

        self.prior = glow.nn.Conv2dZeros(
            channels_in=channels_in, channels_out=channels_out)

        # Add parameters to ChainList
        with self.init_scope():
            for flow in flows:
                self.append(flow)
            self.append(self.prior)

    def __getitem__(self, index):
        return self.flows[index]

    def forward_step(self, x, squeeze_factor):
        sum_logdet = 0
        out = x

        out = squeeze(out, factor=squeeze_factor)

        for flow in self.flows:
            out, logdet = flow.forward_step(out)
            sum_logdet += logdet

        if self.split_output:
            zi, out = split_channel(out)
            prior_in = out
        else:
            zi = out
            out = None
            prior_in = zeros_like(zi)

        z_distritubion = self.prior(prior_in)
        mean, ln_var = split_channel(z_distritubion)

        return out, (zi, mean, ln_var), sum_logdet

    def reverse_step(self, out, gaussian_eps, squeeze_factor, sampling=True):
        sum_logdet = 0
        if self.split_output:
            if sampling:
                z_distritubion = self.prior(out)
                mean, ln_var = split_channel(z_distritubion)
                zi = cf.gaussian(mean, ln_var, eps=gaussian_eps)
            else:
                zi = gaussian_eps
            out = cf.concat((zi, out), axis=1)
        else:
            if sampling:
                zeros = zeros_like(gaussian_eps)
                z_distritubion = self.prior(zeros)
                mean, ln_var = split_channel(z_distritubion)
                out = cf.gaussian(mean, ln_var, eps=gaussian_eps)
            else:
                out = gaussian_eps

        for flow in self.flows[::-1]:
            out, logdet = flow.reverse_step(out)
            sum_logdet += logdet

        out = unsqueeze(out, factor=squeeze_factor)

        return out, sum_logdet


class Glow(chainer.ChainList):
    def __init__(self,
                 hyperparams: Hyperparameters,
                 hdf5_path=None,
                 coupling="additive"):
        super().__init__()
        assert isinstance(hyperparams, Hyperparameters)
        available_couplings = ["additive", "affine"]
        assert coupling in available_couplings

        self.is_reverse_mode = False
        self.hyperparams = hyperparams
        self.blocks = []
        self.need_initialize = True

        channels_x = 3  # RGB

        for level in range(hyperparams.levels):
            if level == 0:
                # squeeze
                channels_x *= hyperparams.squeeze_factor**2
            else:
                # squeeze and split
                channels_x *= hyperparams.squeeze_factor**2 // 2

            flows = []

            for _ in range(hyperparams.depth_per_level):
                actnorm = glow.nn.Actnorm(channels=channels_x)

                if hyperparams.lu_decomposition:
                    conv_1x1 = glow.nn.LUInvertible1x1Conv(channels=channels_x)
                else:
                    conv_1x1 = glow.nn.Invertible1x1Conv(channels=channels_x)

                if coupling == "additive":
                    nonlinear_mapping = glow.nn.AdditiveCouplingNonlinearMapping(
                        channels_x=channels_x // 2,
                        channels_h=hyperparams.nn_hidden_channels)  # NN
                    coupling_layer = glow.nn.AdditiveCoupling(
                        nn=nonlinear_mapping)
                elif coupling == "affine":
                    nonlinear_mapping = glow.nn.AffineCouplingNonlinearMapping(
                        channels_x=channels_x // 2,
                        channels_h=hyperparams.nn_hidden_channels)  # NN
                    coupling_layer = glow.nn.AffineCoupling(
                        nn=nonlinear_mapping)
                else:
                    raise NotImplementedError

                flows.append(Flow(actnorm, conv_1x1, coupling_layer))

            split_output = False if level == hyperparams.levels - 1 else True

            block = Block(
                flows, channels_x=channels_x, split_output=split_output)
            self.blocks.append(block)

            # Add parameters to ChainList
            with self.init_scope():
                self.append(block)

        if hdf5_path:
            try:
                filepath = os.path.join(hdf5_path, self.filename)
                if os.path.exists(filepath) and os.path.isfile(filepath):
                    print("loading {}".format(filepath))
                    load_hdf5(filepath, self)
                    self.need_initialize = False
            except Exception as error:
                print(error)

    def forward_step(self, x):
        z = []
        sum_logdet = 0
        out = x

        for block in self.blocks:
            out, zi_mean_lnvar, logdet = block.forward_step(
                out, squeeze_factor=self.hyperparams.squeeze_factor)
            sum_logdet += logdet
            z.append(zi_mean_lnvar)

        return z, sum_logdet

    # return z of same shape as x
    def merge_factorized_z(self, factorized_z, factor=2):
        z = None
        for zi in reversed(factorized_z):
            xp = cuda.get_array_module(zi.data)
            z = zi.data if z is None else xp.concatenate((zi.data, z), axis=1)
            z = glow.nn.functions.unsqueeze(z, factor, xp)
        return z

    def factor_z(self, z):
        return factor_z(
            z,
            levels=self.hyperparams.levels,
            squeeze_factor=self.hyperparams.squeeze_factor)

    def reverse_step(self, z):
        assert self.is_reverse_mode is True

        if isinstance(z, list):
            factorized_z = z
        else:
            factorized_z = self.factor_z(z)

        assert len(factorized_z) == len(self.blocks)

        out = None
        sum_logdet = 0

        for block, zi in zip(self.blocks[::-1], factorized_z[::-1]):
            out, logdet = block.reverse_step(
                out,
                gaussian_eps=zi,
                squeeze_factor=self.hyperparams.squeeze_factor)
            sum_logdet += logdet

        return out, sum_logdet

    @property
    def filename(self):
        return "model.hdf5"

    def save(self, path):
        self.save_parameter(path, self.filename, self)

    def save_parameter(self, path, filename, params):
        tmp_filename = str(uuid.uuid4())
        tmp_filepath = os.path.join(path, tmp_filename)
        save_hdf5(tmp_filepath, params)
        os.rename(tmp_filepath, os.path.join(path, filename))

    def __getitem__(self, level):
        return self.blocks[level]

    # data dependent initialization
    def initialize_actnorm_weights(self, x):
        assert self.need_initialize

        xp = cuda.get_array_module(x)
        levels = len(self.blocks)
        out = x

        for level, block in enumerate(self.blocks):
            out = squeeze(out, factor=self.hyperparams.squeeze_factor)

            for flow in block.flows:
                mean = xp.mean(out.data, axis=(0, 2, 3), keepdims=True)
                std = xp.std(out.data, axis=(0, 2, 3), keepdims=True)

                flow.actnorm.scale.data = 1.0 / std
                flow.actnorm.bias.data = -mean

                out, _ = flow.forward_step(out)

            if level < levels - 1:
                _, out = split_channel(out)

        self.need_initialize = False

    @contextlib.contextmanager
    def reverse(self):
        self.is_reverse_mode = True

        # compute W^(-1)
        for block in self.blocks:
            for flow in block.flows:
                flow.conv_1x1.update_inverse_weight()

        try:
            yield self
        finally:
            self.is_reverse_mode = False
