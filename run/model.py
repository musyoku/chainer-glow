import os
import sys
import chainer
import uuid
import cupy
import numpy as np
from pathlib import Path
from chainer import functions as cf
from chainer.serializers import load_hdf5, save_hdf5
from chainer.backends import cuda

sys.path.append("..")
import glow

from glow.nn.chainer.functions import squeeze, unsqueeze
from hyperparams import Hyperparameters


def to_gpu(array):
    if isinstance(array, np.ndarray):
        return cuda.to_gpu(array)
    return array


def to_cpu(array):
    if isinstance(array, cupy.ndarray):
        return cuda.to_cpu(array)
    return array


def closure(layer):
    def func(x):
        return layer(x)

    return func


def split_channel(x):
    n = x.shape[1] // 2
    return x[:, :n], x[:, n:]


def zeros_like(x):
    xp = cuda.get_array_module(x)
    if isinstance(x, chainer.Variable):
        x = x.data
    return xp.zeros_like(x)


class Flow(object):
    def __init__(self, actnorm, conv_1x1, coupling_layer, reverse=False):
        self.actnorm = actnorm
        self.conv_1x1 = conv_1x1
        self.coupling_layer = coupling_layer
        self.is_reversed = reverse
        self.params = chainer.ChainList()

        with self.params.init_scope():
            self.params.append(actnorm.params)
            self.params.append(conv_1x1.params)
            self.params.append(coupling_layer.nn.params)

    def forward_step(self, x, reduce_memory=False):
        sum_logdet = 0
        out = x

        if reduce_memory:
            out, logdet = cf.forget(closure(self.actnorm), out)
        else:
            out, logdet = self.actnorm(out)
        sum_logdet += logdet

        out, logdet = self.conv_1x1(out)
        sum_logdet += logdet

        if reduce_memory:
            out, logdet = cf.forget(closure(self.coupling_layer), out)
        else:
            out, logdet = self.coupling_layer(out)
        sum_logdet += logdet

        return out, sum_logdet

    def reverse_step(self, x, reduce_memory=False):
        sum_logdet = 0
        out = x

        out, logdet = self.coupling_layer(out)
        sum_logdet += logdet

        out, logdet = self.conv_1x1(out)
        sum_logdet += logdet

        out, logdet = self.actnorm(out)
        sum_logdet += logdet

        return out, sum_logdet

    def __call__(self, x, reduce_memory=False):
        if self.is_reversed:
            return self.reverse_step(x, reduce_memory)
        else:
            return self.forward_step(x, reduce_memory)

    def __iter__(self):
        yield self.actnorm
        yield self.conv_1x1
        yield self.coupling_layer

    def reverse(self):
        rev_actnorm = self.actnorm.reverse_copy()
        rev_conv_1x1 = self.conv_1x1.reverse_copy()
        rev_coupling_layer = self.coupling_layer.reverse_copy()
        return Flow(
            rev_actnorm, rev_conv_1x1, rev_coupling_layer, reverse=True)


class Block(object):
    def __init__(self, flows, channels_x, split_output=False, reverse=False):
        self.flows = flows
        self.params = chainer.ChainList()
        self.channels_x = channels_x
        self.split_output = split_output
        self.is_reversed = reverse

        channels_in = channels_x // 2 if split_output else channels_x
        channels_out = channels_x if split_output else channels_x * 2
        prior_params = glow.nn.chainer.conv2d_zeros.Parameters(
            channels_in=channels_in, channels_out=channels_out)
        self.prior = glow.nn.chainer.conv2d_zeros.Conv2dZeros(prior_params)

        with self.params.init_scope():
            for flow in flows:
                self.params.append(flow.params)
            self.params.append(self.prior.params)

    def __getitem__(self, index):
        return self.flows[index]

    def forward_step(self, x, squeeze_factor, reduce_memory=False):
        sum_logdet = 0
        out = x

        out = squeeze(out, factor=squeeze_factor)

        for flow in self.flows:
            out, logdet = flow(out, reduce_memory=reduce_memory)
            sum_logdet += logdet

        if self.split_output:
            zi, out = split_channel(out)
            prior_in = out
        else:
            zi = out
            out = None
            xp = cuda.get_array_module(zi)
            prior_in = zeros_like(zi)

        z_distritubion = self.prior(prior_in)
        mean, ln_var = split_channel(z_distritubion)

        return out, (zi, mean, ln_var), sum_logdet

    def reverse_step(self,
                     out,
                     gaussian_eps,
                     squeeze_factor,
                     reduce_memory=False):
        sum_logdet = 0
        xp = cuda.get_array_module(gaussian_eps)

        if self.split_output:
            z_distritubion = self.prior(out)
            mean, ln_var = split_channel(z_distritubion)
            zi = cf.gaussian(mean, ln_var, eps=gaussian_eps.data)
            out = cf.concat((zi, out), axis=1)
        else:
            zeros = zeros_like(gaussian_eps)
            z_distritubion = self.prior(zeros)
            mean, ln_var = split_channel(z_distritubion)
            out = cf.gaussian(mean, ln_var, eps=gaussian_eps.data)

        for flow in self.flows:
            out, logdet = flow(out, reduce_memory=reduce_memory)
            sum_logdet += logdet

        out = unsqueeze(out, factor=squeeze_factor)

        return out, sum_logdet

    def __call__(self,
                 x,
                 squeeze_factor=2,
                 gaussian_eps=None,
                 reduce_memory=False):
        if self.is_reversed:
            return self.reverse_step(
                x,
                gaussian_eps=gaussian_eps,
                squeeze_factor=squeeze_factor,
                reduce_memory=reduce_memory)
        else:
            return self.forward_step(
                x, squeeze_factor=squeeze_factor, reduce_memory=reduce_memory)

    def reverse(self):
        flows = []
        for flow in self.flows:
            rev_flow = flow.reverse()
            flows.append(rev_flow)
        flows.reverse()
        copy = Block(
            flows,
            channels_x=self.channels_x,
            split_output=self.split_output,
            reverse=True)
        if self.params.xp is not np:
            copy.params.to_gpu()
        copy.prior.params.conv.W.data[...] = self.prior.params.conv.W.data
        return copy


class InferenceModel():
    def __init__(self,
                 hyperparams: Hyperparameters,
                 hdf5_path=None,
                 coupling="additive"):
        assert isinstance(hyperparams, Hyperparameters)
        available_couplings = ["additive", "affine"]
        assert coupling in available_couplings

        self.hyperparams = hyperparams
        self.params = chainer.ChainList()
        self.blocks = []
        self.need_initialize = True
        channels_x = 3  # RGB

        with self.params.init_scope():

            for level in range(hyperparams.levels):
                if level == 0:
                    # squeeze
                    channels_x *= hyperparams.squeeze_factor**2
                else:
                    # squeeze and split
                    channels_x *= hyperparams.squeeze_factor**2 // 2

                flows = []

                for _ in range(hyperparams.depth_per_level):
                    params = glow.nn.chainer.actnorm.Parameters(
                        channels=channels_x)
                    actnorm = glow.nn.chainer.actnorm.Actnorm(params)

                    if hyperparams.lu_decomposition:
                        params = glow.nn.chainer.invertible_1x1_conv.LUParameters(
                            channels=channels_x)
                        conv_1x1 = glow.nn.chainer.invertible_1x1_conv.LUInvertible1x1Conv(
                            params)
                    else:
                        params = glow.nn.chainer.invertible_1x1_conv.Parameters(
                            channels=channels_x)
                        conv_1x1 = glow.nn.chainer.invertible_1x1_conv.Invertible1x1Conv(
                            params)

                    if coupling == "additive":
                        params = glow.nn.chainer.additive_coupling.Parameters(
                            channels_x=channels_x // 2,
                            channels_h=hyperparams.nn_hidden_channels)
                        nonlinear_mapping = glow.nn.chainer.additive_coupling.NonlinearMapping(
                            params)  # NN
                        coupling_layer = glow.nn.chainer.additive_coupling.AdditiveCoupling(
                            nn=nonlinear_mapping)
                    elif coupling == "affine":
                        params = glow.nn.chainer.affine_coupling.Parameters(
                            channels_x=channels_x // 2,
                            channels_h=hyperparams.nn_hidden_channels)
                        nonlinear_mapping = glow.nn.chainer.affine_coupling.NonlinearMapping(
                            params)  # NN
                        coupling_layer = glow.nn.chainer.affine_coupling.AffineCoupling(
                            nn=nonlinear_mapping)
                    else:
                        raise NotImplementedError

                    flows.append(Flow(actnorm, conv_1x1, coupling_layer))

                split_output = False if level == hyperparams.levels - 1 else True
                block = Block(
                    flows, channels_x=channels_x, split_output=split_output)
                self.blocks.append(block)
                self.params.append(block.params)

        if hdf5_path:
            try:
                filepath = os.path.join(hdf5_path, self.params_filename)
                if os.path.exists(filepath) and os.path.isfile(filepath):
                    print("loading {}".format(filepath))
                    load_hdf5(filepath, self.params)
                    self.need_initialize = False
            except Exception as error:
                print(error)

    def __call__(self, x, reduce_memory=False):
        z = []
        sum_logdet = 0
        out = x

        for block in self.blocks:
            out, zi_mean_lnvar, logdet = block(
                out, squeeze_factor=self.hyperparams.squeeze_factor)
            sum_logdet += logdet
            z.append(zi_mean_lnvar)

        return z, sum_logdet

    @property
    def params_filename(self):
        return "model.hdf5"

    def to_gpu(self):
        self.params.to_gpu()

    def cleargrads(self):
        self.params.cleargrads()

    def serialize(self, path):
        self.serialize_parameter(path, self.params_filename, self.params)

    def serialize_parameter(self, path, filename, params):
        tmp_filename = str(uuid.uuid4())
        tmp_filepath = os.path.join(path, tmp_filename)
        save_hdf5(tmp_filepath, params)
        os.rename(tmp_filepath, os.path.join(path, filename))

    def __getitem__(self, level):
        return self.blocks[level]

    # data dependent initialization
    def initialize_actnorm_weights(self, x):
        xp = cuda.get_array_module(x)
        levels = len(self.blocks)
        out = x

        for level, block in enumerate(self.blocks):
            out = squeeze(out, factor=self.hyperparams.squeeze_factor)

            for flow in block.flows:
                mean = xp.mean(out.data, axis=(0, 2, 3), keepdims=True)
                std = xp.std(out.data, axis=(0, 2, 3), keepdims=True)

                params = flow.actnorm.params
                params.scale.data = 1.0 / std
                params.bias.data = -mean

                out, _ = flow(out)

            if level < levels - 1:
                _, out = split_channel(out)

    def reverse(self):
        return GenerativeModel(self)


class GenerativeModel():
    def __init__(self, source: InferenceModel):
        self.hyperparams = source.hyperparams
        self.params = chainer.ChainList()
        self.blocks = []

        with self.params.init_scope():
            for block in source.blocks:
                rev_block = block.reverse()
                self.blocks.append(rev_block)
                self.params.append(rev_block.params)

        self.blocks.reverse()

    def __getitem__(self, level):
        return self.blocks[level]

    def factor_z(self, z):
        factorized_z = []
        for level in range(self.hyperparams.levels):
            z = squeeze(z)
            if level == self.hyperparams.levels - 1:
                factorized_z.append(z)
            else:
                zi, z = split_channel(z)
                factorized_z.append(zi)
        return factorized_z

    def __call__(self, z):
        if isinstance(z, list):
            factorized_z = z
        else:
            factorized_z = self.factor_z(z)

        assert len(factorized_z) == len(self.blocks)

        out = None
        sum_logdet = 0

        for block, zi in zip(self.blocks, factorized_z[::-1]):
            out, logdet = block(
                out,
                gaussian_eps=zi,
                squeeze_factor=self.hyperparams.squeeze_factor)
            sum_logdet += logdet

        return out, sum_logdet

    def to_gpu(self):
        self.params.to_gpu()