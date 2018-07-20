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

sys.path.append(os.path.join("..", ".."))
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

    def __call__(self, x, reduce_memory=False):
        sum_logdet = 0
        out = x
        if self.is_reversed is False:
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
        else:
            out, logdet = self.coupling_layer(out)
            sum_logdet += logdet

            out, logdet = self.conv_1x1(out)
            sum_logdet += logdet

            out, logdet = self.actnorm(out)
            sum_logdet += logdet

        return out, sum_logdet

    def layers(self):
        return self.actnorm, self.conv_1x1, self.coupling_layer

    def __iter__(self):
        yield self.actnorm
        yield self.conv_1x1
        yield self.coupling_layer

    def reverse(self):
        rev_actnorm = reverse_actnorm(self.actnorm)
        rev_conv_1x1 = reverse_conv_1x1(self.conv_1x1)
        rev_coupling_layer = reverse_coupling_layer(self.coupling_layer)
        return Flow(
            rev_actnorm, rev_conv_1x1, rev_coupling_layer, reverse=True)


class Block(object):
    def __init__(self, flows):
        self.flows = flows
        self.params = chainer.ChainList()

        with self.params.init_scope():
            for flow in flows:
                self.params.append(flow.params)

    def __getitem__(self, index):
        return self.flows[index]

    def __call__(self, x, reduce_memory=False):
        sum_logdet = 0
        out = x

        for flow in self.flows:
            out, logdet = flow(x, reduce_memory=reduce_memory)
            sum_logdet += logdet

        return out, sum_logdet

    def reverse(self):
        flows = []
        for flow in self.flows:
            rev_flow = flow.reverse()
            flows.append(rev_flow)
        flows.reverse()
        return Block(flows)


class InferenceModel():
    def __init__(self, hyperparams: Hyperparameters, hdf5_path=None):
        assert isinstance(hyperparams, Hyperparameters)
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

                    params = glow.nn.chainer.affine_coupling.Parameters(
                        channels_x=channels_x,
                        channels_h=hyperparams.nn_hidden_channels)
                    nonlinear_mapping = glow.nn.chainer.affine_coupling.NonlinearMapping(
                        params)  # NN
                    coupling_layer = glow.nn.chainer.affine_coupling.AffineCoupling(
                        nn=nonlinear_mapping)

                    flows.append(Flow(actnorm, conv_1x1, coupling_layer))

                block = Block(flows)
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
        num_levels = len(self.blocks)

        for level, block in enumerate(self.blocks):
            # squeeze
            out = squeeze(out, factor=self.hyperparams.squeeze_factor)

            # step of flow
            out, logdet = block(out)
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
                n = out.shape[1]
                out = out[:, n // 2:]

    def reverse(self):
        return GenerativeModel(self)


def reverse_actnorm(layer: glow.nn.chainer.actnorm.Actnorm):
    source = layer.params
    target = glow.nn.chainer.actnorm.Parameters(source.channels)
    target.scale.data[...] = to_cpu(source.scale.data)
    target.bias.data[...] = to_cpu(source.bias.data)
    return glow.nn.chainer.actnorm.ReverseActnorm(params=target)


def reverse_conv_1x1(layer):
    if isinstance(layer,
                  glow.nn.chainer.invertible_1x1_conv.Invertible1x1Conv):
        source = layer.params
        target = glow.nn.chainer.invertible_1x1_conv.Parameters(
            source.channels)
        source_weight = source.conv.W.data
        # square matrix
        weight = source_weight.reshape(source_weight.shape[:2])
        xp = cuda.get_array_module(weight)
        inv_weight = xp.linalg.inv(weight)
        # conv kernel
        target.conv.W.data = to_cpu(
            inv_weight.reshape(inv_weight.shape + (1, 1)))
        return glow.nn.chainer.invertible_1x1_conv.ReverseInvertible1x1Conv(
            params=target)

    if isinstance(layer,
                  glow.nn.chainer.invertible_1x1_conv.LUInvertible1x1Conv):
        source = layer.params
        target = glow.nn.chainer.invertible_1x1_conv.Parameters(
            source.channels)
        source_weight = source.W.data
        # square matrix
        weight = source_weight.reshape(source_weight.shape[:2])
        xp = cuda.get_array_module(weight)
        inv_weight = xp.linalg.inv(weight)
        # conv kernel
        target.conv.W.data = to_cpu(
            inv_weight.reshape(inv_weight.shape + (1, 1)))
        return glow.nn.chainer.invertible_1x1_conv.ReverseInvertible1x1Conv(
            params=target)

    raise NotImplementedError


def reverse_coupling_layer(
        layer: glow.nn.chainer.affine_coupling.AffineCoupling):
    source = layer.nn.params
    target = glow.nn.chainer.affine_coupling.Parameters(
        source.channels_x, source.channels_h)
    target.conv_1.W.data[...] = to_cpu(source.conv_1.W.data)
    target.conv_2.W.data[...] = to_cpu(source.conv_2.W.data)
    target.conv_scale.W.data[...] = to_cpu(source.conv_scale.W.data)
    target.conv_bias.W.data[...] = to_cpu(source.conv_bias.W.data)
    nonlinear_mapping = glow.nn.chainer.affine_coupling.NonlinearMapping(
        params=target)
    return glow.nn.chainer.affine_coupling.ReverseAffineCoupling(
        nn=nonlinear_mapping)


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
                n = z.shape[1]
                zi = z[:, :n // 2]
                z = z[:, n // 2:]
                factorized_z.append(zi)
        return factorized_z

    def __call__(self, z):
        if isinstance(z, list):
            factorized_z = z
        else:
            factorized_z = self.factor_z(z)
        assert len(z) == len(self.blocks)

        i = -1
        out = factorized_z[i]
        sum_logdet = 0
        num_levels = len(self.blocks)

        for level, block in enumerate(self.blocks):
            out, logdet = block(out)
            sum_logdet += logdet
            out = unsqueeze(out)
            if level < num_levels - 1:
                i -= 1
                zi = factorized_z[i]
                out = cf.concat((zi, out), axis=1)

        return out, sum_logdet

    def to_gpu(self):
        self.params.to_gpu()