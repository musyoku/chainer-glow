import os
import sys
import chainer
import uuid
import cupy
import numpy as np
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


class InferenceModel():
    def __init__(self, hyperparams: Hyperparameters, hdf5_path=None):
        assert isinstance(hyperparams, Hyperparameters)
        self.hyperparams = hyperparams
        self.parameters = chainer.Chain()
        channels_x = 3  # RGB

        with self.parameters.init_scope():
            self.map_flows_level = []

            for level in range(hyperparams.levels):
                if level == 0:
                    # squeeze
                    channels_x *= hyperparams.squeeze_factor**2
                else:
                    # squeeze and split
                    channels_x *= hyperparams.squeeze_factor**2 // 2

                map_flow_depth = []

                for depth in range(hyperparams.depth_per_level):
                    ### one step of flow ###
                    flow = []

                    # actnorm
                    params = glow.nn.chainer.actnorm.Parameters(
                        channels=channels_x)
                    actnorm = glow.nn.chainer.actnorm.Actnorm(params)
                    setattr(self.parameters, "actnorm_{}_{}".format(
                        level, depth), params)
                    flow.append(actnorm)

                    # invertible 1x1 convolution
                    params = glow.nn.chainer.invertible_1x1_conv.Parameters(
                        channels=channels_x)
                    conv_1x1 = glow.nn.chainer.invertible_1x1_conv.Invertible1x1Conv(
                        params)
                    setattr(self.parameters,
                            "invertible_1x1_conv_{}_{}".format(level,
                                                               depth), params)
                    flow.append(conv_1x1)

                    # affine coupling layer
                    params = glow.nn.chainer.affine_coupling.Parameters(
                        channels_x=channels_x,
                        channels_h=hyperparams.nn_hidden_channels)
                    nonlinear_mapping = glow.nn.chainer.affine_coupling.NonlinearMapping(
                        params)  # NN
                    coupling_layer = glow.nn.chainer.affine_coupling.AffineCoupling(
                        nn=nonlinear_mapping)
                    setattr(self.parameters, "affine_coupling_{}_{}".format(
                        level, depth), params)
                    flow.append(coupling_layer)

                    map_flow_depth.append(flow)

                self.map_flows_level.append(map_flow_depth)

        if hdf5_path:
            try:
                load_hdf5(
                    os.path.join(hdf5_path, self.params_filename),
                    self.parameters)
            except:
                pass

    def __call__(self, x):
        z = []
        levels = self.hyperparams.levels
        depth_per_level = self.hyperparams.depth_per_level
        sum_logdet = 0

        for level in range(levels):

            # squeeze
            out = squeeze(x, factor=self.hyperparams.squeeze_factor)

            # step of flow
            for depth in range(depth_per_level):
                actnorm, conv_1x1, coupling_layer = self[level][depth]
                out, logdet = actnorm(out)
                sum_logdet += logdet

                out, logdet = conv_1x1(out)
                sum_logdet += logdet

                out, logdet = coupling_layer(out)
                sum_logdet += logdet

            # split
            if level == levels - 1:
                z.append(out)
            else:
                n = out.shape[1]
                zi = out[:, :n // 2]
                x = out[:, n // 2:]
                z.append(zi)

        return z, sum_logdet

    @property
    def params_filename(self):
        return "model.hdf5"

    def to_gpu(self):
        self.parameters.to_gpu()

    def cleargrads(self):
        self.parameters.cleargrads()

    def serialize(self, path):
        self.serialize_parameter(path, self.params_filename, self.parameters)

    def serialize_parameter(self, path, filename, params):
        tmp_filename = str(uuid.uuid4())
        save_hdf5(os.path.join(path, tmp_filename), params)
        os.rename(
            os.path.join(path, tmp_filename), os.path.join(path, filename))

    def __getitem__(self, level):
        return self.map_flows_level[level]

    # data dependent initialization
    def initialize_actnorm_weights(self, x):
        xp = cuda.get_array_module(x)
        levels = self.hyperparams.levels
        depth_per_level = self.hyperparams.depth_per_level
        for level in range(levels):

            # squeeze
            out = squeeze(x, factor=self.hyperparams.squeeze_factor)

            # step of flow
            for depth in range(depth_per_level):
                actnorm, conv_1x1, coupling_layer = self[level][depth]
                mean = xp.mean(out.data, axis=(0, 2, 3))
                std = xp.std(out.data, axis=(0, 2, 3))

                params = actnorm.params
                params.scale.W.data = 1.0 / std
                params.bias.b.data = -mean

                out, _ = actnorm(out)
                out, _ = conv_1x1(out)
                out, _ = coupling_layer(out)

            # split
            if level < levels - 1:
                n = out.shape[1]
                x = out[:, n // 2:]


def reverse_actnorm(layer: glow.nn.chainer.actnorm.Actnorm):
    source = layer.params
    target = glow.nn.chainer.actnorm.Parameters(source.channels)
    target.scale.W.data = 1.0 / source.scale.W.data
    target.bias.b.data = -source.bias.b.data
    return glow.nn.chainer.actnorm.ReverseActnorm(params=target)


def reverse_conv_1x1(
        layer: glow.nn.chainer.invertible_1x1_conv.Invertible1x1Conv):
    source = layer.params
    target = glow.nn.chainer.invertible_1x1_conv.Parameters(source.channels)
    source_weight = source.conv.W.data
    # make it a square matrix
    weight = source_weight.reshape(source_weight.shape[:2])
    xp = cuda.get_array_module(weight)
    inv_weight = xp.linalg.inv(weight)
    # make it a conv kernel
    target.conv.W.data = to_cpu(inv_weight.reshape(inv_weight.shape + (1, 1)))
    return glow.nn.chainer.invertible_1x1_conv.ReverseInvertible1x1Conv(
        params=target)


def reverse_coupling_layer(
        layer: glow.nn.chainer.affine_coupling.AffineCoupling):
    source = layer.nn.params
    target = glow.nn.chainer.affine_coupling.Parameters(
        source.channels_x, source.channels_h)
    target.conv_1.W.data[...] = source.conv_1.W.data
    target.conv_2.W.data[...] = source.conv_2.W.data
    target.conv_3.W.data[...] = source.conv_3.W.data
    nonlinear_mapping = glow.nn.chainer.affine_coupling.NonlinearMapping(
        params=target)
    return glow.nn.chainer.affine_coupling.ReverseAffineCoupling(
        nn=nonlinear_mapping)


class GenerativeModel():
    def __init__(self, model: InferenceModel):
        self.hyperparams = model.hyperparams
        self.parameters = chainer.Chain()

        with self.parameters.init_scope():
            self.map_flows_level = []

            for level in range(self.hyperparams.levels):
                map_flow_depth = []

                for depth in range(self.hyperparams.depth_per_level):
                    actnorm, conv_1x1, coupling_layer = model[level][depth]

                    rev_actnorm = reverse_actnorm(actnorm)
                    rev_conv_1x1 = reverse_conv_1x1(conv_1x1)
                    rev_coupling_layer = reverse_coupling_layer(coupling_layer)

                    setattr(self.parameters, "actnorm_{}_{}".format(
                        level, depth), rev_actnorm.params)
                    setattr(
                        self.parameters, "invertible_1x1_conv_{}_{}".format(
                            level, depth), rev_conv_1x1.params)
                    setattr(self.parameters, "affine_coupling_{}_{}".format(
                        level, depth), rev_coupling_layer.nn.params)

                    flow = []
                    flow.append(rev_coupling_layer)
                    flow.append(rev_conv_1x1)
                    flow.append(rev_actnorm)
                    map_flow_depth.append(flow)

                self.map_flows_level.append(map_flow_depth)

    def __getitem__(self, level):
        return self.map_flows_level[level]

    def factor_z(self, z):
        factorized_z = []
        for level in range(self.hyperparams.levels):
            z = squeeze(z)
            if level == self.hyperparams.levels - 1:
                factorized_z.append(z)
            else:
                zi = z[:, 0::2]
                z = z[:, 1::2]
                factorized_z.append(zi)
        return factorized_z

    def __call__(self, z):
        if isinstance(z, list):
            factorized_z = z
        else:
            factorized_z = self.factor_z(z)

        out = factorized_z.pop(-1)
        for level in range(self.hyperparams.levels - 1, -1, -1):
            for depth in range(self.hyperparams.depth_per_level - 1, -1, -1):
                rev_coupling_layer, rev_conv_1x1, rev_actnorm = self[level][
                    depth]

                out = rev_coupling_layer(out)
                out = rev_conv_1x1(out)
                out = rev_actnorm(out)

            out = unsqueeze(out)
            if level > 0:
                zi = factorized_z.pop(-1)
                out = cf.concat((zi, out), axis=1)

        return out

    def to_gpu(self):
        self.parameters.to_gpu()