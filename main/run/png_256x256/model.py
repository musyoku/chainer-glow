import os
import sys
import chainer
import uuid
from chainer import functions as cf
from chainer.serializers import load_hdf5, save_hdf5

sys.path.append(os.path.join("..", ".."))
import glow

from hyperparams import Hyperparameters


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
                        params, reverse=False)  # NN
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
            map_flow_depth = self.map_flows_level[level]

            # squeeze
            out = glow.nn.chainer.functions.squeeze(
                x, factor=self.hyperparams.squeeze_factor)

            # step of flow
            for depth in range(depth_per_level):
                flow = map_flow_depth[depth]
                actnorm, conv_1x1, coupling_layer = flow
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
                zi = out[:, 0::2]
                x = out[:, 1::2]
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


class GenerativeModel():
    pass


def reverse():
    pass