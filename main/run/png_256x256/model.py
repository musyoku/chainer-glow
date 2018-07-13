import os
import sys
import chainer
import uuid
from chainer.serializers import load_hdf5, save_hdf5

sys.path.append(os.path.join("..", ".."))
import glow

from hyperparams import Hyperparameters


class InferenceModel():
    def __init__(self, hyperparams: Hyperparameters, hdf5_path=None):
        assert isinstance(hyperparams, Hyperparameters)

        self.parameters = chainer.Chain()
        num_channels = 3  # RGB

        with self.parameters.init_scope():
            num_channels *= 2  # squeeze
            self.map_flows_level = []

            for level in range(hyperparams.levels):
                map_flow_depth = []

                for depth in range(hyperparams.depth_per_level):
                    ### one step of flow ###
                    flow = []

                    # actnorm
                    params = glow.nn.chainer.actnorm.Parameters(
                        channels=num_channels)
                    actnorm = glow.nn.chainer.actnorm.Actnorm(params)
                    setattr(self.parameters, "actnorm_{}_{}".format(
                        level, depth), params)
                    flow.append(actnorm)

                    # invertible 1x1 convolution
                    params = glow.nn.chainer.invertible_1x1_conv.Parameters(
                        channels=num_channels)
                    conv_1x1 = glow.nn.chainer.invertible_1x1_conv.Invertible1x1Conv(
                        params)
                    setattr(self.parameters,
                            "invertible_1x1_conv_{}_{}".format(level,
                                                               depth), params)
                    flow.append(conv_1x1)

                    # affine coupling layer
                    params = glow.nn.chainer.affine_coupling.Parameters(
                        channels=num_channels)
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


class GenerationModel():
    pass