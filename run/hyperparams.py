import json
import os
from tabulate import tabulate


class Hyperparameters():
    def __init__(self, path=None):
        self.depth_per_level = 32
        self.levels = 6
        self.num_bits_x = 8
        self.squeeze_factor = 2
        self.nn_hidden_channels = 512
        self.image_size = (256, 256)
        self.lu_decomposition = False

        if path is not None:
            json_path = os.path.join(path, self.params_filename)
            if os.path.exists(json_path) and os.path.isfile(json_path):
                with open(json_path, "r") as f:
                    obj = json.load(f)
                    for (key, value) in obj.items():
                        if isinstance(value, list):
                            value = tuple(value)
                        setattr(self, key, value)

    @property
    def params_filename(self):
        return "model.json"

    def save(self, path):
        with open(os.path.join(path, self.params_filename), "w") as f:
            json.dump(
                {
                    "depth_per_level": self.depth_per_level,
                    "levels": self.levels,
                    "squeeze_factor": self.squeeze_factor,
                    "nn_hidden_channels": self.nn_hidden_channels,
                    "num_bits_x": self.num_bits_x,
                    "image_size": self.image_size,
                    "lu_decomposition": self.lu_decomposition,
                },
                f,
                indent=4,
                sort_keys=True)

    def print(self):
        print(
            tabulate([
                ["levels", self.levels],
                ["squeeze_factor", self.squeeze_factor],
                ["depth_per_level", self.depth_per_level],
                ["nn_hidden_channels", self.nn_hidden_channels],
                ["image_size", self.image_size],
                ["lu_decomposition", self.lu_decomposition],
                ["num_bits_x", self.num_bits_x],
            ]))
