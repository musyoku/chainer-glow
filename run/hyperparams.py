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
            json_path = os.path.join(path, self.filename)
            if os.path.exists(json_path) and os.path.isfile(json_path):
                with open(json_path, "r") as f:
                    obj = json.load(f)
                    for (key, value) in obj.items():
                        if isinstance(value, list):
                            value = tuple(value)
                        setattr(self, key, value)
            else:
                raise Exception("{} does not exist".format(json_path))

    @property
    def filename(self):
        return "hyperparams.json"

    def save(self, path):
        with open(os.path.join(path, self.filename), "w") as f:
            json.dump(self.__dict__, f, indent=4, sort_keys=True)

    def print(self):
        rows = []
        for key, value in self.__dict__.items():
            rows.append([key, value])
        print(tabulate(rows))
