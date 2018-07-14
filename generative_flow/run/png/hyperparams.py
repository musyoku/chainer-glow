import json
from pathlib import Path


class Hyperparameters():
    def __init__(self, path=None):
        self.depth_per_level = 32
        self.levels = 6
        self.squeeze_factor = 2
        self.nn_hidden_channels = 512
        self.image_size = (256, 256)

        if path is not None:
            json_path = Path(path) / self.params_filename
            if Path(json_path).exists():
                with open(json_path, "r") as f:
                    obj = json.load(f)
                    for (key, value) in obj.items():
                        if isinstance(value, list):
                            value = tuple(value)
                        setattr(self, key, value)

    @property
    def params_filename(self):
        return "model.json"

    def serialize(self, path):
        with open(Path(path) / self.params_filename, "w") as f:
            json.dump(
                {
                    "depth_per_level": self.depth_per_level,
                    "levels": self.levels,
                    "squeeze_factor": self.squeeze_factor,
                    "nn_hidden_channels": self.nn_hidden_channels,
                    "image_size": self.image_size,
                },
                f,
                indent=4,
                sort_keys=True)
