import os
import numpy as np


class Dataset():
    def __init__(self, images):
        self.images = images

    def __getitem__(self, indices):
        return self.images[indices]

    def __len__(self):
        return self.images.shape[0]
