class Invertible1x1Conv:
    def __call__(self, x):
        raise NotImplementedError

    def compute_log_determinant(self, h, w):
        raise NotImplementedError


class LUInvertible1x1Conv:
    def __call__(self, x):
        raise NotImplementedError

    def compute_log_determinant(self, h, w):
        raise NotImplementedError