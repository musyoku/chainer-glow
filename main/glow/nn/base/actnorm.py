class Actnorm:
    def __call__(self, x):
        raise NotImplementedError

    def compute_log_determinant(self):
        raise NotImplementedError

class ReverseActnorm:
    def __call__(self, y):
        raise NotImplementedError