class AffineCoupling:
    def __call__(self, x):
        raise NotImplementedError

    def compute_log_determinant(self, h, w):
        raise NotImplementedError


class ReverseAffineCoupling:
    def __call__(self, y):
        raise NotImplementedError


class NonlinearMapping:
    def __call__(self, x):
        raise NotImplementedError
