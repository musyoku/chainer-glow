class AdditiveCoupling:
    def __call__(self, x):
        raise NotImplementedError


class ReverseAdditiveCoupling:
    def __call__(self, y):
        raise NotImplementedError


class NonlinearMapping:
    def __call__(self, x):
        raise NotImplementedError
