import chainer
import chainer.links as L
import chainer.functions as cf
from chainer.backends import cuda
from chainer.utils import type_check
import numpy as np
import scipy


class Parameters(chainer.Chain):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        with self.init_scope():
            shape = (channels, channels)
            rotation_mat = np.linalg.qr(np.random.normal(
                size=shape))[0].astype("float32").reshape(shape + (1, 1))
            self.conv = L.Convolution2D(
                channels,
                channels,
                ksize=1,
                stride=1,
                pad=0,
                nobias=True,
                initialW=rotation_mat)


class Diag(chainer.function.Function):
    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1, )

    def forward(self, inputs):
        vector = inputs[0]
        xp = cuda.get_array_module(vector)
        mat = xp.diag(vector)
        return mat,

    def backward(self, inputs, grad_outputs):
        vector = inputs[0]
        grad = grad_outputs[0]
        xp = cuda.get_array_module(vector)
        return xp.diag(grad),


def diag(vector):
    return Diag()(vector)

class Mask(chainer.function.Function):
    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1, )

    def forward(self, inputs):
        vector = inputs[0]
        xp = cuda.get_array_module(vector)
        mat = xp.diag(vector)
        return mat,

    def backward(self, inputs, grad_outputs):
        vector = inputs[0]
        grad = grad_outputs[0]
        xp = cuda.get_array_module(vector)
        return xp.diag(grad),


def mask(vector):
    return Mask()(vector)


class LUParameters(chainer.Chain):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        shape = (channels, channels)
        rotation_mat = np.linalg.qr(
            np.random.normal(size=shape))[0].astype("float32")
        # w_p: a permutation matrix
        # w_l: a lower triangular matrix with ones on the diagonal
        # w_u: an upper triangular matrix with zeros on the diagonal,
        w_p, w_l, w_u = scipy.linalg.lu(rotation_mat)
        s = np.diag(w_u)
        u_mask = np.triu(np.ones_like(w_u), k=1)
        l_mask = np.tril(np.ones_like(w_u), k=-1)
        l_diag = np.eye(w_l.shape[0])
        w_u = w_u * u_mask

        # w_u = np.ascontiguousarray(w_u)
        # w_l = np.ascontiguousarray(w_l)
        # s = np.ascontiguousarray(s)
        # w_p = np.ascontiguousarray(w_p)
        # self.u_mask = np.ascontiguousarray(u_mask)
        # self.l_mask = np.ascontiguousarray(l_mask)
        # l_diag = np.ascontiguousarray(l_diag)

        with self.init_scope():
            self.w_u = chainer.Parameter(initializer=w_u, shape=w_u.shape)
            self.w_l = chainer.Parameter(initializer=w_l, shape=w_l.shape)
            self.s = chainer.Parameter(initializer=s, shape=s.shape)
            self.add_persistent("w_p", w_p)
            self.add_persistent("u_mask", u_mask)
            self.add_persistent("l_mask", l_mask)
            self.add_persistent("l_diag", l_diag)


    @property
    def W(self):
        kernel = self.w_p @ (self.w_l * self.l_mask + self.l_diag) @ (
            self.w_u * self.u_mask + diag(self.s))
        return cf.reshape(kernel, kernel.shape + (1, 1))
