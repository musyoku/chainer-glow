import chainer
import chainer.functions as cf
import chainer.links as L
import numpy as np
import scipy
from chainer.backends import cuda
from chainer.functions.connection import convolution_2d
from chainer.utils import type_check


class Invertible1x1Conv(chainer.Chain):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        with self.init_scope():
            shape = (channels, channels)
            rotation_mat = np.linalg.qr(np.random.normal(
                size=shape))[0].astype("float32").reshape(shape + (1, 1))
            inv_rotation_mat = np.linalg.inv(rotation_mat)
            self.conv = L.Convolution2D(
                channels,
                channels,
                ksize=1,
                stride=1,
                pad=0,
                nobias=True,
                initialW=rotation_mat)

        # W^(-1) is not a learnable parameter
        self.inverse_conv = L.Convolution2D(
            channels,
            channels,
            ksize=1,
            stride=1,
            pad=0,
            nobias=True,
            initialW=inv_rotation_mat)

    def forward_step(self, x):
        log_det = self.compute_log_determinant(x, self.conv.W)

        y = self.conv(x)
        return y, log_det

    def reverse_step(self, x):
        log_det = self.compute_log_determinant(x, self.inverse_conv.W)

        y = self.inverse_conv(x)
        return y, log_det

    def update_inverse_weight(self):
        weight = self.conv.W.data
        xp = cuda.get_array_module(weight)
        # square matrix
        weight = weight.reshape(weight.shape[:2])
        inv_weight = xp.linalg.inv(weight)
        # conv kernel
        self.inverse_conv.W.data = inv_weight.reshape(inv_weight.shape + (1, 1))

    def compute_log_determinant(self, x, W):
        h, w = x.shape[2:]
        W = self.conv.W
        det = cf.det(W)
        if det.data == 0:
            det += 1e-16  # avoid nan
        return h * w * cf.log(abs(det))


class LUInvertible1x1Conv(chainer.Chain):
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

        with self.init_scope():
            self.w_u = chainer.Parameter(initializer=w_u, shape=w_u.shape)
            self.w_l = chainer.Parameter(initializer=w_l, shape=w_l.shape)
            self.s = chainer.Parameter(initializer=s, shape=s.shape)
            self.add_persistent("w_p", w_p)
            self.add_persistent("u_mask", u_mask)
            self.add_persistent("l_mask", l_mask)
            self.add_persistent("l_diag", l_diag)

        # W^(-1) is not a learnable parameter
        inv_rotation_mat = np.linalg.inv(self.W)
        self.inverse_conv = L.Convolution2D(
            channels,
            channels,
            ksize=1,
            stride=1,
            pad=0,
            nobias=True,
            initialW=inv_rotation_mat)

    @property
    def W(self):
        kernel = self.w_p @ (self.w_l * self.l_mask + self.l_diag) @ (
            self.w_u * self.u_mask + diag(self.s))
        return cf.reshape(kernel, kernel.shape + (1, 1))

    def forward_step(self, x):
        y = convolution_2d.convolution_2d(x, self.W, None, 1, 0)

        log_det = self.compute_log_determinant(x)
        return y, log_det

    def reverse_step(self, x):
        log_det = self.compute_log_determinant(x)

        y = self.inverse_conv(x)
        return y, log_det

    def compute_log_determinant(self, x):
        h, w = x.shape[2:]
        return h * w * cf.sum(cf.log(abs(self.s)))

    def update_inverse_weight(self):
        weight = self.W.data
        xp = cuda.get_array_module(weight)
        # square matrix
        weight = weight.reshape(weight.shape[:2])
        inv_weight = xp.linalg.inv(weight)
        # conv kernel
        self.inverse_conv.W.data = inv_weight.reshape(inv_weight.shape + (1, 1))
