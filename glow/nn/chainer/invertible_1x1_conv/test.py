import unittest

import numpy

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr

from parameters import Diag, diag


@testing.parameterize(
    {
        'n': 10
    },
    {'n': 20},
    {'n': 30},
    {'n': 40},
    {'n': 50},
)
class DiagTest(unittest.TestCase):
    def setUp(self):
        self.vector = numpy.random.uniform(0.1, 1,
                                           (self.n, )).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1,
                                       (self.n, self.n)).astype(numpy.float32)

    def check_forward(self, vector_data):
        vector = chainer.Variable(vector_data)
        y = diag(vector)
        correct_y = vector.xp.diag(vector_data)
        gradient_check.assert_allclose(correct_y, cuda.to_cpu(y.data))

    def test_forward_cpu(self):
        self.check_forward(self.vector)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.vector))

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
            Diag(), x_data, y_grad, eps=1e-2, rtol=1e-2)

    def test_backward_cpu(self):
        self.check_backward((self.vector, ), self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward((cuda.to_gpu(self.vector), ), cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)