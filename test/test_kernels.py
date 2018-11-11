# File: test_kernels.py
# File Created: Sunday, 11th November 2018 1:31:17 pm
# Author: Steven Atkinson (steven@atkinson.mn)
#
# Tests:
# K(x)
# K(x, x2)
# Transpose
# Shift (Stationary only)

from unittest import TestCase
import pytest
import numpy as np
from torch.autograd import Variable

from gptorch import kernels
from gptorch.util import TensorType


def as_variable(x: np.array) -> Variable:
    return Variable(TensorType(x))


class Kern(object):
    def setUp(self, kernel_type):
        self.kernel_type = kernel_type
        self.x1 = as_variable(np.load("data/kernels/x1.npy"))
        self.x2 = as_variable(np.load("data/kernels/x2.npy"))
        self.n1, self.d1 = self.x1.data.numpy().shape
        self.n2, self.d2 = self.x2.data.numpy().shape
        self.kern = self.kernel_type(self.d1)
        self.kern_str = self.kern.__class__.__name__
        self.kx_expected = np.load("data/kernels/{}_kx.npy".format(
            self.kern_str))
        self.kx2_expected = np.load("data/kernels/{}_kx2.npy".format(
            self.kern_str))
        self.kdiag_expected = np.load("data/kernels/{}_kdiag.npy".format(
            self.kern_str))

    def test_K(self):
        """
        Inputs are expected to be torch.autograd.Variables of 
        torch.DoubleTensors

        Output is expected to be a torch.autograd.Variable
        """
        kx_actual = self.kern.K(self.x1).data.numpy()
        kx2_actual = self.kern.K(self.x1, self.x2).data.numpy()
        kx2t_actual = self.kern.K(self.x2, self.x1).data.numpy()
        assert np.allclose(self.kx_expected, kx_actual)
        assert np.allclose(self.kx2_expected, kx2_actual)

        # Test symmetric K()
        assert np.allclose(kx_actual.T, kx_actual)

        # Test transpose of cross-kernel:
        assert np.allclose(self.kx2_expected, kx2t_actual.T)

    def test_Kdiag(self):
        kdiag_actual = self.kern.Kdiag(self.x1).data.numpy()
        assert np.allclose(self.kdiag_expected, kdiag_actual)


class Stationary(Kern):
    def setUp(self, kernel_type):
        super().setUp(kernel_type)
        x_shift = 0.34
        self.x1_shift = self.x1 + x_shift
        self.x2_shift = self.x1 + x_shift

    def test_K(self):
        super().test_K()
        kx_shift_actual = self.kern.K(self.x1_shift).data.numpy()
        assert np.allclose(self.kx_expected, kx_shift_actual)

    def test_Kdiag(self):
        super().test_Kdiag()
        kxdiag_shift_actual = self.kern.Kdiag(self.x1_shift).data.numpy()
        assert np.allclose(self.kdiag_expected, kxdiag_shift_actual)


class ARD(Stationary):
    def setUp(self, kernel_type):
        super().setUp(kernel_type)
        self.ard_length_scales = np.load("data/kernels/ard_length_scales.npy")
        self.kern_ard = self.kernel_type(self.d1, ARD=True, 
            length_scales=self.ard_length_scales)
        self.kx_ard_expected = np.load("data/kernels/{}_kx_ard.npy".format(
            self.kern_str))
        self.kx2_ard_expected = np.load("data/kernels/{}_kx2_ard.npy".format(
            self.kern_str))
        self.kdiag_ard_expected = np.load(
            "data/kernels/{}_kdiag_ard.npy".format(self.kern_str))

    def test_K(self):
        super().test_K()
        kx_ard_actual = self.kern_ard.K(self.x1).data.numpy()
        kx2_ard_actual = self.kern_ard.K(self.x1, self.x2).data.numpy()
        assert np.allclose(self.kx_ard_expected, kx_ard_actual)
        assert np.allclose(self.kx2_ard_expected, kx2_ard_actual)

    def test_Kdiag(self):
        super().test_Kdiag()
        kdiag_ard_actual = self.kern_ard.Kdiag(self.x1).data.numpy()
        assert np.allclose(self.kdiag_ard_expected, kdiag_ard_actual)


class TestWhite(Kern, TestCase):
    def setUp(self):
        super().setUp(kernels.White)    


class TestConstant(Kern, TestCase):
    def setUp(self):
        super().setUp(kernels.Constant) 


class TestBias(Kern, TestCase):
    def setUp(self):
        super().setUp(kernels.Bias)


class TestExp(ARD, TestCase):
    def setUp(self):
        super().setUp(kernels.Exp)


class TestMatern12(ARD, TestCase):
    def setUp(self):
        super().setUp(kernels.Matern12)


class TestMatern32(ARD, TestCase):
    def setUp(self):
        super().setUp(kernels.Matern32)


class TestMatern52(ARD, TestCase):
    def setUp(self):
        super().setUp(kernels.Matern52)


class TestRbf(ARD, TestCase):
    def setUp(self):
        super().setUp(kernels.Rbf)


@pytest.mark.xfail(reason="Linear is different because we use .variance " + 
    "instead of .length_scales (TODO)")
class TestLinear(ARD, TestCase):
    def setUp(self):
        super().setUp(kernels.Linear)
