# File: test_kernels.py
# File Created: Sunday, 11th November 2018 1:31:17 pm
# Author: Steven Atkinson (steven@atkinson.mn)
#
# Tests:
# K(x)
# K(x, x2)
# Transpose
# Shift (Stationary only)

import os

import pytest
import numpy as np

from gptorch import kernels
from gptorch.util import TensorType

data_dir = os.path.join(os.path.dirname(__file__), "data", "kernels")


class Kern(object):
    @classmethod
    def setup_class(cls, kernel_type):
        cls.kernel_type = kernel_type
        cls.x1 = TensorType(np.load(os.path.join(data_dir, "x1.npy")))
        cls.x2 = TensorType(np.load(os.path.join(data_dir, "x2.npy")))
        cls.n1, cls.d1 = cls.x1.data.numpy().shape
        cls.n2, cls.d2 = cls.x2.data.numpy().shape
        cls.kern = cls.kernel_type(cls.d1)
        cls.kern_str = cls.kern.__class__.__name__
        cls.kx_expected = np.load(os.path.join(data_dir, "{}_kx.npy".format(
            cls.kern_str)))
        cls.kx2_expected = np.load(os.path.join(data_dir, "{}_kx2.npy".format(
            cls.kern_str)))
        cls.kdiag_expected = np.load(os.path.join(data_dir, 
            "{}_kdiag.npy".format(cls.kern_str)))

    def test_add(self):
        """
        Test kernel addition operator
        """
        k1 = self.kern + self.kern
        k2 = kernels.Sum(self.kern, self.kern)
        k1x = k1.K(self.x1)
        k2x = k2.K(self.x1)
        assert all([x1 == x2 for x1, x2 in zip(k1x.flatten(), k2x.flatten())])

    def test_mul(self):
        """
        Test kernel product operator
        """
        k1 = self.kern * self.kern
        k2 = kernels.Product(self.kern, self.kern)
        k1x = k1.K(self.x1)
        k2x = k2.K(self.x1)
        assert all([x1 == x2 for x1, x2 in zip(k1x.flatten(), k2x.flatten())])

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
    @classmethod
    def setup_class(cls, kernel_type):
        super(Stationary, cls).setup_class(kernel_type)
        x_shift = 0.34
        cls.x1_shift = cls.x1 + x_shift
        cls.x2_shift = cls.x1 + x_shift

    def test_K(self):
        super().test_K()
        kx_shift_actual = self.kern.K(self.x1_shift).data.numpy()
        assert np.allclose(self.kx_expected, kx_shift_actual)

    def test_Kdiag(self):
        super().test_Kdiag()
        kxdiag_shift_actual = self.kern.Kdiag(self.x1_shift).data.numpy()
        assert np.allclose(self.kdiag_expected, kxdiag_shift_actual)


class ARD(Stationary):
    @classmethod
    def setup_class(cls, kernel_type):
        super(ARD, cls).setup_class(kernel_type)
        cls.ard_length_scales = np.load(os.path.join(data_dir, 
            "ard_length_scales.npy"))
        cls.kern_ard = cls.kernel_type(cls.d1, ARD=True, 
            length_scales=cls.ard_length_scales)
        cls.kx_ard_expected = np.load(os.path.join(data_dir, 
            "{}_kx_ard.npy".format(cls.kern_str)))
        cls.kx2_ard_expected = np.load(os.path.join(data_dir, 
            "{}_kx2_ard.npy".format(cls.kern_str)))
        cls.kdiag_ard_expected = np.load(os.path.join(data_dir,
            "{}_kdiag_ard.npy".format(cls.kern_str)))

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


class TestWhite(Kern):
    @classmethod
    def setup_class(cls):
        super(TestWhite, cls).setup_class(kernels.White)    


class TestConstant(Kern):
    @classmethod
    def setup_class(cls):
        super(TestConstant, cls).setup_class(kernels.Constant) 


class TestBias(Kern):
    @classmethod
    def setup_class(cls):
        super(TestBias, cls).setup_class(kernels.Bias)


class TestExp(ARD):
    @classmethod
    def setup_class(cls):
        super(TestExp, cls).setup_class(kernels.Exp)


class TestMatern12(ARD):
    @classmethod
    def setup_class(cls):
        super(TestMatern12, cls).setup_class(kernels.Matern12)


class TestMatern32(ARD):
    @classmethod
    def setup_class(cls):
        super().setup_class(kernels.Matern32)


class TestMatern52(ARD):
    @classmethod
    def setup_class(cls):
        super().setup_class(kernels.Matern52)


class TestRbf(ARD):
    @classmethod
    def setup_class(cls):
        super().setup_class(kernels.Rbf)


class TestPeriodic(ARD):
    @classmethod
    def setup_class(cls):
        super().setup_class(kernels.Periodic)


class TestLinear(Kern):
    @classmethod
    def setup_class(cls):
        super().setup_class(kernels.Linear)
