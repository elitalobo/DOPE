from unittest import TestCase
import torch
import pytorch_lightning as pl
from torch.autograd.functional import hessian
from sklearn.linear_model import LogisticRegression as SklearnLogReg
import unittest
import numpy as np

from pytorch_influence_functions.influence_functions.utils import (
    conjugate_gradient,
    load_weights,
    make_functional,
    tensor_to_tuple,
)


class TestIHVPGrad(TestCase):
    def test_cg(self):
        size = 5

        A_half = np.random.random_sample((size, size))
        A = A_half + A_half.T + np.eye(size) * 5

        A_inv = np.linalg.inv(A)
        x = np.ones((size,))

        A_inv_x_cg_real = A_inv.dot(x)
        print("Real A inv x: ", A_inv_x_cg_real)

        def ax_fn(x):
            return A.dot(x)
        
        def print_function_value(_, f_linear, f_quadratic):
            print(
                f"Conjugate function value: {f_linear + f_quadratic}, lin: {f_linear}, quad: {f_quadratic}"
            )

        A_inv_x_cg = conjugate_gradient(
            ax_fn,
            x,
            debug_callback=print_function_value,
            avextol=1e-8,
            maxiter=100,
        )

        print("CG A inv x: ", A_inv_x_cg)

        self.assertTrue(np.allclose(A_inv_x_cg_real, A_inv_x_cg))


if __name__ == "__main__":
    unittest.main()
