from unittest import TestCase
import torch
import pytorch_lightning as pl
from torch.autograd.functional import hessian
from sklearn.linear_model import LogisticRegression as SklearnLogReg
import unittest
import numpy as np

from pytorch_influence_functions.influence_functions.hvp_grad import (
    calc_loss,
    s_test_sample,
    grad_z,
    s_test_cg,
)

from pytorch_influence_functions.influence_functions.utils import (
    load_weights,
    make_functional,
    tensor_to_tuple,
    parameters_to_vector,
)
from utils.dummy_dataset import (
    DummyDataset,
)

from utils.logistic_regression import (
    LogisticRegression,
)


class TestIHVPGrad(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        pl.seed_everything(0)

        cls.n_features = 10
        cls.n_classes = 3

        cls.n_params = cls.n_classes * cls.n_features + cls.n_features

        cls.model = LogisticRegression(cls.n_classes, cls.n_features)

        gpus = 1 if torch.cuda.is_available() else 0
        
        trainer = pl.Trainer(gpus=gpus, max_epochs=10)
        # trainer.fit(self.model)

        use_sklearn = True
        if use_sklearn:
            train_dataset = DummyDataset(cls.n_features, cls.n_classes)
            multi_class = "multinomial" if cls.model.n_classes != 2 else "auto"
            clf = SklearnLogReg(C=1e4, tol=1e-8, max_iter=1000, multi_class=multi_class)

            clf.fit(train_dataset.data, train_dataset.targets)

            with torch.no_grad():
                cls.model.linear.weight = torch.nn.Parameter(
                    torch.tensor(clf.coef_, dtype=torch.float)
                )
                cls.model.linear.bias = torch.nn.Parameter(
                    torch.tensor(clf.intercept_, dtype=torch.float)
                )

        # Setup test point data
        cls.test_idx = 5
        cls.x_test = torch.tensor(
            cls.model.test_set.data[[cls.test_idx]], dtype=torch.float
        )
        cls.y_test = torch.tensor(
            cls.model.test_set.targets[[cls.test_idx]], dtype=torch.long
        )

        # Compute estimated IVHP
        cls.gpu = 1 if torch.cuda.is_available() else -1

        if cls.gpu == 1:
            cls.model = cls.model.cuda()

        cls.train_loader = cls.model.train_dataloader(batch_size=40000)
        # Compute anc flatten grad
        grads = grad_z(cls.x_test, cls.y_test, cls.model, gpu=cls.gpu)
        flat_grads = parameters_to_vector(grads)

        print("Grads:")
        print(flat_grads)

        # Make model functional
        params, names = make_functional(cls.model)
        # Make params regular Tensors instead of nn.Parameter
        params = tuple(p.detach().requires_grad_() for p in params)
        flat_params = parameters_to_vector(params)

        # Initialize Hessian
        h = torch.zeros([flat_params.shape[0], flat_params.shape[0]])
        if cls.gpu == 1:
            h = h.cuda()

        # Compute real IHVP
        for x_train, y_train in cls.train_loader:

            if cls.gpu >= 0:
                x_train, y_train = x_train.cuda(), y_train.cuda()

            def f(flat_params_):
                split_params = tensor_to_tuple(flat_params_, params)
                load_weights(cls.model, names, split_params)
                out = cls.model(x_train)
                loss = calc_loss(out, y_train)
                return loss

            batch_h = hessian(f, flat_params, strict=True)

            with torch.no_grad():
                h += batch_h / float(len(cls.train_loader))

        print("Hessian:")
        print(h)

        np.save("hessian_pytorch.npy", h.cpu().numpy())

        # Make the model back `nn`

        with torch.no_grad():
            load_weights(cls.model, names, params, as_params=True)
            inv_h = torch.inverse(h)
            print("Inverse Hessian")
            print(inv_h)
            cls.real_ihvp = inv_h @ flat_grads

        print("Real IHVP")
        print(cls.real_ihvp)

    def test_s_test_cg(self):
        estimated_ihvp = s_test_cg(
            self.x_test,
            self.y_test,
            self.model,
            self.train_loader,
            damp=0.0,
            gpu=self.gpu,
        )

        print("CG")
        self.assertTrue(self.check_estimation(estimated_ihvp))

    def test_s_test_sample(self):

        estimated_ihvp = s_test_sample(
            self.model,
            self.x_test,
            self.y_test,
            self.train_loader,
            gpu=self.gpu,
            damp=0.0,
            r=10,
            recursion_depth=10000,
        )

        flat_estimated_ihvp = parameters_to_vector(estimated_ihvp)

        print("LiSSA")
        self.assertTrue(self.check_estimation(flat_estimated_ihvp))

    def check_estimation(self, estimated_ihvp):

        print(estimated_ihvp)
        print("real / estimate")
        print(self.real_ihvp / estimated_ihvp)

        with torch.no_grad():
            l_2_difference = torch.norm(self.real_ihvp - estimated_ihvp)
            l_infty_difference = torch.norm(
                self.real_ihvp - estimated_ihvp, p=float("inf")
            )
        print(f"L-2 difference: {l_2_difference}")
        print(f"L-infty difference: {l_infty_difference}")

        return torch.allclose(self.real_ihvp, estimated_ihvp)


if __name__ == "__main__":
    unittest.main()
