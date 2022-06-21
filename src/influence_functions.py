import sys

sys.path.append("experiments/")
from argparser_fqe import parse

args = parse()
import torch

torch.manual_seed(args.seed)
import random

random.seed(args.seed)
import numpy as np

np.random.seed(args.seed)

from torch.autograd import grad
from torch.autograd.functional import vhp

from datetime import datetime
from influence_utils import (
    conjugate_gradient
)


def s_test_cg(dataLoader, damp, train_params,  test_params, test_loss_func, train_loss_func, gpu=-1, verbose=True, batch_size=256):
    train_params = train_params.reshape(-1)
    test_params = test_params.reshape(-1)
    v_flat = grad_z(test_params, loss_func=test_loss_func).reshape(-1)


    def hvp_fn(x):

        x_tensor = torch.tensor(x, requires_grad=False).reshape(-1)



        hvp = torch.zeros_like(train_params)

        n = dataLoader.states.shape[0]
        d = dataLoader.states.shape[1]

        num_batches = n/batch_size



        def f(train_p):

            loss = train_loss_func(weights=train_p)
            return loss
        start=datetime.now()
        batch_hvp = vhp(f, train_params, x_tensor, strict=True)[1]
        end = datetime.now(
        )
        print("hvp in cg",(end-start).total_seconds())

        hvp += batch_hvp

        with torch.no_grad():
            damped_hvp = hvp + damp * v_flat

        return damped_hvp

    def print_function_value(_, f_linear, f_quadratic):
        print(
            f"Conjugate function value: {f_linear + f_quadratic}, lin: {f_linear}, quad: {f_quadratic}"
        )

    debug_callback = print_function_value if verbose else None
    start=datetime.now()
    result = conjugate_gradient(
        hvp_fn,
        v_flat.cpu().numpy(),
        debug_callback=debug_callback,
        avextol=1e-8,
        maxiter=100, #100
    )
    end = datetime.now()
    print(" time for cg numoy",(end-start).total_seconds())

    result = torch.tensor(result)
    if gpu >= 0:
        result = result.cuda()

    return result


def s_test(dataLoader, train_params,  test_params, test_loss_func, train_loss_func, gpu=-1, damp=0.01, scale=25.0):
    """s_test can be precomputed for each test point of interest, and then
    multiplied with grad_z to get the desired value for each training point.
    Here, stochastic estimation is used to calculate s_test. s_test is the
    Inverse Hessian Vector Product.
    Arguments:
        x_test: torch tensor, test data points, such as test images
        y_test: torch tensor, contains all test data labels
        model: torch NN, model used to evaluate the dataset
        i: the sample number
        samples_loader: torch DataLoader, can load the training dataset
        gpu: int, GPU id to use if >=0 and -1 means use CPU
        damp: float, dampening factor
        scale: float, scaling factor
    Returns:
        h_estimate: list of torch tensors, s_test"""
    train_params = train_params.reshape(-1)
    test_params = test_params.reshape(-1)
    v = grad_z(test_params, loss_func=test_loss_func).reshape(-1)

    h_estimate = [v]

    # Make params regular Tensors instead of nn.Parameter
    params = train_params.reshape(-1)
    params = params.detach().requires_grad_(True)

    n = dataLoader.states.shape[0]/2
    # TODO: Dynamically set the recursion depth so that iterations stop once h_estimate stabilises
    for i in range(int(n)):


        def f(new_params):
            #start = datetime.now()
            loss= train_loss_func(weights=new_params,single=True)
            #end = datetime.now()
            #print("loss single",(end-start).total_seconds())
            return loss

        #start = datetime.now()
        hv = vhp(f, params, tuple(h_estimate), strict=True)[1]
        #end = datetime.now()
        #print("hvp", (end - start).total_seconds())

        # Recursively calculate h_estimate
        with torch.no_grad():
            h_estimate = [
                _v + (1 - damp) * _h_e - _hv / scale
                for _v, _h_e, _hv in zip(v, h_estimate, hv)
            ]

            if i % 100 == 0:
                norm = sum([h_.norm() for h_ in h_estimate])
                #print("norm",norm)

    return h_estimate



def grad_z(params, gpu=-1, loss_func=None):
    """Calculates the gradient z. One grad_z should be computed for each
    training sample.
    Arguments:
        x: torch tensor, training data points
            e.g. an image sample (batch_size, 3, 256, 256)
        y: torch tensor, training data labels
        model: torch NN, model used to evaluate the dataset
        gpu: int, device id to use for GPU, -1 for CPU
    Returns:
        grad_z: list of torch tensor, containing the gradients
            from model parameters to loss"""

    params = params.detach().requires_grad_(True)
    loss, new_params = loss_func(weights=params)
    #loss = loss_func(weights=params)

    # Compute sum of gradients from model parameters to loss
    return grad(loss, params)[0]


def s_test_sample(
dataLoader, train_params,  test_params, test_loss_func, train_loss_func,
        gpu=-1,
        damp=0.01,
        scale=25,
        recursion_depth=5000,
        r=2,
        loss_func="cross_entropy",
):
    """Calculates s_test for a single test image taking into account the whole
    training dataset. s_test = invHessian * nabla(Loss(test_img, model params))
    Arguments:
        model: pytorch model, for which s_test should be calculated
        x_test: test image
        y_test: test image label
        train_loader: pytorch dataloader, which can load the train data
        gpu: int, device id to use for GPU, -1 for CPU (default)
        damp: float, influence function damping factor
        scale: float, influence calculation scaling factor
        recursion_depth: int, number of recursions to perform during s_test
            calculation, increases accuracy. r*recursion_depth should equal the
            training dataset size.
        r: int, number of iterations of which to take the avg.
            of the h_estimate calculation; r*recursion_depth should equal the
            training dataset size.
    Returns:
        s_test_vec: torch tensor, contains s_test for a single test image"""
    params = train_params.flatten()
    test_params = test_params.requires_grad_(True)
    inverse_hvp = [
        torch.zeros_like(params, dtype=torch.double)
    ]

    n = dataLoader.states.shape[0]/4
    indices = np.arange(n)
    np.random.shuffle(indices)

    start1 = datetime.now()
    for i in range(r):

        start = datetime.now()
        cur_estimate = s_test(
            dataLoader, train_params, test_params, test_loss_func, train_loss_func, gpu=gpu, damp=damp, scale=scale
        )
        end = datetime.now()
        print("time for s_estimate",(end-start).total_seconds())

        with torch.no_grad():
            inverse_hvp = [
                old + (cur / scale) for old, cur in zip(inverse_hvp, cur_estimate)
            ]

    end1 = datetime.now()
    print("total time for s_estimate", (end1 - start1).total_seconds())

    with torch.no_grad():
        inverse_hvp = [component / r for component in inverse_hvp]

    return inverse_hvp