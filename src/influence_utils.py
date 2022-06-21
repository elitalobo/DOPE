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
from scipy.optimize import fmin_ncg


"""
Taken from 
https://github.com/ryokamoi/pytorch_influence_functions/blob/master/pytorch_influence_functions/influence_functions/utils.py
"""

def conjugate_gradient(ax_fn, b, debug_callback=None, avextol=None, maxiter=None):
    """Computes the solution to Ax - b = 0 by minimizing the conjugate objective
    f(x) = x^T A x / 2 - b^T x. This does not require evaluating the matrix A
    explicitly, only the matrix vector product Ax.
    From https://github.com/kohpangwei/group-influence-release/blob/master/influence/conjugate.py.
    Args:
      ax_fn: A function that return Ax given x.
      b: The vector b.
      debug_callback: An optional debugging function that reports the current optimization function. Takes two
          parameters: the current solution and a helper function that evaluates the quadratic and linear parts of the
          conjugate objective separately. (Default value = None)
      avextol:  (Default value = None)
      maxiter:  (Default value = None)
    Returns:
      The conjugate optimization solution.
    """

    cg_callback = None
    if debug_callback:
        cg_callback = lambda x: debug_callback(
            x, -np.dot(b, x), 0.5 * np.dot(x, ax_fn(x))
        )

    result = fmin_ncg(
        f=lambda x: 0.5 * np.dot(x, ax_fn(x)) - np.dot(b, x),
        x0=np.zeros_like(b),
        fprime=lambda x: ax_fn(x) - b,
        fhess_p=lambda x, p: ax_fn(p),
        callback=cg_callback,
        avextol=avextol,
        maxiter=maxiter,
    )

    return result



def make_functional(model):
    orig_params = tuple(model.parameters())
    # Remove all the parameters in the model
    names = []

    for name, p in list(model.named_parameters()):
        del_attr(model, name.split("."))
        names.append(name)

    return orig_params, names


def load_weights(model, names, params, as_params=False):
    for name, p in zip(names, params):
        if not as_params:
            set_attr(model, name.split("."), p)
        else:
            set_attr(model, name.split("."), torch.nn.Parameter(p))



def parameters_to_vector(parameters):
    r"""Convert parameters to one vector
    Arguments:
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    Returns:
        The parameters represented by a single vector
    """

    vec = []
    for param in parameters:
        vec.append(param.view(-1))

    return torch.cat(vec)