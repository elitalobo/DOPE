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

sys.path.append("../")
from utils import *
from projections import *
from influence_functions import *
from influence_utils import *
import timeit
from datetime import datetime


def get_hessian_inverse_vector_product(v,hessian):
    Q, R = np.linalg.qr(hessian)
    Q_inv = Q.transpose()
    R_inv = np.linalg.inv(R)
    temp = np.matmul(v.reshape(1,-1),R_inv)
    hv = np.matmul(temp.reshape(1,-1),Q_inv)
    return hv


class Influence():
    def __init__(self,model=None,eps=0.05,epsilon=5,iters=100,lr=0.01,type="l2"):
        """
        Carries out influence based data poisoning attack on the model provided as input
        :param model: Model object whose data needs to be poisoned
        :param eps: percentage of corrupt points
        :param epsilon: magnitude of data poisoning (depends on the norm used for projecting the corrupt data points)
        :param iters: maximum no of iterations
        :param lr: learning rate
        """
        self.attack_type="influence"
        self.model = model
        self.eps = eps
        self.epsilon = epsilon
        self.iters = iters
        self.lr = lr
        self.type = type





    def compute_influence(self,train_error,weights_train,xtrain,idx,gtest_hess):
        """
         Computes influence of train data points on test error.
        :param train_error:
        :param test_error:
        :param weights_train: weights of the model used for calculating train error (assumes that the weights are linear)
        :param weights_test: weights of the model used for calculating test error (assumes that the weights are linear)
        :param xtrain:
        :param idx:
        :return:
        weights_train and weights_test can represent the same weights.
        """



        #print(np.mean(np.abs(hessian_val_inv.detach().cpu().numpy())))
        """ Computes mixed derivative of train error, first with respect the train dataset and second with respect to weights train"""
        mixed_derivative_val = mixed_derivative(train_error,xtrain,weights_train,idx)

        #assert(torch.sum(torch.abs(mixed_derivative_val))!=0)
        mixed_derivative_val = mixed_derivative_val.detach().cpu().numpy()


        #assert(torch.sum(mixed_derivative_val)!=0)
        """ Computes product of hessian and mixed derivative of train error """

        #assert(torch.sum(gtest)!=0)

        influence = -1.0* np.matmul(gtest_hess.reshape(1,-1),mixed_derivative_val)
        #d_w_d_x = np.matmul(hessian_val_inv,mixed_derivative_val)
        """ Computes influence as  -dtest_error/dparam * (d^2(train_error)/(dparam^2))^{-1}* dtrain_error/(dparam dxtrain) """
        #influence = -torch.mm(gtest.reshape(1,-1).double(),d_w_d_x.double())


        #jacobian = torch.autograd.grad(train_error,weights_train,create_graph=True)[0]
        #elem_prod = torch.matmul(gtest_hess.detach().reshape(1,-1),jacobian.reshape(-1,1))

        #influence = -1.0* torch.autograd.grad(jacobian[0],xtrain[idx,:],retain_graph=True)[0]

        # influence = influence.detach()
        #


        """ return influence (state_dim) """
        return influence.flatten(), gtest_hess


    def compute_all_influences(self,weights, indices,train_func,test_func,dataloader,test=True,hessian=None):
        """

        :param test_error:
        :param train_error:
        :param param:
        :param xtrain_arr:
        :return:
        """

        def test_f(weights=None):

            loss, w, __ = test_func(weights)
            return loss

        def train_f(weights, single=False, grad=False):
            loss, __, ___, _ = train_func(weights=weights, grad=grad, single=single)
            return loss

        if isinstance(weights, np.ndarray):
            weights = torch.tensor(weights, requires_grad=True)

        start = datetime.now()
        """ Compute jacobian of test error with respect to the weights_test """
        gtest = torch.autograd.functional.jacobian(test_f, weights.detach().flatten().requires_grad_(True))
        print("gtest",torch.sum(gtest))
        if hessian is None:
            hessian_val = torch.autograd.functional.hessian(train_f, weights.detach().flatten())
        else:
            print("got hessian")
            hessian_val = hessian

        print("condition",np.linalg.cond(hessian_val.detach().cpu().numpy()))

        print("min eig",np.min(np.linalg.eigvals(hessian_val.detach().cpu().numpy())))
        print("hessian val", np.sum(np.abs(hessian_val.detach().cpu().numpy())))

        end = datetime.now()
        print("Time taken to compute hessian", (start - end).total_seconds())

        # print("rank", np.linalg.matrix_rank(hessian_val.detach().cpu().numpy()))
        """ Computes inverse of hessian of train error """

        hessian_val = hessian_val.detach().cpu().numpy()
        gtest = gtest.detach().cpu().numpy()

        gtest_hess = get_hessian_inverse_vector_product(gtest, hessian_val)
        gtest_hess = torch.tensor(gtest_hess.astype(np.double))
        end = datetime.now()
        print("time taken via backpropagation", (end-start).total_seconds())




        start = datetime.now()

        train_error, xtrain, w_train, l = train_func(weights=weights.detach().requires_grad_(True), grad=True, single=False,indices=indices)
        jac = torch.autograd.grad(train_error,w_train, create_graph=True,retain_graph=True)[0]
        elem_prod = torch.matmul(gtest_hess.reshape(1, -1), jac.reshape(-1, 1))

        influences = -1.0 * torch.autograd.grad(elem_prod, xtrain)[0].detach().cpu().numpy()
        print("influences", np.sum(influences))

        if test==True:
            test_error, w_test, xtest = test_func(weights=weights.detach().requires_grad_(False), test=test)

            test_grad = torch.autograd.grad(test_error,xtest,retain_graph=True)[0].detach().cpu().numpy()
            if test_grad.shape[0] != influences.shape[0]:
                test_grad = test_grad[indices.flatten(),:]
            influences = influences + test_grad

            del test_grad
        del train_error, xtrain, w_train


        influences = np.array(influences)

        end = datetime.now()
        print("Time taken to get influences",(start-end).total_seconds())


        return np.array(influences)







