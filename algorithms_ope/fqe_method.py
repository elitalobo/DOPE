import os
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
sys.path.append("src/")
from influence import *
from influence_utils import *
from influence_functions import *
import copy
from dataloader import *
from utils_nn import *
sys.path.append("../domains_for_bilevel/")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
sys.path.append("../domains_for_bilevel/")


from utils import *

import torch.nn as nn


class FQE_method(nn.Module):
    def __init__(self, dataLoader,env, config,reg=1e-2):
        """
        :param dataLoader:
        :param env:
        :param config:
        :param reg:
        """
        super(FQE_method, self).__init__()
        # set regularization parameter
        self.lamda = reg
        # set environment
        self.env = env
        # set config
        self.config = config
        # set dataloader if given
        self.dataloader = dataLoader
        # reset weights of q-value function
        self.reset_weights()
        # set discount factor
        self.gamma = config.gamma



    def get_weights(self, s_features=None):
        """
        Computes weights using Mean Squared Bellman error
        :param s_features: state feature matrix
        :return:
        """
        # get weights by minimizing squared bellman error
        weights = self.solvedFittedQIteration(s_features=s_features)

        return weights


    def reset_weights(self):
        """
            Computes weights of q-value function of policy pi from train data
            :return: weights  (state_dim |a|,)
        """
        # get weights
        weights = self.solvedFittedQIteration()
        weights = torch.tensor(weights)
        # reset tensor weights
        self.weights = torch.nn.Parameter(weights)

    def compute_train_loss(self,indices):
        """
        :param indices:
        :return:
        """
        # get train error
        train_error, states, w, v_error =  self.get_train_error(s_features=None, grad=True,indices=indices)
        return train_error

    def compute_test_loss(self):
        """
        Computes test error
        :return:
        """
        # get test error
        total_error,_, __ = self.get_test_error()
        return total_error


    def compute_weights(self):
        # derive weights by minimizing mean squared bellman error
        self.weights = self.solvedFittedQIteration()
        self.weights = torch.nn.Parameter(self.weights, requires_grad=True)
        self.weights = torch.tensor(self.weights)



    def solvedFittedQIteration(self,s_features=None, actions=None, nsa_features=None, rewards=None):
        """
        https://github.com/elitalobo/OPE-tools/blob/master/ope/algos/fqe.py
        :param max_epochs:
        :param epsilon:
        :return:
        """

        # construct state action features
        if s_features is None:
            # state features
            s_features = self.dataloader.states
            #get actions for states
        if actions is None:
            actions = self.dataloader.actions

        # construct sa features from states and actions
        sa_features = self.dataloader.construct_state_action_features(s_features, actions)
        # get next state action features
        if nsa_features is None:
            nsa_features = self.dataloader.next_state_action_features

        # get rewards
        if rewards is None:
            rewards = self.dataloader.rewards

        # if next state features is not a numpy array, convert to numpy array
        if isinstance(nsa_features, np.ndarray) == False:
            nsa_features = nsa_features.detach().cpu().numpy()
        # if state features is not a numpy array, convert to numpy array
        if isinstance(sa_features, np.ndarray) ==False:
            sa_features = sa_features.detach().cpu().numpy()
        # if rewards is not a numpy array, convert to numpy array
        if isinstance(rewards, np.ndarray)==False:
            rewards = rewards.detach().cpu().numpy()

        # Solve ( (\phi(s,a) - \gamma \phi_p(s,a))^T (\phi(s,a) - \gamma \phi_p(s,a)) + \lambda I ) w = (\phi(s,a) - \gamma \phi_p(s,a))^t r
        # set temp = \Phi(s,a) - \gamma Phi_p(s,a)
        temp = sa_features - self.dataloader.gamma * nsa_features
        A =  np.matmul(temp.transpose(),temp) + self.lamda * np.eye(sa_features.shape[1])
        b =  np.matmul(temp.transpose(),rewards.reshape(-1,1))



        # solve system of equations
        weights = np.linalg.solve(A,b)
        print(np.allclose(np.dot(A, weights), b))
        print("weights",np.sum(np.abs(weights)))
        # print("weights",weights)
        print("error",np.sum(np.abs(np.matmul(A,weights)-b)))

        actual_error = np.sum(np.square(np.matmul(np.matmul(sa_features.transpose(), (sa_features - self.dataloader.gamma * nsa_features)),weights)-b+1e-10))
        # return weights
        return weights


    def delete_and_recompute_initial_returns(self, indices):
        """
        Deletes influential data points and computes initial returns
        :param indices: Takes list of influential indices as parameters
        :return: initial returns after deleting influential data points
        """
        # get total no of data points
        n = self.dataloader.states.shape[0]
        # get indices of all n data points
        all_indices = np.arange(n).tolist()
        # get list of indices without influential indices
        final_indices = delete_multiple_element(all_indices, indices)
        # get actions for remaining indices
        actions = self.dataloader.actions[final_indices]
        # get rewards for remaining indices
        rewards = self.dataloader.rewards[final_indices]
        # get next state features for remaining indices
        nsa_features = self.dataloader.next_state_action_features[final_indices,:]
        # get state features for remaining indices
        s_features = self.dataloader.states[final_indices,:]
        # get weights with new set of states, actions, rewards and next_states
        weights = self.solvedFittedQIteration(s_features=s_features,actions=actions,rewards=rewards,nsa_features=nsa_features)
        # recompute initial returns with new weights
        rets = self.get_initial_return(weights=weights)
        return rets


    def get_initial_return(self, weights=None, policy_eval=None):

        """
        Evaluates initial returns
        :param weights: weights of the q-value function of policy policy_eval
        :param policy_weights:
        :return:
        """
        # if weights is not provided, get previously saved weights
        if weights is None:
            weights = self.weights.detach().cpu().numpy()

        if isinstance(weights,np.ndarray) == False:
            weights = weights.detach().cpu().numpy()

        # Compute state-action features for initial_state
        isa_features = self.dataloader.init_state_action_features
        # compute returns = init_features x weights
        if isinstance(isa_features,np.ndarray)==False:
            isa_features = isa_features.detach().cpu().numpy()
       #  reteurns is the matrix product of initial state features  and weights
        returns = np.matmul(isa_features,weights.reshape(-1,1))
        # assuming uniform distribution over initial states, compute mean of returns
        returns = np.mean(returns)
        # return rets
        return returns

    def get_train_error(self,s_features=None, grad=True, indices=None, weights=None):
        """
        Computes train error using weights of q-value function of evaluation policy
        :param weights: weights of q-value function (state_dim x 1)
        :return:
        """
        # if weights are not given, load previously saved weights
        if weights is None:
            weights = self.weights
        if isinstance(weights, np.ndarray)==True:
            weights = torch.tensor(weights)
        # weights is dx1 matrix
        weights = weights.reshape((-1,1))


        if indices is not None:
            n = self.dataloader.states.shape[0]
            # get states and  transformed states for indices
            states = self.dataloader.get_state_features_indices(requires_grad=grad, indices=indices)

            # get next-state action featyres
            nsa_features = self.dataloader.next_state_action_features[indices.flatten(), :]
            # get state-action features
            sa_features = self.dataloader.construct_state_action_features(states, self.dataloader.actions)

            # get rewards for indcies
            rewards = self.dataloader.rewards[indices.flatten()]

        else:
            # get all next-state action features
            nsa_features = self.dataloader.next_state_action_features

            # get all rewards
            rewards = self.dataloader.rewards

            # # get original next_states and transformed next_state features
            if s_features is None and grad == True:
                # get all states and transformed states
                states= self.dataloader.get_state_features(requires_grad=grad)
                # construct state-action features
                sa_features = self.dataloader.construct_state_action_features(states, self.dataloader.actions)

            else:
                # No need to transform features here
                if s_features is None:
                    states = self.dataloader.states
                else:
                    states = s_features
                if isinstance(s_features, np.ndarray) == True:
                    # Assuming we got transformed features
                    states = torch.tensor(s_features, requires_grad=grad)
                # construct state_action features
                sa_features = self.dataloader.construct_state_action_features(states, self.dataloader.actions)

        w = weights.reshape(-1, 1)


        # 1-step bellman error
        # Set target_t = r _ \gamma \Phi_p w
        target_t = rewards.detach().flatten() + torch.tensor(self.gamma) * torch.mm(nsa_features,
                                                                                    w).flatten()
        # Q = phi^w
        prediction_t = torch.mm(sa_features, w).flatten()

        # v_error = Phi w - r - \gamma Phi_p w
        bellman_error = (prediction_t - target_t).reshape(-1, 1)

        # print("bellman_error", torch.mean(torch.abs(bellman_error)))
        # v_error = ||Phi^T (Phi w   - \gamma Phi_p w)||_2
        v_error = torch.sum(torch.square(torch.mm(sa_features.t(), bellman_error)))
        # Compute mean of v_error and add l2 regularization


        n = self.dataloader.states.shape[0]
        n1 = nsa_features.shape[0]
        # train error = belman error + l2 regularization
        train_error = torch.mean(torch.square(bellman_error)) +self.lamda * torch.sum(w ** 2) * (n1/n)
        # print("lambda value", self.lamda)
        # train_error = v_error + self.lamda * torch.sum(w ** 2)

        return train_error, states, w, v_error

    def get_test_error(self, weights=None,states=None):
        """
        Computes error on test data
        :param weights: weights of q-value function . If weights=None, self.weights is used
        :return: test error, weights
        """
        if weights is None:
            weights = self.weights
        else:

            if isinstance(weights, np.ndarray)==True:
                weights = torch.tensor(weights)

        # Get features for initial state
        init_features = self.dataloader.init_state_action_features

        # Compute returns   = Phi w
        if isinstance(init_features, np.ndarray):
            init_features = torch.tensor(init_features).double()
        # error = sign * init_features * weights
        returns = torch.mm(init_features, weights.reshape(-1, 1))
        # sign =-1 if we want to decrease the value function and +1 if we want to increase the value function
        error = self.config.sign * torch.mean(returns)
        total_error = error
        print("actual_test_error", error)
        # return error and weights
        return total_error, weights, None


