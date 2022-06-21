
import os

from datetime import datetime
import sys
sys.path.append("experiments/")
from argparser_fqe import parse

args = parse()
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression

torch.manual_seed(args.seed)
import random

random.seed(args.seed)
import numpy as np

np.random.seed(args.seed)
sys.path.append("../")

sys.path.append("src/")
from influence import *
from feature_expansion import *
from influence_utils import *
from influence_functions import *
import copy
from dataloader import *

sys.path.append("../domains_for_bilevel/")

from sklearn.preprocessing import PolynomialFeatures

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
print(sys.path)
# sys.path.append("/Users/elitalobo/PycharmProjects/untitled/soft-robust/interpretable_ope_public/")
sys.path.append("../domains_for_bilevel/")
# import Cancer


from utils import *

import torch.nn as nn


class WDR_method(nn.Module):
    def __init__(self, dataloader,env, config,is_type="is", reg=1e-3,max_epochs=5000,clipped=0.05):
        super(WDR_method,self).__init__()
        self.lamda = reg
        self.env = env
        self.is_type = is_type
        self.config = config
        self.sign = config.sign
        self.max_epochs = max_epochs
        self.clipped = clipped

        self.dataloader = dataloader

        self.gamma = config.gamma
        self.is_operator=None
        if self.is_type=="dr":

            self.is_operator = self.get_consistent_wdr_returns

        elif self.is_type=="wdr":
            print("here wdr")
            self.is_operator = self.get_consistent_weight_dr_returns
        else:
            print("invalid type")
            exit(1)

        # evaluation policy
        self.policy_eval = self.dataloader.policy_eval

        # Estimate initial behavior policy weights
        self.weights = self.estimate_behavior_policy()
        print("weights sum", np.sum(np.abs(self.weights)))
        self.weights = torch.tensor(self.weights)
        # self.weights = torch.nn.Parameter(self.weights)
        # Construct behavior policy
        self.policy_beh =  Policy(self.weights, num_actions=self.config.action_size)

        self.is_weights = None

        self.influence_indices=None
        self.reset_weights()

        # self.weights_q = self.solvedFittedQIteration()
        # self.weights_q = torch.tensor(self.weights_q)
        # self.weights_q = torch.nn.Parameter(self.weights_q, requires_grad=True)


    def compute_train_loss(self,indices):
        """
        computes train loss for given indices
        :param indices: indices of train dataset for which we need to compute train error
        :return:
        """
        train_error, states, w, v_error =  self.get_train_error(s_features=None, grad=True, indices=indices)
        return train_error

    def compute_test_loss(self):
        """
        Computes test loss
        :return:
        """
        total_error,_, __ = self.get_test_error()
        return total_error


    def estimate_behavior_policy(self, states=None, actions=None):
        """
        https://github.com/elitalobo/OPE-tools/blob/master/ope/algos/fqe.py
        :param max_epochs:
        :param epsilon:
        :return:
        """
        # get state features
        if states is None:
            states = self.dataloader.actual_states
        # get action features
        if actions is None:
            actions = self.dataloader.actions

        test_states = self.dataloader.actual_states[:self.config.num_samples, :]
        test_actions = self.dataloader.actions.flatten()[:self.config.num_samples]
        # learn logistic regression model
        clf = LogisticRegression(fit_intercept=False, max_iter=self.max_epochs, multi_class='multinomial').fit(states,
                                                                                                               actions.flatten())
        print("Accuracy", clf.score(test_states, test_actions))

        return clf.coef_.flatten()

    def reset(self):
        # self.weights = self.run_linear_value_iter(self.policy_eval)
        self.weights_q = self.solvedFittedQIteration()
        print("weights sum", np.sum(np.abs(self.weights)))

    def solvedFittedQIteration(self,s_features=None, actions=None, nsa_features=None, rewards=None):
        """
        https://github.com/elitalobo/OPE-tools/blob/master/ope/algos/fqe.py
        :param max_epochs:
        :param epsilon:
        :return:
        """

        # construct state action features
        if s_features is None:
            # get next states
            s_features = self.dataloader.states
            #get actions for next_states
        if actions is None:
            actions = self.dataloader.actions

        sa_features = self.dataloader.construct_state_action_features(s_features, actions)
        # construct next-state-action features
        # get actions for states
        if nsa_features is None:
            nsa_features = self.dataloader.next_state_action_features

        # get rewards
        if rewards is None:
            rewards = self.dataloader.rewards


        if isinstance(nsa_features, np.ndarray) == False:
            nsa_features = nsa_features.detach().cpu().numpy()
        if isinstance(sa_features, np.ndarray) ==False:
            sa_features = sa_features.detach().cpu().numpy()
        if isinstance(rewards, np.ndarray)==False:
            rewards = rewards.detach().cpu().numpy()


        # (phi^T r)
        # b = np.matmul(sa_features.transpose(),rewards.reshape(-1,1))
        # (phi^T phi - r phi^T phi_p) + lambda * I
        # A = np.matmul(sa_features.transpose(), sa_features - self.dataloader.gamma * nsa_features) + self.lamda * np.eye(sa_features.shape[1])
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

        return weights




    def get_weights(self, s_features=None):
        """
            Computes weights of q-value function of policy pi from train data
            :return: weights  (state_dim |a|,)
        """
        # weights = self.run_linear_value_iter(pi, sa_features=sa_features, nsa_features=nsa_features)
        weights = self.solvedFittedQIteration(s_features=s_features)

        return weights

    def reset_weights(self):
        """
            Computes weights of q-value function of policy pi from train data
            :return: weights  (state_dim |a|,)
        """
        #weights = self.run_linear_value_iter(pi, sa_features=sa_features, nsa_features=nsa_features)
        # get q-value weights
        weights_q = self.solvedFittedQIteration()
        weights_q = torch.tensor(weights_q)
        # get tensor weights
        self.weights_q = torch.nn.Parameter(weights_q, requires_grad=True)

    def get_initial_return(self, weights=None, policy_eval=None):
        """
        Evaluates initial returns
        :param weights: weights of the q-value function of policy policy_eval
        :param policy_weights:
        :return:
        """
        # get test error
        returns, weights, states = self.get_test_error(weights=weights)
        returns = self.sign * returns
        return returns.detach().cpu().numpy()

    def get_train_error(self, weights=None, s_features=None, grad=True, indices=None):
        """
        Computes train error using weights of q-value function of evaluation policy
        :param weights: weights of q-value function (state_dim x 1)
        :return:
        """
        # get state-action features
        if indices is not None:
            n = self.dataloader.states.shape[0]
            # get states and  transformed states for indices
            states = self.dataloader.get_state_features_indices(requires_grad=grad, indices=indices)

            states = states
            # get next-state action featyres
            nsa_features = self.dataloader.next_state_action_features[indices.flatten(), :]
            # get state-action features
            sa_features = self.dataloader.construct_state_action_features(states,
                                                                          self.dataloader.actions[indices.flatten()])
            # get rewards for indcies
            rewards = self.dataloader.rewards[indices.flatten()].flatten()

        else:
            # get all next-state action features
            nsa_features = self.dataloader.next_state_action_features
            # get rewards
            # get all rewards
            rewards = self.dataloader.rewards

            # # get original next_states and transformed next_state features
            if s_features is None and grad == True:
                # get all states and transformed states
                states = self.dataloader.get_state_features(requires_grad=grad)
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

        if weights is None:
            w = self.weights_q.reshape(-1, 1)
        else:
            w= weights
            if isinstance(w, np.ndarray):
                w = torch.tensor(w,requires_grad=True)

        w= w.reshape((-1,1))
        # 1-step bellman error
        target_t = rewards.detach().flatten() + torch.tensor(self.gamma) * torch.mm(nsa_features,
                                                                                    w).flatten()
        # Q = phi^w
        prediction_t = torch.mm(sa_features, w).flatten()

        # v_error = Phi w - r - \gamma Phi_p w
        bellman_error = (prediction_t - target_t).reshape(-1, 1)

        # print("bellman_error", torch.mean(torch.abs(bellman_error)))
        # v_error = ||Phi^T (Phi w   - \gamm Phi_p w)||_2
        v_error = torch.sum(torch.square(torch.mm(sa_features.t(), bellman_error)))
        # Compute mean of v_error and add l2 regularization


        n = self.dataloader.states.shape[0]
        n1 = nsa_features.shape[0]

        # train_error = torch.mean(torch.square(bellman_error)) #+self.lamda * torch.sum(w ** 2)
        train_error = torch.mean(torch.square(bellman_error)) + self.lamda * torch.sum(w ** 2) * (n1/n)

        return train_error, states, w, v_error

    def compute_initial_returns(self, weights):
        """
        computes returns using q-value weights
        :param weights: q-value weights
        :return:
        """
        if weights is None:
            weights = self.weights_q.reshape(-1, 1)
        else:
            if isinstance(weights, np.ndarray) is True:
                weights = torch.tensor(weights,requires_grad=True)
            # Get features for initial state
        total_error, weights, states = self.get_test_error(weights=weights)
        wdr_returns = self.sign * total_error

        return wdr_returns

    def get_consistent_wdr_returns(self, policy_beh, v_weights, states=None, debug=False,
                                   next_states=None):
        """
        computes consistent wdr returns
        :param policy_beh: behavior policy
        :param v_weights: value-function weights
        :param states:
        :param debug:
        :param next_states:
        :return:
        """
        num_traj = len(self.dataloader.traj_set.trajectories)
        # if states is none, get state features
        if states is None:
            states = self.dataloader.get_state_features(requires_grad=True)
        # if next states is none, get next-state features
        if next_states is None:
            next_states = self.dataloader.get_next_state_features(requires_grad=False)
        actions = torch.tensor(self.dataloader.actions).long()

        action_values = []
        next_action_values = []
        n = states.shape[0]
        v_weights = v_weights.reshape((-1,1))
        # for each action, compute state-action and next-state action values
        for action_id in range(self.config.action_size):
            action_arr = torch.ones(n) * action_id
            # get s-a features for given action and state
            sa_features = self.dataloader.construct_state_action_features(states, action_arr)
            # get nsa features for given action and next-state
            nsa_features = self.dataloader.construct_state_action_features(next_states, action_arr)
            # get action-values
            value = torch.mm(sa_features, v_weights)
            action_values.append(value)
            #get next-action value for given nsa features
            next_value = torch.mm(nsa_features, v_weights)
            next_action_values.append(next_value)
        action_values = torch.stack(action_values, dim=1).squeeze()
        next_action_values = torch.stack(next_action_values, dim=1).squeeze()

        if isinstance(states, np.ndarray):
            states = torch.tensor(states)
        n = states.shape[0]
        d = states.shape[1]



        # get eval probabilities
        eval_probabilities = self.policy_eval.predict_proba(states, transformed=self.config.transformed,normalized=self.config.normalized,scaler=self.dataloader.scaler)
        # # compute log of eval probabilities
        # log_eval_probs = torch.log(eval_probabilities+ 1e-10)
        # # compute gumbel of log of eval probabilities
        # gumbel_probs = torch.nn.functional.gumbel_softmax(log_eval_probs, tau=0.001, hard=True, eps=1e-10, dim=- 1)
        # # extract gumbel probabilities of selected actions
        # eval_probs = gumbel_probs[
        #     np.arange(n), actions.flatten()]
        # predict behavior probabilities
        beh_probs = policy_beh.predict_proba(states)[
            np.arange(n), actions.flatten()]
        # get eval probabilities
        probabilities = eval_probabilities
        eval_probs = eval_probabilities[
            np.arange(n), actions.flatten()]
        # value_function = eval_probabilties * action_values
        value_fn = torch.sum(probabilities*action_values,dim=1)

        index = 0
        rewards = torch.tensor(self.dataloader.rewards)
        traj_contributions = torch.zeros((num_traj, int(self.dataloader.max_traj_length+1)))
        weights = torch.zeros((num_traj, int(self.dataloader.max_traj_length+1)))
        val=0.0
        ind=0 # count no of transitions
        for i_traj in range(num_traj):
            transitions = self.dataloader.traj_set.trajectories[i_traj].transitions
            l = len(transitions)
            tmp = torch.ones(1)[0]
            is_weight = torch.ones(1)[0]
            discount = torch.ones(1)[0]

            for n in range(l,0,-1):
                # state = transitions[n].state
                if n<= 0:
                    continue
                n = int(n)

                # update is weights by multiplying with pi_e(s,a)/pi_b(s,a)
                is_weight = is_weight * eval_probs[ind+ n-1]
                is_weight = is_weight / (beh_probs[ind + n-1] + self.clipped)
                cur_weight = eval_probs[ind + n-1]/ (beh_probs[ind + n-1] + self.clipped)
                # v(i_traj, l-n+1) = v(ind+n-1) + rho_n *(r(ind+n-1) + \gamma v(i_traj, l-n)- q(ind+n-1,a))
                traj_contributions[i_traj, l-n+1] = value_fn[ind + n-1] + cur_weight * (
                        rewards[ind + n-1] + self.gamma *  traj_contributions[i_traj, l-n] - action_values[ind+ n-1, actions[ind+ n-1]])
                weights[i_traj, l-n+1] = cur_weight
                # update discount factor
                discount = discount * self.gamma
                index = index + 1
            ind = ind + l
            val = val + traj_contributions[i_traj,l]
        # We are not normalizing weights
        traj_cont = traj_contributions / (torch.sum(weights, 0).reshape(1, -1) + 1e-10)
        self.is_weights = weights
        val = val/(num_traj + 1e-20)

        if debug == True:
            num_near_zero = torch.sum(eval_probs <= 0.05)
            print("sum probs", torch.sum(beh_probs))
            print("policy weights ", np.sum(np.abs(self.weights)))
            print("num_near_zero", num_near_zero)
            print("is weights sum", torch.sum(weights))
            print("val", val)

        return val, weights, states


    def get_consistent_weight_dr_returns(self, policy_beh, v_weights, states=None, debug=False,
                                   next_states=None):
        """
        computes consistent weighted doubly robust retusn
        :param policy_beh: behavior policy
        :param v_weights: value-function weights
        :param states:
        :param debug:
        :param next_states:
        :return:
        """
        # get no of trajectories
        num_traj = len(self.dataloader.traj_set.trajectories)

        # get state features
        if states is None:
            states = self.dataloader.get_state_features(requires_grad=True)

        # get next-state features
        if next_states is None:
            next_states = self.dataloader.get_next_state_features(requires_grad=False)
        actions = torch.tensor(self.dataloader.actions).long()

        action_values = []
        next_action_values = []
        n = states.shape[0]
        # for each action, construct state-action feature and next-state action feature matrices
        for action_id in range(self.config.action_size):
            action_arr = torch.ones(n) * action_id
            sa_features = self.dataloader.construct_state_action_features(states, action_arr)
            nsa_features = self.dataloader.construct_state_action_features(next_states, action_arr)
            value = torch.mm(sa_features, v_weights)
            action_values.append(value)
            next_value = torch.mm(nsa_features, v_weights)
            next_action_values.append(next_value)
        # state-action values
        action_values = torch.stack(action_values, dim=1).squeeze()
        # next state-action values
        next_action_values = torch.stack(next_action_values, dim=1).squeeze()

        if isinstance(states, np.ndarray):
            states = torch.tensor(states)
        n = states.shape[0]
        d = states.shape[1]

        # Rescale back before passing to evaluation policy
        # compute next state probabilities
        eval_next_probs = self.policy_eval.predict_proba(next_states, transformed=self.config.transformed,normalized=self.config.normalized,scaler=self.dataloader.scaler)

        # # compute log of next state probabilities
        # log_eval_next_probs = torch.log(eval_next_probs+ 1e-10)
        # # get gumbel of log of next state probabilities
        # gumbel_next_probs = torch.nn.functional.gumbel_softmax(log_eval_next_probs, tau=0.001, hard=True, eps=1e-10, dim=- 1)
        #
        # eval_next_probs = gumbel_next_probs

        # next values = eval_next_probs * next_action values
        next_values = torch.sum(eval_next_probs * next_action_values, dim=1)


        # get eval probabilities for current state
        eval_probabilities = self.policy_eval.predict_proba(states, transformed=self.config.transformed,normalized=self.config.normalized,scaler=self.dataloader.scaler)
        # compute log of eval probabilities
        # log_eval_probs = torch.log(eval_probabilities+ 1e-10)
        # # compute gumbel of log of eval probabilities
        # gumbel_probs = torch.nn.functional.gumbel_softmax(log_eval_probs, tau=0.001, hard=True, eps=1e-10, dim=- 1)
        # # extract probabilities from gumbel probabilities
        eval_probs = eval_probabilities[
            np.arange(n), actions.flatten()]

        # get behavior probabilities
        beh_probs = policy_beh.predict_proba(states)[
            np.arange(n), actions.flatten()]

        index = 0
        rewards = torch.tensor(self.dataloader.rewards)
        traj_contributions = torch.zeros((num_traj, int(self.dataloader.max_traj_length)))
        weights = torch.zeros((num_traj, int(self.dataloader.max_traj_length)))
        for i_traj in range(num_traj):
            transitions = self.dataloader.traj_set.trajectories[i_traj].transitions
            l = len(transitions)
            tmp = torch.ones(1)[0]
            is_weight = torch.ones(1)[0]
            discount = torch.ones(1)[0]

            for n in range(l):
                # multiply is weight by pi_e(s,a)/pi_b(s,a)
                is_weight = is_weight * eval_probs[index]
                is_weight = is_weight / (beh_probs[index] + self.clipped)
                # trajectory cont = \gamma * is_weight * (r(s,a) + \gamma v(s) - q(s,a))
                traj_contributions[i_traj, n] = discount * is_weight * (
                        rewards[index] + self.gamma * next_values[index] - action_values[index, actions[index]])
                weights[i_traj, n] = is_weight
                # update discoutn factor
                discount = discount * self.gamma
                index = index + 1
        traj_cont = traj_contributions / (torch.sum(weights, 0).reshape(1, -1)+ 1e-10)
        val = torch.sum(traj_cont)

        self.is_weights = weights
        if debug == True:
            num_near_zero = torch.sum(eval_probs <= 0.05)
            print("sum probs", torch.sum(beh_probs))
            print("policy weights ", np.sum(np.abs(self.weights)))
            print("num_near_zero", num_near_zero)
            print("is weights sum", torch.sum(weights))
            print("val", val)

        returns = self.get_returns(v_weights)
        val = val + returns
        return val, weights, states

    def get_returns(self, weights):
        # Get features for initial state
        init_features = self.dataloader.init_state_action_features
        # Compute returns   = Phi w
        if isinstance(init_features, np.ndarray):
            init_features = torch.tensor(init_features).double()
        # error = sign * init_features * weights
        returns = torch.mm(init_features, weights.reshape(-1, 1))
        return torch.mean(returns)


    def get_test_error(self, weights=None,states=None):
        """
        Computes error on test data
        :param weights: weights of q-value function . If weights=None, self.weights is used
        :return: test error, weights
        """
        # if weights is None:
        #     # print(self.weights)
        #     weights = torch.tensor(self.weights.astype(np.float64), requires_grad=grad).reshape(-1, 1)
        # else:
        #     if isinstance(weights, np.ndarray):
        #         weights = torch.tensor(weights.astype(np.float64), requires_grad=False).reshape(-1, 1)
        #     else:
        if weights is None:
            weights = self.weights_q.reshape(-1, 1)
        else:
            if isinstance(weights, np.ndarray) is True:
                weights = torch.tensor(weights,requires_grad=True)

        weights= weights.reshape(-1, 1)


        # sign =-1 if we want to decrease the value function and +1 if we want to increase the value function
        wdr_returns, wt, states = self.is_operator(self.policy_beh, v_weights=weights,states=states)
        # error = self.sign * (returns + wdr_returns)
        error = self.sign * wdr_returns
        total_error = error
        print("actual_test_error", error)
        print("wdr returns", wdr_returns)
        # return error and weights
        return total_error, weights, states

